"""Fast unit tests for the physics, estimation and security building blocks."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy.optimize import check_grad

from bms_copilot.cloud.dma import fit_modes
from bms_copilot.cloud.gpr import GaussianProcess
from bms_copilot.cloud.ota import DeviceVerifier, PackageSigner, load_or_create_keys
from bms_copilot.cloud.passport import PassportStore, render
from bms_copilot.config import CELL, SAFETY
from bms_copilot.edge.ica import incremental_capacity, savgol
from bms_copilot.physics.electrode import electrode_model
from bms_copilot.safety.envelope import SafetySupervisor, Tightening
from bms_copilot.sim.sensors import slow_charge_capture


# ------------------------------------------------------------------ physics
def test_fresh_cell_has_rated_capacity():
    assert float(electrode_model().window(0, 0, 0).capacity_ah) == pytest.approx(CELL.rated_capacity_ah, rel=1e-4)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_every_degradation_mode_reduces_capacity(mode):
    em = electrode_model()
    base = float(em.window(0.02, 0.01, 0.01).capacity_ah)
    args = [0.02, 0.01, 0.01]
    args[mode] += 0.05
    assert float(em.window(*args).capacity_ah) < base


def test_ocv_is_monotonic_in_soc():
    from bms_copilot.physics.electrode import DegradationState

    soc, ocv, _ = electrode_model().ocv_table(DegradationState(sei_z=0.05**2, lam_pe=0.02, lam_ne=0.02))
    assert np.all(np.diff(ocv) > 0)
    assert ocv[0] == pytest.approx(CELL.v_min, abs=0.01) and ocv[-1] == pytest.approx(CELL.v_max, abs=0.01)


# ---------------------------------------------------------------------- ICA
def test_savgol_preserves_cubic():
    x = np.linspace(-1, 1, 101)
    y = 2 * x**3 - x + 0.5
    assert np.allclose(savgol(y, 21, 3)[15:-15], y[15:-15], atol=1e-9)


def test_ica_features_repeatable_and_sensitive_to_lli():
    rng = np.random.default_rng(0)
    lli = np.array([0.0, 0.0, 0.10])
    v, dq = slow_charge_capture(lli, np.zeros(3), np.zeros(3), np.ones(3), np.full(3, 25.0), rng)
    f = incremental_capacity(v, dq).features
    area_b1 = f[:, 2]
    assert abs(area_b1[0] - area_b1[1]) < 0.01  # noise repeatability (Ah)
    assert area_b1[2] < area_b1[0] - 0.1  # LLI shrinks the low-voltage band


# ---------------------------------------------------------------------- GPR
def test_gpr_gradient_and_calibrated_interval():
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, (250, 2))
    y = np.sin(X[:, 0]) * 2 + rng.normal(0, 0.2, 250)
    gp = GaussianProcess(n_restarts=2)
    gp.X_, gp.y_ = (X - X.mean(0)) / X.std(0), (y - y.mean()) / y.std()
    th = np.array([0.1, 0.4, 0.1, -1.5])
    assert check_grad(lambda t: gp._nll(t)[0], lambda t: gp._nll(t)[1], th) < 1e-3
    gp.fit(X, y)
    Xt = rng.uniform(-3, 3, (1500, 2))
    yt = np.sin(Xt[:, 0]) * 2 + rng.normal(0, 0.2, 1500)
    m, s = gp.predict(Xt)
    assert np.sqrt(np.mean((m - yt) ** 2)) < 0.25
    assert 0.90 <= np.mean(np.abs(m - yt) <= 1.96 * s) <= 0.99


# ---------------------------------------------------------------------- DMA
@pytest.mark.parametrize("modes", [(0.08, 0.02, 0.01), (0.05, 0.01, 0.12), (0.06, 0.10, 0.02)])
def test_dma_recovers_modes(modes):
    rng = np.random.default_rng(3)
    lli, pe, ne = modes
    v, dq = slow_charge_capture(
        np.array([lli]),
        np.array([pe]),
        np.array([ne]),
        np.array([1 + 2.8 * lli + 1.6 * (pe + ne)]),
        np.array([25.0]),
        rng,
        c_rate=0.05,
    )
    r = fit_modes(v[0], float(dq[0]))
    assert (r.lli, r.lam_pe, r.lam_ne) == pytest.approx(modes, abs=0.01)


# ------------------------------------------------------------------- safety
def test_gate_never_exceeds_envelope_fuzz():
    sup = SafetySupervisor()
    rng = np.random.default_rng(7)
    for _ in range(3000):
        req = float(rng.uniform(-1, 5))
        temp = float(rng.uniform(-10, 70))
        ocv = float(rng.uniform(3.0, 4.3))
        g = sup.gate_charge(req, temp, ocv, 0.06)
        assert 0.0 <= g.allowed_c <= SAFETY.max_charge_c_rate
        assert g.allowed_c <= sup.jeita_limit(temp) + 1e-12
        if temp < SAFETY.charge_temp_min_c or temp >= SAFETY.charge_temp_max_c or ocv >= SAFETY.cell_v_max:
            assert g.allowed_c == 0.0
        arr = sup.gate_charge_array(np.array([req]), np.array([temp]), np.array([ocv]), np.array([0.06]))[0]
        assert arr == pytest.approx(g.allowed_c)


def test_ai_can_only_tighten():
    sup = SafetySupervisor()
    assert not sup.request_tightening(Tightening("ai", "try to loosen", v_max=4.35))
    assert not sup.request_tightening(Tightening("ai", "try to loosen", max_charge_c=3.0))
    assert sup.effective_limits()["v_max"] == SAFETY.cell_v_max
    assert sup.request_tightening(Tightening("ai", "swelling", v_max=4.10, max_charge_c=0.5))
    assert sup.effective_limits()["v_max"] == 4.10
    assert not sup.request_tightening(Tightening("ai", "undo", v_max=4.15))  # cannot relax a tightening
    assert sup.gate_charge(1.5, 25.0).allowed_c == 0.5


def test_hardware_faults_trip_independently():
    faults = {f["fault"] for f in SafetySupervisor().check_faults([4.30, 4.1, 4.1], 70.0, 0.5)}
    assert {"overvoltage", "overtemperature", "cell_imbalance"} <= faults


# ---------------------------------------------------------------------- OTA
def test_ota_signature_binding_rollback_expiry(tmp_path):
    priv, pub, kid = load_or_create_keys(tmp_path)
    signer = PackageSigner(priv, kid)
    now = datetime(2026, 9, 26, tzinfo=UTC)
    pkg = signer.sign("calibration", {"capacity_ah": 4.1}, version=3, device_id="DEV-0001", issued_at=now)
    dev = DeviceVerifier(pub, "DEV-0001", installed={"calibration": 2})
    assert dev.verify(pkg, now)[0]
    bad = {**pkg, "payload": {"capacity_ah": 9.9}}
    assert dev.verify(bad, now) == (False, "signature invalid")
    assert not DeviceVerifier(pub, "DEV-0002").verify(pkg, now)[0]
    assert not dev.verify(pkg, now + timedelta(days=31))[0]
    assert dev.install(pkg, now)[0]
    assert "rollback" in dev.verify(pkg, now)[1]


# ----------------------------------------------------------------- passport
def test_passport_access_tiers_and_hash_chain():
    rec = {
        "battery_identifier": "B1",
        "updated_at": "t0",
        "a": {
            "pub": {"value": 1, "access": "public"},
            "li": {"value": 2, "access": "legitimate_interest"},
            "auth": {"value": 3, "access": "authority"},
        },
    }
    assert set(render(rec, "public")["a"]) == {"pub"}
    assert set(render(rec, "legitimate_interest")["a"]) == {"pub", "li"}
    assert set(render(rec, "authority")["a"]) == {"pub", "li", "auth"}
    store = PassportStore()
    store.publish(rec)
    rec2 = {**rec, "a": {**rec["a"], "pub": {"value": 5, "access": "public"}}, "updated_at": "t1"}
    e2 = store.publish(rec2)
    assert e2["version"] == 2 and store.verify_chain("B1")
    store.versions["B1"][0]["record"]["a"]["pub"]["value"] = 99  # tamper with history
    assert not store.verify_chain("B1")
