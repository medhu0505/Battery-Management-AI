"""Layer 5 - hybrid battery digital twin.

Physics core   : 1RC equivalent-circuit model, re-parameterised per device from
                 fleet data (capacity from the SoH GPR, OCV-SoC table from the
                 degradation-mode fit, R0 from DCIR telemetry).
Data-driven    : a residual model (ridge regression on engineered features)
                 trained on fleet data predicts the error between the ECM and
                 measured voltage - hysteresis, the unmodelled second RC branch,
                 temperature dependence.
What-if engine : projects SoH forward with the semi-empirical ageing model
                 under alternative charging policies for this device's usage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CELL, SERVICE
from ..edge.soc import SoCCalibration, SoCEKF, coulomb_count
from ..physics.aging import AgingModel, DailyStress
from ..physics.ecm import CellSimulator, ECMParams, drive_cycle
from ..physics.electrode import DegradationState, electrode_model
from ..sim.profiles import adaptive_transform

_RES_FEATURES = ["i", "abs_i", "i_soc", "soc", "soc2", "dT", "i_dT", "v_rc", "hyst", "one"]


# Share of fitted LLI attributed to calendar SEI (sqrt law) vs cycling when a
# state is rebuilt from degradation modes (the fit cannot separate them).
CALENDAR_LLI_SHARE = 0.55


def state_from_modes(lli: float, lam_pe: float, lam_ne: float, r_factor: float | None = None) -> DegradationState:
    st = DegradationState(
        sei_z=(CALENDAR_LLI_SHARE * float(lli)) ** 2,
        lli_cycle=(1 - CALENDAR_LLI_SHARE) * float(lli),
        lam_pe=float(lam_pe),
        lam_ne=float(lam_ne),
    )
    if r_factor is not None:
        st.r_extra = max(0.0, float(r_factor) - float(st.resistance_factor()))
    return st


def modes_from_soh(soh: float) -> tuple[float, float, float]:
    """Fallback split when no diagnostic capture exists: LLI-dominated ageing."""
    lo, hi = 0.0, 0.5
    em = electrode_model()
    for _ in range(40):
        m = 0.5 * (lo + hi)
        cap = float(em.window(m, 0.2 * m, 0.12 * m).capacity_ah) / CELL.rated_capacity_ah
        lo, hi = (m, hi) if cap > soh else (lo, m)
    m = 0.5 * (lo + hi)
    return m, 0.2 * m, 0.12 * m


def soc_calibration(state: DegradationState, version: str) -> SoCCalibration:
    soc_t, ocv_t, cap = electrode_model().ocv_table(state, n=101)
    p = ECMParams.for_state(state, 25.0)
    return SoCCalibration(cap, soc_t, ocv_t, p.r0, p.r1 + p.r2 * 0.3, CELL.c1_bol_f, version)


def bol_calibration() -> SoCCalibration:
    return soc_calibration(DegradationState(), "soc-cal-bol")


# ---------------------------------------------------------------------------
# ECM core + residual model
# ---------------------------------------------------------------------------
def ecm_predict(cal: SoCCalibration, current_a: np.ndarray, dt_s: float, soc0: float):
    """Open-loop 1RC voltage prediction; returns (v, soc, v_rc)."""
    a = np.exp(-dt_s / (cal.r1 * cal.c1))
    soc = coulomb_count(current_a, dt_s, soc0, cal.capacity_ah)
    v_rc = np.zeros_like(current_a)
    v = 0.0
    for k, i in enumerate(current_a):
        v = a * v + cal.r1 * (1 - a) * i
        v_rc[k] = v
    ocv = np.interp(soc, cal.soc_table, cal.ocv_table)
    return ocv - current_a * cal.r0 - v_rc, soc, v_rc


def _residual_features(current_a, soc, temp_c, v_rc, dt_s):
    hyst = np.zeros_like(current_a)
    h = 0.0
    k = 1 - np.exp(-dt_s / 600.0)
    for j, i in enumerate(current_a):
        h += (-np.sign(i) - h) * k
        hyst[j] = h
    dT = temp_c - 25.0
    return np.column_stack(
        [
            current_a,
            np.abs(current_a),
            current_a * soc,
            soc,
            soc**2,
            dT,
            current_a * dT,
            v_rc,
            hyst,
            np.ones_like(current_a),
        ]
    )


@dataclass
class ResidualModel:
    weights: np.ndarray
    lam: float = 1e-3

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> ResidualModel:
        A = X.T @ X + lam * np.eye(X.shape[1])
        return cls(np.linalg.solve(A, X.T @ y), lam)


def _twin_run(state: DegradationState, temp_c: float, seed: int, dt_s: float = 5.0, duration_s: int = 5400):
    """Truth simulation plus the twin's ECM prediction for one drive cycle."""
    cal = soc_calibration(state, "twin")
    cur = drive_cycle(duration_s, dt_s, seed, cal.capacity_ah)
    sim = CellSimulator(state, soc0=0.95, temp0_c=temp_c)
    truth = sim.run(cur, dt_s, temp_c)
    v_ecm, soc, v_rc = ecm_predict(cal, cur, dt_s, 0.95)
    X = _residual_features(cur, soc, truth["temp_c"], v_rc, dt_s)
    return X, truth["v"] - v_ecm


def train_residual_model(n_train: int = 36, n_test: int = 12, seed: int = 0) -> tuple[ResidualModel, dict]:
    rng = np.random.default_rng(seed)

    def batch(n, off):
        Xs, ys = [], []
        for j in range(n):
            st = state_from_modes(rng.uniform(0, 0.15), rng.uniform(0, 0.06), rng.uniform(0, 0.06))
            X, y = _twin_run(st, float(rng.uniform(8, 40)), seed=off + j)
            Xs.append(X)
            ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)

    Xtr, ytr = batch(n_train, 1000)
    Xte, yte = batch(n_test, 5000)
    model = ResidualModel.fit(Xtr, ytr)
    rmse_ecm = float(np.sqrt(np.mean(yte**2)) * 1000)
    rmse_hybrid = float(np.sqrt(np.mean((yte - model.predict(Xte)) ** 2)) * 1000)
    return model, {
        "voltage_rmse_mv_ecm_only": rmse_ecm,
        "voltage_rmse_mv_hybrid": rmse_hybrid,
        "improvement_pct": 100 * (1 - rmse_hybrid / rmse_ecm),
        "test_runs": n_test,
        "features": _RES_FEATURES,
    }


# ---------------------------------------------------------------------------
# SoC benchmark (legacy gauge vs EKF with stale vs recalibrated parameters)
# ---------------------------------------------------------------------------
def soc_benchmark(
    state: DegradationState,
    temp_c: float = 25.0,
    seed: int = 3,
    gauge_capacity_ah: float = CELL.rated_capacity_ah,
    dt_s: float = 2.0,
) -> dict:
    rng = np.random.default_rng(seed)
    sim = CellSimulator(state, soc0=1.0, temp0_c=temp_c)
    cur = drive_cycle(int(3.2 * 3600), dt_s, seed, sim.capacity_ah)
    truth = sim.run(cur, dt_s, temp_c)
    alive = truth["v"] > CELL.v_min
    n = int(np.argmin(alive)) if not alive.all() else len(cur)
    cur, v_true, soc_true = cur[:n], truth["v"][:n], truth["soc"][:n]
    i_meas = cur * 1.004 + rng.normal(0, 0.01, n)  # sense gain + noise
    v_meas = v_true + rng.normal(0, 0.002, n)
    out = {"n_steps": n, "duration_h": round(n * dt_s / 3600, 2)}
    runs = {
        "legacy_coulomb_counting": coulomb_count(i_meas, dt_s, 1.0, gauge_capacity_ah),
        "ekf_bol_parameters": SoCEKF(bol_calibration(), 1.0).run(i_meas, v_meas, dt_s),
        "ekf_recalibrated": SoCEKF(soc_calibration(state, "recal"), 1.0).run(i_meas, v_meas, dt_s),
    }
    for k, est in runs.items():
        err = np.abs(est - soc_true) * 100
        out[k] = {"mae_pct": float(err.mean()), "max_pct": float(err.max()), "final_pct": float(err[-1])}
    step = max(1, n // 120)
    out["trace"] = {
        "t_h": (np.arange(0, n, step) * dt_s / 3600).round(3).tolist(),
        "truth": (soc_true[::step] * 100).round(2).tolist(),
        **{k: (v[::step] * 100).round(2).tolist() for k, v in runs.items()},
    }
    return out


# ---------------------------------------------------------------------------
# What-if projections
# ---------------------------------------------------------------------------
def project_soh(
    state: DegradationState,
    usage: dict,
    days: int = 3 * 365,
    adaptive: bool = False,
    step_days: int = 7,
    eol_soh: float = SERVICE.eol_soh,
) -> dict:
    """SoH trajectory under this device's usage with legacy or adaptive charging."""
    model = AgingModel()
    em = electrode_model()
    soc, high, chg = usage["soc_mean"], usage["frac_high_mean"], usage["charge_c_mean"]
    if adaptive and not usage.get("already_adaptive", False):
        s1 = adaptive_transform(np.array(soc), np.array(high), np.array(chg), np.array(False))
        soc, high, chg = (float(x) for x in s1)
    stress = DailyStress(
        efc=usage["efc_rate"],
        mean_soc=soc,
        frac_high_soc=high,
        charge_c=chg,
        temp_c=usage["temp_mean"],
        charge_temp_c=usage["temp_mean"] + 2 + 5 * chg,
        dod=usage.get("dod", 0.7),
    )
    st = state.copy()
    t, soh = [0], [float(em.capacity(st)) / CELL.rated_capacity_ah]
    eol = None
    for d in range(step_days, days + 1, step_days):
        st = model.step(st, stress, days=step_days)
        s = float(em.capacity(st)) / CELL.rated_capacity_ah
        t.append(d)
        soh.append(s)
        if eol is None and s <= eol_soh:
            eol = d
    return {"days": t, "soh": np.round(soh, 4).tolist(), "eol_in_days": eol, "adaptive": adaptive}
