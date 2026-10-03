"""Charging optimisers, agent autonomy gating, grading and the copilot loop."""

import json
from datetime import UTC
from types import SimpleNamespace

import numpy as np
import pytest

from bms_copilot.agents.workforce import Workforce
from bms_copilot.charging.milp import plan_and_verify
from bms_copilot.charging.session import ChargeContext, legacy_policy, plan_policy, rollout
from bms_copilot.cloud.grading import grade_battery
from bms_copilot.cloud.twin import state_from_modes
from bms_copilot.copilot import assistant


# ------------------------------------------------------------------ charging
@pytest.mark.parametrize(
    "ctx",
    [
        ChargeContext(0.25, 9.0, 0.85, 22.0, 24.0, state_from_modes(0.06, 0.02, 0.015)),
        ChargeContext(0.25, 9.0, 1.00, 22.0, 24.0, state_from_modes(0.06, 0.02, 0.015)),
        ChargeContext(0.40, 6.0, 0.90, 20.0, 36.0, state_from_modes(0.10, 0.03, 0.02)),
    ],
)
def test_milp_meets_need_with_less_damage_than_legacy(ctx):
    plan = plan_and_verify(ctx)
    assert np.all(np.asarray(plan["plan"]) >= 0)
    milp = rollout(ctx, plan_policy(plan["plan"]))
    legacy = rollout(ctx, legacy_policy())
    assert milp["met_target"]
    assert milp["damage_pct_capacity"] < 0.8 * legacy["damage_pct_capacity"]
    assert milp["peak_temp_c"] < 45.0


def test_charging_respects_tightened_envelope():
    from bms_copilot.charging.session import CellModelSet, ChargePhysics
    from bms_copilot.safety.envelope import SafetySupervisor, Tightening

    ctx = ChargeContext(0.2, 3.0, 1.0, 22.0, 24.0, state_from_modes(0.05, 0.02, 0.02))
    sup = SafetySupervisor()
    sup.request_tightening(Tightening("test", "derate", max_charge_c=0.3))
    r = rollout(ctx, lambda t, soc, temp, c: 1.5, ChargePhysics(CellModelSet([ctx.state]), sup))
    assert max(r["trace"]["c"]) <= 0.3 + 1e-9


# -------------------------------------------------------------------- agents
class FakeEngine:
    def __init__(self, consent):
        from datetime import datetime

        self.now = datetime(2026, 9, 26, tzinfo=UTC)
        self._consent = consent
        self.effects = []

    def device(self, did):
        return SimpleNamespace(l5_consent=self._consent)

    def __getattr__(self, name):
        if name.startswith("effect_"):
            return lambda a: self.effects.append(a.kind) or {"ok": True}
        raise AttributeError(name)


@pytest.mark.parametrize(
    "level,kind,consent,expected",
    [
        (1, "enable_adaptive_charging", False, "recommended"),
        (2, "enable_adaptive_charging", False, "pending_approval"),
        (3, "enable_adaptive_charging", False, "executed"),
        (3, "warranty_claim", False, "pending_approval"),
        (4, "warranty_claim", False, "executed"),
        (4, "replacement_order", True, "awaiting_user"),
        (5, "replacement_order", False, "awaiting_user"),
        (5, "replacement_order", True, "executed"),
    ],
)
def test_autonomy_gating(level, kind, consent, expected):
    eng = FakeEngine(consent)
    wf = Workforce(eng, level=level)
    a = wf.propose("TestAgent", "DEV-0001", kind, "t", "r", {}, needs_user_consent=kind == "replacement_order")
    assert a.status == expected
    assert (kind in eng.effects) == (expected == "executed")
    assert wf.propose("TestAgent", "DEV-0001", kind, "t", "r", {}, needs_user_consent=kind == "replacement_order") is a


def test_raising_autonomy_promotes_instead_of_duplicating():
    eng = FakeEngine(False)
    wf = Workforce(eng, level=1)
    a = wf.propose("ServiceAgent", "DEV-0001", "enable_adaptive_charging", "t", "r", {})
    assert a.status == "recommended"
    wf.level = 3
    b = wf.propose("ServiceAgent", "DEV-0001", "enable_adaptive_charging", "t", "r", {})
    assert b is a and a.status == "executed" and eng.effects == ["enable_adaptive_charging"]
    assert len(wf.actions) == 1


def test_human_approval_executes_effect():
    eng = FakeEngine(False)
    wf = Workforce(eng, level=3)
    a = wf.propose("ServiceAgent", "DEV-0001", "warranty_claim", "claim", "r", {})
    assert a.status == "pending_approval" and not eng.effects
    wf.decide(a.id, True, "engineer")
    assert a.status == "executed" and eng.effects == ["warranty_claim"]
    with pytest.raises(ValueError):
        wf.decide(a.id, True, "engineer")


# ------------------------------------------------------------------- grading
def test_grading_safety_gate_and_healthy_pack():
    st = state_from_modes(0.05, 0.01, 0.01)
    hist = {
        "overtemp_hours": 0,
        "deep_discharges": 0,
        "temp_mean": 26,
        "frac_high_mean": 0.1,
        "self_discharge_pct_day": 0.04,
    }
    assert grade_battery(0.92, 700, 1200, 10, hist, "critical", st).grade == "F"
    g = grade_battery(0.92, 700, 1200, 10, hist, "normal", st)
    assert g.grade == "A"
    assert g.second_life_days is not None or any("horizon" in r for r in g.reasons)
    assert grade_battery(0.72, 60, 150, 70, {**hist, "overtemp_hours": 300, "temp_mean": 38}, "watch", st).grade in "BC"


# ------------------------------------------------------------------- copilot
class _Block(SimpleNamespace):
    pass


class FakeClaude:
    """Scripted client: first asks for a tool, then answers with its result."""

    def __init__(self):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[_Block(type="tool_use", id="tu_1", name="get_fleet_overview", input={})],
            )
        result = kw["messages"][-1]["content"][0]
        assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_1"
        n = json.loads(result["content"])["kpis"]["devices"]
        return SimpleNamespace(stop_reason="end_turn", content=[_Block(type="text", text=f"{n} devices")])


class StubEngine:
    def fleet_overview(self):
        return {
            "now": "t",
            "workforce": {},
            "kpis": {
                "devices": 7,
                "mean_soh": 0.9,
                "below_85pct": 1,
                "below_eol": 0,
                "rul_under_180d": 0,
                "anomalies": {"critical": 0, "warning": 1, "watch": 2},
                "adaptive_share": 0.4,
                "soh_mae_ai": 0.003,
                "soh_mae_legacy_gauge": 0.05,
            },
        }


def test_copilot_tool_loop_with_fake_client(monkeypatch):
    monkeypatch.setenv("BMS_COPILOT_LLM", "on")
    cp = assistant.Copilot(StubEngine())
    import anthropic

    cp._anthropic = anthropic
    cp.client = FakeClaude()
    out = cp.chat("how many devices?")
    assert out["answer"] == "7 devices"
    assert [t["tool"] for t in out["trace"]] == ["get_fleet_overview"]
    first = cp.client.calls[0]
    assert first["model"] == assistant.MODEL and first["fallbacks"] == "default"
    assert {t["name"] for t in first["tools"]} >= {"get_device_health", "plan_charging", "get_battery_passport"}


def test_copilot_falls_back_offline_without_credentials(monkeypatch):
    monkeypatch.setenv("BMS_COPILOT_LLM", "on")
    cp = assistant.Copilot(StubEngine())

    class NoCreds:
        beta = SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: (_ for _ in ()).throw(TypeError('"Could not resolve authentication method."'))
            )
        )

    import anthropic

    cp._anthropic = anthropic
    cp.client = NoCreds()
    out = cp.chat("fleet overview please")
    assert out["mode"].startswith("offline") and "no Claude credentials" in out["mode"]
    assert out["answer"].startswith("Fleet of 7 devices")
    assert cp.client is None
