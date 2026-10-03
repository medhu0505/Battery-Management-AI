"""The agentic workforce: prescriptive service with agentic triage.

Agents
  PredictiveAnalyticsAgent  forecast health/RUL, detect risks (per device)
  TriageAgent               root cause (diagnostic capture + degradation-mode
                            analysis), warranty eligibility, confidence
  ServiceAgent              prescribe: safety derate, adaptive charging, OTA
                            recalibration, warranty claim, replacement, offer
  LogisticsAgent            shipments, technician slots, returns
  AssetRecoveryAgent        second-life grading, pathway, passport status
  DesignFeedbackAgent       fleet-level insights for next-gen design (L5 loop)

Autonomy ladder
  L1 Tool               estimates only; every action is logged as a recommendation
  L2 Assistant          recommends; engineers approve policy/safety changes
  L3 Supervised agent   executes adaptive charging / derating / OTA inside the
                        certified envelope
  L4 Autonomous agent   also initiates warranty claims; user confirms replacements
  L5 Agentic workforce  with pre-approved consent, orders replacements before
                        failure; fleet loop feeds design

Gating: an action whose required level <= current level executes; one level
short -> pending human approval; further short -> logged recommendation.
Replacement orders always need user consent (pre-approved at L5, else asked).
AI never overrides the safety layer: derates are *tightenings* the
deterministic supervisor may accept or reject.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from datetime import timedelta

import numpy as np

from ..config import SERVICE

LEVELS = {
    1: "L1 Tool - ML estimation; actions logged as recommendations",
    2: "L2 Assistant - anomaly detection; engineers approve recommended changes",
    3: "L3 Supervised agent - executes adaptive charging, derating and OTA tuning inside the certified envelope",
    4: "L4 Autonomous agent - also initiates warranty claims; user confirms replacements",
    5: "L5 Agentic workforce - pre-approved replacements before failure; fleet loop informs design",
}
REQUIRED_LEVEL = {
    "notify_user": 2,
    "enable_adaptive_charging": 3,
    "safety_derate": 3,
    "diagnostic_capture": 3,
    "ota_recalibration": 3,
    "warranty_claim": 4,
    "proactive_replacement_offer": 4,
    "replacement_order": 4,
    "logistics_shipment": 4,
    "asset_recovery": 4,
    "design_insight": 5,
}
OPEN = {"executed", "pending_approval", "awaiting_user", "in_progress", "recommended"}
_RANK = {"recommended": 0, "pending_approval": 1, "awaiting_user": 2, "executed": 3}


@dataclass
class AgentAction:
    id: str
    created_at: str
    agent: str
    device_id: str | None
    kind: str
    title: str
    rationale: str
    evidence: dict
    required_level: int
    status: str
    params: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Finding:
    code: str
    detail: str
    severity: str  # info | warning | critical


class Workforce:
    def __init__(self, engine, level: int = 3, actions: list | None = None):
        self.engine = engine
        self.level = level
        self.actions: list[AgentAction] = [AgentAction(**a) for a in (actions or [])]
        self._ids = itertools.count(len(self.actions) + 1)
        self.predictive = PredictiveAnalyticsAgent()
        self.triage = TriageAgent()
        self.service = ServiceAgent()
        self.logistics = LogisticsAgent()
        self.recovery = AssetRecoveryAgent()

    # ------------------------------------------------------------- plumbing
    def _open(self, device_id, kind, key=None) -> AgentAction | None:
        for a in self.actions:
            if a.device_id == device_id and a.kind == kind and a.status in OPEN and a.params.get("dedupe_key") == key:
                return a
        return None

    def propose(
        self,
        agent: str,
        device_id,
        kind: str,
        title: str,
        rationale: str,
        evidence: dict,
        params: dict | None = None,
        needs_user_consent: bool = False,
    ) -> AgentAction:
        params = dict(params or {})
        key = params.get("lot") or params.get("topic")
        params["dedupe_key"] = key
        req = REQUIRED_LEVEL[kind]
        if self.level >= req:
            status = "executed"
            if needs_user_consent:
                consent = device_id is not None and self.engine.device(device_id).l5_consent
                status = "executed" if (self.level >= 5 and consent) else "awaiting_user"
        elif self.level == req - 1:
            status = "pending_approval"
        else:
            status = "recommended"
        existing = self._open(device_id, kind, key)
        if existing:
            # A higher autonomy level promotes an earlier recommendation / pending
            # item instead of creating a duplicate.
            if existing.status in ("recommended", "pending_approval") and _RANK[status] > _RANK[existing.status]:
                existing.status = status
                existing.history.append(
                    {"at": self.engine.now.isoformat(), "status": status, "by": f"{agent} @ L{self.level} (promoted)"}
                )
                if status == "executed":
                    self._execute(existing)
            return existing
        a = AgentAction(
            id=f"A{next(self._ids):05d}",
            created_at=self.engine.now.isoformat(),
            agent=agent,
            device_id=device_id,
            kind=kind,
            title=title,
            rationale=rationale,
            evidence=evidence,
            required_level=req,
            status=status,
            params=params,
            history=[{"at": self.engine.now.isoformat(), "status": status, "by": f"{agent} @ L{self.level}"}],
        )
        self.actions.append(a)
        if status == "executed":
            self._execute(a)
        return a

    def _execute(self, a: AgentAction) -> None:
        handler = getattr(self.engine, f"effect_{a.kind}", None)
        a.result = handler(a) if handler else {"note": "no side effect"}
        if a.kind == "replacement_order":
            self.logistics.dispatch(self, a)

    def decide(self, action_id: str, approve: bool, actor: str) -> AgentAction:
        a = next(x for x in self.actions if x.id == action_id)
        if a.status not in ("pending_approval", "awaiting_user", "recommended"):
            raise ValueError(f"action {action_id} is {a.status}")
        a.status = "executed" if approve else "rejected"
        a.history.append({"at": self.engine.now.isoformat(), "status": a.status, "by": actor})
        if approve:
            self._execute(a)
        return a

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        before = len(self.actions)
        self.logistics.advance(self)  # time passes: deliveries, returns
        self.recovery.process_returns(self)
        flagged = 0
        for dev in self.engine.devices:
            h = self.engine.health(dev.device_id)
            findings = self.predictive.assess(h)
            if not findings:
                continue
            flagged += 1
            diag = self.triage.diagnose(self, dev, h, findings)
            self.service.prescribe(self, dev, h, findings, diag)
        DesignFeedbackAgent().analyse(self)
        new = self.actions[before:]
        return {
            "level": self.level,
            "devices_flagged": flagged,
            "new_actions": len(new),
            "by_status": {s: sum(a.status == s for a in new) for s in sorted({a.status for a in new})},
        }


# --------------------------------------------------------------------------
class PredictiveAnalyticsAgent:
    name = "PredictiveAnalyticsAgent"

    def assess(self, h: dict) -> list[Finding]:
        f = []
        rul = h["rul"]
        levels = h["anomaly_recent_levels"]
        if h["soh"] <= SERVICE.eol_soh:
            f.append(
                Finding("below_eol", f"SoH {h['soh']:.1%} at or below end-of-life {SERVICE.eol_soh:.0%}", "warning")
            )
        elif rul["lo_days"] <= 30 or rul["median_days"] <= SERVICE.proactive_replacement_horizon_days:
            f.append(
                Finding(
                    "rul_short", f"RUL {rul['median_days']} d (95 %: {rul['lo_days']}-{rul['hi_days']} d)", "warning"
                )
            )
        if h["soh_z"] <= -SERVICE.defect_z_threshold:
            f.append(
                Finding(
                    "accelerated_fade",
                    f"SoH {h['soh']:.1%} vs expected {h['expected_soh']:.1%} for age and usage (z = {h['soh_z']:.1f})",
                    "warning",
                )
            )
        crit = levels[-1:] == ["critical"]
        persistent = sum(lv in ("warning", "critical") for lv in levels[-3:]) >= 2
        kind = h["anomaly"].get("kind")
        if (crit or persistent) and kind == "safety_precursor":
            f.append(Finding("safety_precursor", h["anomaly"]["mechanism"] or "precursor", "critical"))
        elif crit or persistent:
            f.append(Finding("cell_imbalance", f"{levels[-1]}: {h['anomaly']['mechanism']}", "warning"))
        elif levels and levels[-1] in ("watch", "warning"):
            f.append(Finding("anomaly_watch", f"{levels[-1]}: {h['anomaly']['mechanism']}", "info"))
        if h.get("lam_ne_excess_z", 0.0) >= 3.0 and "accelerated_fade" not in {x.code for x in f}:
            f.append(
                Finding(
                    "anode_lam_excess",
                    f"routine-ICA anode LAM {h['modes_est']['lam_ne']:.3f} is "
                    f"{h['lam_ne_excess_z']:.1f} sigma above usage expectation",
                    "info",
                )
            )
        u = h["usage"]
        if not h["adaptive"] and (u["frac_high_mean"] > 0.5 or u["temp_mean"] > 33 or u["charge_c_mean"] > 0.9):
            f.append(
                Finding(
                    "usage_stress",
                    f"time >95 % SoC {u['frac_high_mean']:.0%}, mean cell temp "
                    f"{u['temp_mean']:.1f} C, charge rate {u['charge_c_mean']:.2f}C",
                    "info",
                )
            )
        return f


class TriageAgent:
    name = "TriageAgent"

    def diagnose(self, wf: Workforce, dev, h: dict, findings: list[Finding]) -> dict:
        codes = {x.code for x in findings}
        diag = {"cause": "normal_ageing", "confidence": 0.6, "modes": None, "in_warranty": h["warranty_days_left"] > 0}
        wants_dma = bool(
            codes
            & {"accelerated_fade", "rul_short", "below_eol", "anomaly_watch", "cell_imbalance", "anode_lam_excess"}
        )
        if wants_dma:
            act = wf.propose(
                self.name,
                dev.device_id,
                "diagnostic_capture",
                "Run on-device diagnostic slow-charge capture during idle window",
                "Full-range C/20 capture enables degradation-mode analysis to separate normal ageing "
                "from a defect before any claim is raised.",
                {"findings": [x.code for x in findings]},
            )
            if act.status == "executed":
                diag["modes"] = act.result.get("modes")
        m = diag["modes"]
        if "safety_precursor" in codes:
            diag.update(cause="safety_precursor", confidence=0.9, mechanism=h["anomaly"]["mechanism"])
        elif codes & {"accelerated_fade", "cell_imbalance", "anode_lam_excess"}:
            anode_dma = bool(m and (m["dominant"] == "lam_ne" or m["lam_ne"] > 0.06))
            anode_fleet = h.get("lam_ne_excess_z", 0.0) >= 3.0
            if anode_dma or anode_fleet:
                diag.update(
                    cause="manufacturing_defect",
                    confidence=0.9 if (anode_dma and anode_fleet) else 0.75,
                    mechanism="anode active-material loss far above usage expectation "
                    f"({'DMA' if anode_dma else ''}{' + ' if anode_dma and anode_fleet else ''}"
                    f"{'routine-ICA fleet model' if anode_fleet else ''})",
                )
            elif "usage_stress" in codes and "accelerated_fade" in codes:
                diag.update(cause="usage_induced", confidence=0.7)
            elif "accelerated_fade" in codes:
                diag.update(cause="accelerated_ageing_unexplained", confidence=0.5 if not m else 0.65)
            else:
                diag.update(cause="imbalance_from_ageing", confidence=0.6)
        elif "usage_stress" in codes and codes & {"rul_short", "below_eol"}:
            diag.update(cause="usage_induced", confidence=0.7)
        diag["warrantable"] = bool(
            diag["in_warranty"]
            and (diag["cause"] in ("manufacturing_defect", "safety_precursor") or h["soh"] < SERVICE.warranted_soh)
        )
        diag["findings"] = [asdict(x) for x in findings]
        return diag


class ServiceAgent:
    name = "ServiceAgent"

    def prescribe(self, wf: Workforce, dev, h: dict, findings: list[Finding], diag: dict) -> None:
        did = dev.device_id
        codes = {x.code for x in findings}
        ev = {
            "soh": round(h["soh"], 4),
            "rul": h["rul"],
            "cause": diag["cause"],
            "confidence": diag["confidence"],
            "modes": diag.get("modes"),
            "warranty_days_left": h["warranty_days_left"],
        }
        if diag["cause"] == "safety_precursor":
            wf.propose(
                self.name,
                did,
                "safety_derate",
                "Tighten charge envelope (4.10 V, 0.5C, 40 C)",
                f"Precursor: {diag.get('mechanism')}. Reduce stress until the pack is replaced. The "
                "deterministic supervisor accepts tightenings only.",
                ev,
                {"v_max": 4.10, "max_charge_c": 0.5, "charge_temp_max_c": 40.0},
            )
            wf.propose(
                self.name,
                did,
                "notify_user",
                "Safety notice sent to user",
                "User informed that charging is limited for safety and a free replacement is arranged.",
                ev,
                {
                    "message": "Your battery shows early signs of swelling or an internal fault. We've limited "
                    "charging for safety and arranged a free replacement."
                },
            )
            wf.propose(
                self.name,
                did,
                "warranty_claim",
                "Open safety replacement claim (no cost to user)",
                "Safety precursors are covered regardless of warranty status.",
                ev,
                {"claim_type": "safety", "cost_to_user": 0},
            )
            wf.propose(
                self.name,
                did,
                "replacement_order",
                "Ship replacement battery + technician",
                "Replace before the precursor becomes a thermal event.",
                ev,
                {"reason": "safety"},
                needs_user_consent=True,
            )
            return
        if "cell_imbalance" in codes:
            wf.propose(
                self.name,
                did,
                "safety_derate",
                "Mild derate while cells are imbalanced (4.15 V, 0.7C)",
                "Series-cell divergence above the healthy band: the weakest cell reaches its limits first. "
                "Extended balancing + reduced ceiling until diagnosis.",
                ev,
                {"v_max": 4.15, "max_charge_c": 0.7, "charge_temp_max_c": None},
            )
        if diag["warrantable"]:
            wf.propose(
                self.name,
                did,
                "warranty_claim",
                "Initiate warranty claim with diagnostic evidence",
                f"Cause {diag['cause']} inside warranty ({h['warranty_days_left']} days left).",
                ev,
                {"claim_type": diag["cause"], "cost_to_user": 0},
            )
            wf.propose(
                self.name,
                did,
                "replacement_order",
                "Free replacement battery under warranty",
                "Replace proactively before the user notices degraded runtime.",
                ev,
                {"reason": "warranty"},
                needs_user_consent=True,
            )
        elif diag["cause"] == "manufacturing_defect":
            wf.propose(
                self.name,
                did,
                "proactive_replacement_offer",
                "Goodwill replacement: extended service programme for a confirmed cell defect",
                "Defect signature confirmed after the warranty ended; OEM goodwill avoids a failure in the "
                "field and protects the brand.",
                ev,
                {
                    "discount_pct": 100,
                    "programme": "extended service programme",
                    "message": "We detected a known cell quality issue in your battery. It qualifies for a free "
                    "replacement under our extended service programme.",
                },
            )
        elif codes & {"rul_short", "below_eol"}:
            if "below_eol" in codes:
                why = f"Out of warranty; SoH {h['soh']:.0%} is already below {SERVICE.eol_soh:.0%}."
                when = "has dropped below its optimal capacity"
            else:
                months = max(1, round(h["rul"]["median_days"] / 30))
                why = (
                    f"Out of warranty; SoH projected below {SERVICE.eol_soh:.0%} in about "
                    f"{h['rul']['median_days']} days."
                )
                when = f"is projected to fall below optimal in about {months} month{'s' if months > 1 else ''}"
            wf.propose(
                self.name,
                did,
                "proactive_replacement_offer",
                f"Offer proactive replacement with {SERVICE.loyalty_discount_pct} % loyalty discount",
                why,
                ev,
                {
                    "discount_pct": SERVICE.loyalty_discount_pct,
                    "message": f"Your battery health {when}. Schedule a replacement now with a "
                    f"{SERVICE.loyalty_discount_pct} % loyalty discount.",
                },
            )
        if codes & {"usage_stress"} or diag["cause"] == "usage_induced":
            wf.propose(
                self.name,
                did,
                "enable_adaptive_charging",
                "Enable AI adaptive charging policy",
                "Usage keeps the pack hot / at high SoC; the twin projects a longer life with adaptive charging.",
                {**ev, "whatif": h.get("whatif_summary")},
            )
            wf.propose(
                self.name,
                did,
                "notify_user",
                "Explain adaptive charging to user",
                "Transparent notice: charge will be held lower until shortly before you usually unplug.",
                ev,
                {
                    "message": "We'll hold your battery around 80 % and finish charging shortly before you "
                    "usually unplug, which extends battery life."
                },
            )
        if h["soh"] < 0.90 and h.get("calibration_stale", True):
            wf.propose(
                self.name,
                did,
                "ota_recalibration",
                "Push signed recalibration package (SoC/ICA/anomaly)",
                "Capacity and OCV curve have drifted from beginning-of-life tables; refresh on-device "
                "estimator parameters from the twin.",
                ev,
            )


class LogisticsAgent:
    name = "LogisticsAgent"

    def dispatch(self, wf: Workforce, order: AgentAction) -> None:
        eng = wf.engine
        day = eng.now + timedelta(days=2)
        wf.propose(
            self.name,
            order.device_id,
            "logistics_shipment",
            f"Ship replacement + prepaid return kit; technician slot {day:%a %d %b} 10:00",
            "Coordinated with stock; old pack returned for asset recovery.",
            {"order": order.id},
            {
                "tracking": f"TRK{abs(hash(order.id)) % 10**9:09d}",
                "eta": day.date().isoformat(),
                "technician_slot": day.replace(hour=10, minute=0).isoformat(),
                "state": "in_transit",
            },
        )

    def advance(self, wf: Workforce) -> None:
        for a in wf.actions:
            if a.kind == "logistics_shipment" and a.status == "executed" and a.params.get("state") == "in_transit":
                a.params["state"] = "delivered_and_returned"
                a.history.append(
                    {"at": wf.engine.now.isoformat(), "status": "delivered; old pack returned", "by": self.name}
                )
                wf.engine.mark_returned(a.device_id)


class AssetRecoveryAgent:
    name = "AssetRecoveryAgent"

    def process_returns(self, wf: Workforce) -> None:
        for did in list(wf.engine.returned):
            if wf._open(did, "asset_recovery"):
                continue
            grade = wf.engine.second_life(did)
            wf.propose(
                self.name,
                did,
                "asset_recovery",
                f"Returned pack graded {grade['grade']}: {grade['pathway']}",
                "Grade uses the RUL lower bound and safety history from the passport record.",
                {"grade": grade},
                {"grade": grade["grade"]},
            )


class DesignFeedbackAgent:
    name = "DesignFeedbackAgent"

    def analyse(self, wf: Workforce) -> None:
        ins = wf.engine.fleet_insights()
        for lot in ins["lots"]:
            if lot["n"] < 5 or lot["p_value"] >= 0.05:
                continue
            strong = lot["p_value"] < 0.01
            wf.propose(
                self.name,
                None,
                "design_insight",
                (
                    f"Supplier quality: cell lot {lot['lot']} shows excess anode active-material loss"
                    if strong
                    else f"Monitor cell lot {lot['lot']}: early signs of excess anode LAM"
                ),
                f"Routine-ICA mode estimates: {lot['outliers']}/{lot['n']} packs > 3 sigma anode-LAM excess vs "
                f"fleet rate {ins['outlier_rate_fleet']:.1%} (binomial p = {lot['p_outliers']:.1e}); lot mean "
                f"{lot['lam_ne_excess_z_mean']:.1f} sigma (Mann-Whitney p = {lot['p_shift']:.1e}); combined "
                f"p = {lot['p_value']:.1e}. "
                + (
                    "Recommend supplier 8D, incoming inspection and an extended service programme for the lot."
                    if strong
                    else "Increase diagnostic sampling of this lot."
                ),
                {"lot": lot},
                {"lot": lot["lot"]},
            )
        a = ins["adaptive_vs_legacy"]
        if a["legacy_fade_pct_per_100d"] > 0:
            gain = 100 * (1 - a["adaptive_fade_pct_per_100d"] / a["legacy_fade_pct_per_100d"])
            wf.propose(
                self.name,
                None,
                "design_insight",
                f"Adaptive charging cohort fades {gain:.0f} % slower than legacy",
                "Evidence for making adaptive charging the platform default and for right-sizing the next-gen pack.",
                a,
                {"topic": "adaptive_default"},
            )
        top = ins["stress_drivers"][0] if ins["stress_drivers"] else None
        if top:
            wf.propose(
                self.name,
                None,
                "design_insight",
                f"Largest ageing driver: {top['driver']}",
                "Fleet regression of fade rate on usage stress; informs thermal design of next chassis.",
                {"drivers": ins["stress_drivers"]},
                {"topic": "thermal"},
            )


def recent_levels(results: list) -> list[str]:
    return [r.level for r in results[-6:]]


def fade_rate(days: np.ndarray, soh: np.ndarray) -> float:
    if len(days) < 3:
        return 0.0
    return float(-np.polyfit(days, soh, 1)[0] * 1e4)
