"""Second-life readiness grading (Asset Recovery Agent).

Risk-aware: uses the *lower* bound of the RUL interval, not the median, and
hard safety gates override any score.

    A  premium second life   SoH >= 85 %, gentle history, long RUL lower bound
    B  standard second life  SoH 70-85 %, typical history
    C  limited use           SoH < 70 % or high-stress history -> low-demand use or recycle
    F  recycle               safety precursor, abnormal self-discharge/swelling, very low SoH

Second-life duration is projected with the twin under a gentle stationary
storage profile down to a 60 % second-life end point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CELL
from ..physics.electrode import DegradationState
from .twin import project_soh

SECOND_LIFE_PROFILE = {
    "efc_rate": 0.35,
    "temp_mean": 25.0,
    "soc_mean": 0.55,
    "frac_high_mean": 0.0,
    "charge_c_mean": 0.3,
    "dod": 0.5,
    "already_adaptive": True,
}
SECOND_LIFE_EOL = 0.60
PATHWAYS = {
    "A": "Repurpose: portable power station / UPS module (premium second life)",
    "B": "Repurpose: home or office energy-storage module (standard second life)",
    "C": "Limited use: low-power backup / IoT, or route to recycling if no demand",
    "F": "Recycle immediately via certified recycler (hazard handling required)",
}


@dataclass
class Grade:
    grade: str
    score: float
    pathway: str
    reasons: list
    second_life_days: int | None
    inputs: dict

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "score": round(self.score, 1),
            "pathway": self.pathway,
            "reasons": self.reasons,
            "second_life_days_to_60pct": self.second_life_days,
            "inputs": self.inputs,
        }


def grade_battery(
    soh: float,
    rul_lo_days: float,
    rul_median_days: float,
    r0_growth_pct: float,
    history: dict,
    anomaly_max_level: str,
    state: DegradationState,
) -> Grade:
    reasons = []
    inputs = {
        "soh": round(soh, 4),
        "rul_lower_95_days": round(rul_lo_days),
        "rul_median_days": round(rul_median_days),
        "dcir_growth_pct": round(r0_growth_pct, 1),
        "anomaly_max_level": anomaly_max_level,
        **history,
    }

    # Hard safety gates
    if anomaly_max_level == "critical":
        reasons.append("safety precursor reached CRITICAL (swelling / thermal / micro-short)")
    if history.get("self_discharge_pct_day", 0) > 0.3:
        reasons.append("abnormal self-discharge indicates internal leakage")
    if soh < 0.55:
        reasons.append(f"SoH {soh:.0%} below any useful second-life threshold")
    if reasons:
        return Grade("F", 0.0, PATHWAYS["F"], reasons, None, inputs)

    # Weighted score (0-100)
    s_soh = np.clip((soh - 0.55) / (0.95 - 0.55), 0, 1) * 45
    s_rul = np.clip(rul_lo_days / 730.0, 0, 1) * 20
    s_res = np.clip(1 - r0_growth_pct / 120.0, 0, 1) * 15
    stress = (
        np.clip(history.get("overtemp_hours", 0) / 200.0, 0, 1) * 0.4
        + np.clip((history.get("temp_mean", 28) - 25) / 15.0, 0, 1) * 0.3
        + np.clip(history.get("frac_high_mean", 0.3) / 0.8, 0, 1) * 0.2
        + np.clip(history.get("deep_discharges", 0) / 20.0, 0, 1) * 0.1
    )
    s_hist = (1 - stress) * 15
    s_anom = {"normal": 5, "watch": 2.5, "warning": 0}.get(anomaly_max_level, 0)
    score = float(s_soh + s_rul + s_res + s_hist + s_anom)

    if soh >= 0.85 and score >= 70 and anomaly_max_level == "normal":
        g = "A"
    elif soh >= 0.70 and score >= 50:
        g = "B"
    else:
        g = "C"
    reasons.append(f"SoH {soh:.1%}; RUL lower bound {rul_lo_days:.0f} d (95 % interval) used for risk-aware grading")
    reasons.append(f"DCIR growth {r0_growth_pct:.0f} % vs beginning of life")
    if stress > 0.4:
        reasons.append("high-stress usage history (heat / time at high SoC) lowers the score")
    if anomaly_max_level in ("watch", "warning"):
        reasons.append(f"precursor detector peaked at {anomaly_max_level.upper()} - inspect before reuse")
    reasons.append("chemistry NMC/graphite: good energy density, shorter second life than LFP")

    proj = project_soh(state, SECOND_LIFE_PROFILE, days=8 * 365, step_days=14, eol_soh=SECOND_LIFE_EOL)
    if proj["eol_in_days"] is None:
        reasons.append("projected second life exceeds the 8-year projection horizon (gentle storage profile)")
    return Grade(g, score, PATHWAYS[g], reasons, proj["eol_in_days"], inputs)


def power_capability_pct(r0_now_mohm: float, r0_bol_mohm: float = 1000 * 3 * CELL.r0_bol_ohm) -> float:
    """Remaining power capability ~ R0_bol / R0_now (for a fixed voltage window)."""
    return float(100.0 * min(1.0, r0_bol_mohm / max(r0_now_mohm, 1e-9)))
