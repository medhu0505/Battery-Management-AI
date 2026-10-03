"""Layer 2 - deterministic safety supervisor.

Design rule: *AI never overrides the safety layer.* Every
charge command produced by an AI policy (RL, MILP, agents) passes through
`SafetySupervisor.gate_charge`, which clamps it to the certified envelope.
AI components may request *tightenings* (e.g. a lower voltage ceiling on a
cell showing swelling precursors); a request that would loosen any limit is
rejected and recorded. Hardware trips (`check_faults`) are independent of
any AI state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import SAFETY, SafetyEnvelope


@dataclass(frozen=True)
class Tightening:
    """An AI-requested restriction. None = no change for that limit."""

    source: str
    reason: str
    v_max: float | None = None
    max_charge_c: float | None = None
    charge_temp_max_c: float | None = None


@dataclass
class GateResult:
    requested_c: float
    allowed_c: float
    limiting_factor: str
    notes: list[str] = field(default_factory=list)

    @property
    def clamped(self) -> bool:
        return self.allowed_c < self.requested_c - 1e-12


class SafetySupervisor:
    def __init__(self, limits: SafetyEnvelope = SAFETY):
        self.base = limits
        self.tightenings: list[Tightening] = []
        self.rejected: list[tuple[Tightening, str]] = []

    # ------------------------------------------------------------ envelope
    def request_tightening(self, t: Tightening) -> bool:
        """Accept a restriction only if it does not loosen any limit."""
        eff = self.effective_limits()
        problems = []
        if t.v_max is not None and t.v_max > eff["v_max"]:
            problems.append(f"v_max {t.v_max} > {eff['v_max']}")
        if t.max_charge_c is not None and t.max_charge_c > eff["max_charge_c"]:
            problems.append(f"max_charge_c {t.max_charge_c} > {eff['max_charge_c']}")
        if t.charge_temp_max_c is not None and t.charge_temp_max_c > eff["charge_temp_max_c"]:
            problems.append(f"charge_temp_max_c {t.charge_temp_max_c} > {eff['charge_temp_max_c']}")
        if any(v is not None and v < 0 for v in (t.v_max, t.max_charge_c)):
            problems.append("negative limit")
        if problems:
            self.rejected.append((t, "; ".join(problems)))
            return False
        self.tightenings.append(t)
        return True

    def clear_tightenings(self, source: str | None = None) -> None:
        self.tightenings = [t for t in self.tightenings if source is not None and t.source != source]

    def effective_limits(self) -> dict:
        v_max = min([self.base.cell_v_max] + [t.v_max for t in self.tightenings if t.v_max is not None])
        c_max = min(
            [self.base.max_charge_c_rate] + [t.max_charge_c for t in self.tightenings if t.max_charge_c is not None]
        )
        t_max = min(
            [self.base.charge_temp_max_c]
            + [t.charge_temp_max_c for t in self.tightenings if t.charge_temp_max_c is not None]
        )
        return {
            "v_max": v_max,
            "max_charge_c": c_max,
            "charge_temp_max_c": t_max,
            "charge_temp_min_c": self.base.charge_temp_min_c,
        }

    def jeita_limit(self, temp_c: float, charge_temp_max_c: float | None = None) -> float:
        t_max = self.base.charge_temp_max_c if charge_temp_max_c is None else charge_temp_max_c
        if temp_c < self.base.charge_temp_min_c or temp_c >= t_max:
            return 0.0
        for lo, hi, c in self.base.jeita_bands:
            if lo <= temp_c < hi:
                return c
        return 0.0

    # --------------------------------------------------------------- gating
    def gate_charge(
        self,
        requested_c: float,
        cell_temp_c: float,
        ocv_v: float | None = None,
        r_dc_ohm: float | None = None,
        capacity_ah: float = 4.5,
    ) -> GateResult:
        """Clamp a requested charge C-rate to the effective envelope.

        If OCV and DC resistance are given, the current is also limited so the
        terminal voltage stays below the (possibly tightened) v_max - the CV
        phase of a CC-CV charger.
        """
        eff = self.effective_limits()
        req = float(requested_c)
        if not np.isfinite(req) or req < 0:
            return GateResult(req, 0.0, "invalid_request", ["non-finite or negative request"])
        caps = {
            "request": req,
            "abs_max": eff["max_charge_c"],
            "jeita": self.jeita_limit(cell_temp_c, eff["charge_temp_max_c"]),
        }
        if ocv_v is not None and r_dc_ohm is not None:
            headroom = eff["v_max"] - ocv_v
            caps["cv_limit"] = max(0.0, headroom / (r_dc_ohm * capacity_ah))
        factor = min(caps, key=caps.get)
        notes = [f"tightening:{t.source}" for t in self.tightenings]
        return GateResult(req, max(0.0, caps[factor]), factor, notes)

    def gate_charge_array(self, requested_c, cell_temp_c, ocv_v, r_dc_ohm, capacity_ah: float = 4.5) -> np.ndarray:
        """Vectorised `gate_charge` (identical limits) for simulators and RL."""
        eff = self.effective_limits()
        req = np.nan_to_num(np.asarray(requested_c, float), nan=0.0)
        temp = np.asarray(cell_temp_c, float)
        jeita = np.zeros_like(temp)
        for lo, hi, c in self.base.jeita_bands:
            jeita = np.where((temp >= lo) & (temp < hi), c, jeita)
        jeita = np.where((temp < self.base.charge_temp_min_c) | (temp >= eff["charge_temp_max_c"]), 0.0, jeita)
        cv = np.maximum(0.0, (eff["v_max"] - np.asarray(ocv_v)) / (np.asarray(r_dc_ohm) * capacity_ah))
        out = np.minimum(np.minimum(np.maximum(req, 0.0), eff["max_charge_c"]), np.minimum(jeita, cv))
        return np.maximum(out, 0.0)

    # ------------------------------------------------------------- hardware
    def check_faults(self, cell_v, cell_temp_c: float, current_c: float) -> list[dict]:
        """Hardware-level protection. Returns trip actions (independent of AI)."""
        cell_v = np.atleast_1d(cell_v)
        b = self.base
        faults = []
        if np.max(cell_v) >= b.cell_v_overvoltage_trip:
            faults.append({"fault": "overvoltage", "action": "open_charge_fet"})
        if np.min(cell_v) <= b.cell_v_undervoltage_trip:
            faults.append({"fault": "undervoltage", "action": "open_discharge_fet"})
        if cell_temp_c >= b.overtemp_trip_c:
            faults.append({"fault": "overtemperature", "action": "open_all_fets"})
        if current_c < 0 and -current_c > b.max_discharge_c_rate:
            faults.append({"fault": "discharge_overcurrent", "action": "open_discharge_fet"})
        if current_c > b.max_charge_c_rate * 1.1:
            faults.append({"fault": "charge_overcurrent", "action": "open_charge_fet"})
        if (np.max(cell_v) - np.min(cell_v)) * 1000 > b.max_cell_divergence_mv:
            faults.append({"fault": "cell_imbalance", "action": "suspend_charge_and_balance"})
        return faults
