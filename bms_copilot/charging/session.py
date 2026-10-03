"""Charging-session physics shared by every policy (legacy, MILP, RL).

One step = 15 minutes. Each step:
1. the policy requests a charge C-rate,
2. the deterministic safety supervisor clamps it (JEITA, CV headroom, ceilings),
3. cell temperature, SoC, grid energy and losses update,
4. capacity damage is computed from the same semi-empirical ageing model that
   ages the fleet (calendar + cycling + plating), so the optimisers' costs are
   grounded in the physics rather than hand-tuned penalties.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..config import CELL, PACK
from ..physics.aging import AgingModel, DailyStress
from ..physics.ecm import resistance_temp_factor
from ..physics.electrode import DegradationState, electrode_model
from ..safety.envelope import SafetySupervisor

DT_H = 0.25
CHARGER_EFF = 0.90
EUR_PER_CAPACITY = 650.0  # value of cell capacity: ~EUR 6.5 per 1 % (pack ~EUR 130 per 20 %)
UNMET_EUR_PER_UNIT = 2000.0  # EUR 20 per 1 % SoC short of the user's need at departure
CARBON_EUR_PER_KG = 0.10  # shadow carbon price
R_DC_BOL = CELL.r0_bol_ohm + CELL.r1_bol_ohm + CELL.r2_bol_ohm
SEI_Z_FLOOR = 4e-4  # linearise calendar ageing away from the sqrt singularity


def tou_price(hour) -> np.ndarray:
    """Time-of-use tariff [EUR/kWh]: off-peak 22-07, peak 17-21."""
    h = np.mod(np.asarray(hour, float), 24)
    return np.where((h >= 22) | (h < 7), 0.15, np.where((h >= 17) & (h < 21), 0.38, 0.26))


def grid_carbon(hour) -> np.ndarray:
    """Illustrative grid carbon intensity [gCO2/kWh]: solar dip midday, evening peak."""
    h = np.mod(np.asarray(hour, float), 24)
    return 300 - 110 * np.exp(-(((h - 13) / 3.0) ** 2)) + 90 * np.exp(-(((h - 19) / 2.0) ** 2))


class CellModelSet:
    """A small set of cell ageing states with pre-computed OCV tables."""

    def __init__(self, states: list[DegradationState]):
        em = electrode_model()
        self.states = states
        tabs = [em.ocv_table(s, n=121) for s in states]
        self.soc_tab = np.array([t[0] for t in tabs])
        self.ocv_tab = np.array([t[1] for t in tabs])
        self.capacity = np.array([t[2] for t in tabs])
        self.r_factor = np.array([float(s.resistance_factor()) for s in states])
        self.soh = self.capacity / CELL.rated_capacity_ah
        self.state_arr = DegradationState(
            **{
                k: np.array([float(getattr(s, k)) for s in states])
                for k in ("sei_z", "lli_cycle", "lli_plating", "lam_pe", "lam_ne", "r_extra")
            }
        )
        self.state_arr.sei_z = np.maximum(self.state_arr.sei_z, SEI_Z_FLOOR)
        # Capacity sensitivity to each mode (fraction of rated per unit mode).
        base = em.window(0.05, 0.02, 0.02).capacity_ah
        h = 1e-3
        self.sens = (
            np.array(
                [
                    (base - em.window(0.05 + h, 0.02, 0.02).capacity_ah) / h,
                    (base - em.window(0.05, 0.02 + h, 0.02).capacity_ah) / h,
                    (base - em.window(0.05, 0.02, 0.02 + h).capacity_ah) / h,
                ]
            )
            / CELL.rated_capacity_ah
        )

    def ocv(self, idx: np.ndarray, soc: np.ndarray) -> np.ndarray:
        out = np.empty_like(soc, dtype=float)
        for k in np.unique(idx):
            m = idx == k
            out[m] = np.interp(soc[m], self.soc_tab[k], self.ocv_tab[k])
        return out


class ChargePhysics:
    def __init__(self, models: CellModelSet, supervisor: SafetySupervisor | None = None):
        self.m = models
        self.sup = supervisor or SafetySupervisor()
        self.aging = AgingModel()

    def step(self, idx, soc, temp_prev, amb, c_req):
        m = self.m
        r_dc = R_DC_BOL * m.r_factor[idx] * resistance_temp_factor(temp_prev)
        ocv = m.ocv(idx, soc)
        c = self.sup.gate_charge_array(c_req, temp_prev, ocv, r_dc, CELL.rated_capacity_ah)
        current = c * CELL.rated_capacity_ah
        heat = current**2 * r_dc
        temp = amb + 3.0 + heat / CELL.ha_w_per_k
        soc_new = np.minimum(soc + current * DT_H / m.capacity[idx], 1.0)
        mid = 0.5 * (soc + soc_new)
        days = DT_H / 24.0
        st = m.state_arr.take(idx)
        stress = DailyStress(
            efc=(current * DT_H / CELL.rated_capacity_ah / 2.0) / days,
            mean_soc=mid,
            frac_high_soc=(mid > 0.95).astype(float),
            charge_c=c,
            temp_c=temp,
            charge_temp_c=temp,
            dod=0.7,
        )
        nxt = self.aging.step(st, stress, days=days)
        d_modes = np.stack([nxt.lli - st.lli, nxt.lam_pe - st.lam_pe, nxt.lam_ne - st.lam_ne])
        damage = (m.sens[:, None] * d_modes).sum(0)
        n_cells = PACK.series * PACK.parallel
        e_in_wh = n_cells * (ocv + current * r_dc) * current * DT_H
        loss_wh = n_cells * heat * DT_H
        return soc_new, temp, c, damage, loss_wh, e_in_wh / CHARGER_EFF / 1000.0


@dataclass
class ChargeContext:
    soc0: float
    hours: float  # until the predicted unplug / departure
    target_soc: float  # energy the user will need
    start_hour: float = 22.0  # clock time at plug-in
    ambient_c: float = 24.0
    state: DegradationState = field(default_factory=DegradationState)
    mode: str = "balanced"  # balanced | max_life | ready_asap
    reason: str = ""

    @property
    def n_steps(self) -> int:
        return max(1, int(np.ceil(self.hours / DT_H - 1e-9)))

    def clock(self) -> np.ndarray:
        return self.start_hour + DT_H * np.arange(self.n_steps)

    def to_dict(self) -> dict:
        return {
            "soc0": self.soc0,
            "hours": self.hours,
            "target_soc": self.target_soc,
            "start_hour": self.start_hour,
            "ambient_c": self.ambient_c,
            "mode": self.mode,
            "reason": self.reason,
        }


Policy = Callable[[int, float, float, "ChargeContext"], float]  # (step, soc, temp) -> C-rate


def rollout(ctx: ChargeContext, policy: Policy, physics: ChargePhysics | None = None) -> dict:
    physics = physics or ChargePhysics(CellModelSet([ctx.state]))
    idx = np.zeros(1, dtype=int)
    soc, temp = np.array([ctx.soc0]), np.array([ctx.ambient_c + 3.0])
    clock = ctx.clock()
    rec = {k: [] for k in ("soc", "temp", "c_req", "c", "damage", "loss_wh", "grid_kwh")}
    for t in range(ctx.n_steps):
        req = float(policy(t, float(soc[0]), float(temp[0]), ctx))
        soc, temp, c, dmg, loss, kwh = physics.step(idx, soc, temp, np.array([ctx.ambient_c]), np.array([req]))
        for k, v in (
            ("soc", soc),
            ("temp", temp),
            ("c_req", [req]),
            ("c", c),
            ("damage", dmg),
            ("loss_wh", loss),
            ("grid_kwh", kwh),
        ):
            rec[k].append(float(v[0]))
    return summarize(ctx, {k: np.array(v) for k, v in rec.items()}, clock)


def summarize(ctx: ChargeContext, r: dict, clock: np.ndarray) -> dict:
    soc_path = np.r_[ctx.soc0, r["soc"]]
    reached = np.nonzero(r["soc"] >= ctx.target_soc - 1e-3)[0]
    shortfall = max(0.0, ctx.target_soc - r["soc"][-1])
    damage = float(r["damage"].sum())
    energy_eur = float((r["grid_kwh"] * tou_price(clock)).sum())
    carbon_g = float((r["grid_kwh"] * grid_carbon(clock)).sum())
    return {
        "final_soc": float(r["soc"][-1]),
        "met_target": bool(shortfall <= 0.01),
        "shortfall_pct": 100 * shortfall,
        "ready_after_h": float((reached[0] + 1) * DT_H) if len(reached) else None,
        "damage_pct_capacity": 100 * damage,
        "damage_eur": EUR_PER_CAPACITY * damage,
        "energy_eur": energy_eur,
        "carbon_g": carbon_g,
        "loss_wh": float(r["loss_wh"].sum()),
        "hours_above_95": float(DT_H * np.sum(r["soc"] > 0.95)),
        "peak_temp_c": float(r["temp"].max()),
        "total_cost_eur": EUR_PER_CAPACITY * damage
        + energy_eur
        + CARBON_EUR_PER_KG * carbon_g / 1000
        + UNMET_EUR_PER_UNIT * shortfall,
        "trace": {
            "clock": np.round(np.r_[clock, clock[-1] + DT_H], 2).tolist(),
            "soc": np.round(soc_path, 4).tolist(),
            "c": np.round(r["c"], 3).tolist(),
            "c_req": np.round(r["c_req"], 3).tolist(),
            "temp": np.round(r["temp"], 2).tolist(),
        },
    }


def legacy_policy(fast: bool = False) -> Policy:
    """Legacy behaviour: charge at full adapter rate to 100 % immediately, then float."""
    rate = 1.0 if fast else 0.7
    return lambda t, soc, temp, ctx: rate


def plan_policy(plan: np.ndarray) -> Policy:
    plan = np.asarray(plan, float)
    return lambda t, soc, temp, ctx: float(plan[t]) if t < len(plan) else 0.0


GUARD_REF_C = 0.5  # rate the guard assumes is always available
GUARD_BUFFER_STEPS = 2  # extra steps reserved for the CV taper near full


def deadline_guard_c(soc: float, hours_left: float, target: float, capacity_ah: float) -> float:
    """Minimum C-rate needed now so the target is still reachable in time.

    Deterministic executor rule (like the safety layer, but for availability):
    learned policies may charge later/slower to save the battery, but never so
    late that the user's energy need can no longer be met at a moderate rate.
    """
    need = target - soc
    if need <= 1e-4:
        return 0.0
    steps_left = hours_left / DT_H
    steps_needed = need * capacity_ah / (GUARD_REF_C * CELL.rated_capacity_ah * DT_H) + GUARD_BUFFER_STEPS
    if steps_left > steps_needed:
        return 0.0
    required = need * capacity_ah / (CELL.rated_capacity_ah * max(hours_left, DT_H))
    return float(min(1.2, max(GUARD_REF_C, 1.3 * required)))


def with_deadline_guard(policy: Policy, capacity_ah: float) -> Policy:
    def guarded(t, soc, temp, ctx: ChargeContext):
        left = ctx.n_steps * DT_H - t * DT_H
        return max(float(policy(t, soc, temp, ctx)), deadline_guard_c(soc, left, ctx.target_soc, capacity_ah))

    return guarded
