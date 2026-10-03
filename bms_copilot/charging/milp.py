"""Mixed-Integer Linear Programming charge scheduler.

Decision variables per 15-min step t (horizon = until predicted unplug):
    p1_t in [0, 0.5]       charge C-rate, gentle segment
    p2_t in [0, cmax-0.5]  charge C-rate, fast segment (costlier per unit)
    b_t  in {0, 1}         charger on/off (minimum current when on -> integer)
    s_t                    state of charge at the end of step t
    h1_t, h2_t >= 0        SoC above 80 % / 95 % (calendar-stress dwell)
    u >= 0                 unmet energy at departure (soft, heavily penalised)
    g_t >= 0               gap to target at each step ("ready ASAP" mode only)
    f >= 0                 shortfall below the reserve floor after the first hour
                           (hedges against an earlier-than-predicted unplug)

Objective (EUR): linearised capacity damage (cycling segments + SoC dwell) +
time-of-use electricity + carbon shadow price + unmet-demand penalty.
Constraints: SoC dynamics, on/off coupling, CV-phase taper (linear in SoC),
JEITA thermal derating via cmax_t, departure target.

The piecewise-linear cost coefficients are *fitted from the ageing model* at
this device's state and forecast temperature, so the MILP optimises the same
physics the evaluation (and the fleet) uses. Solver: HiGHS via scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from ..config import CELL, PACK
from ..physics.ecm import resistance_temp_factor
from .session import (
    CARBON_EUR_PER_KG,
    CHARGER_EFF,
    DT_H,
    EUR_PER_CAPACITY,
    R_DC_BOL,
    UNMET_EUR_PER_UNIT,
    CellModelSet,
    ChargeContext,
    ChargePhysics,
    grid_carbon,
    tou_price,
)

C_MIN_ON = 0.10
ADAPTER_MAX_C = 1.2
RESERVE_FLOOR = 0.50  # hold at least min(target, 50 %) after the first hour
FLOOR_EUR_PER_UNIT = UNMET_EUR_PER_UNIT / 4
MODE_WEIGHTS = {"balanced": (1.0, 0.0), "max_life": (2.0, 0.0), "ready_asap": (1.0, 0.25)}


def _cost_coefficients(ctx: ChargeContext, physics: ChargePhysics, cmax: float) -> dict:
    idx = np.zeros(1, dtype=int)
    amb = np.array([ctx.ambient_c])
    t0 = amb + 3.0

    def dmg(soc, c):
        _, _, _, d, _, _ = physics.step(idx, np.array([soc]), t0, amb, np.array([c]))
        return float(d[0])

    cal_mid = dmg(0.5, 0.0)
    f_half = dmg(0.5, 0.5) - cal_mid
    f_max = dmg(0.5, cmax) - cal_mid
    slope1 = f_half / 0.5
    slope2 = max(slope1, (f_max - f_half) / max(cmax - 0.5, 1e-6))
    g = {s: dmg(s, 0.0) for s in (0.5, 0.8, 0.95, 1.0)}
    a = max(0.0, (g[0.8] - g[0.5]) / 0.3)
    b1 = max(0.0, (g[0.95] - g[0.8]) / 0.15 - a)
    b2 = max(0.0, (g[1.0] - g[0.95]) / 0.05 - a - b1)
    return {"slope1": slope1, "slope2": slope2, "dwell_a": a, "dwell_b1": b1, "dwell_b2": b2}


def _cv_lines(ctx: ChargeContext, models: CellModelSet, cmax: float) -> list[tuple[float, float]]:
    """Chord constraints c <= alpha_i - beta_i * s for the constant-voltage phase.

    The CV headroom (v_max - OCV(s)) / (R * Q) is wavy (graphite staging), so
    chords are anchored where it first drops below cmax and only kept if they
    do not restrict charging below that point. Residual approximation error is
    caught by `plan_and_verify` (safety-gated rollout + re-plan / top-up).
    """
    s = np.linspace(0.3, 1.0, 141)
    ocv = np.interp(s, models.soc_tab[0], models.ocv_tab[0])
    r = R_DC_BOL * models.r_factor[0] * resistance_temp_factor(ctx.ambient_c + 3.0)
    c_cv = np.maximum(0.0, (CELL.v_max - ocv) / (r * CELL.rated_capacity_ah))
    below = np.nonzero(c_cv < cmax)[0]
    if len(below) == 0:
        return []
    k0 = below[0]
    s0, c0 = s[k0], min(cmax, c_cv[k0])
    lines = []
    for s1 in (s0 + 0.5 * (1.0 - s0), 1.0):
        c1 = float(np.interp(s1, s, c_cv))
        if s1 - s0 < 1e-3:
            continue
        beta = (c0 - c1) / (s1 - s0)
        alpha = c0 + beta * s0
        if beta > 0 and alpha - beta * s0 >= cmax - 1e-6:
            lines.append((float(alpha), float(beta)))
    return lines


def solve_milp(
    ctx: ChargeContext, physics: ChargePhysics | None = None, time_limit_s: float = 5.0, lead_steps: int = 0
) -> dict:
    """lead_steps: require the target that many 15-min steps before unplug."""
    models = CellModelSet([ctx.state])
    physics = physics or ChargePhysics(models)
    T = ctx.n_steps
    eff = physics.sup.effective_limits()
    jeita = physics.sup.jeita_limit(ctx.ambient_c + 3.0, eff["charge_temp_max_c"])
    # Quasi-steady cell temperature amb + 3 + I^2 R / hA must stay below the
    # charge-temperature limit, or the safety gate will stop the next step.
    r_hot = R_DC_BOL * models.r_factor[0] * resistance_temp_factor(eff["charge_temp_max_c"])
    headroom = max(0.0, eff["charge_temp_max_c"] - 0.5 - ctx.ambient_c - 3.0)
    c_thermal = np.sqrt(headroom * CELL.ha_w_per_k / r_hot) / CELL.rated_capacity_ah
    cmax = float(min(ADAPTER_MAX_C, eff["max_charge_c"], jeita, c_thermal))
    w_dmg, w_asap = MODE_WEIGHTS.get(ctx.mode, (1.0, 0.0))
    coef = _cost_coefficients(ctx, physics, max(cmax, 0.51))
    cv_lines = _cv_lines(ctx, models, cmax)
    k = CELL.rated_capacity_ah * DT_H / float(models.capacity[0])
    clock = ctx.clock()
    kwh_per_c = PACK.series * 3.9 * CELL.rated_capacity_ah * DT_H / CHARGER_EFF / 1000.0
    energy_eur = kwh_per_c * (tou_price(clock) + CARBON_EUR_PER_KG * grid_carbon(clock) / 1000.0)
    target = min(1.0, ctx.target_soc + 0.005)

    # Variable layout
    P1, P2, B, S, H1, H2, G = (np.arange(T) + j * T for j in range(7))
    U = 7 * T
    F = 7 * T + 1
    nv = 7 * T + 2
    c = np.zeros(nv)
    c[P1] = w_dmg * EUR_PER_CAPACITY * coef["slope1"] + energy_eur
    c[P2] = w_dmg * EUR_PER_CAPACITY * coef["slope2"] + energy_eur
    c[S] = w_dmg * EUR_PER_CAPACITY * coef["dwell_a"]
    c[H1] = w_dmg * EUR_PER_CAPACITY * coef["dwell_b1"]
    c[H2] = w_dmg * EUR_PER_CAPACITY * coef["dwell_b2"]
    c[B] = 1e-5
    c[G] = w_asap
    c[U] = UNMET_EUR_PER_UNIT
    c[F] = FLOOR_EUR_PER_UNIT

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[P1] = min(0.5, cmax)
    ub[P2] = max(0.0, cmax - 0.5)
    ub[B] = 1.0
    ub[S] = 1.0
    if w_asap == 0:
        ub[G] = 0.0
    integrality = np.zeros(nv)
    integrality[B] = 1

    lo_b, hi_b = [], []
    A = lil_matrix(((7 + len(cv_lines)) * T + 1, nv))
    floor = min(ctx.target_soc, RESERVE_FLOOR)
    r = 0

    def add(entries, lo, hi):
        nonlocal r
        for j, v in entries:
            A[r, j] = v
        lo_b.append(lo)
        hi_b.append(hi)
        r += 1

    for t in range(T):
        prev = [] if t == 0 else [(S[t - 1], -1.0)]
        s_prev_const = ctx.soc0 if t == 0 else 0.0
        add([(S[t], 1.0), (P1[t], -k), (P2[t], -k)] + prev, s_prev_const, s_prev_const)  # dynamics
        add([(P1[t], 1.0), (P2[t], 1.0), (B[t], -cmax)], -np.inf, 0.0)  # on/off max
        add([(P1[t], 1.0), (P2[t], 1.0), (B[t], -C_MIN_ON)], 0.0, np.inf)  # min current
        for alpha, beta in cv_lines:  # CV taper
            cv_prev = [] if t == 0 else [(S[t - 1], beta)]
            add([(P1[t], 1.0), (P2[t], 1.0)] + cv_prev, -np.inf, alpha - (beta * ctx.soc0 if t == 0 else 0.0))
        add([(S[t], 1.0), (H1[t], -1.0)], -np.inf, 0.80)
        add([(S[t], 1.0), (H2[t], -1.0)], -np.inf, 0.95)
        if w_asap:
            add([(G[t], 1.0), (S[t], 1.0)], target, np.inf)
        if (t + 1) * DT_H >= 1.0 and floor > ctx.soc0:
            add([(S[t], 1.0), (F, 1.0)], floor, np.inf)
    add([(S[max(0, T - 1 - lead_steps)], 1.0), (U, 1.0)], target, np.inf)

    A = A[:r].tocsr()
    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=[LinearConstraint(A, lo_b, hi_b)],
        options={"time_limit": time_limit_s, "mip_rel_gap": 1e-4},
    )
    if res.x is None:
        return {"status": "infeasible", "message": res.message, "plan": np.zeros(T)}
    x = res.x
    plan = np.clip(x[P1] + x[P2], 0.0, None) * (x[B] > 0.5)
    return {
        "status": "optimal" if res.status == 0 else res.message,
        "plan": plan,
        "objective_eur": float(res.fun),
        "coefficients": coef,
        "cmax": cmax,
        "c_thermal_limit": float(c_thermal),
        "cv_lines": [{"alpha": a, "beta": b} for a, b in cv_lines],
        "n_variables": int(nv),
        "n_binary": int(T),
        "n_constraints": int(r),
    }


def plan_and_verify(ctx: ChargeContext, physics: ChargePhysics | None = None, max_iter: int = 6) -> dict:
    """Solve, verify the plan on the twin's safety-gated physics, re-plan if short.

    The MILP's linear CV-taper approximation can over-estimate the current the
    safety gate will allow near full charge; any resulting shortfall is added
    to the internal target and the problem is re-solved.
    """
    from dataclasses import replace

    from .session import plan_policy, rollout

    physics = physics or ChargePhysics(CellModelSet([ctx.state]))
    bump, lead = 0.0, 0
    result, best = None, None
    max_lead = max(0, ctx.n_steps // 3)
    for it in range(max_iter):
        trial = replace(ctx, target_soc=min(1.0, ctx.target_soc + bump))
        sol = solve_milp(trial, physics, lead_steps=lead)
        sim = rollout(ctx, plan_policy(sol["plan"]), physics)
        result = {
            **sol,
            "verification": {k: v for k, v in sim.items() if k != "trace"},
            "replans": it,
            "lead_steps": lead,
            "trace": sim["trace"],
        }
        short = ctx.target_soc - sim["final_soc"]
        if short > 0.002:
            # Gentle top-up in the idle steps before unplug, verified again.
            plan = np.array(sol["plan"], float)
            soc_path = np.array(sim["trace"]["soc"][1:])
            active = np.nonzero(plan > 0)[0]
            start = active[-1] + 1 if len(active) else 0
            for t in range(start, len(plan)):
                if soc_path[t - 1 if t else 0] < ctx.target_soc:
                    plan[t] = 0.2
            sim2 = rollout(ctx, plan_policy(plan), physics)
            if sim2["final_soc"] > sim["final_soc"]:
                sol = {**sol, "plan": plan}
                sim = sim2
                result = {
                    **sol,
                    "verification": {k: v for k, v in sim.items() if k != "trace"},
                    "replans": it,
                    "lead_steps": lead,
                    "trace": sim["trace"],
                    "topped_up": True,
                }
                short = ctx.target_soc - sim["final_soc"]
        key = (round(max(short, 0.0), 4), sim["total_cost_eur"])
        if best is None or key < best[0]:
            best = (key, result)
        if short <= 0.002:
            break
        if trial.target_soc < 1.0:
            bump += short + 0.01  # aim higher
        elif lead < max_lead:
            lead += 1  # already aiming for full: finish earlier (CV taper time)
        else:
            break
    return best[1]
