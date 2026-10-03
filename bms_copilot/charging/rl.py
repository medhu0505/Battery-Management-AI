"""Reinforcement-learning charge policy (model-free Q-learning).

MDP
  state   : SoC, hours to predicted unplug, ambient temperature, target SoC,
            SoH band, "first hour elapsed" flag (discretised, 4 320 states)
  action  : requested charge C-rate in {0, 0.2, 0.5, 0.8, 1.2}
  reward  : -(capacity damage EUR + energy EUR) each step
            - reserve-floor penalty after the first hour
            - unmet-demand penalty at unplug
  env     : ChargePhysics (safety-gated, same ageing model as the fleet)

Training runs thousands of episodes in parallel (vectorised environment) with
epsilon-greedy exploration; duplicate (state, action) updates within a batch
are averaged. The learned table is the *teacher* for the TinyML student that
runs on the NPU (edge/tinyml.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cloud.twin import modes_from_soh, state_from_modes
from ..config import MODELS
from .session import DT_H, EUR_PER_CAPACITY, UNMET_EUR_PER_UNIT, CellModelSet, ChargeContext, ChargePhysics

ACTIONS = np.array(MODELS.rl_c_rates)
SOC_EDGES = np.array([0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
HOUR_EDGES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
AMB_EDGES = np.array([15.0, 25.0, 32.0])
TARGET_EDGES = np.array([0.85, 0.95])
SOH_EDGES = np.array([0.9])
SOH_LEVELS = np.array([1.0, 0.92, 0.85, 0.78])
DIMS = (len(SOC_EDGES) + 1, len(HOUR_EDGES) + 1, len(AMB_EDGES) + 1, len(TARGET_EDGES) + 1, len(SOH_EDGES) + 1, 2)
N_STATES = int(np.prod(DIMS))
FLOOR = 0.5
FLOOR_EUR_PER_UNIT_STEP = UNMET_EUR_PER_UNIT / 4 * DT_H
FLAT_PRICE = 0.22


def encode(soc, hours_left, amb, target, soh, elapsed_1h) -> np.ndarray:
    parts = (
        np.digitize(soc, SOC_EDGES),
        np.digitize(hours_left - 1e-9, HOUR_EDGES),
        np.digitize(amb, AMB_EDGES),
        np.digitize(target, TARGET_EDGES),
        np.digitize(soh, SOH_EDGES),
        np.asarray(elapsed_1h, int),
    )
    return np.ravel_multi_index(parts, DIMS)


def level_models() -> CellModelSet:
    return CellModelSet(
        [state_from_modes(*modes_from_soh(s)) if s < 1 else state_from_modes(0.02, 0.005, 0.005) for s in SOH_LEVELS]
    )


@dataclass
class QPolicy:
    q: np.ndarray

    def act(self, soc, hours_left, amb, target, soh, elapsed_1h) -> np.ndarray:
        s = encode(soc, hours_left, amb, target, soh, elapsed_1h)
        return ACTIONS[np.argmax(self.q[s], axis=-1)]

    def as_session_policy(self, soh: float):
        def policy(t, soc, temp, ctx: ChargeContext):
            left = ctx.n_steps * DT_H - t * DT_H
            return float(
                self.act(
                    np.array([soc]),
                    np.array([left]),
                    np.array([ctx.ambient_c]),
                    np.array([ctx.target_soc]),
                    np.array([soh]),
                    np.array([t * DT_H >= 1.0]),
                )[0]
            )

        return policy


def _sample(rng, B):
    return {
        "soc": rng.uniform(0.05, 0.75, B),
        "n": np.maximum(1, np.round(rng.uniform(0.5, 11.0, B) / DT_H)).astype(int),
        "target": rng.choice([0.8, 0.9, 1.0], B, p=[0.45, 0.35, 0.2]),
        "amb": rng.uniform(8.0, 38.0, B),
        "level": rng.integers(0, len(SOH_LEVELS), B),
    }


def train_q_policy(
    n_batches: int = 220, batch: int = 4096, alpha: float = 0.25, seed: int = 0, verbose: bool = False
) -> tuple[QPolicy, dict]:
    rng = np.random.default_rng(seed)
    models = level_models()
    phys = ChargePhysics(models)
    soh_of_level = models.soh
    kwh_eur = FLAT_PRICE
    Q = np.zeros((N_STATES, len(ACTIONS)))
    visits = np.zeros(N_STATES, dtype=int)
    history = []
    for it in range(n_batches):
        eps = max(0.05, 1.0 - it / (0.6 * n_batches))
        ep = _sample(rng, batch)
        soc, amb, lvl, tgt, n = ep["soc"], ep["amb"], ep["level"], ep["target"], ep["n"]
        temp = amb + 3.0
        soh = soh_of_level[lvl]
        ret = np.zeros(batch)
        for t in range(int(n.max())):
            act = t < n
            if not act.any():
                break
            left = (n - t) * DT_H
            s = encode(soc, left, amb, tgt, soh, t * DT_H >= 1.0)
            greedy = np.argmax(Q[s], axis=1)
            a = np.where(rng.random(batch) < eps, rng.integers(0, len(ACTIONS), batch), greedy)
            soc2, temp2, c, dmg, loss, kwh = phys.step(lvl, soc, temp, amb, ACTIONS[a])
            r = -(EUR_PER_CAPACITY * dmg + kwh_eur * kwh)
            if (t + 1) * DT_H >= 1.0:
                r -= FLOOR_EUR_PER_UNIT_STEP * np.maximum(0.0, np.minimum(tgt, FLOOR) - soc2)
            terminal = (t + 1) == n
            r = r - np.where(terminal, UNMET_EUR_PER_UNIT * np.maximum(0.0, tgt - soc2), 0.0)
            s2 = encode(soc2, np.maximum(left - DT_H, DT_H), amb, tgt, soh, (t + 1) * DT_H >= 1.0)
            y = r + np.where(terminal, 0.0, Q[s2].max(axis=1))
            m = act
            flat = s[m] * len(ACTIONS) + a[m]
            err = y[m] - Q.flat[flat]
            sums = np.bincount(flat, weights=err, minlength=Q.size)
            cnt = np.bincount(flat, minlength=Q.size)
            nz = cnt > 0
            Q.flat[nz] += alpha * sums[nz] / cnt[nz]
            visits += np.bincount(s[m], minlength=N_STATES)
            ret += np.where(m, r, 0.0)
            soc, temp = np.where(m, soc2, soc), np.where(m, temp2, temp)
        history.append(float(ret.mean()))
        if verbose and it % 20 == 0:
            print(f"batch {it:4d} eps {eps:.2f} mean return EUR {ret.mean():.4f}")
    info = {
        "episodes": n_batches * batch,
        "states": N_STATES,
        "actions": ACTIONS.tolist(),
        "visited_states": int((visits > 0).sum()),
        "return_curve": history[:: max(1, n_batches // 50)],
    }
    return QPolicy(Q), info


def evaluate_policies(q: QPolicy, n: int = 300, seed: int = 99) -> dict:
    """Legacy vs MILP vs RL on the same random sessions (closed-loop rollouts)."""
    from .milp import plan_and_verify
    from .session import legacy_policy, plan_policy, rollout

    rng = np.random.default_rng(seed)
    ep = _sample(rng, n)
    models = level_models()
    rows = {"legacy": [], "milp": [], "rl": []}
    for i in range(n):
        lvl = int(ep["level"][i])
        ctx = ChargeContext(
            float(ep["soc"][i]),
            float(ep["n"][i] * DT_H),
            float(ep["target"][i]),
            float(rng.uniform(0, 24)),
            float(ep["amb"][i]),
            models.states[lvl],
        )
        phys = ChargePhysics(CellModelSet([models.states[lvl]]))
        rows["legacy"].append(rollout(ctx, legacy_policy(), phys))
        rows["milp"].append(rollout(ctx, plan_policy(plan_and_verify(ctx, phys)["plan"]), phys))
        rows["rl"].append(rollout(ctx, q.as_session_policy(float(models.soh[lvl])), phys))
    out = {}
    for k, rs in rows.items():
        out[k] = {
            "damage_pct_capacity_mean": float(np.mean([r["damage_pct_capacity"] for r in rs])),
            "met_target_rate": float(np.mean([r["met_target"] for r in rs])),
            "mean_shortfall_pct": float(np.mean([r["shortfall_pct"] for r in rs])),
            "energy_eur_mean": float(np.mean([r["energy_eur"] for r in rs])),
            "hours_above_95_mean": float(np.mean([r["hours_above_95"] for r in rs])),
            "peak_temp_c_mean": float(np.mean([r["peak_temp_c"] for r in rs])),
        }
    base = out["legacy"]["damage_pct_capacity_mean"]
    for k in ("milp", "rl"):
        out[k]["damage_reduction_vs_legacy_pct"] = 100 * (1 - out[k]["damage_pct_capacity_mean"] / base)
    out["sessions"] = n
    return out


def onpolicy_states(q: QPolicy, n_episodes: int = 6000, eps: float = 0.1, seed: int = 5, act_fn=None) -> dict:
    """States visited when running a policy (default: the noisy greedy teacher).

    `act_fn(soc, hours_left, amb, target, soh, elapsed) -> action index` lets
    DAgger collect the states a *student* visits, which the teacher then labels.
    """
    rng = np.random.default_rng(seed)
    models = level_models()
    phys = ChargePhysics(models)
    ep = _sample(rng, n_episodes)
    soc, amb, lvl, tgt, n = ep["soc"], ep["amb"], ep["level"], ep["target"], ep["n"]
    temp = amb + 3.0
    soh = models.soh[lvl]
    out = {k: [] for k in ("soc", "hours", "amb", "target", "soh", "elapsed")}
    for t in range(int(n.max())):
        m = t < n
        if not m.any():
            break
        left = (n - t) * DT_H
        el = np.full(n_episodes, t * DT_H >= 1.0)
        for k, v in (("soc", soc), ("hours", left), ("amb", amb), ("target", tgt), ("soh", soh), ("elapsed", el)):
            out[k].append(v[m])
        if act_fn is None:
            a = np.argmax(q.q[encode(soc, left, amb, tgt, soh, el)], axis=1)
        else:
            a = act_fn(soc, left, amb, tgt, soh, el)
        a = np.where(rng.random(n_episodes) < eps, rng.integers(0, len(ACTIONS), n_episodes), a)
        soc2, temp2, *_ = phys.step(lvl, soc, temp, amb, ACTIONS[a])
        soc, temp = np.where(m, soc2, soc), np.where(m, temp2, temp)
    return {k: np.concatenate(v) for k, v in out.items()}
