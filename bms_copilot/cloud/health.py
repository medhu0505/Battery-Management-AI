"""Cloud health models: SoH from electrochemical features, RUL with an interval.

Stage 1 - SoH GPR:  ICA/dQ/dV features + DCIR + capture temperature -> SoH.
          Replaces the legacy gauge that infers SoH from a cycle counter.
Stage 1b - Mode GPRs: the same routine features -> degradation modes (LLI,
          LAM_pe, LAM_ne) of the limiting cell. Gives every device a
          mechanism estimate without a dedicated diagnostic capture, which is
          what fleet-level lot screening needs.
Stage 2 - RUL GPR:  SoH trajectory (level, recent and lifetime fade rate) +
          usage stress (throughput, temperature, time at high SoC, C-rate)
          -> days until SoH crosses the end-of-life threshold.
          Trained in log space, so the 95 % interval is asymmetric (it can
          stretch far into the future but never below zero).

Training data comes from the history cohort (devices observed to end of life).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MODELS, SERVICE
from ..edge.ica import FEATURE_NAMES
from ..sim.fleet import FleetData
from .gpr import GaussianProcess

SOH_FEATURES = FEATURE_NAMES + ["r0_mohm", "capture_temp_c"]
RUL_FEATURES = [
    "soh_est",
    "soh_rate_recent",
    "soh_rate_life",
    "r0_mohm",
    "efc_rate",
    "temp_mean",
    "frac_high_mean",
    "charge_c_mean",
    "age_days",
]
_RUL_OFFSET = 30.0
_RECENT_SNAPS = 13  # ~6 months


def soh_inputs(fleet: FleetData, dev: np.ndarray, snap: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [fleet.ica[dev, snap], fleet.tele["r0_mohm"][dev, snap], fleet.tele["capture_temp_c"][dev, snap]]
    )


def _fade_rates(days: np.ndarray, soh: np.ndarray) -> tuple[float, float]:
    """(recent, lifetime) SoH fade in % per 100 days, from the estimate history."""
    life = (1.0 - soh[-1]) / max(days[-1], 1.0) * 1e4
    if len(soh) < 4:
        return life, life
    d, s = days[-_RECENT_SNAPS:], soh[-_RECENT_SNAPS:]
    slope = np.polyfit(d, s, 1)[0]
    return float(-slope * 1e4), float(life)


@dataclass
class RULPrediction:
    median_days: float
    lo_days: float
    hi_days: float
    std_log: float

    def to_dict(self) -> dict:
        return {
            "median_days": round(self.median_days),
            "lo_days": round(self.lo_days),
            "hi_days": round(self.hi_days),
            "interval": "95%",
            "std_log": round(self.std_log, 3),
        }


MODES = ("lli", "lam_pe", "lam_ne")


class HealthModels:
    def __init__(self, soh_gp: GaussianProcess, rul_gp: GaussianProcess, mode_gps: dict | None = None):
        self.soh_gp = soh_gp
        self.rul_gp = rul_gp
        self.mode_gps = mode_gps or {}

    def estimate_modes(self, X: np.ndarray) -> dict:
        out = {}
        for m, gp in self.mode_gps.items():
            mu, sd = gp.predict(np.atleast_2d(X))
            out[m] = (np.maximum(mu, 0.0), sd)
        return out

    # ------------------------------------------------------------------ SoH
    def estimate_soh(self, X: np.ndarray):
        mean, std = self.soh_gp.predict(X)
        return np.clip(mean, 0.0, 1.05), std

    def soh_history(self, fleet: FleetData, i: int, upto: int | None = None):
        """SoH estimates for device i over its observed snapshots."""
        n = int(fleet.n_observed[i]) if upto is None else upto
        snaps = np.arange(n)
        mean, std = self.estimate_soh(soh_inputs(fleet, np.full(n, i), snaps))
        return fleet.snap_days[:n], mean, std

    # ------------------------------------------------------------------ RUL
    def rul_features(self, fleet: FleetData, i: int, k: int, soh_hist: np.ndarray | None = None) -> np.ndarray:
        if soh_hist is None:
            _, soh_hist, _ = self.soh_history(fleet, i, upto=k + 1)
        recent, life = _fade_rates(fleet.snap_days[: k + 1], soh_hist[: k + 1])
        t = fleet.tele
        return np.array(
            [
                soh_hist[k],
                recent,
                life,
                t["r0_mohm"][i, k],
                t["efc_rate"][i, k],
                t["temp_mean"][i, k],
                t["frac_high_mean"][i, k],
                t["charge_c_mean"][i, k],
                t["age_days"][i, k],
            ]
        )

    def predict_rul(self, X: np.ndarray) -> list[RULPrediction]:
        mu, sd = self.rul_gp.predict(np.atleast_2d(X))
        out = []
        for m, s in zip(mu, sd):
            f = lambda z: max(0.0, float(np.exp(z) - _RUL_OFFSET))  # noqa: E731
            out.append(RULPrediction(f(m), f(m - 1.96 * s), f(m + 1.96 * s), float(s)))
        return out

    # ---------------------------------------------------------- persistence
    def to_arrays(self) -> dict:
        out = {**self.soh_gp.to_arrays("soh_"), **self.rul_gp.to_arrays("rul_")}
        for m, gp in self.mode_gps.items():
            out.update(gp.to_arrays(f"mode_{m}_"))
        return out

    @classmethod
    def from_arrays(cls, z) -> HealthModels:
        modes = {m: GaussianProcess.from_arrays(z, f"mode_{m}_") for m in MODES if f"mode_{m}_theta" in z.files}
        return cls(GaussianProcess.from_arrays(z, "soh_"), GaussianProcess.from_arrays(z, "rul_"), modes)


def truth_modes(fleet: FleetData, dev: np.ndarray, snap: np.ndarray) -> dict:
    lim = fleet.truth_limiting_cell[dev, snap]
    st = {k: v[dev, snap, lim] for k, v in fleet.truth_state.items()}
    return {
        "lli": np.sqrt(st["sei_z"]) + st["lli_cycle"] + st["lli_plating"],
        "lam_pe": st["lam_pe"],
        "lam_ne": st["lam_ne"],
    }


def _rul_dataset(models: HealthModels, fleet: FleetData, devices: np.ndarray, min_snap: int = 2):
    X, y, dev = [], [], []
    for i in devices:
        eol = fleet.eol_day[i]
        if np.isnan(eol):
            continue
        n = int(np.searchsorted(fleet.snap_days, eol))  # snapshots before end of life
        if n <= min_snap:
            continue
        _, soh_hist, _ = models.soh_history(fleet, i, upto=n)
        for k in range(min_snap, n):
            X.append(models.rul_features(fleet, i, k, soh_hist))
            y.append(eol - fleet.snap_days[k])
            dev.append(i)
    return np.array(X), np.array(y), np.array(dev)


def train_health_models(history: FleetData, seed: int = 0, test_share: float = 0.2) -> tuple[HealthModels, dict]:
    rng = np.random.default_rng(seed)
    n_dev = len(history.devices)
    perm = rng.permutation(n_dev)
    n_test = int(round(test_share * n_dev))
    test_dev, train_dev = np.sort(perm[:n_test]), np.sort(perm[n_test:])
    cap = MODELS.gpr_max_train_points

    # ---- Stage 1: SoH from ICA features (pre- and slightly post-EOL snapshots)
    S = len(history.snap_days)
    dd, ss = np.meshgrid(train_dev, np.arange(S), indexing="ij")
    dd, ss = dd.ravel(), ss.ravel()
    keep = history.truth_soh[dd, ss] > 0.65
    dd, ss = dd[keep], ss[keep]
    pick = rng.choice(len(dd), size=min(cap, len(dd)), replace=False)
    n_soh = len(pick)
    soh_gp = GaussianProcess(n_restarts=3, seed=seed).fit(
        soh_inputs(history, dd[pick], ss[pick]), history.truth_soh[dd[pick], ss[pick]]
    )

    Xs = soh_inputs(history, dd[pick], ss[pick])
    tm = truth_modes(history, dd[pick], ss[pick])
    mode_gps = {m: GaussianProcess(n_restarts=2, seed=seed).fit(Xs, tm[m]) for m in MODES}
    models = HealthModels(soh_gp, GaussianProcess(), mode_gps)

    # ---- Stage 2: RUL from SoH trajectory + stress
    Xr, yr, _ = _rul_dataset(models, history, train_dev)
    pick = rng.choice(len(yr), size=min(cap, len(yr)), replace=False)
    models.rul_gp = GaussianProcess(n_restarts=3, seed=seed).fit(Xr[pick], np.log(yr[pick] + _RUL_OFFSET))

    report = evaluate(models, history, test_dev)
    report["train_points"] = {"soh": int(n_soh), "rul": int(len(pick))}
    report["soh_length_scales"] = dict(zip(SOH_FEATURES, np.round(soh_gp.length_scales(), 3).tolist()))
    report["rul_length_scales"] = dict(zip(RUL_FEATURES, np.round(models.rul_gp.length_scales(), 3).tolist()))
    return models, report


def evaluate(models: HealthModels, fleet: FleetData, devices: np.ndarray) -> dict:
    """Held-out accuracy of both stages plus the legacy-gauge baseline."""
    dd, ss = [], []
    for i in devices:
        n = int(fleet.n_observed[i])
        dd += [i] * n
        ss += list(range(n))
    dd, ss = np.array(dd), np.array(ss)
    truth = fleet.truth_soh[dd, ss]
    m = truth > 0.65
    est, std = models.estimate_soh(soh_inputs(fleet, dd[m], ss[m]))
    gauge = fleet.tele["gauge_soh"][dd[m], ss[m]]
    soh = {
        "n": int(m.sum()),
        "mae_ai": float(np.mean(np.abs(est - truth[m]))),
        "mae_gauge_baseline": float(np.mean(np.abs(gauge - truth[m]))),
        "coverage95": float(np.mean(np.abs(est - truth[m]) <= 1.96 * std)),
    }

    modes = {}
    if models.mode_gps:
        Xm = soh_inputs(fleet, dd[m], ss[m])
        tm = truth_modes(fleet, dd[m], ss[m])
        est = models.estimate_modes(Xm)
        for k in MODES:
            mu = est[k][0]
            modes[k] = {
                "mae": float(np.mean(np.abs(mu - tm[k]))),
                "r2": float(1 - np.mean((mu - tm[k]) ** 2) / max(np.var(tm[k]), 1e-12)),
            }

    Xr, yr, _ = _rul_dataset(models, fleet, devices)
    rul = {"n": int(len(yr))}
    if len(yr):
        preds = models.predict_rul(Xr)
        med = np.array([p.median_days for p in preds])
        lo = np.array([p.lo_days for p in preds])
        hi = np.array([p.hi_days for p in preds])
        rel = np.abs(med - yr) / np.maximum(yr, 60)
        rul.update(
            {
                "mae_days": float(np.mean(np.abs(med - yr))),
                "median_rel_error": float(np.median(rel)),
                "coverage95": float(np.mean((yr >= lo) & (yr <= hi))),
                "mae_days_last_year": float(np.mean(np.abs(med - yr)[yr <= 365])) if np.any(yr <= 365) else None,
            }
        )
    return {"soh": soh, "modes_from_routine_ica": modes, "rul": rul, "eol_soh": SERVICE.eol_soh}
