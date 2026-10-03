"""On-device precursor detection for swelling, thermal events and micro-shorts.

Today's BMS reacts *after* a limit is breached. This detector looks for the
precursors instead, using five signals the AFE (plus an optional strain / gas
sensor) already produces:

    cell_divergence_mv       rest-voltage spread between series cells
    self_discharge_pct_day   SoC loss while resting (internal leakage)
    thermal_residual_c       measured minus twin-predicted temperature rise
    strain_excess_ue         pack strain above the healthy ageing baseline
    r0_growth_mohm_100d      DCIR growth rate over recent snapshots

Scoring:
* Mahalanobis distance against a healthy-fleet baseline (fit in the cloud,
  shipped in the calibration package) -> watch / warning levels
* one-sided CUSUM on each standardised signal -> catches slow drifts early
* hard physical rules -> critical (independent of the statistics)
Each alert carries per-signal contributions and a mechanism hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

SIGNALS = [
    "cell_divergence_mv",
    "self_discharge_pct_day",
    "thermal_residual_c",
    "strain_excess_ue",
    "r0_growth_mohm_100d",
]
MECHANISM = {
    "cell_divergence_mv": "internal micro-short / cell imbalance",
    "self_discharge_pct_day": "internal micro-short (leakage current)",
    "thermal_residual_c": "abnormal heat generation (side reactions / short)",
    "strain_excess_ue": "cell swelling (gas generation)",
    "r0_growth_mohm_100d": "accelerated impedance rise (contact loss / electrolyte decomposition)",
}
HARD_RULES = {  # signal: critical threshold
    "strain_excess_ue": 600.0,
    "thermal_residual_c": 3.0,
    "self_discharge_pct_day": 1.0,
    "cell_divergence_mv": 60.0,
}
# Signals that are memoryless under health; only these feed the CUSUM (the others
# carry persistent per-device offsets that a CUSUM would slowly accumulate).
CUSUM_SIGNALS = ["self_discharge_pct_day", "thermal_residual_c", "strain_excess_ue"]
LEVELS = ["normal", "watch", "warning", "critical"]
# Signals that indicate a *safety* precursor. Divergence and DCIR growth on
# their own indicate imbalance / ageing and are routed to diagnostics instead.
PRECURSOR_SIGNALS = {"self_discharge_pct_day", "thermal_residual_c", "strain_excess_ue"}


@dataclass
class AnomalyCalibration:
    """Healthy baseline: per-signal linear ageing trend + residual covariance."""

    mean: np.ndarray  # residual mean (~0)
    cov_inv: np.ndarray
    std: np.ndarray
    strain_intercept: float
    strain_slope_per_day: float
    trend_intercept: np.ndarray
    trend_slope: np.ndarray
    watch_d2: float = float(chi2.ppf(0.99, len(SIGNALS)))
    warning_d2: float = float(chi2.ppf(0.9999, len(SIGNALS)))
    cusum_k: float = 0.5
    cusum_h: float = 8.0

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "cov_inv": self.cov_inv.tolist(),
            "std": self.std.tolist(),
            "strain_intercept": self.strain_intercept,
            "strain_slope_per_day": self.strain_slope_per_day,
            "trend_intercept": self.trend_intercept.tolist(),
            "trend_slope": self.trend_slope.tolist(),
            "watch_d2": self.watch_d2,
            "warning_d2": self.warning_d2,
            "cusum_k": self.cusum_k,
            "cusum_h": self.cusum_h,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnomalyCalibration:
        return cls(
            np.array(d["mean"]),
            np.array(d["cov_inv"]),
            np.array(d["std"]),
            d["strain_intercept"],
            d["strain_slope_per_day"],
            np.array(d["trend_intercept"]),
            np.array(d["trend_slope"]),
            d["watch_d2"],
            d["warning_d2"],
            d["cusum_k"],
            d["cusum_h"],
        )


@dataclass
class AnomalyResult:
    day: int
    level: str
    d2: float
    contributions: dict
    signals: dict
    mechanism: str | None
    rules_fired: list = field(default_factory=list)
    cusum_fired: list = field(default_factory=list)
    kind: str | None = None  # safety_precursor | imbalance_or_ageing

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "level": self.level,
            "d2": round(self.d2, 2),
            "contributions": {k: round(v, 2) for k, v in self.contributions.items()},
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "mechanism": self.mechanism,
            "kind": self.kind,
            "rules_fired": self.rules_fired,
            "cusum_fired": self.cusum_fired,
        }


def signal_matrix(tele: dict, days: np.ndarray, strain_intercept: float, strain_slope: float) -> np.ndarray:
    """Build the (n_snapshots, 5) signal matrix for one device from its telemetry rows."""
    r0 = np.asarray(tele["r0_mohm"], float)
    growth = np.zeros_like(r0)
    for k in range(len(r0)):
        j = max(0, k - 6)
        if k - j >= 2:
            growth[k] = np.polyfit(days[j : k + 1], r0[j : k + 1], 1)[0] * 100.0
    return np.column_stack(
        [
            tele["cell_divergence_mv"],
            tele["self_discharge_pct_day"],
            tele["thermal_residual_c"],
            np.asarray(tele["strain_ue"]) - (strain_intercept + strain_slope * np.asarray(days)),
            growth,
        ]
    )


def fit_calibration(signals: np.ndarray, days: np.ndarray, strain: np.ndarray) -> AnomalyCalibration:
    """Cloud-side: fit the healthy baseline from defect-free fleet snapshots."""
    slope, intercept = np.polyfit(days, strain, 1)
    x = signals.copy()
    x[:, 3] = strain - (intercept + slope * days)
    t_slope = np.zeros(x.shape[1])
    t_int = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        t_slope[j], t_int[j] = np.polyfit(days, x[:, j], 1)
    resid = x - (t_int + np.outer(days, t_slope))
    cov = np.cov(resid.T) + 1e-9 * np.eye(x.shape[1])
    return AnomalyCalibration(
        resid.mean(0), np.linalg.inv(cov), np.sqrt(np.diag(cov)), float(intercept), float(slope), t_int, t_slope
    )


class PrecursorDetector:
    def __init__(self, cal: AnomalyCalibration):
        self.cal = cal
        self.cusum = np.zeros(len(SIGNALS))

    def reset(self) -> None:
        self.cusum[:] = 0.0

    def score(self, x: np.ndarray, day: int) -> AnomalyResult:
        c = self.cal
        dev = x - (c.trend_intercept + c.trend_slope * day) - c.mean
        d2 = float(dev @ c.cov_inv @ dev)
        z = dev / c.std
        # Only increases are dangerous for every signal here.
        mask = np.array([s in CUSUM_SIGNALS for s in SIGNALS])
        self.cusum = np.where(mask, np.maximum(0.0, self.cusum + z - c.cusum_k), 0.0)
        cusum_fired = [SIGNALS[i] for i in np.nonzero(self.cusum > c.cusum_h)[0]]
        rules = [s for s, thr in HARD_RULES.items() if x[SIGNALS.index(s)] >= thr]
        if rules:
            level = "critical"
        elif d2 > c.warning_d2:
            level = "warning"
        elif d2 > c.watch_d2 or cusum_fired:
            level = "watch"
        else:
            level = "normal"
        contrib = {s: float(max(zi, 0.0)) for s, zi in zip(SIGNALS, z)}
        mechanism = None
        kind = None
        if level != "normal":
            pre_rules = [r for r in rules if r in PRECURSOR_SIGNALS]
            top = pre_rules[0] if pre_rules else (rules[0] if rules else max(contrib, key=contrib.get))
            mechanism = MECHANISM[top]
            kind = (
                "safety_precursor"
                if (top in PRECURSOR_SIGNALS or set(cusum_fired) & PRECURSOR_SIGNALS)
                else "imbalance_or_ageing"
            )
        res = AnomalyResult(day, level, d2, contrib, dict(zip(SIGNALS, map(float, x))), mechanism, rules, cusum_fired)
        res.kind = kind
        return res

    def scan(self, X: np.ndarray, days: np.ndarray) -> list[AnomalyResult]:
        self.reset()
        return [self.score(x, int(d)) for x, d in zip(X, days)]
