"""On-device Incremental Capacity Analysis (ICA / dQ/dV).

Runs during an opportunistic slow constant-current charge (e.g. the overnight
segment of an adaptive charge plan). Designed for an MCU/NPU budget: numpy only,
fixed-size buffers, no root finding. Pipeline:

1. Savitzky-Golay smoothing of the raw voltage samples (sensor noise + ADC steps)
2. Enforce monotonic voltage (CC charge)
3. Voltage-histogram integration: each sample carries dQ = I*dt; summing dQ per
   voltage bin and dividing by the bin width gives dQ/dV directly
4. Savitzky-Golay smoothing of the dQ/dV curve
5. Per-band features: peak height, peak position, band area

The voltage bands are part of the calibration package the cloud pushes over the
air, so the extractor keeps tracking peaks as they drift with ageing.

All functions accept a batch dimension (rows = captures).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..config import MODELS

FEATURE_NAMES = [f"{kind}_{band}" for band in ("b1", "b2", "b3") for kind in ("peak_height", "peak_v", "area_ah")]


def savgol_coeffs(window: int, order: int) -> np.ndarray:
    """Smoothing coefficients of a Savitzky-Golay filter (centre point)."""
    if window % 2 == 0 or window <= order:
        raise ValueError("window must be odd and larger than order")
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    vander = np.vander(x, order + 1, increasing=True)
    return np.linalg.pinv(vander)[0]


def savgol(y: np.ndarray, window: int, order: int) -> np.ndarray:
    """Row-wise SG smoothing with edge replication."""
    c = savgol_coeffs(window, order)
    half = window // 2
    padded = np.pad(np.atleast_2d(y), ((0, 0), (half, half)), mode="edge")
    out = sliding_window_view(padded, window, axis=-1) @ c
    return out if np.ndim(y) > 1 else out[0]


@dataclass
class ICACalibration:
    """Parameters the cloud may retune via signed OTA packages."""

    window_v: tuple = MODELS.ica_capture_window_v
    bands_v: tuple = MODELS.ica_bands_v
    grid_mv: float = MODELS.ica_grid_mv
    sg_window_v: int = MODELS.ica_savgol_window
    sg_order_v: int = MODELS.ica_savgol_order
    sg_window_dqdv: int = 15
    sg_order_dqdv: int = 2
    version: str = "ica-cal-1.0"

    def to_dict(self) -> dict:
        return {
            "window_v": list(self.window_v),
            "bands_v": [list(b) for b in self.bands_v],
            "grid_mv": self.grid_mv,
            "sg_window_v": self.sg_window_v,
            "sg_order_v": self.sg_order_v,
            "sg_window_dqdv": self.sg_window_dqdv,
            "sg_order_dqdv": self.sg_order_dqdv,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ICACalibration:
        return cls(
            window_v=tuple(d["window_v"]),
            bands_v=tuple(tuple(b) for b in d["bands_v"]),
            grid_mv=d["grid_mv"],
            sg_window_v=d["sg_window_v"],
            sg_order_v=d["sg_order_v"],
            sg_window_dqdv=d["sg_window_dqdv"],
            sg_order_dqdv=d["sg_order_dqdv"],
            version=d.get("version", "ica-cal"),
        )


@dataclass
class ICAResult:
    v_grid: np.ndarray  # bin centres [V]
    dqdv: np.ndarray  # (rows, bins) [Ah/V]
    features: np.ndarray  # (rows, 9) ordered as FEATURE_NAMES
    window_charge_ah: np.ndarray  # (rows,) charge accumulated inside the window
    valid: np.ndarray = field(default=None)  # (rows,) enough samples in window

    def feature_dict(self, row: int = 0) -> dict:
        return {k: float(v) for k, v in zip(FEATURE_NAMES, self.features[row])}


def incremental_capacity(
    v_samples: np.ndarray, dq_per_sample: np.ndarray, cal: ICACalibration | None = None
) -> ICAResult:
    """dQ/dV and features from CC-charge voltage samples.

    v_samples: (rows, n) terminal voltage samples at a fixed sampling period.
    dq_per_sample: (rows,) or scalar, charge added between samples [Ah] (I * dt).
    """
    cal = cal or ICACalibration()
    v = np.atleast_2d(np.asarray(v_samples, dtype=float))
    rows = v.shape[0]
    dq = np.broadcast_to(
        np.asarray(dq_per_sample, dtype=float).reshape(-1, 1)
        if np.ndim(dq_per_sample)
        else np.full((rows, 1), float(dq_per_sample)),
        v.shape,
    )

    v_s = np.maximum.accumulate(savgol(v, cal.sg_window_v, cal.sg_order_v), axis=1)

    lo, hi = cal.window_v
    step = cal.grid_mv / 1000.0
    edges = np.arange(lo, hi + step / 2, step)
    n_bins = len(edges) - 1
    idx = np.digitize(v_s, edges) - 1
    inside = (idx >= 0) & (idx < n_bins)
    flat = (np.arange(rows)[:, None] * n_bins + np.clip(idx, 0, n_bins - 1))[inside]
    dq_bins = np.bincount(flat, weights=dq[inside], minlength=rows * n_bins).reshape(rows, n_bins)
    dqdv = savgol(dq_bins / step, cal.sg_window_dqdv, cal.sg_order_dqdv)
    dqdv = np.maximum(dqdv, 0.0)
    centres = 0.5 * (edges[:-1] + edges[1:])

    feats = []
    for b_lo, b_hi in cal.bands_v:
        m = (centres >= b_lo) & (centres < b_hi)
        seg = dqdv[:, m]
        k = np.argmax(seg, axis=1)
        feats += [seg[np.arange(rows), k], centres[m][k], dq_bins[:, m].sum(axis=1)]
    features = np.stack(feats, axis=1)
    counts = inside.sum(axis=1)
    return ICAResult(centres, dqdv, features, dq_bins.sum(axis=1), valid=counts > 0.5 * n_bins)
