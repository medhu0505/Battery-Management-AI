"""Degradation Mode Analysis (DMA): *why* a battery is losing capacity.

Given a full diagnostic slow-charge capture (requested by the Triage Agent and
run on the device during an idle window), fit the electrode model's three
degradation modes plus an ohmic offset so the modelled charge curve Q(V)
matches the measured one. The fitted modes separate normal ageing (SEI-driven
LLI) from manufacturing-type defects (e.g. anode LAM) and usage-driven cathode
stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..edge.ica import savgol
from ..physics.electrode import electrode_model

_V_GRID = np.linspace(3.30, 4.12, 160)


@dataclass
class DMAResult:
    lli: float
    lam_pe: float
    lam_ne: float
    offset_mv: float
    rmse_mah: float
    capacity_ah: float

    def dominant(self) -> str:
        modes = {"lli": self.lli, "lam_pe": self.lam_pe, "lam_ne": self.lam_ne}
        return max(modes, key=modes.get)

    def to_dict(self) -> dict:
        return {
            "lli": round(self.lli, 4),
            "lam_pe": round(self.lam_pe, 4),
            "lam_ne": round(self.lam_ne, 4),
            "offset_mv": round(self.offset_mv, 1),
            "rmse_mah": round(self.rmse_mah, 2),
            "capacity_ah": round(self.capacity_ah, 3),
            "dominant": self.dominant(),
        }


def measured_q_of_v(v_samples: np.ndarray, dq_per_sample: float) -> np.ndarray:
    """Charge-from-empty at each grid voltage, from raw CC-charge samples."""
    v = np.maximum.accumulate(savgol(np.asarray(v_samples, float), 21, 3))
    q = np.arange(len(v)) * dq_per_sample
    return np.interp(_V_GRID, v, q)


def _model_q_of_v(params: np.ndarray) -> np.ndarray:
    lli, pe, ne, off = params
    q, v = electrode_model().charge_curve(lli, pe, ne, n=500)
    return np.interp(_V_GRID - off, v, q)


GOOD_FIT_MAH = 3.0  # measurement-noise level: no need to try further starts


def fit_modes(v_samples: np.ndarray, dq_per_sample: float) -> DMAResult:
    target = measured_q_of_v(v_samples, dq_per_sample)
    lo, hi = [0.0, 0.0, 0.0, 0.0], [0.45, 0.45, 0.45, 0.25]
    best = None
    for start in (
        [0.05, 0.02, 0.02, 0.03],
        [0.15, 0.05, 0.05, 0.05],
        [0.05, 0.02, 0.15, 0.04],
        [0.05, 0.15, 0.02, 0.04],
    ):
        r = least_squares(
            lambda p: _model_q_of_v(p) - target,
            start,
            bounds=(lo, hi),
            diff_step=1e-3,
            x_scale=[0.05, 0.05, 0.05, 0.02],
        )
        if best is None or r.cost < best.cost:
            best = r
        if np.sqrt(np.mean(best.fun**2)) * 1000 <= GOOD_FIT_MAH:
            break
    lli, pe, ne, off = best.x
    cap = float(electrode_model().window(lli, pe, ne).capacity_ah)
    rmse = float(np.sqrt(np.mean(best.fun**2)) * 1000)
    return DMAResult(float(lli), float(pe), float(ne), float(off * 1000), rmse, cap)
