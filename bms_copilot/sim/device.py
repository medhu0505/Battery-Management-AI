"""Device-side measurement simulation for on-demand captures.

When the Triage Agent requests a diagnostic capture, or the dashboard asks for
dQ/dV curves, the *device* performs a slow charge and uploads samples. In this
simulator the device's hidden true state (from the fleet simulation) produces
those samples through the same AFE model; everything downstream (ICA, DMA)
sees only the measurements.
"""

from __future__ import annotations

import numpy as np

from ..edge.ica import ICACalibration, incremental_capacity
from .fleet import FleetData
from .sensors import slow_charge_capture


def _limiting_state(fleet: FleetData, i: int, k: int):
    c = int(fleet.truth_limiting_cell[i, k])
    st = {f: float(v[i, k, c]) for f, v in fleet.truth_state.items()}
    lli = np.sqrt(st["sei_z"]) + st["lli_cycle"] + st["lli_plating"]
    rf = 1.0 + 2.8 * lli + 1.6 * (st["lam_pe"] + st["lam_ne"]) + st["r_extra"]
    return lli, st["lam_pe"], st["lam_ne"], rf


def capture(fleet: FleetData, i: int, k: int, c_rate: float, seed: int, temp_c: float | None = None):
    lli, pe, ne, rf = _limiting_state(fleet, i, k)
    temp = fleet.tele["capture_temp_c"][i, k] if temp_c is None else temp_c
    rng = np.random.default_rng(seed)
    v, dq = slow_charge_capture(
        np.array([lli]),
        np.array([pe]),
        np.array([ne]),
        np.array([rf]),
        np.array([temp]),
        rng,
        c_rate=c_rate,
        current_gain=fleet.devices[i].current_gain,
    )
    return v[0], float(dq[0])


def ica_curve(fleet: FleetData, i: int, k: int, cal: ICACalibration | None = None, seed: int = 0) -> dict:
    v, dq = capture(fleet, i, k, 0.1, seed=seed + 1000 * i + k)
    r = incremental_capacity(v[None, :], dq, cal)
    return {
        "day": int(fleet.snap_days[k]),
        "v": np.round(r.v_grid, 4).tolist(),
        "dqdv": np.round(r.dqdv[0], 3).tolist(),
        "features": r.feature_dict(0),
    }


def diagnostic_capture(fleet: FleetData, i: int, k: int, seed: int = 0):
    """Full-range C/20 capture at a controlled 25 C (device waits for an idle, cool window)."""
    return capture(fleet, i, k, 0.05, seed=seed + 7919 * i + k, temp_c=25.0)
