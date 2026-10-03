"""Equivalent-circuit + lumped thermal cell model.

`CellSimulator` is the high-fidelity "truth" used to generate telemetry:
2RC Thevenin network, OCV from the electrode model at the current ageing state,
charge/discharge hysteresis, temperature-dependent resistances and a lumped
thermal node. The digital twin (cloud/twin.py) deliberately uses a simpler 1RC
model so its data-driven residual layer has real unmodelled dynamics to learn.

Sign convention: current > 0 is discharge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CELL
from .electrode import DegradationState, electrode_model

_R_GAS = 8.314


def arrhenius(temp_c, ea_j_per_mol: float, ref_c: float = 25.0):
    """exp(Ea/R (1/T - 1/Tref)): > 1 below the reference temperature."""
    return np.exp(ea_j_per_mol / _R_GAS * (1.0 / (np.asarray(temp_c) + 273.15) - 1.0 / (ref_c + 273.15)))


def resistance_temp_factor(temp_c):
    """Resistances roughly double between 25 C and 0 C."""
    return arrhenius(temp_c, 22_000.0)


@dataclass
class ECMParams:
    r0: float
    r1: float
    c1: float
    r2: float
    c2: float

    @classmethod
    def for_state(cls, state: DegradationState, temp_c: float) -> ECMParams:
        f_age = float(state.resistance_factor())
        f_t = float(resistance_temp_factor(temp_c))
        # Charge-transfer (R1) ages a little faster than the ohmic term.
        return cls(
            r0=CELL.r0_bol_ohm * f_age * f_t,
            r1=CELL.r1_bol_ohm * (1 + 1.3 * (f_age - 1)) * f_t,
            c1=CELL.c1_bol_f,
            r2=CELL.r2_bol_ohm * f_age * f_t,
            c2=CELL.c2_bol_f,
        )

    @property
    def dc_resistance(self) -> float:
        return self.r0 + self.r1 + self.r2


class CellSimulator:
    """Time-stepping truth model of one cell."""

    HYSTERESIS_V = 0.008
    HYSTERESIS_GAMMA = 30.0

    def __init__(self, state: DegradationState, soc0: float = 1.0, temp0_c: float = 25.0, extra_heat_w: float = 0.0):
        self.state = state
        self.soc_tab, self.ocv_tab, self.capacity_ah = electrode_model().ocv_table(state)
        self.soc = float(soc0)
        self.temp_c = float(temp0_c)
        self.v1 = 0.0
        self.v2 = 0.0
        self.h = 0.0
        self.extra_heat_w = extra_heat_w  # e.g. an internal micro-short

    def ocv(self, soc) -> np.ndarray:
        return np.interp(soc, self.soc_tab, self.ocv_tab)

    def step(self, current_a: float, dt_s: float, t_amb_c: float) -> dict:
        p = ECMParams.for_state(self.state, self.temp_c)
        a1 = np.exp(-dt_s / (p.r1 * p.c1))
        a2 = np.exp(-dt_s / (p.r2 * p.c2))
        self.v1 = a1 * self.v1 + p.r1 * (1 - a1) * current_a
        self.v2 = a2 * self.v2 + p.r2 * (1 - a2) * current_a
        self.soc = float(np.clip(self.soc - current_a * dt_s / 3600.0 / self.capacity_ah, 0.0, 1.0))
        target = -np.sign(current_a) * self.HYSTERESIS_V
        k = 1 - np.exp(-abs(current_a) * dt_s / 3600.0 / self.capacity_ah * self.HYSTERESIS_GAMMA)
        self.h += (target - self.h) * k
        v = float(self.ocv(self.soc) - current_a * p.r0 - self.v1 - self.v2 + self.h)
        heat = current_a**2 * p.r0 + self.v1**2 / p.r1 + self.v2**2 / p.r2 + self.extra_heat_w
        self.temp_c += (heat - CELL.ha_w_per_k * (self.temp_c - t_amb_c)) * dt_s / (CELL.mass_kg * CELL.cp_j_per_kgk)
        return {"v": v, "soc": self.soc, "temp_c": self.temp_c, "heat_w": heat, "r0": p.r0}

    def run(self, current_a: np.ndarray, dt_s: float, t_amb_c: float) -> dict:
        out = [self.step(float(i), dt_s, t_amb_c) for i in current_a]
        return {k: np.array([o[k] for o in out]) for k in out[0]}


def drive_cycle(
    duration_s: int = 7200, dt_s: float = 1.0, seed: int = 0, capacity_ah: float = CELL.rated_capacity_ah
) -> np.ndarray:
    """A laptop-like load: idle, browsing, compile/render bursts, video calls.

    Returns cell current [A] per step (positive = discharge).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt_s)
    current = np.empty(n)
    i = 0
    levels = {"idle": 0.05, "browse": 0.18, "call": 0.35, "compile": 0.9, "render": 1.3}
    names = list(levels)
    probs = np.array([0.25, 0.35, 0.2, 0.12, 0.08])
    while i < n:
        mode = names[rng.choice(len(names), p=probs)]
        seg = int(rng.integers(30, 400) / dt_s)
        base = levels[mode] * capacity_ah
        current[i : i + seg] = np.clip(base * (1 + 0.25 * rng.standard_normal(min(seg, n - i))), 0.01, None)
        i += seg
    return current
