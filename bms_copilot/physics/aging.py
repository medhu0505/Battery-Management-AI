"""Semi-empirical degradation model driven by daily stress factors.

Mechanisms (each maps to a degradation mode the ICA can observe):

* Calendar SEI growth  -> LLI, sqrt-of-time law, Arrhenius in T, worse at high SoC
* Cycling SEI/cracking -> LLI, linear in throughput, worse at high C-rate
* Lithium plating      -> LLI, cold + fast charging
* Cathode degradation  -> LAM_pe, accelerated by time at high voltage
* Anode degradation    -> LAM_ne, accelerated by charge rate and depth of discharge

All functions are vectorised over cells/devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ecm import arrhenius
from .electrode import DegradationState


@dataclass
class DailyStress:
    """Aggregated usage of one day. Fields may be arrays (one per device/cell)."""

    efc: float = 1.0  # equivalent full cycles
    mean_soc: float = 0.7  # time-weighted mean SoC
    frac_high_soc: float = 0.3  # fraction of the day above 95 % SoC
    charge_c: float = 0.7  # typical charge C-rate
    temp_c: float = 30.0  # time-weighted mean cell temperature
    charge_temp_c: float = 30.0
    dod: float = 0.6  # typical depth of discharge per cycle


@dataclass(frozen=True)
class AgingParams:
    k_cal: float = 0.0016  # 1/sqrt(day) at 25 C, S(soc)=1
    ea_cal: float = 50_000.0
    k_cyc: float = 4.0e-5  # LLI per EFC
    ea_cyc: float = 28_000.0
    k_plating: float = 3.0e-5
    k_lam_pe: float = 1.8e-5
    k_lam_ne: float = 1.3e-5


@dataclass
class DefectMultipliers:
    """Per-cell acceleration factors used to inject manufacturing defects."""

    lli: float | np.ndarray = 1.0
    lam_ne: float | np.ndarray = 1.0
    lam_pe: float | np.ndarray = 1.0


class AgingModel:
    def __init__(self, params: AgingParams | None = None):
        self.p = params or AgingParams()

    def soc_stress(self, mean_soc, frac_high):
        return 0.3 + 1.2 * np.asarray(mean_soc) ** 2 + 2.5 * np.asarray(frac_high)

    def step(
        self, state: DegradationState, s: DailyStress, days: float = 1.0, mult: DefectMultipliers | None = None
    ) -> DegradationState:
        p = self.p
        mult = mult or DefectMultipliers()
        # Aging is faster when hot: arrhenius() > 1 below the reference, so invert.
        a_cal = 1.0 / arrhenius(s.temp_c, p.ea_cal)
        a_cyc = 1.0 / arrhenius(s.temp_c, p.ea_cyc)
        efc = np.asarray(s.efc) * days
        c_over = np.maximum(0.0, np.asarray(s.charge_c) - 0.5)

        new = state.copy()
        new.sei_z = state.sei_z + p.k_cal**2 * a_cal * self.soc_stress(s.mean_soc, s.frac_high_soc) * days * mult.lli
        new.lli_cycle = (
            state.lli_cycle + p.k_cyc * a_cyc * efc * (1 + 1.2 * c_over**2) * (0.6 + 0.8 * np.asarray(s.dod)) * mult.lli
        )
        cold = np.maximum(0.0, 15.0 - np.asarray(s.charge_temp_c)) / 10.0
        new.lli_plating = state.lli_plating + p.k_plating * efc * cold * np.asarray(s.charge_c) ** 2
        new.lam_pe = (
            state.lam_pe
            + p.k_lam_pe
            * a_cyc
            * efc
            * (1 + 2.5 * np.asarray(s.frac_high_soc) + 1.5 * np.maximum(0.0, np.asarray(s.mean_soc) - 0.6))
            * mult.lam_pe
        )
        new.lam_ne = (
            state.lam_ne + p.k_lam_ne * a_cyc * efc * (1 + 1.5 * c_over) * (0.5 + np.asarray(s.dod)) * mult.lam_ne
        )
        return new

    def degradation_rate(self, state: DegradationState, s: DailyStress) -> float:
        """Scalar 'damage' of one day, used as the cost signal by the charging
        optimisers (weighted sum of mode increments; LLI dominates capacity)."""
        nxt = self.step(state, s)
        return float(
            1.0 * (nxt.lli - state.lli) + 0.8 * (nxt.lam_pe - state.lam_pe) + 0.8 * (nxt.lam_ne - state.lam_ne)
        )
