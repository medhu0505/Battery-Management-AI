"""Electrode-level open-circuit-voltage model with degradation modes.

The full-cell OCV is the difference of the cathode and anode open-circuit
potentials, each evaluated at its own lithium stoichiometry. Three degradation
modes change how the two electrodes line up, which is exactly what moves the
peaks of the incremental-capacity (dQ/dV) curve:

* LLI    - loss of lithium inventory (SEI growth, plating)
* LAM_pe - loss of active material in the positive electrode
* LAM_ne - loss of active material in the negative electrode

LAM is modelled as loss of *lithiated* material, so it also consumes lithium.

OCP fits: Chen et al. 2020 (LG M50, NMC811 / graphite-SiOx), as used in PyBaMM.
All solvers are vectorised so a whole fleet can be evaluated in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ..config import CELL


def anode_ocp(x):
    """Graphite OCP vs Li/Li+ [V]; the tanh steps are the staging plateaus."""
    return (
        1.9793 * np.exp(-39.3631 * x)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
    )


def cathode_ocp(y):
    """NMC811 OCP vs Li/Li+ [V]."""
    return (
        -0.8090 * y
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (y - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (y - 0.3120))
    )


# Fresh-cell stoichiometry windows (Chen 2020) for 100 % and 0 % SoC.
_X100, _X0 = 0.9014, 0.0279
_Y100, _Y0 = 0.2661, 0.9084
# Fraction of lithiation of the material lost to LAM (lithiated-LAM assumption).
_LAM_PE_LI_FRACTION = 0.55
_LAM_NE_LI_FRACTION = 0.45


@dataclass
class DegradationState:
    """Internal ageing state of a cell. Fields may be floats or numpy arrays
    (one entry per cell) so the same code ages a single cell or a fleet."""

    sei_z: float = 0.0  # calendar-SEI state; LLI_cal = sqrt(sei_z)
    lli_cycle: float = 0.0  # cycling-driven SEI growth / particle cracking
    lli_plating: float = 0.0  # lithium plating (cold or fast charging)
    lam_pe: float = 0.0
    lam_ne: float = 0.0
    r_extra: float = 0.0  # extra ohmic resistance from defects, relative to R0_bol

    @property
    def lli(self):
        return np.sqrt(self.sei_z) + self.lli_cycle + self.lli_plating

    def resistance_factor(self):
        """R0 multiplier vs beginning of life (SEI film + contact loss + defects)."""
        return 1.0 + 2.8 * self.lli + 1.6 * (self.lam_pe + self.lam_ne) + self.r_extra

    def copy(self) -> DegradationState:
        return DegradationState(**{f.name: np.copy(getattr(self, f.name)) for f in fields(self)})

    def take(self, idx) -> DegradationState:
        """Select one or more cells from an array-valued state."""
        return DegradationState(**{f.name: np.asarray(getattr(self, f.name))[idx] for f in fields(self)})

    @classmethod
    def zeros(cls, shape) -> DegradationState:
        return cls(**{f.name: np.zeros(shape) for f in fields(cls)})


@dataclass(frozen=True)
class Window:
    """Electrode alignment at the top of charge plus usable capacity."""

    x100: np.ndarray
    y100: np.ndarray
    cne: np.ndarray
    cpe: np.ndarray
    capacity_ah: np.ndarray


class ElectrodeModel:
    """Maps degradation modes to capacity and OCV curves for one cell design."""

    def __init__(
        self, rated_capacity_ah: float = CELL.rated_capacity_ah, v_max: float = CELL.v_max, v_min: float = CELL.v_min
    ):
        self.v_max, self.v_min = v_max, v_min
        # Electrode capacities for a 1 Ah reference window, then rescale so the
        # fresh cell measures exactly the rated capacity between v_min and v_max.
        self.cne0 = 1.0 / (_X100 - _X0)
        self.cpe0 = 1.0 / (_Y0 - _Y100)
        scale = rated_capacity_ah / float(self.window(0.0, 0.0, 0.0).capacity_ah)
        self.cne0 *= scale
        self.cpe0 *= scale
        self.rated_capacity_ah = rated_capacity_ah

    def window(self, lli, lam_pe, lam_ne) -> Window:
        lli, lam_pe, lam_ne = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (lli, lam_pe, lam_ne)))
        cne = self.cne0 * (1.0 - lam_ne)
        cpe = self.cpe0 * (1.0 - lam_pe)
        n_li = (
            (_X100 * self.cne0 + _Y100 * self.cpe0) * (1.0 - lli)
            - lam_pe * self.cpe0 * _LAM_PE_LI_FRACTION
            - lam_ne * self.cne0 * _LAM_NE_LI_FRACTION
        )
        # Top of charge: anode stoichiometry x at which OCV = v_max.
        lo = np.full(lli.shape, 1e-4)
        hi = np.full(lli.shape, 1.0 - 1e-4)
        for _ in range(50):
            xm = 0.5 * (lo + hi)
            above = cathode_ocp((n_li - xm * cne) / cpe) - anode_ocp(xm) > self.v_max
            hi = np.where(above, xm, hi)
            lo = np.where(above, lo, xm)
        x100 = 0.5 * (lo + hi)
        y100 = (n_li - x100 * cne) / cpe
        # Bottom of discharge: v_min, or an electrode running out of lithium/sites.
        q_limit = np.minimum(x100 * cne, (1.0 - y100) * cpe) * 0.9999
        lo = np.zeros(lli.shape)
        hi = q_limit.copy()
        for _ in range(50):
            qm = 0.5 * (lo + hi)
            above = cathode_ocp(y100 + qm / cpe) - anode_ocp(x100 - qm / cne) > self.v_min
            lo = np.where(above, qm, lo)
            hi = np.where(above, hi, qm)
        cap = 0.5 * (lo + hi)
        return Window(x100, y100, cne, cpe, cap)

    def capacity(self, state: DegradationState):
        return self.window(state.lli, state.lam_pe, state.lam_ne).capacity_ah

    def ocv_vs_discharged(self, lli, lam_pe, lam_ne, n: int = 600):
        """OCV vs Ah discharged from full. Shapes: q, v -> (..., n); cap -> (...)."""
        w = self.window(lli, lam_pe, lam_ne)
        frac = np.linspace(0.0, 1.0, n)
        q = w.capacity_ah[..., None] * frac
        v = cathode_ocp(w.y100[..., None] + q / w.cpe[..., None]) - anode_ocp(w.x100[..., None] - q / w.cne[..., None])
        return q, v, w.capacity_ah

    def ocv_table(self, state: DegradationState, n: int = 201):
        """Scalar state -> (soc ascending 0..1, ocv, capacity)."""
        q, v, cap = self.ocv_vs_discharged(state.lli, state.lam_pe, state.lam_ne, n)
        cap = float(cap)
        soc = 1.0 - np.ravel(q) / cap
        return soc[::-1].copy(), np.ravel(v)[::-1].copy(), cap

    def charge_curve(self, lli, lam_pe, lam_ne, n: int = 1200):
        """Ah charged from empty vs OCV (ascending voltage). The ICA source signal."""
        q, v, cap = self.ocv_vs_discharged(lli, lam_pe, lam_ne, n)
        return (cap[..., None] - q)[..., ::-1], v[..., ::-1]


_MODEL: ElectrodeModel | None = None


def electrode_model() -> ElectrodeModel:
    """Process-wide singleton (construction runs a root solve)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = ElectrodeModel()
    return _MODEL
