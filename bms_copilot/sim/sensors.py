"""Analog-front-end (AFE) simulation: what the BMS actually measures.

Turns a true electrochemical state into noisy, quantised sensor samples.
"""

from __future__ import annotations

import numpy as np

from ..config import CELL
from ..physics.ecm import resistance_temp_factor
from ..physics.electrode import anode_ocp, cathode_ocp, electrode_model

V_NOISE_V = 0.0010  # 1 mV rms voltage noise
ADC_LSB_V = 0.0010  # 1 mV quantisation
CAPTURE_C_RATE = 0.1  # slow overnight segment used for ICA
N_SAMPLES = 1800  # samples across a full charge (~2.5 mAh each at 4.5 Ah)


def slow_charge_capture(
    lli,
    lam_pe,
    lam_ne,
    r_factor,
    temp_c,
    rng: np.random.Generator,
    c_rate: float = CAPTURE_C_RATE,
    n: int = N_SAMPLES,
    current_gain=1.0,
):
    """Terminal-voltage samples of a CC charge from empty to full, per row.

    Returns (v_samples (rows, n), measured dq_per_sample (rows,)).
    Terminal voltage = OCV + I * R_dc(T, age) + charge hysteresis + noise, then ADC.
    `current_gain` is the (uncalibrated) gain error of the current sense path; it
    scales the charge the BMS *believes* it added, as in real coulomb counting.
    """
    em = electrode_model()
    w = em.window(lli, lam_pe, lam_ne)
    cap = w.capacity_ah
    q_charged = cap[:, None] * np.linspace(0.0, 1.0, n)
    q_dis = cap[:, None] - q_charged
    ocv = cathode_ocp(w.y100[:, None] + q_dis / w.cpe[:, None]) - anode_ocp(w.x100[:, None] - q_dis / w.cne[:, None])
    current = c_rate * CELL.rated_capacity_ah
    r_dc = (
        (CELL.r0_bol_ohm + CELL.r1_bol_ohm + CELL.r2_bol_ohm)
        * np.asarray(r_factor)
        * resistance_temp_factor(np.asarray(temp_c))
    )
    v = ocv + (current * r_dc)[:, None] + 0.008
    v = v + V_NOISE_V * rng.standard_normal(v.shape)
    v = np.round(v / ADC_LSB_V) * ADC_LSB_V
    return v, cap / (n - 1) * np.asarray(current_gain)
