"""State-of-charge estimation: legacy coulomb counting vs a recalibrated EKF.

Legacy: SoC = SoC0 - integral(I)/Q_rated. Accuracy decays as the cell loses
capacity (Q_rated is stale) and the current sensor drifts.

Here: an extended Kalman filter on a 1RC equivalent circuit whose parameters
(capacity, OCV-SoC table, R0, R1, C1) come from the cloud twin via a signed
OTA calibration package. The voltage measurement continuously corrects the
coulomb-counting drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CELL


@dataclass
class SoCCalibration:
    capacity_ah: float
    soc_table: np.ndarray
    ocv_table: np.ndarray
    r0: float
    r1: float
    c1: float
    version: str = "soc-cal-bol"

    def to_dict(self) -> dict:
        return {
            "capacity_ah": self.capacity_ah,
            "soc_table": np.round(self.soc_table, 5).tolist(),
            "ocv_table": np.round(self.ocv_table, 5).tolist(),
            "r0": self.r0,
            "r1": self.r1,
            "c1": self.c1,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SoCCalibration:
        return cls(
            d["capacity_ah"],
            np.array(d["soc_table"]),
            np.array(d["ocv_table"]),
            d["r0"],
            d["r1"],
            d["c1"],
            d.get("version", "soc-cal"),
        )


def coulomb_count(
    current_a: np.ndarray, dt_s: float, soc0: float, capacity_ah: float = CELL.rated_capacity_ah
) -> np.ndarray:
    return np.clip(soc0 - np.cumsum(current_a) * dt_s / 3600.0 / capacity_ah, 0.0, 1.0)


class SoCEKF:
    """2-state EKF: x = [SoC, v_rc]. Current > 0 is discharge."""

    def __init__(
        self,
        cal: SoCCalibration,
        soc0: float,
        p0: float = 0.01,
        q_soc: float = 1e-9,
        q_v: float = 1e-7,
        r_v: float = 4e-5,
    ):
        self.cal = cal
        self.x = np.array([soc0, 0.0])
        self.P = np.diag([p0, 1e-4])
        self.Q = np.diag([q_soc, q_v])
        self.R = r_v

    def _ocv(self, soc):
        return np.interp(soc, self.cal.soc_table, self.cal.ocv_table)

    def _docv(self, soc, h=1e-3):
        return (self._ocv(min(soc + h, 1.0)) - self._ocv(max(soc - h, 0.0))) / (2 * h)

    def step(self, current_a: float, v_meas: float, dt_s: float) -> float:
        c = self.cal
        a = np.exp(-dt_s / (c.r1 * c.c1))
        F = np.array([[1.0, 0.0], [0.0, a]])
        self.x = np.array(
            [self.x[0] - current_a * dt_s / 3600.0 / c.capacity_ah, a * self.x[1] + c.r1 * (1 - a) * current_a]
        )
        self.P = F @ self.P @ F.T + self.Q
        H = np.array([self._docv(self.x[0]), -1.0])
        v_pred = self._ocv(self.x[0]) - current_a * c.r0 - self.x[1]
        S = H @ self.P @ H + self.R
        K = self.P @ H / S
        self.x = self.x + K * (v_meas - v_pred)
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        return float(self.x[0])

    def run(self, current_a: np.ndarray, v_meas: np.ndarray, dt_s: float) -> np.ndarray:
        return np.array([self.step(float(i), float(v), dt_s) for i, v in zip(current_a, v_meas)])
