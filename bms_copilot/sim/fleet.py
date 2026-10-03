"""Vectorised fleet simulator.

Ages every device day by day (all devices x 3 series cells in lockstep) under
its usage profile, climate, charging policy and any injected defect, and records
a telemetry snapshot every `snapshot_every_days`. Snapshots contain what the
device would upload (ICA features, DCIR, precursor signals, usage statistics)
and, separately, the hidden ground truth used for training labels and scoring.

Two cohorts are generated:
* history - retired devices simulated to end of life; the cloud trains on them
* live    - today's installed base at mixed ages; the future is simulated too
            but is only used to *score* predictions, never as model input.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ..config import CELL, MODELS, SERVICE
from ..edge.ica import incremental_capacity
from ..physics.aging import AgingModel, DailyStress, DefectMultipliers
from ..physics.electrode import DegradationState, electrode_model
from .profiles import CLIMATE_WEIGHTS, CLIMATES, PROFILES, adaptive_transform
from .sensors import slow_charge_capture

TODAY = date(2026, 9, 26)
GOOD_LOTS = [f"L23{n:02d}" for n in range(1, 13)]
# Scenario: one supplier lot with an anode coating defect (40 % of its packs
# develop accelerated anode active-material loss 30-300 days into service).
BAD_LOT = "L2317"
BAD_LOT_SHARE = 0.12
BAD_LOT_DEFECT_RATE = 0.40
STATE_FIELDS = ["sei_z", "lli_cycle", "lli_plating", "lam_pe", "lam_ne", "r_extra"]
TELEMETRY_FIELDS = [
    "age_days",
    "efc_total",
    "efc_rate",
    "temp_mean",
    "soc_mean",
    "frac_high_mean",
    "charge_c_mean",
    "p_efc",
    "p_temp_mean",
    "p_temp_max",
    "p_soc_mean",
    "p_frac_high",
    "p_charge_c",
    "r0_mohm",
    "capture_temp_c",
    "cell_divergence_mv",
    "self_discharge_pct_day",
    "thermal_residual_c",
    "strain_ue",
    "overtemp_hours",
    "deep_discharges",
    "gauge_soh",
]


@dataclass
class DeviceSpec:
    device_id: str
    serial: str
    profile: str
    climate: str
    lot: str
    adaptive: bool
    extended_warranty: bool
    l5_consent: bool
    defect: str | None
    defect_onset_day: int | None
    defect_cell: int | None
    age_days: int
    sale_date: str
    current_gain: float = 1.0  # current-sense gain error (hidden truth)

    def warranty_days(self) -> int:
        return SERVICE.warranty_days_extended if self.extended_warranty else SERVICE.warranty_days_standard


@dataclass
class FleetData:
    cohort: str
    devices: list[DeviceSpec]
    snap_days: np.ndarray  # (S,) device-relative day of each snapshot
    tele: dict[str, np.ndarray]  # field -> (N, S)
    ica: np.ndarray  # (N, S, 9)
    truth_soh: np.ndarray  # (N, S)
    truth_state: dict[str, np.ndarray]  # field -> (N, S, 3)
    truth_limiting_cell: np.ndarray  # (N, S)
    eol_day: np.ndarray  # (N,) nan if not reached in horizon
    meta: dict = field(default_factory=dict)

    @property
    def n_observed(self) -> np.ndarray:
        """Snapshots already uploaded (live cohort: up to today)."""
        return np.searchsorted(self.snap_days, [d.age_days for d in self.devices], side="right")

    def index(self, device_id: str) -> int:
        for i, d in enumerate(self.devices):
            if d.device_id == device_id:
                return i
        raise KeyError(device_id)

    def save(self, path: Path) -> None:
        arrays = {
            "snap_days": self.snap_days,
            "ica": self.ica,
            "truth_soh": self.truth_soh,
            "truth_limiting_cell": self.truth_limiting_cell,
            "eol_day": self.eol_day,
        }
        arrays.update({f"tele__{k}": v for k, v in self.tele.items()})
        arrays.update({f"state__{k}": v for k, v in self.truth_state.items()})
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(
                {"cohort": self.cohort, "meta": self.meta, "devices": [asdict(d) for d in self.devices]}, indent=1
            )
        )

    @classmethod
    def load(cls, path: Path) -> FleetData:
        z = np.load(path.with_suffix(".npz"))
        j = json.loads(path.with_suffix(".json").read_text())
        return cls(
            cohort=j["cohort"],
            devices=[DeviceSpec(**d) for d in j["devices"]],
            snap_days=z["snap_days"],
            tele={k[6:]: z[k] for k in z.files if k.startswith("tele__")},
            ica=z["ica"],
            truth_soh=z["truth_soh"],
            truth_state={k[7:]: z[k] for k in z.files if k.startswith("state__")},
            truth_limiting_cell=z["truth_limiting_cell"],
            eol_day=z["eol_day"],
            meta=j["meta"],
        )


class FleetSimulator:
    def __init__(
        self,
        n_devices: int,
        cohort: str = "live",
        seed: int = 7,
        horizon_days: int = 8 * 365,
        adaptive_share: float = 0.35,
    ):
        self.n = n_devices
        self.cohort = cohort
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon_days
        self.adaptive_share = adaptive_share
        self.aging = AgingModel()
        self.em = electrode_model()

    # ------------------------------------------------------------------ devices
    def _sample_devices(self) -> list[DeviceSpec]:
        rng = self.rng
        pkeys = list(PROFILES)
        pw = np.array([PROFILES[k].weight for k in pkeys])
        ckeys = list(CLIMATES)
        cw = np.array([CLIMATE_WEIGHTS[k] for k in ckeys])
        devices = []
        for i in range(self.n):
            lot = BAD_LOT if rng.random() < BAD_LOT_SHARE else str(rng.choice(GOOD_LOTS))
            u = rng.random()
            defect = None
            if lot == BAD_LOT and u < BAD_LOT_DEFECT_RATE or lot != BAD_LOT and u < 0.012:
                defect = "anode_defect"
            elif u > 0.965:
                defect = "swelling"
            elif u > 0.950:
                defect = "micro_short"
            if self.cohort == "live":
                age = int(rng.integers(42, 1184))
                sale = TODAY - timedelta(days=age)
            else:
                age = self.horizon
                sale = date(2017, 1, 1) + timedelta(days=int(rng.integers(0, 900)))
            onset = None
            if defect == "anode_defect":
                onset = int(rng.integers(30, 300 if lot == BAD_LOT else 420))
            elif defect is not None:
                onset = int(rng.integers(60, 1000))
            adaptive = bool(rng.random() < self.adaptive_share)
            prefix = "DEV" if self.cohort == "live" else "HIS"
            devices.append(
                DeviceSpec(
                    device_id=f"{prefix}-{i + 1:04d}",
                    serial=f"BP{lot[1:]}{rng.integers(10**6, 10**7)}",
                    profile=str(rng.choice(pkeys, p=pw / pw.sum())),
                    climate=str(rng.choice(ckeys, p=cw / cw.sum())),
                    lot=lot,
                    adaptive=adaptive,
                    extended_warranty=bool(rng.random() < 0.30),
                    l5_consent=bool(adaptive and rng.random() < 0.45),
                    defect=defect,
                    defect_onset_day=onset,
                    defect_cell=int(rng.integers(0, 3)) if defect else None,
                    age_days=age,
                    sale_date=sale.isoformat(),
                )
            )
        return devices

    # --------------------------------------------------------------------- run
    def run(self) -> FleetData:
        rng, n = self.rng, self.n
        devs = self._sample_devices()
        prof = [PROFILES[d.profile] for d in devs]
        col = lambda attr: np.array([getattr(p, attr) for p in prof], dtype=float)  # noqa: E731
        efc_mean, wkend, soc0, high0 = col("efc_mean"), col("weekend_factor"), col("mean_soc"), col("frac_high_soc")
        chg0, dod, rise, fast_share = col("charge_c"), col("dod"), col("load_temp_rise_c"), col("fast_charge_share")
        ambient_mean = np.array([CLIMATES[d.climate] for d in devs])
        adaptive = np.array([d.adaptive for d in devs])
        phase = np.array([date.fromisoformat(d.sale_date).timetuple().tm_yday for d in devs], dtype=float)
        weekday0 = np.array([date.fromisoformat(d.sale_date).weekday() for d in devs])

        defect = np.array([d.defect or "" for d in devs])
        onset = np.array([d.defect_onset_day if d.defect_onset_day is not None else 10**9 for d in devs])
        dcell = np.array([d.defect_cell if d.defect_cell is not None else -1 for d in devs])
        cell_is_def = np.arange(3)[None, :] == dcell[:, None]

        # Cell-to-cell manufacturing spread.
        m_lli = rng.lognormal(0.0, 0.05, (n, 3))
        m_lam = rng.lognormal(0.0, 0.08, (n, 3))

        # Fresh cells are not identical: electrode balance / capacity spread.
        state = DegradationState.zeros((n, 3))
        state.lam_pe = np.abs(rng.normal(0.0, 0.008, (n, 3)))
        state.lam_ne = np.abs(rng.normal(0.0, 0.008, (n, 3)))
        state.sei_z = rng.normal(0.0, 0.006, (n, 3)) ** 2
        self.current_gain = rng.normal(1.0, 0.006, n)  # per-device current-sense gain error
        for dev, g in zip(devs, self.current_gain):
            dev.current_gain = float(g)
        every = MODELS.snapshot_every_days
        snap_days = np.arange(every, self.horizon + 1, every)
        S = len(snap_days)
        tele = {k: np.zeros((n, S)) for k in TELEMETRY_FIELDS}
        ica = np.zeros((n, S, 9))
        truth_soh = np.zeros((n, S))
        truth_state = {k: np.zeros((n, S, 3)) for k in STATE_FIELDS}
        limiting = np.zeros((n, S), dtype=int)

        efc_total = np.zeros(n)
        life = {k: np.zeros(n) for k in ("temp", "soc", "high", "chg")}
        per = {k: np.zeros(n) for k in ("efc", "temp", "soc", "high", "chg")}
        per_tmax = np.full(n, -99.0)
        overtemp_h = np.zeros(n)
        deep = np.zeros(n)
        s = 0
        for d in range(1, self.horizon + 1):
            weekend = ((weekday0 + d) % 7) >= 5
            efc = efc_mean * np.where(weekend, wkend, 1.0) * rng.lognormal(0.0, 0.25, n)
            outdoor = ambient_mean + 6.0 * np.sin(2 * np.pi * (d + phase - 110) / 365.0)
            amb = 21.0 + 0.35 * (outdoor - 16.0) + rng.normal(0, 1.0, n)  # indoor ambient
            fast = rng.random(n) < fast_share
            chg = np.where(fast, np.maximum(chg0, 1.0), chg0)
            soc_m, high = soc0.copy(), high0.copy()
            a_soc, a_high, a_chg = adaptive_transform(soc_m, high, chg, fast)
            soc_m = np.where(adaptive, a_soc, soc_m)
            high = np.where(adaptive, a_high, high)
            chg = np.where(adaptive, a_chg, chg)
            temp = amb + 3.0 + rise * np.clip(efc / np.maximum(efc_mean, 1e-3), 0.3, 2.0)
            chg_temp = amb + 2.0 + 5.0 * chg

            # Defect evolution
            t_def = d - onset
            active = t_def > 0
            anode = active & (defect == "anode_defect")
            swell = active & (defect == "swelling")
            mult = DefectMultipliers(
                lli=m_lli * np.where(cell_is_def & anode[:, None], 1.5, 1.0),
                lam_ne=m_lam * np.where(cell_is_def & anode[:, None], 8.0, 1.0),
                lam_pe=m_lam * np.where(cell_is_def & swell[:, None], 2.0, 1.0),
            )
            stress = DailyStress(
                efc=efc[:, None],
                mean_soc=soc_m[:, None],
                frac_high_soc=high[:, None],
                charge_c=chg[:, None],
                temp_c=temp[:, None],
                charge_temp_c=chg_temp[:, None],
                dod=dod[:, None],
            )
            state = self.aging.step(state, stress, mult=mult)
            state.r_extra = state.r_extra + np.where(cell_is_def & swell[:, None], 0.0035, 0.0)

            efc_total += efc
            for k, v in (("temp", temp), ("soc", soc_m), ("high", high), ("chg", chg)):
                life[k] += v
            for k, v in (("efc", efc), ("temp", temp), ("soc", soc_m), ("high", high), ("chg", chg)):
                per[k] += v
            per_tmax = np.maximum(per_tmax, temp + 4.0)
            overtemp_h += np.maximum(0.0, temp + 4.0 - 45.0) * 0.5
            deep += rng.random(n) < 0.004 * efc

            if d == snap_days[s]:
                self._snapshot(
                    s,
                    d,
                    state,
                    devs,
                    tele,
                    ica,
                    truth_soh,
                    truth_state,
                    limiting,
                    efc_total,
                    life,
                    per,
                    per_tmax,
                    overtemp_h,
                    deep,
                    amb,
                    t_def,
                    defect,
                    cell_is_def,
                )
                per = {k: np.zeros(n) for k in per}
                per_tmax = np.full(n, -99.0)
                s += 1
                if s == S:
                    break

        eol = np.full(n, np.nan)
        for i in range(n):
            below = np.nonzero(truth_soh[i] <= SERVICE.eol_soh)[0]
            if len(below):
                k = below[0]
                if k == 0:
                    eol[i] = snap_days[0]
                else:
                    s0, s1 = truth_soh[i, k - 1], truth_soh[i, k]
                    frac = (s0 - SERVICE.eol_soh) / max(s0 - s1, 1e-9)
                    eol[i] = snap_days[k - 1] + frac * (snap_days[k] - snap_days[k - 1])
        return FleetData(
            self.cohort,
            devs,
            snap_days,
            tele,
            ica,
            truth_soh,
            truth_state,
            limiting,
            eol,
            meta={"today": TODAY.isoformat(), "horizon_days": self.horizon},
        )

    def _snapshot(
        self,
        s,
        d,
        state,
        devs,
        tele,
        ica,
        truth_soh,
        truth_state,
        limiting,
        efc_total,
        life,
        per,
        per_tmax,
        overtemp_h,
        deep,
        amb,
        t_def,
        defect,
        cell_is_def,
    ):
        rng, n, every = self.rng, self.n, MODELS.snapshot_every_days
        caps = self.em.capacity(state)  # (n, 3)
        lim = np.argmin(caps, axis=1)
        rows = np.arange(n)
        lim_state = state.take((rows, lim))
        truth_soh[:, s] = caps[rows, lim] / CELL.rated_capacity_ah
        limiting[:, s] = lim
        for k in STATE_FIELDS:
            truth_state[k][:, s] = getattr(state, k)

        rf = state.resistance_factor()
        capture_temp = amb + 1.5
        v, dq = slow_charge_capture(
            lim_state.lli,
            lim_state.lam_pe,
            lim_state.lam_ne,
            rf[rows, lim],
            capture_temp,
            rng,
            current_gain=self.current_gain,
        )
        ica[:, s] = incremental_capacity(v, dq).features

        tdef = np.maximum(t_def, 0)
        swell = (defect == "swelling") & (t_def > 0)
        short = (defect == "micro_short") & (t_def > 0)
        short_level = np.where(short, 1.0 - np.exp(-tdef / 90.0), 0.0)
        spread = np.std(caps / caps.mean(axis=1, keepdims=True), axis=1)

        tele["age_days"][:, s] = d
        tele["efc_total"][:, s] = efc_total
        tele["efc_rate"][:, s] = efc_total / d
        tele["temp_mean"][:, s] = life["temp"] / d
        tele["soc_mean"][:, s] = life["soc"] / d
        tele["frac_high_mean"][:, s] = life["high"] / d
        tele["charge_c_mean"][:, s] = life["chg"] / d
        tele["p_efc"][:, s] = per["efc"] / every
        tele["p_temp_mean"][:, s] = per["temp"] / every
        tele["p_temp_max"][:, s] = per_tmax
        tele["p_soc_mean"][:, s] = per["soc"] / every
        tele["p_frac_high"][:, s] = per["high"] / every
        tele["p_charge_c"][:, s] = per["chg"] / every
        tele["r0_mohm"][:, s] = 1000 * CELL.r0_bol_ohm * rf.sum(axis=1) * rng.lognormal(0, 0.03, n)
        tele["capture_temp_c"][:, s] = capture_temp
        tele["cell_divergence_mv"][:, s] = 4.0 + 900.0 * spread + 60.0 * short_level + np.abs(rng.normal(0, 1.5, n))
        tele["self_discharge_pct_day"][:, s] = np.abs(0.04 + rng.normal(0, 0.008, n) + 1.5 * short_level)
        tele["thermal_residual_c"][:, s] = (
            rng.normal(0, 0.35, n) + np.where(swell, 0.012 * tdef, 0.0) + 0.8 * short_level
        )
        tele["strain_ue"][:, s] = (
            40 + 0.025 * d + rng.normal(0, 4, n) + np.where(swell, 3.0 * tdef + 0.02 * tdef**2, 0.0)
        )
        tele["overtemp_hours"][:, s] = overtemp_h
        tele["deep_discharges"][:, s] = deep
        # Legacy gauge: SoH inferred from the cycle counter via a static table.
        tele["gauge_soh"][:, s] = np.clip(1.0 - 0.2 * efc_total / CELL.expected_cycle_life, 0.0, 1.0)
