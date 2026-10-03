"""Runtime engine: loads trained artifacts and serves every copilot operation.

Used by the REST API, the agentic workforce and the LLM copilot. The cloud
side only consumes telemetry the devices upload; when an agent asks a device
for a capture, `sim.device` produces it from the simulator's hidden state
through the same sensor model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

from .agents.workforce import LEVELS, Workforce, fade_rate, recent_levels
from .charging.context import predict_need
from .charging.milp import plan_and_verify
from .charging.rl import QPolicy
from .charging.session import CellModelSet, ChargeContext, ChargePhysics, legacy_policy, rollout, with_deadline_guard
from .cloud.dma import fit_modes
from .cloud.grading import grade_battery, power_capability_pct
from .cloud.health import HealthModels, soh_inputs
from .cloud.ota import DeviceVerifier, PackageSigner, load_or_create_keys
from .cloud.passport import PassportStore, build_record, qr_svg, render
from .cloud.twin import ResidualModel, modes_from_soh, project_soh, soc_benchmark, soc_calibration, state_from_modes
from .config import ARTIFACTS, CELL, PACK, SERVICE
from .edge.anomaly import AnomalyCalibration, PrecursorDetector, signal_matrix
from .edge.ica import ICACalibration
from .edge.tinyml import int8_session_policy, load_int8
from .physics.aging import AgingModel, DailyStress
from .physics.electrode import DegradationState, electrode_model
from .safety.envelope import SafetySupervisor, Tightening
from .sim.device import diagnostic_capture, ica_curve
from .sim.fleet import FleetData
from .sim.profiles import CLIMATES, PROFILES

SIM_NOW = datetime(2026, 9, 26, 21, 30, tzinfo=UTC)


class Engine:
    def __init__(self, artifacts: Path = ARTIFACTS, now: datetime = SIM_NOW):
        if not (artifacts / "model_report.json").exists():
            raise FileNotFoundError("Artifacts missing - run `python -m bms_copilot.train` first.")
        self.art = artifacts
        self.now = now
        self.fleet = FleetData.load(artifacts / "fleet_live")
        self.models = HealthModels.from_arrays(np.load(artifacts / "health_models.npz"))
        self.anomaly_cal = AnomalyCalibration.from_dict(
            json.loads((artifacts / "anomaly_calibration.json").read_text())
        )
        self.residual = ResidualModel(np.load(artifacts / "twin_residual.npz")["weights"])
        self.q = QPolicy(np.load(artifacts / "q_policy.npy"))
        self.int8 = load_int8(artifacts / "edge" / "charge_policy_int8.npz")
        self.report = json.loads((artifacts / "model_report.json").read_text())
        priv, self.public_key, key_id = load_or_create_keys(artifacts / "keys")
        self.signer = PackageSigner(priv, key_id)
        self.state_dir = artifacts / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.dev_state: dict = self._load("device_state.json", {})
        self.passports = PassportStore(self._load("passports.json", {}))
        wf = self._load("workforce.json", {"level": 3, "actions": []})
        self.workforce = Workforce(self, level=wf["level"], actions=wf["actions"])
        self._ids = {d.device_id: i for i, d in enumerate(self.fleet.devices)}
        self.battery = None  # LiveBatteryService, attached by the API server
        self._cache: dict = {}
        self._cohort_fit()

    # ------------------------------------------------------------ persistence
    def _load(self, name, default):
        p = self.state_dir / name
        return json.loads(p.read_text()) if p.exists() else copy.deepcopy(default)

    def save_state(self) -> None:
        (self.state_dir / "device_state.json").write_text(json.dumps(self.dev_state, indent=1, default=float))
        (self.state_dir / "passports.json").write_text(json.dumps(self.passports.versions, default=float))
        (self.state_dir / "workforce.json").write_text(
            json.dumps(
                {"level": self.workforce.level, "actions": [a.to_dict() for a in self.workforce.actions]},
                indent=1,
                default=float,
            )
        )

    def reset_state(self) -> None:
        for p in self.state_dir.glob("*.json"):
            p.unlink()
        self.dev_state, self.passports = {}, PassportStore()
        self.workforce = Workforce(self, level=3)
        self._cache.clear()
        self._cohort_fit()

    def _ds(self, did: str) -> dict:
        return self.dev_state.setdefault(
            did, {"adaptive_by_agent": False, "tightenings": [], "ota": {}, "status": "original", "events": []}
        )

    # --------------------------------------------------------------- lookups
    @property
    def devices(self):
        return self.fleet.devices

    def idx(self, did: str) -> int:
        try:
            return self._ids[did]
        except KeyError:
            raise KeyError(f"unknown device {did}") from None

    def device(self, did: str):
        return self.devices[self.idx(did)]

    def _k(self, i: int) -> int:
        return int(self.fleet.n_observed[i]) - 1

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ------------------------------------------------------------- analytics
    def soh_history(self, i: int):
        return self._memo(("soh", i), lambda: self.models.soh_history(self.fleet, i))

    def usage(self, i: int) -> dict:
        k, t = self._k(i), self.fleet.tele
        return {
            "efc_rate": float(t["efc_rate"][i, k]),
            "temp_mean": float(t["temp_mean"][i, k]),
            "soc_mean": float(t["soc_mean"][i, k]),
            "frac_high_mean": float(t["frac_high_mean"][i, k]),
            "charge_c_mean": float(t["charge_c_mean"][i, k]),
            "efc_total": float(t["efc_total"][i, k]),
            "recent_daily_efc": float(t["p_efc"][i, max(0, k - 3) : k + 1].mean()),
            "dod": PROFILES[self.devices[i].profile].dod,
        }

    def _expected_state(self, i: int):
        """Physics expectation for this device's age and usage: (SoH, LAM_ne)."""
        u, age = self.usage(i), int(self.fleet.tele["age_days"][i, self._k(i)])
        s = DailyStress(
            u["efc_rate"],
            u["soc_mean"],
            u["frac_high_mean"],
            u["charge_c_mean"],
            u["temp_mean"],
            u["temp_mean"] + 2 + 5 * u["charge_c_mean"],
            u["dod"],
        )
        st, model = DegradationState(), AgingModel()
        for _ in range(age // 30):
            st = model.step(st, s, days=30)
        st = model.step(st, s, days=age % 30)
        return float(electrode_model().capacity(st)) / CELL.rated_capacity_ah, float(st.lam_ne)

    def modes_est(self, i: int) -> dict:
        def calc():
            k = self._k(i)
            est = self.models.estimate_modes(soh_inputs(self.fleet, np.array([i]), np.array([k])))
            return {m: {"value": float(v[0][0]), "std": float(v[1][0])} for m, v in est.items()}

        return self._memo(("modes", i), calc)

    def _cohort_fit(self) -> None:
        """Expected SoH / anode LAM for each device's age and usage (physics),
        and the fleet spread of residuals, so shortfalls become z-scores."""
        exp = [self._expected_state(i) for i in range(len(self.devices))]
        exp_soh = np.array([e[0] for e in exp])
        exp_ne = np.array([e[1] for e in exp])
        est = np.array([self.soh_history(i)[1][-1] for i in range(len(self.devices))])

        def robust(r):
            med = float(np.median(r))
            return med, max(float(1.4826 * np.median(np.abs(r - med))), 1e-6)

        med, sigma = robust(est - exp_soh)
        self._expected = exp_soh + med
        self._sigma = max(sigma, 0.004)
        if self.models.mode_gps:
            ne = np.array([self.modes_est(i)["lam_ne"]["value"] for i in range(len(self.devices))])
            med_ne, sig_ne = robust(ne - exp_ne)
            self._lam_ne_z = (ne - exp_ne - med_ne) / max(sig_ne, 0.002)
        else:
            self._lam_ne_z = np.zeros(len(self.devices))

    def anomaly(self, i: int):
        def scan():
            t = {k: self.fleet.tele[k][i] for k in self.fleet.tele}
            n = self._k(i) + 1
            X = signal_matrix(
                t, self.fleet.snap_days, self.anomaly_cal.strain_intercept, self.anomaly_cal.strain_slope_per_day
            )[:n]
            return PrecursorDetector(self.anomaly_cal).scan(X, self.fleet.snap_days[:n])

        return self._memo(("anom", i), scan)

    def rul(self, i: int) -> dict:
        def calc():
            _, soh, std = self.soh_history(i)
            if soh[-1] + 1.0 * std[-1] <= SERVICE.eol_soh:
                # Outside the RUL model's training domain: first life already over.
                return {
                    "median_days": 0,
                    "lo_days": 0,
                    "hi_days": 0,
                    "interval": "95%",
                    "std_log": 0.0,
                    "status": "past_end_of_first_life",
                }
            k = self._k(i)
            out = self.models.predict_rul(self.models.rul_features(self.fleet, i, k, soh)[None, :])[0].to_dict()
            out["status"] = "forecast"
            return out

        return self._memo(("rul", i), calc)

    def health(self, did: str) -> dict:
        def calc():
            i = self.idx(did)
            dev, k = self.devices[i], self._k(i)
            days, soh, std = self.soh_history(i)
            t = self.fleet.tele
            res = self.anomaly(i)
            r0 = t["r0_mohm"][i, : k + 1]
            r0_growth = float(100 * (np.median(r0[-3:]) / np.median(r0[:3]) - 1))
            age = int(t["age_days"][i, k])
            ds = self._ds(did)
            return {
                "device_id": did,
                "profile": dev.profile,
                "profile_label": PROFILES[dev.profile].label,
                "climate": dev.climate,
                "lot": dev.lot,
                "age_days": age,
                "sale_date": dev.sale_date,
                "adaptive": bool(dev.adaptive or ds["adaptive_by_agent"]),
                "adaptive_by_agent": ds["adaptive_by_agent"],
                "l5_consent": dev.l5_consent,
                "soh": float(soh[-1]),
                "soh_std": float(std[-1]),
                "soh_interval": [float(soh[-1] - 1.96 * std[-1]), float(soh[-1] + 1.96 * std[-1])],
                "gauge_soh_legacy": float(t["gauge_soh"][i, k]),
                "expected_soh": float(self._expected[i]),
                "soh_z": float((soh[-1] - self._expected[i]) / self._sigma),
                "fade_pct_per_100d": fade_rate(days[-13:], soh[-13:]),
                "modes_est": {m: v["value"] for m, v in self.modes_est(i).items()},
                "modes_est_std": {m: v["std"] for m, v in self.modes_est(i).items()},
                "lam_ne_excess_z": float(self._lam_ne_z[i]),
                "rul": self.rul(i),
                "warranty_days_left": dev.warranty_days() - age,
                "extended_warranty": dev.extended_warranty,
                "anomaly": res[-1].to_dict(),
                "anomaly_recent_levels": recent_levels(res),
                "anomaly_max_level": max((r.level for r in res), key=["normal", "watch", "warning", "critical"].index),
                "usage": self.usage(i),
                "r0_mohm": float(r0[-1]),
                "r0_growth_pct": r0_growth,
                "calibration_stale": "calibration" not in ds["ota"],
                "derated": bool(ds["tightenings"]),
                "status": ds["status"],
            }

        return self._memo(("health", did), calc)

    def invalidate(self, did: str | None = None) -> None:
        if did is None:
            self._cache.clear()
        else:
            self._cache.pop(("health", did), None)

    # ------------------------------------------------------------ fleet view
    def fleet_overview(self) -> dict:
        rows = [self.health(d.device_id) for d in self.devices]
        soh = np.array([r["soh"] for r in rows])
        levels = [r["anomaly"]["level"] for r in rows]
        flagged = [r for r in rows if self.workforce.predictive.assess(r)]
        hp = self.report["health"]
        return {
            "now": self.now.isoformat(),
            "kpis": {
                "devices": len(rows),
                "mean_soh": float(soh.mean()),
                "below_85pct": int((soh < 0.85).sum()),
                "below_eol": int((soh <= SERVICE.eol_soh).sum()),
                "rul_under_180d": int(sum(r["rul"]["median_days"] < 180 for r in rows)),
                "anomalies": {lv: levels.count(lv) for lv in ("watch", "warning", "critical")},
                "adaptive_share": float(np.mean([r["adaptive"] for r in rows])),
                "in_warranty": int(sum(r["warranty_days_left"] > 0 for r in rows)),
                "flagged_for_triage": len(flagged),
                "soh_mae_ai": hp["soh"]["mae_ai"],
                "soh_mae_legacy_gauge": hp["soh"]["mae_gauge_baseline"],
            },
            "soh_histogram": np.histogram(soh, bins=np.arange(0.6, 1.0001, 0.025))[0].tolist(),
            "soh_bins": np.round(np.arange(0.6, 1.0001, 0.025), 3).tolist(),
            "devices": [
                {
                    "device_id": r["device_id"],
                    "profile": r["profile"],
                    "profile_label": r["profile_label"],
                    "climate": r["climate"],
                    "lot": r["lot"],
                    "age_days": r["age_days"],
                    "soh": r["soh"],
                    "soh_std": r["soh_std"],
                    "gauge_soh_legacy": r["gauge_soh_legacy"],
                    "soh_z": r["soh_z"],
                    "rul": r["rul"],
                    "anomaly": r["anomaly"]["level"],
                    "adaptive": r["adaptive"],
                    "warranty_days_left": r["warranty_days_left"],
                    "status": r["status"],
                    "derated": r["derated"],
                }
                for r in rows
            ],
            "workforce": {
                "level": self.workforce.level,
                "level_description": LEVELS[self.workforce.level],
                "open_actions": sum(a.status in ("pending_approval", "awaiting_user") for a in self.workforce.actions),
            },
        }

    def fleet_insights(self) -> dict:
        rows = [self.health(d.device_id) for d in self.devices]
        flagged = {r["device_id"] for r in rows if r["soh_z"] <= -SERVICE.defect_z_threshold}
        z = np.array([r["lam_ne_excess_z"] for r in rows])
        lot_of = np.array([r["lot"] for r in rows])
        base_rate = max(float(np.mean(z > 3)), 0.5 / len(z))
        lots = []
        for lot in sorted(set(lot_of)):
            inside, rest = z[lot_of == lot], z[lot_of != lot]
            if len(inside) < 3:
                continue
            p_shift = float(mannwhitneyu(inside, rest, alternative="greater").pvalue)
            k = int((inside > 3).sum())
            p_out = float(binomtest(k, len(inside), base_rate, alternative="greater").pvalue)
            lots.append(
                {
                    "lot": str(lot),
                    "n": int(len(inside)),
                    "lam_ne_excess_z_mean": float(inside.mean()),
                    "outliers": k,
                    "p_outliers": p_out,
                    "p_shift": p_shift,
                    "p_value": min(1.0, 2 * min(p_out, p_shift)),  # Bonferroni over the two tests
                    "soh_shortfall_flags": int(sum(r["device_id"] in flagged for r in rows if r["lot"] == lot)),
                }
            )
        lots.sort(key=lambda x: x["p_value"])
        # Adaptive vs legacy fade, matched by profile (weights = fleet profile mix).
        per = {}
        for r in rows:
            if r["device_id"] in flagged or r["age_days"] < 120:
                continue
            per.setdefault(r["profile"], {True: [], False: []})[r["adaptive"] and not r["adaptive_by_agent"]].append(
                r["fade_pct_per_100d"]
            )
        ad, lg, w = [], [], []
        for d in per.values():
            if d[True] and d[False]:
                ad.append(np.mean(d[True]))
                lg.append(np.mean(d[False]))
                w.append(len(d[True]) + len(d[False]))
        w = np.array(w, float)
        adaptive = {
            "adaptive_fade_pct_per_100d": float(np.average(ad, weights=w)) if ad else 0.0,
            "legacy_fade_pct_per_100d": float(np.average(lg, weights=w)) if lg else 0.0,
            "profiles_compared": len(ad),
        }
        X = np.array(
            [
                [
                    r["usage"]["temp_mean"],
                    r["usage"]["frac_high_mean"],
                    r["usage"]["charge_c_mean"],
                    r["usage"]["efc_rate"],
                ]
                for r in rows
                if r["device_id"] not in flagged and r["age_days"] >= 120
            ]
        )
        y = np.array([r["fade_pct_per_100d"] for r in rows if r["device_id"] not in flagged and r["age_days"] >= 120])
        drivers = []
        if len(y) > 10:
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
            coef = np.linalg.lstsq(np.c_[Xs, np.ones(len(y))], y, rcond=None)[0][:4]
            names = ["cell temperature", "time above 95 % SoC", "charge rate", "cycling throughput"]
            drivers = sorted(
                [{"driver": n, "effect_pct_per_100d_per_sd": float(c)} for n, c in zip(names, coef)],
                key=lambda d: -abs(d["effect_pct_per_100d_per_sd"]),
            )
        return {
            "lots": lots,
            "lam_ne_excess_z_rest": float(np.median(z)),
            "outlier_rate_fleet": base_rate,
            "soh_shortfall_rate_fleet": len(flagged) / len(rows),
            "adaptive_vs_legacy": adaptive,
            "stress_drivers": drivers,
        }

    # ------------------------------------------------------------ device view
    def _state_for(self, did: str) -> DegradationState:
        h = self.health(did)
        dma = self._cache.get(("dma", did))
        if dma:
            return state_from_modes(dma["lli"], dma["lam_pe"], dma["lam_ne"])
        return state_from_modes(*modes_from_soh(h["soh"]))

    def device_detail(self, did: str) -> dict:
        i, h = self.idx(did), self.health(did)
        days, soh, std = self.soh_history(i)
        k = self._k(i)
        res = self.anomaly(i)
        st = self._state_for(did)
        u = dict(h["usage"])
        legacy_u = dict(u)
        if h["adaptive"]:
            p = PROFILES[self.devices[i].profile]
            legacy_u.update(soc_mean=p.mean_soc, frac_high_mean=p.frac_high_soc, charge_c_mean=p.charge_c)
            u["already_adaptive"] = True
        proj_legacy = project_soh(st, legacy_u, days=3 * 365, adaptive=False, step_days=14)
        proj_adaptive = project_soh(st, u, days=3 * 365, adaptive=True, step_days=14)
        return {
            **h,
            "series": {
                "day": days.tolist(),
                "soh_est": np.round(soh, 4).tolist(),
                "soh_std": np.round(std, 4).tolist(),
                "gauge_soh_legacy": np.round(self.fleet.tele["gauge_soh"][i, : k + 1], 4).tolist(),
                "soh_truth_sim_only": np.round(self.fleet.truth_soh[i, : k + 1], 4).tolist(),
                "r0_mohm": np.round(self.fleet.tele["r0_mohm"][i, : k + 1], 1).tolist(),
                "anomaly_d2": [round(r.d2, 2) for r in res],
                "anomaly_level": [r.level for r in res],
                "strain_ue": np.round(self.fleet.tele["strain_ue"][i, : k + 1], 1).tolist(),
                "thermal_residual_c": np.round(self.fleet.tele["thermal_residual_c"][i, : k + 1], 2).tolist(),
            },
            "whatif": {
                "legacy": proj_legacy,
                "adaptive": proj_adaptive,
                "note": "twin projection from the current estimated state and this device's usage",
            },
            "ica": self.ica_curves(did),
            "diagnosis": self._cache.get(("dma", did)),
            "actions": [a.to_dict() for a in self.workforce.actions if a.device_id == did],
            "tightenings": self._ds(did)["tightenings"],
            "ota_installed": self._ds(did)["ota"],
            "defect_truth_sim_only": self.devices[i].defect,
        }

    def ica_curves(self, did: str) -> list:
        i = self.idx(did)
        k = self._k(i)
        ks = sorted({0, k // 2, k})
        return self._memo(("ica", did), lambda: [ica_curve(self.fleet, i, kk) for kk in ks])

    def diagnose(self, did: str) -> dict:
        def run():
            i = self.idx(did)
            v, dq = diagnostic_capture(self.fleet, i, self._k(i))
            return fit_modes(v, dq).to_dict()

        return self._memo(("dma", did), run)

    def soc_benchmark(self, did: str) -> dict:
        h = self.health(did)
        st = self._state_for(did)
        st.r_extra = max(0.0, 1 + h["r0_growth_pct"] / 100 - float(st.resistance_factor()))
        return soc_benchmark(st)

    # ------------------------------------------------------------- charging
    def supervisor(self, did: str) -> SafetySupervisor:
        sup = SafetySupervisor()
        for t in self._ds(did)["tightenings"]:
            sup.request_tightening(Tightening(**t))
        return sup

    def charge_plan(
        self,
        did: str,
        calendar: list | None = None,
        soc_now: float | None = None,
        ambient_c: float | None = None,
        mode: str | None = None,
    ) -> dict:
        i, h = self.idx(did), self.health(did)
        u = h["usage"]
        need = predict_need(self.devices[i].profile, u["recent_daily_efc"], h["soh"], self.now, calendar)
        soc0 = (
            float(np.clip(1.0 - u["recent_daily_efc"] / max(h["soh"], 0.5), 0.1, 0.8)) if soc_now is None else soc_now
        )
        amb = ambient_c if ambient_c is not None else round(21 + 0.35 * (CLIMATES[self.devices[i].climate] - 16), 1)
        st = self._state_for(did)
        ctx = ChargeContext(
            soc0,
            need.hours_to_unplug,
            need.target_soc,
            self.now.hour + self.now.minute / 60,
            amb,
            st,
            mode or need.mode,
            need.reason,
        )
        models = CellModelSet([st])
        phys = ChargePhysics(models, self.supervisor(did))
        legacy = rollout(ctx, legacy_policy(), phys)
        milp = plan_and_verify(ctx, phys)
        edge = rollout(
            ctx, with_deadline_guard(int8_session_policy(self.int8, h["soh"]), float(models.capacity[0])), phys
        )
        milp_eval = milp["verification"]
        base = max(legacy["damage_pct_capacity"], 1e-12)
        return {
            "context": {
                **need.to_dict(),
                "soc_now": soc0,
                "ambient_c": amb,
                "mode": ctx.mode,
                "effective_limits": phys.sup.effective_limits(),
            },
            "legacy": legacy,
            "milp": {
                **milp_eval,
                "trace": milp["trace"],
                "status": milp["status"],
                "replans": milp["replans"],
                "solver": {
                    "n_variables": milp["n_variables"],
                    "n_binary": milp["n_binary"],
                    "n_constraints": milp["n_constraints"],
                    "objective_eur": milp.get("objective_eur"),
                },
            },
            "edge_rl_int8": edge,
            "comparison": {
                "milp_damage_reduction_pct": 100 * (1 - milp_eval["damage_pct_capacity"] / base),
                "edge_rl_damage_reduction_pct": 100 * (1 - edge["damage_pct_capacity"] / base),
                "note": "damage = capacity loss attributable to this charging session (ageing model)",
            },
        }

    # ------------------------------------------------------ passport & 2nd life
    def _dyn(self, did: str) -> dict:
        i, h = self.idx(did), self.health(did)
        k, t = self._k(i), self.fleet.tele
        r_cell = h["r0_mohm"] / 3 / 1000 + CELL.r1_bol_ohm + CELL.r2_bol_ohm
        rte = 1 - 2 * 0.5 * CELL.rated_capacity_ah * r_cell / CELL.v_nominal
        res = self.anomaly(i)
        return {
            "soh": round(h["soh"], 4),
            "soh_interval": [round(x, 4) for x in h["soh_interval"]],
            "remaining_capacity_ah": round(h["soh"] * PACK.rated_capacity_ah, 3),
            "power_capability_pct": round(power_capability_pct(h["r0_mohm"]), 1),
            "power_fade_pct": round(100 - power_capability_pct(h["r0_mohm"]), 1),
            "round_trip_efficiency": round(rte, 3),
            "r0_mohm": round(h["r0_mohm"], 1),
            "r0_growth_pct": round(h["r0_growth_pct"], 1),
            "self_discharge_history": np.round(t["self_discharge_pct_day"][i, max(0, k - 5) : k + 1], 3).tolist(),
            "efc_total": round(h["usage"]["efc_total"], 1),
            "energy_throughput_kwh": round(h["usage"]["efc_total"] * PACK.rated_energy_wh / 1000, 2),
            "capacity_fade_pct": round(100 * (1 - h["soh"]), 2),
            "rul": h["rul"],
            "modes": self._cache.get(("dma", did)),
            "deep_discharges": int(t["deep_discharges"][i, k]),
            "overtemp_hours": round(float(t["overtemp_hours"][i, k]), 1),
            "anomaly_events": [
                {"day": r.day, "level": r.level, "mechanism": r.mechanism} for r in res if r.level != "normal"
            ][-5:],
            "status": self._ds(did)["status"],
        }

    def second_life(self, did: str) -> dict:
        i, h = self.idx(did), self.health(did)
        k = self._k(i)
        hist = {
            "overtemp_hours": float(self.fleet.tele["overtemp_hours"][i, k]),
            "deep_discharges": float(self.fleet.tele["deep_discharges"][i, k]),
            "temp_mean": h["usage"]["temp_mean"],
            "frac_high_mean": h["usage"]["frac_high_mean"],
            "self_discharge_pct_day": float(self.fleet.tele["self_discharge_pct_day"][i, k]),
        }
        return grade_battery(
            h["soh"],
            h["rul"]["lo_days"],
            h["rul"]["median_days"],
            h["r0_growth_pct"],
            hist,
            h["anomaly_max_level"],
            self._state_for(did),
        ).to_dict()

    def passport(self, did: str, role: str = "public") -> dict:
        dev = self.device(did)
        sl = self._ds(did).get("second_life")
        record = build_record(dev, self._dyn(did), sl, self.now)
        entry = self.passports.publish(record)
        out = render(entry["record"], role)
        out["_version"] = {
            "version": entry["version"],
            "hash": entry["hash"],
            "prev_hash": entry["prev_hash"],
            "chain_valid": self.passports.verify_chain(dev.serial),
        }
        out["_qr_svg"] = qr_svg(record["link"])
        return out

    # ------------------------------------------------------------------ OTA
    def ota_package(self, did: str) -> dict:
        i, h = self.idx(did), self.health(did)
        ds = self._ds(did)
        k = self._k(i)
        feats = self.fleet.ica[i, : k + 1]
        drift = float(np.clip(np.median(feats[-3:, 1]) - np.median(feats[:3, 1]), -0.06, 0.06))
        base = ICACalibration()
        bands = tuple(
            (round(lo + drift, 3), round(hi + drift, 3)) if j < 2 else (lo, hi)
            for j, (lo, hi) in enumerate(base.bands_v)
        )
        version = ds["ota"].get("calibration", 0) + 1
        st = self._state_for(did)
        onnx_path = self.art / "edge" / "charge_policy_int8.onnx"
        payload = {
            "soc_calibration": soc_calibration(st, f"soc-cal-v{version}").to_dict(),
            "ica_calibration": ICACalibration(bands_v=bands, version=f"ica-cal-v{version}").to_dict(),
            "anomaly_calibration": self.anomaly_cal.to_dict(),
            "charge_policy_model": {
                "format": "onnx-int8-qdq",
                "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest() if onnx_path.exists() else None,
            },
            "source": {"soh": h["soh"], "diagnosis": self._cache.get(("dma", did))},
        }
        pkg = self.signer.sign("calibration", payload, version, device_id=did, issued_at=self.now)
        dev = DeviceVerifier(self.public_key, did, installed=dict(ds["ota"]))
        tampered = copy.deepcopy(pkg)
        tampered["payload"]["soc_calibration"]["capacity_ah"] *= 1.2
        ok, why = dev.install(pkg, now=self.now)
        replay_ok, replay_why = dev.verify(pkg, now=self.now)
        t_ok, t_why = DeviceVerifier(self.public_key, did).verify(tampered, now=self.now)
        other_ok, other_why = DeviceVerifier(self.public_key, "DEV-9999").verify(pkg, now=self.now)
        if ok:
            ds["ota"]["calibration"] = version
            self.invalidate(did)
        return {
            "header": pkg["header"],
            "signature": pkg["signature"][:32] + "...",
            "payload_summary": {
                "capacity_ah": payload["soc_calibration"]["capacity_ah"],
                "ocv_points": len(payload["soc_calibration"]["ocv_table"]),
                "ica_bands_v": payload["ica_calibration"]["bands_v"],
                "ica_peak_drift_mv": round(1000 * drift, 1),
                "policy_sha256": payload["charge_policy_model"]["sha256"],
            },
            "device_verification": {"installed": ok, "reason": why},
            "security_checks": {
                "tampered_payload": {"accepted": t_ok, "reason": t_why},
                "replayed_same_version": {"accepted": replay_ok, "reason": replay_why},
                "wrong_device": {"accepted": other_ok, "reason": other_why},
            },
        }

    # --------------------------------------------------------- agent effects
    def _note(self, did, text):
        if did:
            self._ds(did)["events"].append({"at": self.now.isoformat(), "event": text})

    def effect_diagnostic_capture(self, a):
        return {"modes": self.diagnose(a.device_id)}

    def effect_safety_derate(self, a):
        t = {
            "source": a.agent,
            "reason": a.title,
            "v_max": a.params.get("v_max"),
            "max_charge_c": a.params.get("max_charge_c"),
            "charge_temp_max_c": a.params.get("charge_temp_max_c"),
        }
        sup = self.supervisor(a.device_id)
        accepted = sup.request_tightening(Tightening(**t))
        if accepted:
            self._ds(a.device_id)["tightenings"].append(t)
            self.invalidate(a.device_id)
        return {
            "accepted_by_safety_supervisor": accepted,
            "effective_limits": sup.effective_limits(),
            "rejections": [r[1] for r in sup.rejected],
        }

    def effect_enable_adaptive_charging(self, a):
        self._ds(a.device_id)["adaptive_by_agent"] = True
        self.invalidate(a.device_id)
        self._note(a.device_id, "adaptive charging enabled")
        return {"policy": "MILP schedule + INT8 RL edge policy with deadline guard"}

    def effect_ota_recalibration(self, a):
        r = self.ota_package(a.device_id)
        return {
            "version": r["header"]["version"],
            "installed": r["device_verification"]["installed"],
            "reason": r["device_verification"]["reason"],
        }

    def effect_warranty_claim(self, a):
        h = self.health(a.device_id)
        cid = f"WC-{self.now:%y%m%d}-{a.id[1:]}"
        self._note(a.device_id, f"warranty claim {cid}")
        return {
            "claim_id": cid,
            "status": "filed",
            "evidence_bundle": {
                "soh": h["soh"],
                "soh_interval": h["soh_interval"],
                "expected_soh": h["expected_soh"],
                "soh_z": h["soh_z"],
                "rul": h["rul"],
                "modes": self._cache.get(("dma", a.device_id)),
                "anomaly": h["anomaly"],
            },
        }

    def effect_replacement_order(self, a):
        oid = f"RO-{self.now:%y%m%d}-{a.id[1:]}"
        self._note(a.device_id, f"replacement order {oid}")
        return {"order_id": oid, "part": f"{PACK.series}S{PACK.parallel}P pack", "reason": a.params.get("reason")}

    def effect_notify_user(self, a):
        return {"channel": "OS notification + email", "message": a.params.get("message")}

    def effect_proactive_replacement_offer(self, a):
        return {
            "offer_id": f"OF-{a.id[1:]}",
            "discount_pct": a.params.get("discount_pct"),
            "message": a.params.get("message"),
        }

    def effect_logistics_shipment(self, a):
        return {
            "tracking": a.params.get("tracking"),
            "eta": a.params.get("eta"),
            "technician_slot": a.params.get("technician_slot"),
        }

    def effect_asset_recovery(self, a):
        g = a.evidence["grade"]
        status = {"A": "repurposed", "B": "repurposed", "C": "re-used", "F": "waste"}[g["grade"]]
        ds = self._ds(a.device_id)
        ds["status"], ds["second_life"] = status, g
        self.invalidate(a.device_id)
        entry = self.passports.publish(build_record(self.device(a.device_id), self._dyn(a.device_id), g, self.now))
        return {"passport_status": status, "passport_version": entry["version"], "pathway": g["pathway"]}

    def effect_design_insight(self, a):
        return {"filed_to": "next-gen battery design review backlog"}

    def mark_returned(self, did: str) -> None:
        self._ds(did)["status"] = "returned - awaiting assessment"
        self.invalidate(did)

    @property
    def returned(self) -> list[str]:
        return [d for d, s in self.dev_state.items() if s.get("status") == "returned - awaiting assessment"]

    # --------------------------------------------------------------- workforce
    def set_level(self, level: int) -> dict:
        if level not in LEVELS:
            raise ValueError("level must be 1-5")
        self.workforce.level = level
        self.save_state()
        return {"level": level, "description": LEVELS[level]}

    def run_workforce(self) -> dict:
        out = self.workforce.run()
        self.save_state()
        return out

    def decide(self, action_id: str, approve: bool, actor: str = "human") -> dict:
        a = self.workforce.decide(action_id, approve, actor)
        self.save_state()
        return a.to_dict()

    def actions(self, status: str | None = None, device_id: str | None = None, limit: int = 500) -> list[dict]:
        out = [
            a.to_dict()
            for a in reversed(self.workforce.actions)
            if (status is None or a.status == status) and (device_id is None or a.device_id == device_id)
        ]
        return out[:limit]
