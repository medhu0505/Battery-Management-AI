"""Build every artifact the copilot needs: `python -m bms_copilot.train`.

1. Simulate the history cohort (retired devices, observed to end of life) and
   the live installed base (mixed ages; future simulated for scoring only).
2. Train the cloud health models (SoH GPR, RUL GPR) on history; score on
   held-out history devices and on the live fleet (true future hidden).
3. Fit the anomaly baseline on healthy history devices; score false-alarm
   rate and detection lead time on defect devices.
4. Train the twin's residual model; benchmark SoC estimation.
5. Train the RL charge policy; compare legacy vs MILP vs RL.
6. TinyML: distil -> prune -> QAT INT8 -> ONNX, closed-loop parity.
7. Generate OTA signing keys.
All metrics land in artifacts/model_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .charging.rl import evaluate_policies, train_q_policy
from .cloud.health import evaluate, train_health_models
from .cloud.ota import load_or_create_keys
from .cloud.twin import soc_benchmark, state_from_modes, train_residual_model
from .config import ARTIFACTS
from .edge.anomaly import PrecursorDetector, fit_calibration, signal_matrix
from .edge.tinyml import run_pipeline, save_int8
from .sim.fleet import FleetData, FleetSimulator


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def device_signals(fleet: FleetData, i: int, cal=None) -> np.ndarray:
    t = {k: fleet.tele[k][i] for k in fleet.tele}
    return signal_matrix(
        t, fleet.snap_days, cal.strain_intercept if cal else 0.0, cal.strain_slope_per_day if cal else 0.0
    )


def anomaly_report(history: FleetData, live: FleetData, cal) -> dict:
    det = PrecursorDetector(cal)
    L = ["normal", "watch", "warning", "critical"]
    healthy = [i for i, d in enumerate(history.devices) if d.defect is None]
    n = alarms = critical = 0
    for i in healthy[len(healthy) // 2 :]:
        m = history.truth_soh[i] > 0.75
        res = det.scan(device_signals(history, i, cal)[m], history.snap_days[m])
        n += len(res)
        alarms += sum(r.level in ("warning", "critical") for r in res)
        critical += sum(r.level == "critical" for r in res)
    leads, mech_ok = [], []
    for fleet in (history, live):
        for i, d in enumerate(fleet.devices):
            if d.defect not in ("swelling", "micro_short") or d.defect_onset_day + 40 > fleet.snap_days[-1]:
                continue
            res = det.scan(device_signals(fleet, i, cal), fleet.snap_days)
            post = [r for r in res if r.day > d.defect_onset_day]
            first_warn = next((r for r in post if L.index(r.level) >= 2), None)
            first_crit = next((r for r in post if r.level == "critical"), None)
            if first_warn:
                leads.append(
                    {
                        "defect": d.defect,
                        "days_onset_to_warning": first_warn.day - d.defect_onset_day,
                        "warning_before_critical_days": (first_crit.day - first_warn.day) if first_crit else None,
                    }
                )
                expect = "swelling" if d.defect == "swelling" else "short"
                mech_ok.append(expect in (first_warn.mechanism or ""))
    return {
        "healthy_snapshots": n,
        "false_warning_rate": alarms / max(n, 1),
        "false_critical_rate": critical / max(n, 1),
        "defect_devices_detected": len(leads),
        "median_days_onset_to_warning": float(np.median([x["days_onset_to_warning"] for x in leads]))
        if leads
        else None,
        "median_warning_lead_before_critical_days": float(
            np.median([x["warning_before_critical_days"] for x in leads if x["warning_before_critical_days"]])
        )
        if any(x["warning_before_critical_days"] for x in leads)
        else None,
        "mechanism_attribution_accuracy": float(np.mean(mech_ok)) if mech_ok else None,
        "cases": leads,
    }


def live_rul_report(models, live: FleetData) -> dict:
    rows = []
    for i, d in enumerate(live.devices):
        k = int(live.n_observed[i]) - 1
        if k < 2 or np.isnan(live.eol_day[i]) or live.truth_soh[i, k] <= 0.8:
            continue
        _, soh_hist, _ = models.soh_history(live, i, upto=k + 1)
        p = models.predict_rul(models.rul_features(live, i, k, soh_hist)[None, :])[0]
        truth = live.eol_day[i] - live.snap_days[k]
        rows.append((d.adaptive, truth, p.median_days, p.lo_days, p.hi_days))
    out = {}
    for name, sel in (
        ("all", lambda r: True),
        ("adaptive_cohort", lambda r: r[0]),
        ("legacy_cohort", lambda r: not r[0]),
    ):
        r = [x for x in rows if sel(x)]
        if not r:
            continue
        t, m, lo, hi = (np.array([x[j] for x in r]) for j in (1, 2, 3, 4))
        out[name] = {
            "n": len(r),
            "mae_days": float(np.mean(np.abs(m - t))),
            "median_rel_error": float(np.median(np.abs(m - t) / np.maximum(t, 60))),
            "coverage95": float(np.mean((t >= lo) & (t <= hi))),
            "mean_interval_width_days": float(np.mean(hi - lo)),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", type=int, default=200)
    ap.add_argument("--live", type=int, default=240)
    ap.add_argument("--rl-batches", type=int, default=220)
    ap.add_argument("--policy-eval-sessions", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="small, fast build for tests/demos")
    ap.add_argument("--out", default=str(ARTIFACTS), help="artifact directory (default: ./artifacts)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.quick:
        args.history, args.live, args.rl_batches, args.policy_eval_sessions = 80, 60, 80, 30

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "disclaimer": "All metrics are measured on simulated fleets (physics-based simulator). They "
        "demonstrate the pipeline, not field accuracy.",
    }

    history = FleetSimulator(args.history, cohort="history", seed=11).run()
    history.save(out / "fleet_history")
    _log(f"history cohort simulated ({args.history} devices)", t0)
    live = FleetSimulator(args.live, cohort="live", seed=23).run()
    live.save(out / "fleet_live")
    _log(f"live fleet simulated ({args.live} devices)", t0)

    models, rep = train_health_models(history, seed=0)
    np.savez_compressed(out / "health_models.npz", **models.to_arrays())
    rep["live_fleet_rul"] = live_rul_report(models, live)
    rep["live_fleet_soh"] = evaluate(models, live, np.arange(len(live.devices)))["soh"]
    report["health"] = rep
    _log(
        f"health models: SoH MAE {rep['soh']['mae_ai']:.4f} (gauge {rep['soh']['mae_gauge_baseline']:.4f}), "
        f"RUL MAE {rep['rul'].get('mae_days', float('nan')):.0f} d, coverage {rep['rul'].get('coverage95', 0):.2f}",
        t0,
    )

    healthy = [i for i, d in enumerate(history.devices) if d.defect is None]
    fit_ids = healthy[: len(healthy) // 2]
    X, D, S = [], [], []
    for i in fit_ids:
        m = history.truth_soh[i] > 0.75
        X.append(device_signals(history, i)[m])
        D.append(history.snap_days[m])
        S.append(history.tele["strain_ue"][i][m])
    cal = fit_calibration(np.vstack(X), np.concatenate(D), np.concatenate(S))
    (out / "anomaly_calibration.json").write_text(json.dumps(cal.to_dict(), indent=1))
    report["anomaly"] = anomaly_report(history, live, cal)
    _log(f"anomaly detector: false-warning rate {report['anomaly']['false_warning_rate']:.4f}", t0)

    resid, twin_rep = train_residual_model()
    np.savez(out / "twin_residual.npz", weights=resid.weights)
    twin_rep["soc_benchmark_aged_cell"] = {
        k: v for k, v in soc_benchmark(state_from_modes(0.10, 0.03, 0.02, r_factor=1.4)).items() if k != "trace"
    }
    report["twin"] = twin_rep
    _log(
        f"twin residual: {twin_rep['voltage_rmse_mv_ecm_only']:.1f} -> {twin_rep['voltage_rmse_mv_hybrid']:.1f} mV", t0
    )

    q, rl_info = train_q_policy(n_batches=args.rl_batches, batch=4096, seed=0)
    np.save(out / "q_policy.npy", q.q)
    report["charging"] = {
        "rl_training": rl_info,
        "policy_comparison": evaluate_policies(q, n=args.policy_eval_sessions),
    }
    pc = report["charging"]["policy_comparison"]
    _log(
        f"charging: MILP -{pc['milp']['damage_reduction_vs_legacy_pct']:.0f} %, "
        f"RL -{pc['rl']['damage_reduction_vs_legacy_pct']:.0f} % damage vs legacy",
        t0,
    )

    int8, tiny = run_pipeline(q, out / "edge", rollout_sessions=max(20, args.policy_eval_sessions // 2))
    save_int8(int8, out / "edge" / "charge_policy_int8.npz")
    report["tinyml"] = tiny
    _log(
        f"tinyml: {tiny['stages'][-1]['bytes']} B INT8, closed-loop met "
        f"{tiny['closed_loop']['int8_student+guard']['met_target_rate']:.3f}",
        t0,
    )

    _, _, key_id = load_or_create_keys(out / "keys")
    report["ota"] = {"alg": "Ed25519", "key_id": key_id}
    for f in ("state",):
        d = out / f
        if d.exists():
            for p in d.glob("*.json"):
                p.unlink()
    (out / "model_report.json").write_text(json.dumps(report, indent=1, default=float))
    _log("done - artifacts written to " + str(out), t0)


if __name__ == "__main__":
    main()
