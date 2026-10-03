"""The AI models applied to a real laptop battery.

* Capacity-fade forecast: Gaussian process (smooth RBF + linear trend + noise)
  on the Windows capacity history; posterior trajectories are sampled to get
  the distribution of the date the battery crosses 80 / 70 / 60 % of design.
* Usage profile: AC share, time-weighted state of charge, time near full,
  typical unplug hour, daily energy use, and detection of a firmware charge
  limit (e.g. an OEM "battery care" 60 / 80 % mode).
* Calibrated what-if: the fleet ageing model, scaled by one factor fitted to
  this battery's observed fade, projects current habits vs AI adaptive
  charging vs always-100 %.
* Tonight's plan: MILP charge schedule from the live state of charge and the
  learned unplug time.
* Experimental dQ/dV from logged live charging samples.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import minimize

from ..charging.milp import plan_and_verify
from ..charging.session import CellModelSet, ChargeContext, ChargePhysics, legacy_policy, rollout
from ..cloud.twin import modes_from_soh, state_from_modes
from ..edge.ica import savgol
from ..physics.aging import AgingModel, DailyStress
from ..physics.electrode import DegradationState, electrode_model
from .windows_battery import BatteryReport

THRESHOLDS = (0.80, 0.70, 0.60)
REPLACE_AT = 0.60


# --------------------------------------------------------------------------
class TrendGP:
    """1-D GP: sf^2 RBF + sl^2 (t t' + 1) + sn^2 I. The linear term carries the
    fade trend (and its growing uncertainty) beyond the data; the RBF term
    absorbs gauge recalibration wiggles."""

    def fit(self, t, y):
        t, y = np.asarray(t, float), np.asarray(y, float)
        self.tm, self.ts = t.mean(), t.std() or 1.0
        self.ym, self.ys = y.mean(), y.std() or 1.0
        self.t = (t - self.tm) / self.ts
        self.y = (y - self.ym) / self.ys
        best = None
        for start in ([np.log(0.5), 0.0, 0.0, np.log(0.1)], [np.log(2.0), -1.0, 0.0, np.log(0.05)]):
            r = minimize(
                self._nll,
                start,
                method="L-BFGS-B",
                bounds=[
                    (np.log(0.05), np.log(10)),
                    (np.log(1e-3), np.log(10)),
                    (np.log(1e-3), np.log(10)),
                    (np.log(1e-3), np.log(1.0)),
                ],
            )
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        K = self._k(self.t, self.t) + (np.exp(self.theta[3]) ** 2 + 1e-8) * np.eye(len(self.t))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def _k(self, a, b, theta=None):
        ls, sf, sl, _ = np.exp(self.theta if theta is None else theta)
        d = a[:, None] - b[None, :]
        return sf**2 * np.exp(-0.5 * d**2 / ls**2) + sl**2 * (a[:, None] * b[None, :] + 1.0)

    def _nll(self, theta):
        K = self._k(self.t, self.t, theta) + (np.exp(theta[3]) ** 2 + 1e-8) * np.eye(len(self.t))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return 0.5 * self.y @ a + np.log(np.diag(L)).sum()

    def predict(self, tq, full_cov=False):
        tq = (np.asarray(tq, float) - self.tm) / self.ts
        ks = self._k(tq, self.t)
        mean = ks @ self.alpha
        v = np.linalg.solve(self.L, ks.T)
        if full_cov:
            cov = self._k(tq, tq) - v.T @ v
            return mean * self.ys + self.ym, cov * self.ys**2
        var = np.maximum(np.diag(self._k(tq, tq)) - (v * v).sum(0), 1e-12)
        return mean * self.ys + self.ym, np.sqrt(var) * self.ys


def capacity_forecast(rep: BatteryReport, horizon_days: int = 3 * 365, n_samples: int = 4000, seed: int = 0) -> dict:
    if len(rep.history) < 4:
        return {"available": False, "reason": "fewer than 4 capacity-history entries in the Windows report"}
    t0 = rep.history[0].start
    pts = [
        ((h.start + (h.end - h.start) / 2 - t0).total_seconds() / 86400, h.fcc_mwh / h.design_mwh, h.cycles)
        for h in rep.history
    ]
    pts.append(((rep.generated - t0).total_seconds() / 86400, rep.soh, rep.cycles))
    t = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    gp = TrendGP().fit(t, y)
    now = t[-1]
    past = np.linspace(t[0], now, 60)
    fut = np.linspace(now, now + horizon_days, 181)
    m_past, s_past = gp.predict(past)
    m_fut, cov = gp.predict(fut, full_cov=True)
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(cov + 1e-10 * np.eye(len(fut)))
    samples = m_fut[None, :] + (Lc @ rng.standard_normal((len(fut), n_samples))).T
    s_fut = np.sqrt(np.clip(np.diag(cov), 0, None))
    soh_now = float(m_past[-1])
    crossings = {}
    for thr in THRESHOLDS:
        key = f"{int(thr * 100)}"
        if soh_now <= thr:
            crossings[key] = {"status": "already_below", "threshold": thr}
            continue
        below = samples <= thr
        hit = below.any(axis=1)
        days = np.where(hit, fut[np.argmax(below, axis=1)] - now, np.inf)
        med, lo, hi = (float(np.percentile(days, p)) for p in (50, 2.5, 97.5))
        crossings[key] = {
            "status": "forecast",
            "threshold": thr,
            "median_days": med if np.isfinite(med) else None,
            "lo_days": lo if np.isfinite(lo) else None,
            "hi_days": hi if np.isfinite(hi) else None,
            "beyond_horizon_share": float(np.mean(~hit)),
            "median_date": (rep.generated + timedelta(days=med)).date().isoformat() if np.isfinite(med) else None,
        }
    # Fade rate the forecast actually extrapolates (next 12 months of the posterior mean),
    # plus the observed change over the last 60 days of data.
    yr = fut <= now + 365
    slope = np.polyfit(fut[yr], m_fut[yr], 1)[0]
    last60 = [p for p in pts if p[0] >= now - 60]
    recent_change = float(last60[-1][1] - last60[0][1]) if len(last60) > 1 else None
    cyc = [(p[0], p[2]) for p in pts if p[2] is not None]
    cycles_per_day = (cyc[-1][1] - cyc[0][1]) / max(cyc[-1][0] - cyc[0][0], 1) if len(cyc) > 1 else None
    date = lambda d: (t0 + timedelta(days=float(d))).date().isoformat()  # noqa: E731
    return {
        "available": True,
        "points": [{"date": date(p[0]), "day": float(p[0]), "soh": float(p[1]), "cycles": p[2]} for p in pts],
        "fit_past": {
            "day": past.tolist(),
            "mean": m_past.tolist(),
            "lo": (m_past - 1.96 * s_past).tolist(),
            "hi": (m_past + 1.96 * s_past).tolist(),
        },
        "forecast": {
            "day": fut.tolist(),
            "mean": m_fut.tolist(),
            "lo": (m_fut - 1.96 * s_fut).tolist(),
            "hi": (m_fut + 1.96 * s_fut).tolist(),
        },
        "start_date": t0.date().isoformat(),
        "now_day": float(now),
        "soh_now_smoothed": soh_now,
        "fade_pct_per_100d": float(-slope * 1e4),
        "cycles_per_day": cycles_per_day,
        "observed_change_60d_pct": None if recent_change is None else 100 * recent_change,
        "fade_pct_per_100_cycles": float((y[0] - y[-1]) / max((cyc[-1][1] - cyc[0][1]), 1) * 1e4)
        if len(cyc) > 1
        else None,
        "crossings": crossings,
        "model": "GP: RBF + linear trend + noise, hyperparameters by marginal likelihood; 4000 posterior trajectories",
    }


# --------------------------------------------------------------------------
def usage_profile(rep: BatteryReport) -> dict:
    ev = sorted(rep.usage, key=lambda e: e.time)
    seg_on_ac, seg_soc, seg_w = [], [], []
    unplugs = []
    for a, b in zip(ev, ev[1:]):
        dur = (b.time - a.time).total_seconds()
        if dur <= 0 or not a.fcc_mwh:
            continue
        seg_on_ac.append(a.on_ac)
        seg_soc.append(min(1.0, a.charge_mwh / a.fcc_mwh))
        seg_w.append(dur)
        if a.on_ac and not b.on_ac:
            unplugs.append(b.local_time)
    w = np.array(seg_w)
    soc = np.array(seg_soc)
    ac = np.array(seg_on_ac, bool)
    out = {
        "events": len(ev),
        "window_days": round((ev[-1].time - ev[0].time).total_seconds() / 86400, 1) if len(ev) > 1 else 0,
    }
    if w.sum() > 0:
        out.update(
            mean_soc=float(np.average(soc, weights=w)),
            frac_high_soc=float(w[soc > 0.95].sum() / w.sum()),
            ac_share_recent=float(w[ac].sum() / w.sum()),
        )
        if ac.any():
            p90 = float(np.percentile(np.repeat(soc[ac], np.maximum(1, (w[ac] / 600).astype(int))), 90))
            out["soc_on_ac_p90"] = p90
            out["charge_limit_detected"] = bool(p90 <= 0.85 and out["ac_share_recent"] > 0.2)
            out["charge_limit_pct"] = int(5 * round(100 * p90 / 5)) if out["charge_limit_detected"] else None
    first_unplug = {}
    for u in unplugs:
        first_unplug.setdefault(u.date(), u)
    hours = [u.hour + u.minute / 60 for u in first_unplug.values() if 5 <= u.hour <= 14]
    out["typical_unplug_hour"] = float(np.median(hours)) if hours else None
    out["unplug_events"] = len(unplugs)
    recent = [h for h in rep.history if h.end >= rep.generated - timedelta(days=45) and h.times_valid] or [
        h for h in rep.history if h.times_valid
    ][-5:]
    out["invalid_history_entries"] = sum(not h.times_valid for h in rep.history)
    days = sum(h.days for h in recent)
    ac_s = sum(h.active_ac_s + h.standby_ac_s for h in recent)
    dc_s = sum(h.active_dc_s + h.standby_dc_s for h in recent)
    energy = sum(h.dc_energy_mwh for h in recent)
    out["ac_share_45d"] = ac_s / max(ac_s + dc_s, 1)
    out["battery_hours_per_day"] = dc_s / 3600 / max(days, 1e-6)
    out["battery_energy_wh_per_day"] = energy / 1000 / max(days, 1e-6)
    out["daily_need_frac"] = energy / max(days, 1e-6) / max(rep.fcc_mwh, 1)
    return out


# --------------------------------------------------------------------------
def _stress(efc, soc, high, charge_c=0.5, temp=30.0):
    return DailyStress(
        efc=efc,
        mean_soc=soc,
        frac_high_soc=high,
        charge_c=charge_c,
        temp_c=temp,
        charge_temp_c=temp + 2 + 5 * charge_c,
        dod=0.5,
    )


def _fade(state: DegradationState, s: DailyStress, days: float, k: float, step: float = 30.0) -> DegradationState:
    m = AgingModel()
    st = state.copy()
    left = days
    while left > 1e-9:
        d = min(step, left)
        st = m.step(st, s, days=d * k)
        left -= d
    return st


def calibrated_projection(rep: BatteryReport, prof: dict, fc: dict, horizon_days: int = 3 * 365) -> dict:
    """Scale the fleet ageing model to this battery's observed fade, then compare habits."""
    if not fc.get("available"):
        return {"available": False}
    h0, h1 = rep.history[0], rep.history[-1]
    soh0, soh1 = h0.fcc_mwh / h0.design_mwh, h1.fcc_mwh / h1.design_mwh
    span = (h1.end - h0.start).total_seconds() / 86400
    efc = fc["cycles_per_day"] or 0.5
    cur = _stress(efc, prof.get("mean_soc", 0.7), prof.get("frac_high_soc", 0.2))
    em = electrode_model()
    rated = em.rated_capacity_ah
    start = state_from_modes(*modes_from_soh(soh0))

    def model_soh(k):
        return float(em.capacity(_fade(start, cur, span, k))) / rated

    lo, hi = 0.02, 50.0
    if soh1 >= soh0 or not (model_soh(hi) < soh1 < model_soh(lo)):
        k = 1.0
        note = "observed fade outside the model's range; uncalibrated (k = 1)"
    else:
        for _ in range(40):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if model_soh(mid) > soh1 else (lo, mid)
        k = float(np.sqrt(lo * hi))
        note = f"one rate factor k = {k:.2f} fitted so the model reproduces the observed fade over {span:.0f} days"
    scen = {
        "current_habits": cur,
        "ai_adaptive": _stress(efc, min(prof.get("mean_soc", 0.7), 0.58), 0.02, charge_c=0.35),
        "always_100pct": _stress(efc, 0.9, 0.6, charge_c=0.7),
    }
    now_state = state_from_modes(*modes_from_soh(rep.soh))
    out = {
        "available": True,
        "k": k,
        "calibration": note,
        "observed": {"soh_start": soh0, "soh_end": soh1, "days": span},
        "scenarios": {},
    }
    for name, s in scen.items():
        st, days, sohs, eol = now_state.copy(), [0], [rep.soh], None
        for d in range(30, horizon_days + 1, 30):
            st = _fade(st, s, 30, k)
            v = float(em.capacity(st)) / rated
            days.append(d)
            sohs.append(v)
            if eol is None and v <= REPLACE_AT:
                eol = d
        out["scenarios"][name] = {
            "days": days,
            "soh": sohs,
            "days_to_60pct": eol,
            "stress": {
                "mean_soc": s.mean_soc,
                "frac_high_soc": s.frac_high_soc,
                "charge_c": s.charge_c,
                "efc_per_day": s.efc,
            },
        }
    return out


# --------------------------------------------------------------------------
def charge_plan(
    rep: BatteryReport,
    prof: dict,
    live: dict | None,
    now: datetime,
    calendar: list | None = None,
    ambient_c: float = 25.0,
) -> dict:
    soc_now = (
        (live["remaining_mwh"] / live["fcc_mwh"])
        if live and live.get("remaining_mwh") and live.get("fcc_mwh")
        else (live["percent"] / 100 if live and live.get("percent") is not None else 0.4)
    )
    unplug_h = prof.get("typical_unplug_hour") or 9.0
    unplug = now.replace(hour=int(unplug_h), minute=int(60 * (unplug_h % 1)), second=0, microsecond=0)
    if unplug <= now + timedelta(minutes=30):
        unplug += timedelta(days=1)
    need = prof.get("daily_need_frac", 0.5)
    target = float(np.clip(need * 1.25 + 0.15, 0.6, 1.0))
    reason = (
        f"learned routine: first unplug ~{unplug:%H:%M}, you use about {100 * need:.0f} % of a full charge "
        f"on battery per day"
    )
    limit = prof.get("charge_limit_pct")
    if prof.get("charge_limit_detected") and limit and need * 1.1 <= limit / 100:
        target = min(target, limit / 100)
        reason += f"; your ~{limit} % charge limit covers that, so the plan respects it"
    for ev in calendar or []:
        start = datetime.fromisoformat(ev["start"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=now.tzinfo)
        if now < start <= unplug + timedelta(hours=12) and ev.get("kind") in ("flight", "travel", "offsite"):
            unplug, target = start - timedelta(hours=1), 1.0
            reason = f"calendar: '{ev.get('title', 'trip')}' at {start:%a %H:%M} - full charge just before leaving"
    hours = max(0.5, (unplug - now).total_seconds() / 3600)
    state = state_from_modes(*modes_from_soh(rep.soh))
    ctx = ChargeContext(float(soc_now), hours, target, now.hour + now.minute / 60, ambient_c, state, "balanced", reason)
    phys = ChargePhysics(CellModelSet([state]))
    legacy = rollout(ctx, legacy_policy(), phys)
    milp = plan_and_verify(ctx, phys)
    ai = milp["verification"]
    red = 100 * (1 - ai["damage_pct_capacity"] / max(legacy["damage_pct_capacity"], 1e-12))
    return {
        "soc_now": soc_now,
        "target_soc": target,
        "unplug_at": unplug.isoformat(),
        "hours": hours,
        "reason": reason,
        "legacy": legacy,
        "ai": {**ai, "trace": milp["trace"]},
        "damage_reduction_pct": red,
        "note": "rates are expressed as C-rates of the modelled cell; the laptop's OEM charger firmware enforces the "
        "real limits. Actuation needs integration with the OEM's embedded-controller firmware.",
    }


# --------------------------------------------------------------------------
def recommendations(rep: BatteryReport, prof: dict, fc: dict, proj: dict) -> list[dict]:
    recs = []
    soh = rep.soh
    lost = (rep.runtime_design_s - rep.runtime_full_s) / 60
    if soh < 0.80:
        recs.append(
            {
                "level": "warning",
                "title": f"Battery at {soh:.0%} of design capacity",
                "detail": f"Below the common 80 % service threshold. A full charge now gives about "
                f"{rep.runtime_full_s / 3600:.1f} h of active use vs {rep.runtime_design_s / 3600:.1f} h "
                f"when new ({lost:.0f} min lost)."
                if rep.runtime_design_s
                else "Below the common 80 % service threshold.",
            }
        )
    c60 = fc.get("crossings", {}).get("60", {}) if fc.get("available") else {}
    if c60.get("status") == "forecast" and c60.get("median_days") is not None:
        months = c60["median_days"] / 30.4
        lo = (c60["lo_days"] or 0) / 30.4
        hi = c60["hi_days"] / 30.4 if c60.get("hi_days") else None
        recs.append(
            {
                "level": "warning" if months < 9 else "info",
                "title": f"Plan a replacement in about {months:.0f} months",
                "detail": f"The capacity forecast reaches 60 % of design around {c60['median_date']} "
                f"(95 %: {lo:.0f}-{f'{hi:.0f}' if hi else '>36'} months), at the current fade of "
                f"{fc['fade_pct_per_100d']:.1f} % per 100 days.",
            }
        )
    if prof.get("charge_limit_detected"):
        recs.append(
            {
                "level": "good",
                "title": f"A ~{prof['charge_limit_pct']} % charge limit is protecting the battery",
                "detail": "While plugged in the battery stays well below full, which slows calendar ageing. Keep it "
                "on. Before a trip, charge to full the night before - the copilot's calendar-aware plan "
                "does this automatically.",
            }
        )
    elif prof.get("frac_high_soc", 0) > 0.3:
        recs.append(
            {
                "level": "warning",
                "title": "Battery sits near 100 % much of the time",
                "detail": f"{prof['frac_high_soc']:.0%} of the time above 95 %. Enable the OEM battery-care mode "
                "(60-80 % limit) or AI adaptive charging.",
            }
        )
    if fc.get("available") and prof.get("charge_limit_detected"):
        pts = [p for p in fc["points"] if p["day"] >= fc["now_day"] - 45]
        if len(pts) > 3 and max(p["soh"] for p in pts) - min(p["soh"] for p in pts) < 0.005:
            recs.append(
                {
                    "level": "info",
                    "title": "Windows' capacity figure may be stale",
                    "detail": "The reported full-charge capacity has barely moved for 45 days while a charge limit "
                    "is active. Fuel gauges re-learn capacity only after near-full charges/discharges, "
                    "so a flat line does not mean the battery stopped ageing. The on-device ICA in this "
                    "PoC measures health from partial slow charges instead.",
                }
            )
    if proj.get("available"):
        sc = proj["scenarios"]
        a, b = sc["always_100pct"]["days_to_60pct"], sc["current_habits"]["days_to_60pct"]
        if a and b and b > a:
            recs.append(
                {
                    "level": "info",
                    "title": f"Your charging habits add about {(b - a) / 30.4:.0f} months of life",
                    "detail": "Compared with keeping the battery at 100 % (calibrated physics projection).",
                }
            )
    if prof.get("ac_share_45d", 0) > 0.7:
        recs.append(
            {
                "level": "info",
                "title": "Mostly used on AC power",
                "detail": f"{prof['ac_share_45d']:.0%} of powered-on time on AC over the last 45 days.",
            }
        )
    return recs


# --------------------------------------------------------------------------
def ica_from_samples(samples: list[dict], series_cells: int | None = None) -> dict:
    """Experimental: dQ/dV from logged live samples during a charging session."""
    chg = [
        s
        for s in samples
        if s.get("power_online")
        and (s.get("charge_rate_mw") or 0) > 0
        and s.get("voltage_mv")
        and s.get("remaining_mwh") is not None
    ]
    if len(chg) < 25:
        return {
            "available": False,
            "reason": f"{len(chg)} charging samples logged; need at least 25 "
            "(keep the dashboard open while the laptop charges)",
        }
    q = np.array([s["remaining_mwh"] for s in chg], float)
    v = np.array([s["voltage_mv"] for s in chg], float) / 1000.0
    n = series_cells or max(1, int(round(np.median(v) / 3.85)))
    v /= n
    if q.max() - q.min() < 0.1 * max(chg[-1].get("fcc_mwh") or q.max(), 1):
        return {"available": False, "reason": "charging span below 10 % of capacity"}
    order = np.argsort(v)
    v, q = v[order], np.maximum.accumulate(q[order])
    grid = np.arange(np.ceil(v.min() * 200) / 200, v.max(), 0.005)
    if len(grid) < 15:
        return {"available": False, "reason": "voltage span too small"}
    qg = np.interp(grid, v, q)
    dqdv = np.maximum(savgol(np.gradient(qg, grid), 9, 2), 0) / 1000.0 / n
    return {
        "available": True,
        "v_cell": grid.tolist(),
        "dqdv_wh_per_v": dqdv.tolist(),
        "samples": len(chg),
        "series_cells": n,
        "note": "coarse (gauge-reported capacity, ~20 s sampling); on-device firmware "
        "capture at mV resolution is what the product needs",
    }
