"""BMS Copilot: conversational interface over the battery-management engine.

With Claude credentials (ANTHROPIC_API_KEY, or an `ant auth login` profile) the
copilot runs a tool-use loop on Claude: it decides which engine tools to call
(fleet overview, device health, diagnosis, charge planning, passport, second
life, agent actions, model metrics) and answers with the numbers they return.
Without credentials - or if the API is unreachable - a deterministic,
tool-backed responder answers the common questions so the product still works
offline.

Environment:
  BMS_COPILOT_MODEL   default claude-opus-5
  BMS_COPILOT_EFFORT  default medium (low | medium | high | xhigh | max)
  BMS_COPILOT_LLM     set to "off" to force the offline responder
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import timedelta

from .tools import TOOLS, run_tool, run_tool_json

MODEL = os.environ.get("BMS_COPILOT_MODEL", "claude-opus-5")
EFFORT = os.environ.get("BMS_COPILOT_EFFORT", "medium")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are the Next-Gen BMS Copilot, an assistant for battery engineers, service teams and \
sustainability/compliance staff who manage a fleet of premium laptops. You work on top of an AI battery-management \
stack:
- on-device incremental capacity analysis (dQ/dV) and a precursor detector (swelling, thermal, micro-short),
- cloud Gaussian-process models for state of health (SoH), degradation modes and remaining useful life (RUL, with a \
95 % interval),
- a hybrid digital twin (equivalent-circuit core + learned residual),
- adaptive charging (MILP schedule and an INT8 reinforcement-learning policy on the NPU) that always runs inside a \
deterministic safety envelope,
- an agentic workforce (predictive, triage, service, logistics, asset recovery, design feedback) gated by an autonomy \
level L1-L5,
- a passport-style battery record structured after EU Regulation 2023/1542 Annex XIII.

How to work:
- Use the tools to get facts; never invent device data or metrics. Quote the numbers you used.
- Always report uncertainty where the tools provide it (SoH interval, RUL 95 % interval). Treat the lower RUL bound \
as the planning figure for risk decisions.
- Explain mechanisms plainly: LLI = lithium inventory loss (normal SEI ageing), LAM_pe / LAM_ne = cathode / anode \
active-material loss. Anode LAM far above cohort expectation suggests a manufacturing defect.
- You cannot approve or reject agent actions, change the autonomy level, push OTA packages or relax safety limits. \
If asked, explain what a human should do in the dashboard (Workforce tab) and why.
- The fleet data comes from a physics-based simulator; say so if someone treats metrics as field results.
- Laptop batteries are portable batteries: the EU battery passport (Art. 77) is not mandatory for them; the record is \
voluntary. Be precise about that.
- Be concise: lead with the answer, then the evidence. Use short lists or a compact table when comparing options."""


class Copilot:
    def __init__(self, engine):
        self.engine = engine
        self.sessions: dict[str, list] = {}
        self.client = None
        self.llm_error: str | None = None
        if os.environ.get("BMS_COPILOT_LLM", "").lower() == "off":
            self.llm_error = "LLM disabled (BMS_COPILOT_LLM=off)"
            return
        if not _credentials_present():
            self.llm_error = "no Claude credentials configured"
            return
        try:
            import anthropic

            self._anthropic = anthropic
            self.client = anthropic.Anthropic()
        except Exception as e:  # missing package or no resolvable credentials
            self.llm_error = f"Claude client unavailable: {type(e).__name__}"

    @property
    def mode(self) -> str:
        return f"claude ({MODEL})" if self.client is not None else "offline (rule-based)"

    # ----------------------------------------------------------------- public
    def chat(self, message: str, session_id: str | None = None) -> dict:
        session_id = session_id or uuid.uuid4().hex[:12]
        if self.client is not None:
            try:
                return {**self._chat_claude(message, session_id), "session_id": session_id, "mode": self.mode}
            except _FallbackToOffline as e:
                self.client = None
                self.llm_error = str(e)
        out = OfflineResponder(self.engine).answer(message)
        note = f" ({self.llm_error})" if self.llm_error else ""
        return {**out, "session_id": session_id, "mode": "offline (rule-based)" + note}

    # ----------------------------------------------------------------- claude
    def _chat_claude(self, message: str, session_id: str) -> dict:
        a = self._anthropic
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": message})
        trace = []
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                resp = self.client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=history,
                    thinking={"type": "adaptive"},
                    output_config={"effort": EFFORT},
                    cache_control={"type": "ephemeral"},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except TypeError as e:
                # The SDK raises TypeError while building the request when no
                # credential source (API key, auth token, profile) resolves.
                history.pop()
                if "authentication" not in str(e).lower():
                    raise
                raise _FallbackToOffline("no Claude credentials configured") from e
            except (a.AuthenticationError, a.PermissionDeniedError) as e:
                history.pop()
                raise _FallbackToOffline(f"Claude credentials rejected: {e.__class__.__name__}") from e
            except a.APIConnectionError as e:
                history.pop()
                raise _FallbackToOffline("Claude API unreachable") from e
            except a.RateLimitError:
                history.pop()
                return {"answer": "The Claude API is rate-limited right now; please retry in a moment.", "trace": trace}
            except a.APIStatusError as e:
                history.pop()
                return {"answer": f"Claude API error {e.status_code}; please retry.", "trace": trace}

            history.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                return {"answer": "I can't help with that request.", "trace": trace}
            if resp.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if resp.stop_reason == "tool_use" and tool_uses:
                results = []
                for tu in tool_uses:
                    text, is_err = run_tool_json(self.engine, tu.name, tu.input if isinstance(tu.input, dict) else {})
                    trace.append({"tool": tu.name, "input": tu.input, "error": is_err, "output_chars": len(text)})
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": text,
                            **({"is_error": True} if is_err else {}),
                        }
                    )
                history.append({"role": "user", "content": results})  # all results in one message
                continue
            answer = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
            if resp.stop_reason == "max_tokens":
                answer += "\n\n(answer truncated)"
            return {"answer": answer.strip(), "trace": trace}
        return {"answer": "I stopped after too many tool calls; please narrow the question.", "trace": trace}


class _FallbackToOffline(Exception):
    pass


def _credentials_present() -> bool:
    """Cheap check of the SDK's credential sources (API key, auth token, a named
    or on-disk `ant auth login` profile, workload identity federation). A present
    but invalid credential is still caught on the first request."""
    env = os.environ
    if any(
        env.get(k)
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID")
    ):
        return True
    from pathlib import Path

    return (Path.home() / ".config" / "anthropic").exists()


# ---------------------------------------------------------------------------
class OfflineResponder:
    """Deterministic intent router used when Claude is unavailable."""

    DEV = re.compile(r"\b(?:DEV|dev)[- ]?(\d{1,4})\b")

    def __init__(self, engine):
        self.e = engine

    def _call(self, trace, name, args=None):
        out = run_tool(self.e, name, args or {})
        trace.append(
            {"tool": name, "input": args or {}, "error": False, "output_chars": len(json.dumps(out, default=float))}
        )
        return out

    def answer(self, msg: str) -> dict:
        m, t = msg.lower(), []
        dev = self.DEV.search(msg)
        did = f"DEV-{int(dev.group(1)):04d}" if dev else None
        try:
            if any(w in m for w in ("my battery", "my laptop", "this laptop", "this computer", "my pc", "my device")):
                return {"answer": self._mybattery(m, t), "trace": t}
            if did:
                if any(w in m for w in ("charge", "plan", "flight", "tonight", "trip")):
                    return {"answer": self._plan(did, m, t), "trace": t}
                if "passport" in m:
                    return {"answer": self._passport(did, m, t), "trace": t}
                if any(w in m for w in ("second life", "second-life", "grade", "recycl", "reuse")):
                    return {"answer": self._grade(did, t), "trace": t}
                if any(w in m for w in ("why", "diagnos", "cause", "root")):
                    return {"answer": self._diagnose(did, t), "trace": t}
                return {"answer": self._health(did, t), "trace": t}
            if any(w in m for w in ("lot", "supplier", "insight", "design")):
                return {"answer": self._insights(t), "trace": t}
            if any(w in m for w in ("action", "approve", "pending", "agent", "workforce", "claim")):
                return {"answer": self._actions(t), "trace": t}
            if any(w in m for w in ("accura", "model", "metric", "validat", "performance")):
                return {"answer": self._metrics(t), "trace": t}
            if any(w in m for w in ("swell", "thermal", "anomal", "safety", "critical", "short")):
                return {"answer": self._risky(t, "anomaly"), "trace": t}
            if any(w in m for w in ("worst", "risk", "replace", "rul", "failing", "weak")):
                return {"answer": self._risky(t, "rul"), "trace": t}
            return {"answer": self._overview(t), "trace": t}
        except KeyError as e:
            return {"answer": f"I couldn't find that: {e}.", "trace": t}

    def _mybattery(self, m, t):
        if any(w in m for w in ("charge", "plan", "tonight", "flight", "trip")):
            cal = None
            if "flight" in m or "trip" in m:
                from datetime import datetime as _dt

                dep = (_dt.now().astimezone() + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
                cal = [{"title": "Trip", "start": dep.isoformat(), "kind": "flight"}]
            p = self._call(t, "plan_my_charge", {"calendar_events": cal} if cal else {})
            if not p.get("available"):
                return f"I can't read this computer's battery: {p.get('reason')}"
            return (
                f"Tonight for this laptop: {p['reason']}. Now at {p['soc_now']:.0%}; target {p['target_soc']:.0%} by "
                f"{p['unplug_at'][11:16]}. The AI schedule causes {p['damage_reduction_pct']:.0f} % less wear than "
                f"charging to 100 % straight away. {p['note']}"
            )
        b = self._call(t, "get_my_battery")
        if not b.get("available"):
            return f"I can't read this computer's battery: {b.get('reason')}"
        bat, fc, live = b["battery"], b["forecast"], b.get("live") or {}
        c60 = fc["crossings"].get("60", {})
        lines = [
            f"This laptop's {bat['manufacturer']} battery holds {bat['full_charge_mwh'] / 1000:.1f} Wh of its "
            f"{bat['design_mwh'] / 1000:.1f} Wh design capacity ({bat['soh_reported']:.1%}) after {bat['cycles']} "
            f"cycles; a full charge now lasts about {bat['runtime_now_h']:.1f} h vs {bat['runtime_new_h']:.1f} h new."
        ]
        if live:
            lines.append(
                f"Right now: {live.get('percent')} % charged, {(live.get('voltage_mv') or 0) / 1000:.2f} V, "
                f"{'on AC' if live.get('power_online') else 'on battery'}."
            )
        if c60.get("status") == "forecast" and c60.get("median_days") is not None:
            hi = (c60.get("hi_days") or 1100) / 30.4
            lines.append(
                f"Forecast (GP on {bat['history_entries']} capacity-history points): 60 % of design around "
                f"{c60['median_date']} (95 %: {c60['lo_days'] / 30.4:.0f}-{hi:.0f} months), fade "
                f"{fc['fade_pct_per_100d']:.1f} % per 100 days."
            )
        lines += [f"- {r['title']}: {r['detail']}" for r in b["recommendations"]]
        return chr(10).join(lines)

    def _overview(self, t):
        k = self._call(t, "get_fleet_overview")["kpis"]
        return (
            f"Fleet of {k['devices']} devices, mean SoH {k['mean_soh']:.1%}. {k['below_85pct']} are below 85 % and "
            f"{k['below_eol']} at or below end of first life (80 %); {k['rul_under_180d']} have a median RUL under "
            f"180 days. Precursor detector: {k['anomalies']['critical']} critical, {k['anomalies']['warning']} "
            f"warning, {k['anomalies']['watch']} watch. Adaptive charging is active on {k['adaptive_share']:.0%} "
            f"of devices. SoH error: AI {k['soh_mae_ai']:.2%} vs legacy gauge {k['soh_mae_legacy_gauge']:.2%} "
            f"(simulated fleet). Ask about a device (e.g. 'DEV-0041'), charging plans, passports, or agent actions."
        )

    def _risky(self, t, by):
        rows = self._call(t, "list_devices", {"sort_by": by, "limit": 6})

        def rul(r):
            if r["rul_status"] == "past_end_of_first_life":
                return "past end of first life"
            return f"RUL {r['rul_days']} d (95 %: {r['rul_95'][0]}-{r['rul_95'][1]})"

        lines = [
            f"- {r['device_id']} ({r['profile']}, {r['age_days']} d): SoH {r['soh']:.1%}, {rul(r)}, "
            f"precursor level {r['anomaly']}"
            for r in rows
        ]
        head = "Most severe precursor alerts:" if by == "anomaly" else "Shortest remaining useful life:"
        return head + "\n" + "\n".join(lines)

    def _health(self, did, t):
        h = self._call(t, "get_device_health", {"device_id": did})
        r = h["rul"]
        rul = (
            "first life already over (SoH at or below 80 %)"
            if r.get("status") == "past_end_of_first_life"
            else f"RUL {r['median_days']} days (95 %: {r['lo_days']}-{r['hi_days']})"
        )
        a = h["anomaly"]
        return (
            f"{did} ({h['profile_label']}, {h['age_days']} days old, lot {h['lot']}): SoH {h['soh']:.1%} "
            f"(95 %: {h['soh_interval'][0]:.1%}-{h['soh_interval'][1]:.1%}); legacy gauge would show "
            f"{h['gauge_soh_legacy']:.1%}. Expected for its age and usage: {h['expected_soh']:.1%} "
            f"(z = {h['soh_z']:.1f}). {rul}. Precursor detector: {a['level']}"
            f"{' - ' + a['mechanism'] if a.get('mechanism') else ''}. Estimated modes: LLI "
            f"{h['modes_est']['lli']:.3f}, LAM_pe {h['modes_est']['lam_pe']:.3f}, LAM_ne "
            f"{h['modes_est']['lam_ne']:.3f}. "
            f"Warranty days left: {h['warranty_days_left']}."
        )

    def _diagnose(self, did, t):
        d = self._call(t, "diagnose_device", {"device_id": did})
        h = self._call(t, "get_device_health", {"device_id": did})
        names = {
            "lli": "lithium inventory loss (normal SEI ageing)",
            "lam_pe": "cathode active-material loss",
            "lam_ne": "anode active-material loss (defect signature when far above cohort)",
        }
        if h["lam_ne_excess_z"] >= 3 or d["dominant"] == "lam_ne":
            verdict = (
                f"Verdict: anode active-material loss is {h['lam_ne_excess_z']:.1f} sigma above what this device's "
                f"usage explains - a manufacturing-defect signature (cell lot {h['lot']}); treat as warrantable / "
                "goodwill and check other packs from the lot."
            )
        elif h["soh_z"] <= -2.5 and (h["usage"]["frac_high_mean"] > 0.5 or h["usage"]["temp_mean"] > 33):
            verdict = (
                "Verdict: faster than expected, explained by usage stress (heat / time at high SoC): "
                "enable adaptive charging."
            )
        elif h["soh_z"] <= -2.5:
            verdict = "Verdict: faster than expected with no clear mechanism - keep under observation."
        else:
            verdict = "Verdict: consistent with normal ageing for its age and usage."
        return (
            f"Diagnostic capture of {did}: LLI {d['lli']:.3f}, LAM_pe {d['lam_pe']:.3f}, LAM_ne {d['lam_ne']:.3f} "
            f"(fit RMSE {d['rmse_mah']:.1f} mAh); largest mode: {names[d['dominant']]}. SoH {h['soh']:.1%} vs "
            f"expected {h['expected_soh']:.1%} (z = {h['soh_z']:.1f}); anode-LAM excess z = "
            f"{h['lam_ne_excess_z']:.1f}. "
            f"Usage: {h['usage']['temp_mean']:.1f} C mean cell temperature, {h['usage']['frac_high_mean']:.0%} of "
            f"time above 95 % SoC.\n{verdict}"
        )

    def _plan(self, did, m, t):
        cal = None
        if "flight" in m or "trip" in m:
            dep = (self.e.now + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
            cal = [{"title": "Flight", "start": dep.isoformat(), "kind": "flight", "duration_h": 8}]
        p = self._call(t, "plan_charging", {"device_id": did, **({"calendar_events": cal} if cal else {})})
        c, L, M, R = p["context"], p["legacy"], p["milp"], p["edge_rl_int8"]
        return (
            f"Tonight for {did}: {c['reason']}. Target {c['target_soc']:.0%} by {c['unplug_at'][11:16]} "
            f"(from {c['soc_now']:.0%}).\n"
            f"- Legacy charge-to-100 %: damage {L['damage_pct_capacity']:.4f} % capacity, {L['hours_above_95']:.1f} h "
            "above 95 %\n"
            f"- MILP schedule: damage {M['damage_pct_capacity']:.4f} % "
            f"({p['comparison']['milp_damage_reduction_pct']:.0f} % less), "
            f"final {M['final_soc']:.0%}, target met: {M['met_target']}\n"
            f"- On-device INT8 RL policy: damage {R['damage_pct_capacity']:.4f} % "
            f"({p['comparison']['edge_rl_damage_reduction_pct']:.0f} % less), target met: {R['met_target']}"
        )

    def _passport(self, did, m, t):
        role = "authority" if "authorit" in m else "legitimate_interest" if "legitimate" in m else "public"
        p = self._call(t, "get_battery_passport", {"device_id": did, "role": role})
        s = p["state_of_health"]
        return (
            f"Passport {p['passport_id']} (v{p['_version']['version']}, chain valid: {p['_version']['chain_valid']}), "
            f"viewed as {role}: SoH {s['soh_capacity']['value']:.1%}, remaining capacity "
            f"{s['remaining_capacity_ah']['value']} Ah, power capability "
            f"{s['remaining_power_capability_pct']['value']} %, "
            f"{s['full_equivalent_cycles']['value']} full equivalent cycles, status "
            f"{p['status']['battery_status']['value']}. Note: {p['applicability']}"
        )

    def _grade(self, did, t):
        g = self._call(t, "grade_second_life", {"device_id": did})
        return (
            f"{did}: grade {g['grade']} (score {g['score']}). {g['pathway']}. Reasons: "
            + "; ".join(g["reasons"])
            + (
                f". Projected second life to 60 % SoH: {g['second_life_days_to_60pct']} days."
                if g.get("second_life_days_to_60pct")
                else ""
            )
        )

    def _actions(self, t):
        acts = self._call(t, "list_agent_actions", {"limit": 12})
        pend = self._call(t, "list_agent_actions", {"status": "pending_approval", "limit": 50})
        if not acts:
            return "No agent actions yet - run the workforce from the Workforce tab."
        lines = [f"- [{a['status']}] {a['device_id'] or 'fleet'}: {a['title']} ({a['agent']})" for a in acts]
        return (
            f"{len(pend)} action(s) await human approval. Latest:\n"
            + "\n".join(lines)
            + "\nApprovals are made by a person in the Workforce tab; I can't approve on your behalf."
        )

    def _insights(self, t):
        ins = self._call(t, "get_fleet_insights")
        lot = ins["lots"][0]
        a = ins["adaptive_vs_legacy"]
        d = ins["stress_drivers"][0] if ins["stress_drivers"] else None
        return (
            f"Most suspicious lot: {lot['lot']} - anode-LAM excess {lot['lam_ne_excess_z_mean']:.1f} sigma "
            f"(p = {lot['p_value']:.1e}, {lot['outliers']}/{lot['n']} outliers). Adaptive charging cohort fades "
            f"{a['adaptive_fade_pct_per_100d']:.2f} vs {a['legacy_fade_pct_per_100d']:.2f} %/100 days for legacy."
            + (f" Strongest ageing driver: {d['driver']}." if d else "")
        )

    def _metrics(self, t):
        r = self._call(t, "get_model_performance")
        s, rul, an = r["soh"], r["rul_live_fleet"]["all"], r["anomaly"]
        ch = r["charging"]
        return (
            f"(Simulated fleets.) SoH MAE {s['mae_ai']:.2%} vs legacy gauge {s['mae_gauge_baseline']:.2%}; "
            f"live-fleet RUL MAE {rul['mae_days']:.0f} days with {rul['coverage95']:.0%} of truths inside the 95 % "
            f"interval; precursor detection {an['median_days_onset_to_warning']:.0f} days after onset, "
            f"{an['median_warning_lead_before_critical_days']:.0f} days before critical, false-warning rate "
            f"{an['false_warning_rate']:.2%}; charging damage vs legacy: MILP "
            f"-{ch['milp']['damage_reduction_vs_legacy_pct']:.0f} %, RL "
            f"-{ch['rl']['damage_reduction_vs_legacy_pct']:.0f} %."
        )
