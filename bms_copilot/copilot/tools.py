"""Tools the BMS Copilot (LLM) may call, backed by the engine.

Deliberately read/analyse-only: the copilot can explain, diagnose, plan and
simulate, but approving agent actions, changing the autonomy level, pushing
OTA packages or touching safety limits stays with a human in the dashboard.
"""

from __future__ import annotations

import json

TOOLS = [
    {
        "name": "get_fleet_overview",
        "description": "Fleet KPIs: device count, mean SoH, devices below 85 % / end of life, short RUL, anomaly "
        "counts, "
        "adaptive-charging share, warranty exposure, workforce autonomy level and open approvals.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_devices",
        "description": "List devices with SoH, RUL and anomaly level. Use to find at-risk devices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "enum": ["soh", "rul", "anomaly", "soh_z"],
                    "description": "soh/rul ascending (worst first); anomaly = most severe first; soh_z = largest "
                    "shortfall vs expectation first",
                },
                "profile": {"type": "string", "enum": ["office", "docked", "road", "creator", "student"]},
                "anomaly_level": {"type": "string", "enum": ["watch", "warning", "critical"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["sort_by"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_device_health",
        "description": "Health of one device: SoH with 95 % interval (ICA + GPR), legacy gauge SoH, expected SoH for "
        "its "
        "age/usage and z-score, RUL median + 95 % interval, degradation-mode estimates, anomaly/precursor "
        "status, usage stress, warranty, derating, status.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string", "description": "e.g. DEV-0041"}},
            "required": ["device_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "diagnose_device",
        "description": "Run degradation-mode analysis on a full diagnostic slow-charge capture: loss of lithium "
        "inventory "
        "(LLI), cathode and anode active-material loss (LAM_pe, LAM_ne), ohmic offset, dominant mechanism.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "plan_charging",
        "description": "Adaptive charge plan for tonight: predicts unplug time and energy need (optionally from "
        "calendar "
        "events), then compares legacy charge-to-100 %, the MILP schedule and the on-device INT8 RL policy "
        "on capacity damage, readiness and energy cost.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "calendar_events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "start": {"type": "string", "description": "ISO datetime, e.g. 2026-09-27T07:00"},
                            "kind": {"type": "string", "enum": ["flight", "travel", "offsite", "light_day", "meeting"]},
                            "duration_h": {"type": "number"},
                        },
                        "required": ["title", "start", "kind"],
                        "additionalProperties": False,
                    },
                },
                "soc_now": {"type": "number", "minimum": 0, "maximum": 1},
                "ambient_c": {"type": "number"},
                "mode": {"type": "string", "enum": ["balanced", "max_life", "ready_asap"]},
            },
            "required": ["device_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "whatif_adaptive_charging",
        "description": "Digital-twin projection of SoH over 3 years for this device with legacy vs adaptive charging.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_battery_passport",
        "description": "Passport-style record (EU 2023/1542 Annex XIII structure) as seen by a role.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "role": {"type": "string", "enum": ["public", "legitimate_interest", "authority"]},
            },
            "required": ["device_id", "role"],
            "additionalProperties": False,
        },
    },
    {
        "name": "grade_second_life",
        "description": "Second-life readiness grade (A/B/C/F), score, pathway and reasons for a device's pack.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_agent_actions",
        "description": "Actions proposed/executed by the agentic workforce (triage, service, logistics, asset "
        "recovery, "
        "design feedback) with status: executed, pending_approval, awaiting_user, recommended, rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["executed", "pending_approval", "awaiting_user", "recommended", "rejected"],
                },
                "device_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_fleet_insights",
        "description": "Fleet analytics for design feedback: per-lot anomaly tests (anode-LAM excess), adaptive vs "
        "legacy "
        "fade rates, and the strongest ageing drivers.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_my_battery",
        "description": "The REAL battery of the computer running the copilot (Windows battery report + live WMI): "
        "design vs full-charge capacity, cycles, live charge/voltage/power, GP capacity-fade forecast with "
        "95 % interval for reaching 70 / 60 % of design, usage profile (AC share, daily energy, detected "
        "charge limit, typical unplug time), calibrated what-if and recommendations.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "plan_my_charge",
        "description": "Tonight's adaptive charge plan for the REAL battery of this computer, from its live state of "
        "charge and learned unplug time; an optional calendar trip forces a full charge before leaving.",
        "input_schema": {
            "type": "object",
            "properties": {
                "calendar_events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "start": {"type": "string"},
                            "kind": {"type": "string", "enum": ["flight", "travel", "offsite"]},
                        },
                        "required": ["title", "start", "kind"],
                        "additionalProperties": False,
                    },
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_model_performance",
        "description": "Validation metrics of every model (SoH, RUL, modes, anomaly detection, twin, charging "
        "policies, "
        "TinyML) measured on simulated fleets.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

_ORDER = {"critical": 0, "warning": 1, "watch": 2, "normal": 3}


def _r(x, n=4):
    return round(float(x), n)


def _device_row(r: dict) -> dict:
    return {
        "device_id": r["device_id"],
        "profile": r["profile"],
        "age_days": r["age_days"],
        "soh": _r(r["soh"], 3),
        "rul_days": r["rul"]["median_days"],
        "rul_95": [r["rul"]["lo_days"], r["rul"]["hi_days"]],
        "rul_status": r["rul"].get("status", "forecast"),
        "anomaly": r["anomaly"],
        "adaptive": r["adaptive"],
        "status": r["status"],
    }


def run_tool(engine, name: str, args: dict):
    if name == "get_fleet_overview":
        fo = engine.fleet_overview()
        return {"now": fo["now"], "kpis": fo["kpis"], "workforce": fo["workforce"]}
    if name == "list_devices":
        rows = engine.fleet_overview()["devices"]
        if args.get("profile"):
            rows = [r for r in rows if r["profile"] == args["profile"]]
        if args.get("anomaly_level"):
            rows = [r for r in rows if r["anomaly"] == args["anomaly_level"]]
        key = {
            "soh": lambda r: r["soh"],
            "rul": lambda r: (r["rul"]["median_days"], r["soh"]),
            "anomaly": lambda r: (_ORDER[r["anomaly"]], r["soh"]),
            "soh_z": lambda r: r["soh_z"],
        }[args["sort_by"]]
        return [_device_row(r) for r in sorted(rows, key=key)[: args.get("limit", 10)]]
    if name == "get_device_health":
        h = engine.health(args["device_id"])
        keep = [
            "device_id",
            "profile_label",
            "climate",
            "lot",
            "age_days",
            "adaptive",
            "soh",
            "soh_interval",
            "gauge_soh_legacy",
            "expected_soh",
            "soh_z",
            "fade_pct_per_100d",
            "rul",
            "modes_est",
            "lam_ne_excess_z",
            "anomaly",
            "anomaly_recent_levels",
            "usage",
            "r0_mohm",
            "r0_growth_pct",
            "warranty_days_left",
            "extended_warranty",
            "derated",
            "status",
            "l5_consent",
        ]
        return {k: h[k] for k in keep if k in h}
    if name == "diagnose_device":
        return engine.diagnose(args["device_id"])
    if name == "plan_charging":
        p = engine.charge_plan(
            args["device_id"], args.get("calendar_events"), args.get("soc_now"), args.get("ambient_c"), args.get("mode")
        )
        slim = lambda r: {k: (_r(v) if isinstance(v, float) else v) for k, v in r.items() if k != "trace"}  # noqa: E731
        return {
            "context": p["context"],
            "legacy": slim(p["legacy"]),
            "milp": slim(p["milp"]),
            "edge_rl_int8": slim(p["edge_rl_int8"]),
            "comparison": p["comparison"],
        }
    if name == "whatif_adaptive_charging":
        d = engine.device_detail(args["device_id"])["whatif"]

        def pick(pr):
            return {
                "eol_in_days": pr["eol_in_days"],
                "soh_in_1y": pr["soh"][min(26, len(pr["soh"]) - 1)],  # projections step 14 days
                "soh_in_3y": pr["soh"][-1],
            }

        return {"legacy": pick(d["legacy"]), "adaptive": pick(d["adaptive"]), "note": d["note"]}
    if name == "get_battery_passport":
        p = engine.passport(args["device_id"], args["role"])
        p.pop("_qr_svg", None)
        return p
    if name == "grade_second_life":
        return engine.second_life(args["device_id"])
    if name == "list_agent_actions":
        acts = engine.actions(args.get("status"), args.get("device_id"), args.get("limit", 15))
        return [{k: a[k] for k in ("id", "agent", "device_id", "kind", "title", "status", "rationale")} for a in acts]
    if name == "get_fleet_insights":
        ins = engine.fleet_insights()
        ins["lots"] = ins["lots"][:5]
        return ins
    if name == "get_my_battery":
        b = getattr(engine, "battery", None)
        if b is None:
            return {"available": False, "reason": "battery service not running"}
        m = b.summary()
        if not m.get("available"):
            return m
        fc = m["forecast"]
        return {
            "available": True,
            "battery": m["battery"],
            "live": m["live"],
            "source": m["source"],
            "forecast": {
                k: fc.get(k)
                for k in (
                    "soh_now_smoothed",
                    "fade_pct_per_100d",
                    "cycles_per_day",
                    "observed_change_60d_pct",
                    "crossings",
                    "model",
                )
            },
            "profile": m["profile"],
            "whatif_days_to_60pct": {k: v["days_to_60pct"] for k, v in m["projection"].get("scenarios", {}).items()},
            "calibration": m["projection"].get("calibration"),
            "recommendations": m["recommendations"],
        }
    if name == "plan_my_charge":
        b = getattr(engine, "battery", None)
        if b is None:
            return {"available": False, "reason": "battery service not running"}
        p = b.charge_plan(args.get("calendar_events"))
        slim = lambda r: {k: (_r(v) if isinstance(v, float) else v) for k, v in r.items() if k != "trace"}  # noqa: E731
        keep = ("soc_now", "target_soc", "unplug_at", "hours", "reason", "damage_reduction_pct", "note")
        return {"available": True, **{k: p[k] for k in keep}, "legacy": slim(p["legacy"]), "ai": slim(p["ai"])}
    if name == "get_model_performance":
        r = engine.report
        return {
            "disclaimer": r["disclaimer"],
            "soh": r["health"]["soh"],
            "modes_from_routine_ica": r["health"].get("modes_from_routine_ica"),
            "rul_heldout": r["health"]["rul"],
            "rul_live_fleet": r["health"]["live_fleet_rul"],
            "anomaly": {k: v for k, v in r["anomaly"].items() if k != "cases"},
            "twin": {k: v for k, v in r["twin"].items() if k != "features"},
            "charging": r["charging"]["policy_comparison"],
            "tinyml": {"final_bytes": r["tinyml"]["stages"][-1]["bytes"], "closed_loop": r["tinyml"]["closed_loop"]},
        }
    raise KeyError(f"unknown tool {name}")


def run_tool_json(engine, name: str, args: dict) -> tuple[str, bool]:
    """Returns (json text, is_error)."""
    try:
        return json.dumps(run_tool(engine, name, args or {}), default=float), False
    except KeyError as e:
        return json.dumps({"error": str(e)}), True
    except Exception as e:  # tool failures are reported to the model, not raised
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
