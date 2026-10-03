"""REST API + dashboard for the Next-Gen BMS Copilot.

Run:  python -m bms_copilot.api   (http://127.0.0.1:8000)
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agents.workforce import LEVELS
from ..config import ARTIFACTS
from ..copilot.assistant import Copilot
from ..engine import Engine
from ..live.service import LiveBatteryService

WEB = Path(__file__).resolve().parent.parent / "web"
_lock = threading.Lock()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    art = Path(os.environ.get("BMS_ARTIFACTS", ARTIFACTS))
    state["engine"] = Engine(art)
    state["battery"] = LiveBatteryService(art / "live")
    state["engine"].battery = state["battery"]
    state["battery"].start()
    state["copilot"] = Copilot(state["engine"])
    yield
    state["battery"].stop()
    state.clear()


app = FastAPI(title="Next-Gen BMS Copilot", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Browsers must revalidate the dashboard's own files, so an update is never
    hidden behind a cached script (ETag keeps unchanged files cheap)."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def eng() -> Engine:
    return state["engine"]


def guarded(fn, *args, **kwargs):
    with _lock:
        try:
            return fn(*args, **kwargs)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'")) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None


class CalendarEvent(BaseModel):
    title: str
    start: str
    kind: str = Field("travel", pattern="^(flight|travel|offsite|light_day|meeting)$")
    duration_h: float = 0.0


class ChargeRequest(BaseModel):
    calendar_events: list[CalendarEvent] | None = None
    soc_now: float | None = Field(None, ge=0.0, le=1.0)
    ambient_c: float | None = Field(None, ge=-20, le=50)
    mode: str | None = Field(None, pattern="^(balanced|max_life|ready_asap)$")


class LevelRequest(BaseModel):
    level: int = Field(ge=1, le=5)


class Decision(BaseModel):
    approve: bool
    actor: str = "engineer@dashboard"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    e = eng()
    return {
        "now": e.now.isoformat(),
        "devices": len(e.devices),
        "copilot_mode": state["copilot"].mode,
        "autonomy_level": e.workforce.level,
        "built_at": e.report.get("built_at"),
        "disclaimer": e.report.get("disclaimer"),
    }


@app.get("/api/fleet")
def fleet():
    return guarded(eng().fleet_overview)


@app.get("/api/fleet/insights")
def insights():
    return guarded(eng().fleet_insights)


@app.get("/api/devices/{device_id}")
def device(device_id: str):
    return guarded(eng().device_detail, device_id)


@app.post("/api/devices/{device_id}/diagnose")
def diagnose(device_id: str):
    return guarded(eng().diagnose, device_id)


@app.post("/api/devices/{device_id}/charge-plan")
def charge_plan(device_id: str, req: ChargeRequest):
    cal = [c.model_dump() for c in req.calendar_events] if req.calendar_events else None
    return guarded(eng().charge_plan, device_id, cal, req.soc_now, req.ambient_c, req.mode)


@app.get("/api/devices/{device_id}/passport")
def passport(device_id: str, role: str = "public"):
    return guarded(eng().passport, device_id, role)


@app.get("/api/devices/{device_id}/second-life")
def second_life(device_id: str):
    return guarded(eng().second_life, device_id)


@app.post("/api/devices/{device_id}/ota")
def ota(device_id: str):
    def run():
        out = eng().ota_package(device_id)
        eng().save_state()
        return out

    return guarded(run)


@app.get("/api/devices/{device_id}/soc-benchmark")
def soc_bench(device_id: str):
    return guarded(eng().soc_benchmark, device_id)


@app.get("/api/workforce")
def workforce(status: str | None = None, device_id: str | None = None, limit: int = 5000):
    e = eng()

    def run():
        counts: dict = {}
        for a in e.workforce.actions:
            counts[a.status] = counts.get(a.status, 0) + 1
        return {
            "level": e.workforce.level,
            "levels": LEVELS,
            "counts": counts,
            "total": len(e.workforce.actions),
            "actions": e.actions(status, device_id, limit),
        }

    return guarded(run)


@app.put("/api/workforce/level")
def set_level(req: LevelRequest):
    return guarded(eng().set_level, req.level)


@app.post("/api/workforce/run")
def run_workforce():
    return guarded(eng().run_workforce)


@app.post("/api/workforce/actions/{action_id}/decision")
def decide(action_id: str, d: Decision):
    def run():
        try:
            return eng().decide(action_id, d.approve, d.actor)
        except StopIteration:
            raise KeyError(f"unknown action {action_id}") from None

    return guarded(run)


@app.post("/api/workforce/reset")
def reset():
    return guarded(lambda: (eng().reset_state(), {"reset": True})[1])


class BatteryPlanRequest(BaseModel):
    calendar_events: list[CalendarEvent] | None = None


@app.get("/api/mybattery")
def my_battery(refresh: bool = False):
    return state["battery"].summary(refresh)


@app.get("/api/mybattery/live")
def my_battery_live():
    b = state["battery"]
    return {"latest": b.latest(), "trace": b.trace(), "samples_logged": len(b.samples)}


@app.post("/api/mybattery/charge-plan")
def my_battery_plan(req: BatteryPlanRequest):
    cal = [c.model_dump() for c in req.calendar_events] if req.calendar_events else None
    try:
        return state["battery"].charge_plan(cal)
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from None


@app.get("/api/models")
def models():
    return eng().report


@app.post("/api/copilot")
def copilot(req: ChatRequest):
    return guarded(state["copilot"].chat, req.message, req.session_id)
