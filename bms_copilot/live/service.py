"""Runtime service for the real battery of the machine running the copilot.

* regenerates the Windows battery report at most every 30 minutes (or on demand)
* polls live telemetry in a background thread and logs it to live_samples.csv
  (the log is what the experimental real-hardware dQ/dV uses)
* caches the analysis per report so the dashboard and copilot stay fast

`BMS_BATTERY_REPORT=<path.xml>` analyses a saved `powercfg /batteryreport /xml`
file instead (e.g. collected from another laptop) - no live polling then.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from . import analysis as A
from .windows_battery import available, generate_report, parse_report, read_live

FIELDS = [
    "time",
    "percent",
    "remaining_mwh",
    "fcc_mwh",
    "voltage_mv",
    "charge_rate_mw",
    "discharge_rate_mw",
    "power_online",
    "charging",
]


class LiveBatteryService:
    def __init__(self, workdir: Path, poll_s: float = 20.0, report_max_age_s: float = 1800.0):
        self.dir = workdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.poll_s = poll_s
        self.max_age = report_max_age_s
        self.file_mode = os.environ.get("BMS_BATTERY_REPORT")
        self.samples: deque = deque(maxlen=3000)
        self._lock = threading.Lock()
        self._report = None
        self._report_at = 0.0
        self._analysis = None
        self._stop = threading.Event()
        self._thread = None
        self.error: str | None = None
        self._load_log()

    # ------------------------------------------------------------- telemetry
    @property
    def log_path(self) -> Path:
        return self.dir / "live_samples.csv"

    def _load_log(self) -> None:
        if not self.log_path.exists():
            return
        with self.log_path.open(newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append({k: _coerce(v) for k, v in row.items()})

    def _append(self, s: dict) -> None:
        self.samples.append(s)
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(s)

    def start(self) -> None:
        if self.file_mode or not available() or self._thread:
            return
        self._thread = threading.Thread(target=self._loop, name="battery-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = read_live()
                if s:
                    with self._lock:
                        self._append(s)
            except Exception as e:  # keep polling; surface the last error
                self.error = f"live read failed: {type(e).__name__}"
            self._stop.wait(self.poll_s)

    def latest(self) -> dict | None:
        if self.samples:
            return dict(self.samples[-1])
        if self.file_mode or not available():
            return None
        s = read_live()
        if s:
            with self._lock:
                self._append(s)
        return s

    # ---------------------------------------------------------------- report
    def report(self, refresh: bool = False):
        if self.file_mode:
            if self._report is None:
                self._report = parse_report(Path(self.file_mode))
            return self._report
        if not available():
            raise RuntimeError("live battery access needs Windows (or set BMS_BATTERY_REPORT to a report XML)")
        if refresh or self._report is None or time.time() - self._report_at > self.max_age:
            path = generate_report(self.dir / "battery-report.xml")
            self._report = parse_report(path)
            self._report_at = time.time()
            self._analysis = None
        return self._report

    # -------------------------------------------------------------- analysis
    def summary(self, refresh: bool = False) -> dict:
        try:
            rep = self.report(refresh)
        except Exception as e:
            return {"available": False, "reason": str(e)}
        if self._analysis is None or refresh:
            fc = A.capacity_forecast(rep)
            prof = A.usage_profile(rep)
            proj = A.calibrated_projection(rep, prof, fc)
            self._analysis = {
                "forecast": fc,
                "profile": prof,
                "projection": proj,
                "recommendations": A.recommendations(rep, prof, fc, proj),
            }
        live = self.latest()
        return {
            "available": True,
            "source": "saved report file" if self.file_mode else "this computer (Windows battery report + WMI)",
            "battery": {
                "id": rep.battery_id,
                "manufacturer": rep.manufacturer,
                "chemistry": rep.chemistry,
                "design_mwh": rep.design_mwh,
                "full_charge_mwh": rep.fcc_mwh,
                "cycles": rep.cycles,
                "soh_reported": rep.soh,
                "runtime_new_h": rep.runtime_design_s / 3600,
                "runtime_now_h": rep.runtime_full_s / 3600,
                "report_time": rep.generated.isoformat(),
                "history_entries": len(rep.history),
            },
            "live": live,
            "live_trace": self.trace(),
            **self._analysis,
            "ica_experimental": A.ica_from_samples(list(self.samples)),
            "poller": {
                "running": bool(self._thread and self._thread.is_alive()),
                "interval_s": self.poll_s,
                "samples_logged": len(self.samples),
                "error": self.error,
            },
        }

    def trace(self, n: int = 360) -> dict:
        rows = list(self.samples)[-n:]
        return {
            "time": [r.get("time") for r in rows],
            "percent": [r.get("percent") for r in rows],
            "voltage_mv": [r.get("voltage_mv") for r in rows],
            "power_w": [((r.get("charge_rate_mw") or 0) - (r.get("discharge_rate_mw") or 0)) / 1000 for r in rows],
            "on_ac": [r.get("power_online") for r in rows],
        }

    def charge_plan(self, calendar: list | None = None) -> dict:
        rep = self.report()
        prof = (self._analysis or {}).get("profile") or A.usage_profile(rep)
        return A.charge_plan(rep, prof, self.latest(), datetime.now().astimezone(), calendar)


def _coerce(v: str):
    if v in ("", None):
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except ValueError:
        return v
