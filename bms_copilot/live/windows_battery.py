"""Read a real laptop battery on Windows (no admin rights needed).

Two sources:
* `powercfg /batteryreport /xml` - static data (design / full-charge capacity,
  cycle count), up to a year of capacity history with daily AC/DC time and
  energy, the last ~7 days of AC/battery events and discharge segments.
* WMI `root\\wmi` BatteryStatus / BatteryFullChargedCapacity + Win32_Battery -
  live remaining capacity, pack voltage, charge/discharge rate, AC state.

The report's SystemInformation block (computer name, BIOS) is never parsed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_DUR = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?)?")


def iso_duration_s(text: str | None) -> float:
    """'P1DT16H31M13S' -> seconds."""
    if not text:
        return 0.0
    m = _DUR.fullmatch(text.strip())
    if not m:
        return 0.0
    d, h, mi, s = (float(x) if x else 0.0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def _local(name: str) -> str:
    return name.split("}", 1)[-1]


def _ts(text: str | None) -> datetime | None:
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@dataclass
class HistoryEntry:
    start: datetime
    end: datetime
    design_mwh: float
    fcc_mwh: float
    cycles: int | None
    active_ac_s: float
    active_dc_s: float
    standby_ac_s: float
    standby_dc_s: float
    dc_energy_mwh: float  # energy drawn from the battery in the period

    @property
    def days(self) -> float:
        return max((self.end - self.start).total_seconds() / 86400.0, 1e-6)

    @property
    def times_valid(self) -> bool:
        """Windows occasionally logs impossible durations (e.g. 592 623 h of
        standby in one day); such entries are excluded from usage statistics."""
        period = self.days * 86400 * 1.05
        parts = (self.active_ac_s, self.active_dc_s, self.standby_ac_s, self.standby_dc_s)
        return all(0 <= x <= period for x in parts) and sum(parts) <= 2 * period


@dataclass
class UsageEvent:
    time: datetime  # UTC
    local_time: datetime
    on_ac: bool
    kind: str
    charge_mwh: float
    fcc_mwh: float


@dataclass
class BatteryReport:
    generated: datetime
    battery_id: str
    manufacturer: str
    chemistry: str
    design_mwh: float
    fcc_mwh: float
    cycles: int | None
    history: list[HistoryEntry] = field(default_factory=list)
    usage: list[UsageEvent] = field(default_factory=list)
    runtime_design_s: float = 0.0
    runtime_full_s: float = 0.0
    source: str = ""

    @property
    def soh(self) -> float:
        return self.fcc_mwh / self.design_mwh if self.design_mwh else float("nan")


def parse_report(path: Path) -> BatteryReport:
    root = ET.parse(path).getroot()
    sec = {_local(c.tag): c for c in root}
    bats = [b for b in sec.get("Batteries", []) if _local(b.tag) == "Battery"]
    if not bats:
        raise ValueError("battery report contains no battery")
    b = {_local(c.tag): (c.text or "").strip() for c in bats[0]}
    num = lambda s: float(s) if s not in (None, "") else float("nan")  # noqa: E731

    history = []
    for e in sec.get("History", []):
        a = e.attrib
        if a.get("BatteryChanged") == "1":
            history.clear()  # only the current pack's history is relevant
        try:
            history.append(
                HistoryEntry(
                    _ts(a["StartDate"]),
                    _ts(a["EndDate"]),
                    num(a.get("DesignCapacity")),
                    num(a.get("FullChargeCapacity")),
                    int(a["CycleCount"]) if a.get("CycleCount", "").isdigit() else None,
                    iso_duration_s(a.get("ActiveAcTime")),
                    iso_duration_s(a.get("ActiveDcTime")),
                    iso_duration_s(a.get("CsAcTime")),
                    iso_duration_s(a.get("CsDcTime")),
                    max(0.0, num(a.get("ActiveDcEnergy", 0))) + max(0.0, num(a.get("CsDcEnergy", 0))),
                )
            )
        except (KeyError, ValueError):
            continue
    usage = []
    for e in sec.get("RecentUsage", []):
        a = e.attrib
        try:
            usage.append(
                UsageEvent(
                    _ts(a["Timestamp"]),
                    datetime.fromisoformat(a["LocalTimestamp"]),
                    a.get("Ac") == "1",
                    a.get("EntryType", ""),
                    num(a.get("ChargeCapacity")),
                    num(a.get("FullChargeCapacity")),
                )
            )
        except (KeyError, ValueError):
            continue
    rt = {}
    for e in sec.get("RuntimeEstimates", []):
        vals = {_local(c.tag): (c.text or "") for c in e}
        rt[_local(e.tag)] = iso_duration_s(vals.get("ActiveRuntime"))
    gen = usage[-1].time if usage else (history[-1].end if history else datetime.now().astimezone())
    return BatteryReport(
        generated=gen,
        battery_id=b.get("Id", "battery"),
        manufacturer=b.get("Manufacturer", ""),
        chemistry=b.get("Chemistry", ""),
        design_mwh=num(b.get("DesignCapacity")),
        fcc_mwh=num(b.get("FullChargeCapacity")),
        cycles=int(b["CycleCount"]) if b.get("CycleCount", "").isdigit() else None,
        history=[h for h in history if h.design_mwh > 0 and h.fcc_mwh > 0],
        usage=usage,
        runtime_design_s=rt.get("DesignCapacity", 0.0),
        runtime_full_s=rt.get("FullChargeCapacity", 0.0),
        source=str(path),
    )


def available() -> bool:
    return sys.platform == "win32"


def generate_report(out: Path, timeout_s: float = 90.0) -> Path:
    """Run `powercfg /batteryreport /xml`. Raises RuntimeError on failure."""
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["powercfg", "/batteryreport", "/xml", "/output", str(out)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"powercfg failed: {r.stdout.strip() or r.stderr.strip()}")
    return out


_LIVE_PS = r"""
$s = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1
$f = Get-CimInstance -Namespace root\wmi -ClassName BatteryFullChargedCapacity -ErrorAction SilentlyContinue |
  Select-Object -First 1
$w = Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
[pscustomobject]@{
  remaining_mwh = $s.RemainingCapacity; voltage_mv = $s.Voltage; charge_rate_mw = $s.ChargeRate
  discharge_rate_mw = $s.DischargeRate; power_online = $s.PowerOnline; charging = $s.Charging
  critical = $s.Critical; fcc_mwh = $f.FullChargedCapacity; percent = $w.EstimatedChargeRemaining
  status_code = $w.BatteryStatus
} | ConvertTo-Json -Compress
"""


def read_live(timeout_s: float = 20.0) -> dict | None:
    """One live sample, or None if no battery / not Windows."""
    if not available():
        return None
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _LIVE_PS],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    d = json.loads(r.stdout)
    if d.get("remaining_mwh") is None and d.get("percent") is None:
        return None
    d["time"] = datetime.now().astimezone().isoformat()
    return d
