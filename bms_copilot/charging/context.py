"""User-context model: when will the laptop be unplugged and how much energy
will the user need? Feeds the adaptive-charge optimisers.

Learned from the device's own history (typical unplug hour per weekday /
weekend, recent daily energy use) and overridden by calendar events: a long
flight tomorrow means 100 % just before departure; a light day means holding
a lower charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..sim.profiles import PROFILES

RESERVE = 0.15  # always keep this on top of the predicted need
MIN_TARGET = 0.60


@dataclass
class ChargeNeed:
    hours_to_unplug: float
    target_soc: float
    mode: str
    reason: str
    unplug_at: str

    def to_dict(self) -> dict:
        return {
            "hours_to_unplug": round(self.hours_to_unplug, 2),
            "target_soc": round(self.target_soc, 3),
            "mode": self.mode,
            "reason": self.reason,
            "unplug_at": self.unplug_at,
        }


def predict_need(
    profile_key: str, daily_efc: float, soh: float, now: datetime, calendar: list[dict] | None = None
) -> ChargeNeed:
    prof = PROFILES[profile_key]
    nxt = now + timedelta(days=1) if now.hour >= prof.unplug_hour else now
    weekend = nxt.weekday() >= 5
    unplug_hour = prof.unplug_hour + (1.5 if weekend else 0.0)
    unplug = nxt.replace(hour=int(unplug_hour), minute=int(60 * (unplug_hour % 1)), second=0, microsecond=0)
    # Energy need as a fraction of *current* capacity (SoC is relative to it).
    need = daily_efc * (prof.weekend_factor if weekend else 1.0) / max(soh, 0.5)
    target = float(np.clip(need * 1.25 + RESERVE, MIN_TARGET, 1.0))
    use = (
        f"typical use {100 * need:.0f} % of capacity/day (+25 % margin, +{100 * RESERVE:.0f} % reserve)"
        if need < 0.8
        else "heavy use: more than one full charge per day - charge to full"
    )
    mode, reason = "balanced", f"learned routine: unplug ~{unplug:%a %H:%M}, {use}"
    if profile_key == "docked" and need < 0.35:
        mode = "max_life"
        target = max(MIN_TARGET, min(target, 0.65))
        reason = "mostly on AC power: hold a lower charge to minimise calendar ageing"
    for ev in calendar or []:
        start = datetime.fromisoformat(ev["start"])
        if start.tzinfo is None and now.tzinfo is not None:
            start = start.replace(tzinfo=now.tzinfo)
        if now < start <= unplug + timedelta(hours=12):
            if ev.get("kind") in ("flight", "travel", "offsite") or ev.get("duration_h", 0) >= 4:
                unplug = min(unplug, start - timedelta(hours=float(ev.get("lead_h", 1.0))))
                target, mode = 1.0, "balanced"
                reason = f"calendar: '{ev.get('title', 'event')}' at {start:%a %H:%M} - full charge just before leaving"
            elif ev.get("kind") == "light_day":
                target = max(MIN_TARGET, min(target, 0.7))
                reason = f"calendar: light day ('{ev.get('title', 'event')}') - cap at {target:.0%}"
    hours = max(0.25, (unplug - now).total_seconds() / 3600.0)
    return ChargeNeed(hours, target, mode, reason, unplug.isoformat())
