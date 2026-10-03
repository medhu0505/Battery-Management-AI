"""User behaviour archetypes and climates for fleet simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageProfile:
    key: str
    label: str
    efc_mean: float  # equivalent full cycles per weekday
    weekend_factor: float
    mean_soc: float  # with the legacy "charge to 100 % ASAP" policy
    frac_high_soc: float  # fraction of the day above 95 % SoC (legacy policy)
    charge_c: float  # typical charge rate (legacy policy)
    dod: float
    load_temp_rise_c: float  # cell temperature rise under this workload
    fast_charge_share: float  # share of days that genuinely need a fast top-up
    unplug_hour: float  # typical first unplug of the day (for context model)
    weight: float  # share of the fleet


PROFILES: dict[str, UsageProfile] = {
    p.key: p
    for p in [
        UsageProfile("office", "Hybrid office worker", 1.0, 0.3, 0.75, 0.35, 0.7, 0.70, 4.0, 0.10, 8.0, 0.34),
        UsageProfile("docked", "Always docked (AC-dweller)", 0.25, 0.5, 0.96, 0.85, 0.5, 0.20, 9.0, 0.02, 9.0, 0.16),
        UsageProfile("road", "Road warrior", 1.4, 0.8, 0.65, 0.20, 1.0, 0.80, 4.5, 0.45, 6.5, 0.14),
        UsageProfile("creator", "Creator / gamer (heavy load)", 1.3, 1.1, 0.70, 0.30, 1.0, 0.80, 9.0, 0.25, 10.0, 0.14),
        UsageProfile("student", "Student / light use", 0.7, 0.8, 0.70, 0.30, 0.7, 0.60, 3.0, 0.10, 8.5, 0.22),
    ]
}

CLIMATES = {"temperate": 16.0, "warm": 22.0, "hot": 29.0, "cold": 9.0}
CLIMATE_WEIGHTS = {"temperate": 0.45, "warm": 0.25, "hot": 0.15, "cold": 0.15}


def adaptive_transform(mean_soc, frac_high, charge_c, fast_needed):
    """How the AI adaptive-charge policy changes daily stress.

    Holds charge at ~80 % until shortly before the predicted unplug time, charges
    overnight at a gentle rate, and only fast-charges when the context model
    says the energy is actually needed.
    """
    import numpy as np

    mean_soc = np.minimum(mean_soc, 0.62 + 0.1 * np.asarray(fast_needed))
    frac_high = np.asarray(frac_high) * np.where(fast_needed, 0.5, 0.12)
    charge_c = np.where(fast_needed, charge_c, np.minimum(charge_c, 0.35))
    return mean_soc, frac_high, charge_c
