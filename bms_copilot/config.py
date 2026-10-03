"""Central configuration for the Next-Gen BMS Copilot.

Every number that encodes an engineering or business decision lives here so it
can be reviewed in one place (cell spec, safety envelope, warranty terms,
autonomy defaults). Values are representative of a 3S1P Li-ion polymer laptop
pack; replace them with the target SKU's datasheet values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


@dataclass(frozen=True)
class CellSpec:
    chemistry: str = "NMC811 / graphite (Li-ion polymer)"
    rated_capacity_ah: float = 4.5  # per cell, C/10 between v_min and v_max
    v_max: float = 4.20  # charge termination voltage
    v_min: float = 3.00  # discharge cut-off
    v_nominal: float = 3.70
    r0_bol_ohm: float = 0.028  # ohmic resistance at 25 C, beginning of life
    r1_bol_ohm: float = 0.016  # fast RC branch
    c1_bol_f: float = 1800.0
    r2_bol_ohm: float = 0.012  # slow RC branch (only in the "truth" simulator)
    c2_bol_f: float = 40000.0
    mass_kg: float = 0.070
    cp_j_per_kgk: float = 1000.0
    ha_w_per_k: float = 0.12  # effective convective cooling inside a laptop chassis
    expected_cycle_life: int = 1000  # to 80 % SoH under the reference profile


@dataclass(frozen=True)
class PackSpec:
    series: int = 3
    parallel: int = 1
    cell: CellSpec = field(default_factory=CellSpec)

    @property
    def rated_energy_wh(self) -> float:
        return self.series * self.parallel * self.cell.rated_capacity_ah * self.cell.v_nominal

    @property
    def rated_capacity_ah(self) -> float:
        return self.parallel * self.cell.rated_capacity_ah


@dataclass(frozen=True)
class SafetyEnvelope:
    """Deterministic, certified limits. AI policies may only tighten these."""

    cell_v_max: float = 4.20
    cell_v_min: float = 3.00
    cell_v_overvoltage_trip: float = 4.25  # hardware cut-off
    cell_v_undervoltage_trip: float = 2.80
    charge_temp_min_c: float = 0.0
    charge_temp_max_c: float = 45.0
    discharge_temp_min_c: float = -20.0
    discharge_temp_max_c: float = 60.0
    overtemp_trip_c: float = 65.0
    max_charge_c_rate: float = 1.5  # absolute ceiling (fast charge)
    max_discharge_c_rate: float = 3.0
    # JEITA-style charge-current derating: (t_low, t_high, max C-rate)
    jeita_bands: tuple = ((0.0, 10.0, 0.3), (10.0, 20.0, 0.7), (20.0, 45.0, 1.5))
    max_cell_divergence_mv: float = 80.0  # balancing fault


@dataclass(frozen=True)
class ServicePolicy:
    eol_soh: float = 0.80  # end of first life
    warranty_days_standard: int = 365
    warranty_days_extended: int = 3 * 365  # extended care plan
    warranted_soh: float = 0.80  # capacity below this inside warranty is a claim
    defect_z_threshold: float = 2.5  # SoH shortfall vs cohort (in sigma) to call a defect
    proactive_replacement_horizon_days: int = 90
    loyalty_discount_pct: int = 15


@dataclass(frozen=True)
class ModelConfig:
    ica_capture_window_v: tuple = (3.55, 4.15)  # opportunistic slow-charge capture window
    ica_bands_v: tuple = ((3.55, 3.75), (3.75, 4.02), (4.02, 4.15))
    ica_grid_mv: float = 2.0
    ica_savgol_window: int = 21
    ica_savgol_order: int = 3
    snapshot_every_days: int = 14
    gpr_max_train_points: int = 900
    rl_c_rates: tuple = (0.0, 0.2, 0.5, 0.8, 1.2)
    rl_step_minutes: int = 15


PACK = PackSpec()
CELL = PACK.cell
SAFETY = SafetyEnvelope()
SERVICE = ServicePolicy()
MODELS = ModelConfig()

# Brand-neutral placeholders: set these for the real programme.
MANUFACTURER = {
    "name": "Example OEM Ltd.",
    "address": "1 Example Way, 1000 Brussels, BE",
    "contact": "battery-compliance@example-oem.test",
    "facility_id": "FAC-EU-0042",
}
PRODUCT_LINE = "Premium Laptop 15"
PASSPORT_BASE_URL = "https://passport.example-oem.test/b/"
