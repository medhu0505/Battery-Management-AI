"""Battery passport record, structured after Regulation (EU) 2023/1542 Annex XIII.

Applicability: Art. 77 makes the battery passport mandatory from 18 Feb 2027
for LMT batteries, industrial batteries > 2 kWh and EV batteries. Portable
batteries in laptops are *not* in that scope. This module therefore produces a
voluntary, passport-style record that follows the Annex XIII data categories,
so the programme is ready if scope widens and can reuse the same data for
second-life certification and customer transparency.

Access tiers (Annex XIII 1-3): public, persons with a legitimate interest,
and notified bodies / market-surveillance authorities / Commission. Each field
is tagged with the minimum tier; `render(role)` filters accordingly.

Every published version is hash-chained to the previous one (data lineage).
Declared values that only the manufacturer can supply (carbon footprint,
due-diligence report, recycled content) are explicit placeholders.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..config import CELL, MANUFACTURER, PACK, PASSPORT_BASE_URL, PRODUCT_LINE, SAFETY
from .ota import digest

ROLES = ["public", "legitimate_interest", "authority"]
APPLICABILITY_NOTE = (
    "Voluntary passport-style record. Regulation (EU) 2023/1542 Art. 77 mandates battery passports "
    "from 18 Feb 2027 for LMT, industrial (>2 kWh) and EV batteries; portable laptop batteries are "
    "outside that scope. Structure follows Annex XIII to enable second-life certification."
)
DECLARED = "to be declared by manufacturer"


def _f(value, tier: str = "public", unit: str | None = None, source: str | None = None) -> dict:
    d = {"value": value, "access": tier}
    if unit:
        d["unit"] = unit
    if source:
        d["source"] = source
    return d


def build_record(device, dyn: dict, second_life: dict | None, now: datetime) -> dict:
    """device: DeviceSpec; dyn: dynamic metrics computed by the engine."""
    rated_wh = PACK.rated_energy_wh
    return {
        "passport_id": f"PP-{device.serial}",
        "battery_identifier": device.serial,
        "link": PASSPORT_BASE_URL + device.serial,
        "applicability": APPLICABILITY_NOTE,
        "general": {
            "manufacturer": _f(MANUFACTURER["name"]),
            "manufacturer_address": _f(MANUFACTURER["address"]),
            "manufacturing_facility": _f(MANUFACTURER["facility_id"]),
            "battery_category": _f("portable battery (general use) - laptop"),
            "model": _f(f"{PRODUCT_LINE} pack {PACK.series}S{PACK.parallel}P"),
            "cell_lot": _f(device.lot, "legitimate_interest"),
            "date_of_manufacture": _f(device.sale_date[:7], source="approximated from first activation"),
            "weight_kg": _f(round(PACK.series * PACK.parallel * CELL.mass_kg + 0.03, 3), unit="kg"),
            "chemistry": _f(CELL.chemistry),
            "critical_raw_materials": _f(["cobalt", "lithium", "natural graphite", "nickel"]),
            "hazardous_substances": _f(["LiPF6 electrolyte salt (hydrofluoric acid on decomposition)"]),
        },
        "carbon_footprint": {
            "total_kgco2e": _f(DECLARED),
            "per_kwh_kgco2e": _f(DECLARED),
            "lifecycle_stage_shares": _f(DECLARED),
            "performance_class": _f(DECLARED),
        },
        "supply_chain_due_diligence": {"report_link": _f(DECLARED)},
        "circularity": {
            "recycled_content_share": _f({"cobalt": DECLARED, "lithium": DECLARED, "nickel": DECLARED}),
            "renewable_content_share": _f(DECLARED),
            "user_replaceable": _f(True, source="design requirement Art. 11 (portable batteries in appliances)"),
            "dismantling_information": _f(
                "pack removable after base-cover removal (T5 screws); disconnect BMS flex before cell handling",
                "legitimate_interest",
            ),
            "part_numbers": _f(
                {"pack": f"{PRODUCT_LINE[:3].upper()}-BAT-{PACK.series}S", "bms_board": "BMS-AI-01"},
                "legitimate_interest",
            ),
            "safety_measures": _f(
                "discharge to <30 % SoC before transport; UN38.3 packaging; isolate if swollen", "legitimate_interest"
            ),
        },
        "rated_performance": {
            "rated_capacity_ah": _f(CELL.rated_capacity_ah * PACK.parallel, unit="Ah"),
            "rated_energy_wh": _f(round(rated_wh, 1), unit="Wh"),
            "voltage_min_nominal_max_v": _f(
                [round(PACK.series * v, 2) for v in (CELL.v_min, CELL.v_nominal, CELL.v_max)], unit="V"
            ),
            "original_power_capability_w": _f(
                round(PACK.series * CELL.v_nominal * CELL.rated_capacity_ah * SAFETY.max_discharge_c_rate), unit="W"
            ),
            "expected_lifetime_cycles": _f(CELL.expected_cycle_life),
            "capacity_threshold_for_exhaustion": _f(0.80),
            "temperature_range_c": _f(
                {
                    "charge": [SAFETY.charge_temp_min_c, SAFETY.charge_temp_max_c],
                    "discharge": [SAFETY.discharge_temp_min_c, SAFETY.discharge_temp_max_c],
                }
            ),
            "commercial_warranty_days": _f(device.warranty_days()),
            "initial_internal_resistance_mohm": _f(round(1000 * PACK.series * CELL.r0_bol_ohm, 1), unit="mOhm"),
            "initial_round_trip_efficiency": _f(0.95),
        },
        "state_of_health": {  # dynamic - Annex VII parameters
            "soh_capacity": _f(dyn["soh"], source="ICA features + GPR (cloud twin)"),
            "soh_interval_95": _f(dyn["soh_interval"]),
            "remaining_capacity_ah": _f(dyn["remaining_capacity_ah"], unit="Ah"),
            "remaining_power_capability_pct": _f(dyn["power_capability_pct"], unit="%"),
            "remaining_round_trip_efficiency": _f(dyn["round_trip_efficiency"]),
            "internal_resistance_mohm": _f(dyn["r0_mohm"], unit="mOhm"),
            "internal_resistance_increase_pct": _f(dyn["r0_growth_pct"], unit="%"),
            "self_discharge_evolution_pct_day": _f(dyn["self_discharge_history"], "legitimate_interest"),
            "full_equivalent_cycles": _f(dyn["efc_total"]),
            "energy_throughput_kwh": _f(dyn["energy_throughput_kwh"], unit="kWh"),
            "capacity_fade_pct": _f(dyn["capacity_fade_pct"], unit="%"),
            "power_fade_pct": _f(dyn["power_fade_pct"], unit="%"),
            "rul_days_median_and_95": _f(dyn["rul"], source="GPR on SoH trajectory + usage stress"),
            "degradation_modes": _f(
                dyn.get("modes"), "legitimate_interest", source="degradation-mode analysis of diagnostic capture"
            ),
        },
        "negative_events": {
            "deep_discharge_events": _f(dyn["deep_discharges"]),
            "time_above_45c_hours": _f(dyn["overtemp_hours"], unit="h"),
            "safety_precursor_alerts": _f(dyn["anomaly_events"], "legitimate_interest"),
            "accidents": _f([]),
        },
        "status": {
            "battery_status": _f(dyn.get("status", "original")),
            "second_life_assessment": _f(second_life, "legitimate_interest"),
        },
        "compliance": {
            "eu_declaration_of_conformity": _f(f"DoC-{PRODUCT_LINE[:3].upper()}-2026-{PACK.series}S"),
            "separate_collection_symbol": _f(True),
            "test_reports": _f(["IEC 62133-2", "UN 38.3", "BMS HIL campaign HIL-2026-07"], "authority"),
            "bms_functional_safety_evidence": _f("safety case + MBSE trace export", "authority"),
        },
        "updated_at": now.isoformat(),
    }


def render(record: dict, role: str) -> dict:
    """Filter a record to the fields visible to `role`."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    level = ROLES.index(role)

    def walk(node):
        if isinstance(node, dict) and "access" in node and "value" in node:
            return node if ROLES.index(node["access"]) <= level else None
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items()}
            return {k: v for k, v in out.items() if v is not None and v != {}}
        return node

    out = walk(record)
    out["viewer_role"] = role
    return out


class PassportStore:
    """Append-only, hash-chained passport versions per battery."""

    def __init__(self, data: dict | None = None):
        self.versions: dict[str, list] = data or {}

    def publish(self, record: dict) -> dict:
        chain = self.versions.setdefault(record["battery_identifier"], [])
        body = {k: v for k, v in record.items() if k != "updated_at"}
        h = digest(body)
        if chain and chain[-1]["hash"] == h:
            return chain[-1]
        entry = {
            "version": len(chain) + 1,
            "hash": h,
            "prev_hash": chain[-1]["hash"] if chain else None,
            "published_at": record.get("updated_at", datetime.now(UTC).isoformat()),
            "record": record,
        }
        chain.append(entry)
        return entry

    def latest(self, battery_id: str) -> dict | None:
        chain = self.versions.get(battery_id)
        return chain[-1] if chain else None

    def verify_chain(self, battery_id: str) -> bool:
        prev = None
        for e in self.versions.get(battery_id, []):
            body = {k: v for k, v in e["record"].items() if k != "updated_at"}
            if e["prev_hash"] != prev or digest(body) != e["hash"]:
                return False
            prev = e["hash"]
        return True


def qr_svg(url: str) -> str | None:
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    return img.to_string(encoding="unicode")
