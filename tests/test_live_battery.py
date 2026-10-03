"""Real-battery module, tested on a synthetic Windows battery report."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bms_copilot.live import analysis as A
from bms_copilot.live.service import LiveBatteryService
from bms_copilot.live.windows_battery import iso_duration_s, parse_report

DESIGN = 60000


def _report_xml(path: Path, weeks: int = 40, corrupt: bool = True) -> Path:
    t0 = datetime(2025, 10, 1, 8, 0, tzinfo=UTC)
    hist = []
    for w in range(weeks):
        s, e = t0 + timedelta(days=7 * w), t0 + timedelta(days=7 * (w + 1))
        fcc = DESIGN * (0.90 - 0.0022 * w)  # ~3.1 % per 100 days
        standby_dc = "PT592623H36M0S" if (corrupt and w == weeks - 3) else "PT10H0M0S"
        hist.append(
            f'<HistoryEntry StartDate="{s:%Y-%m-%dT%H:%M:%SZ}" EndDate="{e:%Y-%m-%dT%H:%M:%SZ}" '
            f'DesignCapacity="{DESIGN}" FullChargeCapacity="{fcc:.0f}" CycleCount="{200 + 3 * w}" '
            f'ActiveAcTime="PT30H0M0S" ActiveDcTime="PT5H0M0S" CsAcTime="PT40H0M0S" CsDcTime="{standby_dc}" '
            f'ActiveDcEnergy="80000" CsDcEnergy="5000" BatteryChanged="0" />'
        )
    end = t0 + timedelta(days=7 * weeks)
    fcc_now = DESIGN * (0.90 - 0.0022 * (weeks - 1))
    usage = []
    for d in range(7):
        day = end - timedelta(days=7 - d)
        for hh, ac, frac in ((0, 1, 0.60), (8, 0, 0.60), (12, 0, 0.45), (13, 1, 0.45), (18, 1, 0.60)):
            ts = day.replace(hour=hh)
            usage.append(
                f'<UsageEntry Timestamp="{ts:%Y-%m-%dT%H:%M:%SZ}" LocalTimestamp="{ts:%Y-%m-%dT%H:%M:%S}" '
                f'Ac="{ac}" EntryType="Active" ChargeCapacity="{frac * fcc_now:.0f}" '
                f'FullChargeCapacity="{fcc_now:.0f}" Discharge="0" />'
            )
    path.write_text(
        '<?xml version="1.0"?><BatteryReport xmlns="http://schemas.microsoft.com/battery/2012">'
        "<SystemInformation><ComputerName>SHOULD-NOT-BE-READ</ComputerName></SystemInformation>"
        f"<Batteries><Battery><Id>Test Battery</Id><Manufacturer>ACME</Manufacturer><Chemistry>LIon</Chemistry>"
        f"<DesignCapacity>{DESIGN}</DesignCapacity><FullChargeCapacity>{fcc_now:.0f}</FullChargeCapacity>"
        f"<CycleCount>{200 + 3 * weeks}</CycleCount></Battery></Batteries>"
        "<RuntimeEstimates><DesignCapacity><Capacity>60000</Capacity><ActiveRuntime>PT5H0M0S</ActiveRuntime></DesignCapacity>"
        "<FullChargeCapacity><Capacity>48000</Capacity><ActiveRuntime>PT4H0M0S</ActiveRuntime></FullChargeCapacity></RuntimeEstimates>"
        f"<RecentUsage>{''.join(usage)}</RecentUsage><History>{''.join(hist)}</History></BatteryReport>"
    )
    return path


@pytest.fixture
def rep(tmp_path):
    return parse_report(_report_xml(tmp_path / "r.xml"))


def test_iso_durations():
    assert iso_duration_s("P1DT2H3M4S") == 86400 + 7200 + 180 + 4
    assert iso_duration_s("PT0S") == 0 and iso_duration_s(None) == 0


def test_parse_and_sanitise(rep):
    assert rep.manufacturer == "ACME" and rep.cycles == 320 and len(rep.history) == 40
    assert rep.soh == pytest.approx(0.90 - 0.0022 * 39, abs=1e-3)
    assert sum(not h.times_valid for h in rep.history) == 1  # the 592 623 h entry
    prof = A.usage_profile(rep)
    assert prof["invalid_history_entries"] == 1
    assert 0 < prof["battery_hours_per_day"] < 24 and prof["ac_share_45d"] > 0.5


def test_forecast_consistent_with_trend(rep):
    fc = A.capacity_forecast(rep)
    assert fc["fade_pct_per_100d"] == pytest.approx(3.14, abs=0.6)
    c60 = fc["crossings"]["60"]
    assert c60["status"] == "forecast"
    expected = (rep.soh - 0.60) / 0.00031  # days at 0.0022 per week
    assert c60["lo_days"] < expected < c60["hi_days"]
    assert fc["crossings"]["80"]["status"] == "forecast"


def test_profile_detects_limit_and_routine(rep):
    prof = A.usage_profile(rep)
    assert prof["charge_limit_detected"] and prof["charge_limit_pct"] == 60
    assert prof["typical_unplug_hour"] == pytest.approx(8.0)


def test_plan_respects_limit_unless_trip(rep):
    prof = A.usage_profile(rep)
    live = {"remaining_mwh": 0.4 * rep.fcc_mwh, "fcc_mwh": rep.fcc_mwh, "percent": 40}
    now = datetime(2026, 7, 1, 22, 0).astimezone()
    p = A.charge_plan(rep, prof, live, now)
    assert p["target_soc"] <= 0.60 + 1e-9 and p["ai"]["met_target"]
    trip = [{"title": "Flight", "start": (now + timedelta(hours=9)).isoformat(), "kind": "flight"}]
    q = A.charge_plan(rep, prof, live, now, trip)
    assert q["target_soc"] == 1.0 and q["ai"]["final_soc"] >= 0.99
    assert q["damage_reduction_pct"] > 0


def test_calibrated_projection_orders_scenarios(rep):
    prof = A.usage_profile(rep)
    fc = A.capacity_forecast(rep)
    pr = A.calibrated_projection(rep, prof, fc)
    sc = pr["scenarios"]
    full = sc["always_100pct"]["days_to_60pct"] or 10**6
    ai = sc["ai_adaptive"]["days_to_60pct"] or 10**6
    assert ai >= full


def test_service_file_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("BMS_BATTERY_REPORT", str(_report_xml(tmp_path / "r.xml")))
    svc = LiveBatteryService(tmp_path / "live")
    svc.start()  # no poller in file mode
    s = svc.summary()
    assert s["available"] and s["source"] == "saved report file" and not s["poller"]["running"]
    assert any("charge limit" in r["title"] for r in s["recommendations"])
    assert "SHOULD-NOT-BE-READ" not in str(s)
