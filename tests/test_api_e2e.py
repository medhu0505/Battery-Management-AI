"""End-to-end: build a small artifact set, then drive every API route.

Slow (~2 min): builds its own artifacts in a temp dir; skip with `-m "not slow"`.
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    out = tmp_path_factory.mktemp("artifacts")
    from bms_copilot import train

    train.main(["--quick", "--out", str(out)])
    monkeypatch_module.setenv("BMS_ARTIFACTS", str(out))
    monkeypatch_module.setenv("BMS_COPILOT_LLM", "off")
    from bms_copilot.api.server import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_fleet_and_device(client):
    fleet = client.get("/api/fleet").json()
    assert fleet["kpis"]["devices"] == len(fleet["devices"]) > 0
    did = min(fleet["devices"], key=lambda d: d["soh"])["device_id"]
    d = client.get(f"/api/devices/{did}").json()
    assert len(d["series"]["soh_est"]) == len(d["series"]["day"]) and len(d["ica"]) >= 2
    assert client.get("/api/devices/DEV-9999").status_code == 404
    dma = client.post(f"/api/devices/{did}/diagnose").json()
    assert set(dma) >= {"lli", "lam_pe", "lam_ne", "dominant"}


def test_charge_plan_and_validation(client):
    did = client.get("/api/fleet").json()["devices"][0]["device_id"]
    body = {"calendar_events": [{"title": "Flight", "start": "2026-09-27T07:00:00", "kind": "flight"}], "soc_now": 0.3}
    p = client.post(f"/api/devices/{did}/charge-plan", json=body).json()
    assert p["context"]["target_soc"] == 1.0
    assert p["milp"]["met_target"] and p["milp"]["damage_pct_capacity"] < p["legacy"]["damage_pct_capacity"]
    assert client.post(f"/api/devices/{did}/charge-plan", json={"soc_now": 3}).status_code == 422


def test_passport_ota_second_life(client):
    did = client.get("/api/fleet").json()["devices"][0]["device_id"]
    pub = client.get(f"/api/devices/{did}/passport", params={"role": "public"}).json()
    auth = client.get(f"/api/devices/{did}/passport", params={"role": "authority"}).json()
    assert "test_reports" not in pub["compliance"] and "test_reports" in auth["compliance"]
    assert pub["_version"]["chain_valid"]
    assert client.get(f"/api/devices/{did}/passport", params={"role": "hacker"}).status_code == 422
    ota = client.post(f"/api/devices/{did}/ota").json()
    assert ota["device_verification"]["installed"]
    assert not any(v["accepted"] for v in ota["security_checks"].values())
    assert client.get(f"/api/devices/{did}/second-life").json()["grade"] in "ABCF"


def test_workforce_levels_and_approval(client):
    client.post("/api/workforce/reset")
    client.put("/api/workforce/level", json={"level": 3})
    run = client.post("/api/workforce/run").json()
    assert run["level"] == 3
    wf = client.get("/api/workforce").json()
    assert wf["total"] == len(wf["actions"])
    assert all(a["status"] != "executed" for a in wf["actions"] if a["required_level"] > 3)
    pending = [a for a in wf["actions"] if a["status"] == "pending_approval"]
    if pending:
        a = client.post(f"/api/workforce/actions/{pending[0]['id']}/decision", json={"approve": True}).json()
        assert a["status"] == "executed"
    assert client.put("/api/workforce/level", json={"level": 9}).status_code == 422


def test_copilot_offline(client):
    r = client.post("/api/copilot", json={"message": "give me a fleet overview"}).json()
    assert r["mode"].startswith("offline") and r["answer"].startswith("Fleet of")
    assert r["trace"][0]["tool"] == "get_fleet_overview"
