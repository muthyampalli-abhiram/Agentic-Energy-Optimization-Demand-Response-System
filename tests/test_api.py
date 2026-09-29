import pytest
from fastapi.testclient import TestClient
from api.main import app, in_flight_runs


@pytest.fixture
def client():
    return TestClient(app)


def test_get_building_state(client):
    response = client.get("/building/state")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "occupancy" in data
    assert "outdoor_temperature" in data
    assert "electricity_price" in data
    assert "indoor_temperatures" in data


def test_get_forecast(client):
    response = client.get("/forecast?hours=3")
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_horizon_hours"] == 3
    assert len(data["forecast_items"]) == 12
    assert "peak_predicted_kwh" in data


def test_agent_run_and_approve_happy_path(client):
    # 1. Trigger agent run (which pauses for approval when peak risk detected)
    run_resp = client.post("/agent/run", json={"initial_time": "2026-06-01 13:00:00"})
    assert run_resp.status_code == 200
    run_data = run_resp.json()

    assert "run_id" in run_data
    run_id = run_data["run_id"]
    assert run_data["status"] == "PENDING_APPROVAL"
    assert "approval_proposal" in run_data
    assert "top_strategy" in run_data["approval_proposal"]

    # 2. Approve run with top strategy
    approve_resp = client.post("/agent/approve", json={"run_id": run_id, "approved": True})
    assert approve_resp.status_code == 200
    approve_data = approve_resp.json()

    assert approve_data["status"] == "COMPLETED"
    assert len(approve_data["executed_actions"]) > 0
    assert approve_data["cost_savings_estimate"]["projected_cost_savings_usd"] > 0.0


def test_agent_approve_alternative_strategy(client):
    # 1. Trigger agent run
    run_resp = client.post("/agent/run", json={"initial_time": "2026-06-01 13:00:00"})
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    run_id = run_data["run_id"]

    # 2. Approve with alternative strategy (e.g. STRAT_02)
    approve_resp = client.post("/agent/approve", json={
        "run_id": run_id,
        "approved": True,
        "chosen_strategy_id": "STRAT_02",
        "notes": "Facility manager chose aggressive battery shaving"
    })
    assert approve_resp.status_code == 200
    approve_data = approve_resp.json()

    assert approve_data["status"] == "COMPLETED"
    assert approve_data["selected_strategy"]["strategy_id"] == "STRAT_02"


def test_agent_rejection_path(client):
    # 1. Trigger agent run
    run_resp = client.post("/agent/run", json={"initial_time": "2026-06-01 13:00:00"})
    run_data = run_resp.json()
    run_id = run_data["run_id"]

    # 2. Reject 1 -> triggers replan retry with conservative batch proposal
    reject_resp1 = client.post("/agent/approve", json={"run_id": run_id, "approved": False})
    assert reject_resp1.status_code == 200
    reject_data1 = reject_resp1.json()
    assert reject_data1["status"] == "PENDING_APPROVAL_RETRY"

    # 3. Reject 2 -> final safe cancellation with 0 actions executed
    reject_resp2 = client.post("/agent/approve", json={"run_id": run_id, "approved": False})
    assert reject_resp2.status_code == 200
    reject_data2 = reject_resp2.json()
    assert reject_data2["status"] == "REJECTED_CANCELLED"
    assert len(reject_data2["executed_actions"]) == 0


def test_get_actions_log(client):
    response = client.get("/actions/log?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total_logs" in data
    assert "logs" in data


def test_get_savings_summary(client):
    response = client.get("/savings/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_actions_executed" in data
    assert "total_cost_savings_usd" in data


def test_agent_auto_approve_below_threshold(client):
    # 1. Trigger agent run during off-peak night time when peak risk is not detected
    run_resp = client.post("/agent/run", json={"initial_time": "2026-06-01 02:00:00"})
    assert run_resp.status_code == 200
    run_data = run_resp.json()

    assert "run_id" in run_data
    assert run_data["status"] == "COMPLETED"
    assert "selected_strategy" in run_data


def test_get_dashboard(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Commercial Building Energy" in response.text


