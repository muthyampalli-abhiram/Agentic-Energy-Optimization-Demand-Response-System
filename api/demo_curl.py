import json
import httpx
import time
from fastapi.testclient import TestClient
from api.main import app

def run_curl_demos():
    client = TestClient(app)
    
    print("================================================================================")
    print("                 FASTAPI BACKEND HTTP ENDPOINTS DEMONSTRATION                   ")
    print("================================================================================\n")

    # 1. GET /building/state
    print("--------------------------------------------------------------------------------")
    print("1. GET /building/state — Current Building Sensor Telemetry & Thermal State")
    print("--------------------------------------------------------------------------------")
    print("cURL Command:")
    print("  curl -X GET 'http://localhost:8000/building/state' -H 'accept: application/json'\n")
    resp = client.get("/building/state")
    print(f"HTTP Status: {resp.status_code}")
    print("Response Body:")
    print(json.dumps(resp.json(), indent=2))
    print("\n")

    # 2. GET /forecast
    print("--------------------------------------------------------------------------------")
    print("2. GET /forecast — Demand Forecast for Next N Hours")
    print("--------------------------------------------------------------------------------")
    print("cURL Command:")
    print("  curl -X GET 'http://localhost:8000/forecast?hours=3' -H 'accept: application/json'\n")
    resp = client.get("/forecast?hours=3")
    print(f"HTTP Status: {resp.status_code}")
    print("Response Body:")
    body = resp.json()
    # Truncate intervals for display brevity
    if "forecast_items" in body and len(body["forecast_items"]) > 4:
        body_display = dict(body)
        body_display["forecast_items"] = body["forecast_items"][:3] + ["... (9 remaining 15-min intervals) ..."]
        print(json.dumps(body_display, indent=2))
    else:
        print(json.dumps(body, indent=2))
    print("\n")

    # 3. POST /agent/run
    print("--------------------------------------------------------------------------------")
    print("3. POST /agent/run — Trigger One Full Agent Graph Cycle")
    print("--------------------------------------------------------------------------------")
    print("cURL Command:")
    print("  curl -X POST 'http://localhost:8000/agent/run' \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"initial_time\": \"2026-06-01 13:00:00\"}'\n")
    resp = client.post("/agent/run", json={"initial_time": "2026-06-01 13:00:00"})
    print(f"HTTP Status: {resp.status_code}")
    run_body = resp.json()
    print("Response Body:")
    print(json.dumps(run_body, indent=2))
    print("\n")

    run_id = run_body.get("run_id")

    # 4. POST /agent/approve
    print("--------------------------------------------------------------------------------")
    print("4. POST /agent/approve — Facility Manager Sign-off & Control Execution")
    print("--------------------------------------------------------------------------------")
    print("cURL Command:")
    print("  curl -X POST 'http://localhost:8000/agent/approve' \\")
    print("       -H 'Content-Type: application/json' \\")
    print(f"       -d '{{\"run_id\": \"{run_id}\", \"approved\": true, \"notes\": \"Approved by facility manager\"}}'\n")
    resp = client.post("/agent/approve", json={"run_id": run_id, "approved": True, "notes": "Approved by facility manager"})
    print(f"HTTP Status: {resp.status_code}")
    print("Response Body:")
    print(json.dumps(resp.json(), indent=2))
    print("\n")

    # 5. GET /actions/log
    print("--------------------------------------------------------------------------------")
    print("5. GET /actions/log — Recent Execution Audit Logs")
    print("--------------------------------------------------------------------------------")
    print("cURL Command:")
    print("  curl -X GET 'http://localhost:8000/actions/log?limit=5' -H 'accept: application/json'\n")
    resp = client.get("/actions/log?limit=5")
    print(f"HTTP Status: {resp.status_code}")
    print("Response Body:")
    print(json.dumps(resp.json(), indent=2))
    print("\n")

    # 6. GET /savings/summary
    print("--------------------------------------------------------------------------------")
    print("6. GET /savings/summary — Aggregated Cost Savings Summary")
    print("--------------------------------------------------------------------------------")
    print("cURL Command:")
    print("  curl -X GET 'http://localhost:8000/savings/summary' -H 'accept: application/json'\n")
    resp = client.get("/savings/summary")
    print(f"HTTP Status: {resp.status_code}")
    print("Response Body:")
    print(json.dumps(resp.json(), indent=2))
    print("================================================================================\n")

if __name__ == "__main__":
    run_curl_demos()
