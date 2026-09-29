import json
from fastapi.testclient import TestClient
from api.main import app

def demo_alternative_strategy_selection():
    client = TestClient(app)
    
    print("================================================================================")
    print("      END-TO-END DEMO: EXPLICIT ALTERNATIVE STRATEGY APPROVAL (STRAT_02)       ")
    print("================================================================================\n")

    # Step 1: POST /agent/run
    print("1. Triggering Agent Run at Peak Risk Window (13:00)...")
    print("POST /agent/run Payload: {\"initial_time\": \"2026-06-01 13:00:00\"}\n")
    run_resp = client.post("/agent/run", json={"initial_time": "2026-06-01 13:00:00"})
    assert run_resp.status_code == 200, f"Expected 200, got {run_resp.status_code}"
    
    run_data = run_resp.json()
    run_id = run_data["run_id"]
    status = run_data["status"]
    proposal = run_data["approval_proposal"]
    
    top_strat = proposal["top_strategy"]
    alt_strat = proposal["alternative_strategy"]
    
    print(f"-> Received Status: {status}")
    print(f"-> Generated Run ID: {run_id}")
    print(f"-> Top Ranked Strategy (#1): '{top_strat['strategy_id']}' - {top_strat['strategy_name']} (Score: {top_strat['score']})")
    print(f"-> Alternative Strategy (#2): '{alt_strat['strategy_id']}' - {alt_strat['strategy_name']} (Score: {alt_strat['score']})")
    print("\nFull Proposal Payload:")
    print(json.dumps(proposal, indent=2))
    print("\n" + "-"*80 + "\n")

    # Step 2: POST /agent/approve with chosen_strategy_id = "STRAT_02"
    approve_payload = {
        "run_id": run_id,
        "approved": True,
        "chosen_strategy_id": "STRAT_02",
        "notes": "Facility manager overriding top choice to select Aggressive Battery Shaving (STRAT_02)."
    }
    
    print("2. Facility Manager Overriding & Approving ALTERNATIVE Strategy ('STRAT_02')...")
    print(f"POST /agent/approve Payload:\n{json.dumps(approve_payload, indent=2)}\n")
    
    approve_resp = client.post("/agent/approve", json=approve_payload)
    assert approve_resp.status_code == 200, f"Expected 200, got {approve_resp.status_code}"
    
    approve_data = approve_resp.json()
    print(f"-> Execution Status: {approve_data['status']}")
    print(f"-> Message: {approve_data['message']}\n")
    print("Response Body:")
    print(json.dumps(approve_data, indent=2))
    print("\n" + "="*80)

    # Verification Checks
    executed_actions = approve_data.get("executed_actions", [])
    selected_strat = approve_data.get("selected_strategy", {})

    print("\n[VERIFICATION CONFIRMATION]")
    print(f"1. Selected Strategy ID: '{selected_strat.get('strategy_id')}' (Expected: 'STRAT_02')")
    assert selected_strat.get("strategy_id") == "STRAT_02"
    
    # Extract action details
    thermostat_action = next((a for a in executed_actions if a["action_type"] == "adjust_thermostat"), None)
    lighting_action = next((a for a in executed_actions if a["action_type"] == "dim_lighting"), None)
    battery_action = next((a for a in executed_actions if a["action_type"] == "dispatch_battery"), None)
    
    print(f"2. Executed Thermostat Action: {thermostat_action['message'] if thermostat_action else 'MISSING'}")
    print(f"3. Executed Lighting Dimming Action: {lighting_action['message'] if lighting_action else 'MISSING'}")
    print(f"4. Executed Battery Dispatch Action: {battery_action['message'] if battery_action else 'MISSING'}")
    
    assert thermostat_action is not None and thermostat_action["parameters"]["new_setpoint_c"] == 23.5
    assert lighting_action is not None and lighting_action["parameters"]["percent"] == 35
    assert battery_action is not None and battery_action["parameters"]["kw"] == 40.0
    
    print("\nSUCCESS: Confirmed that executed_actions match STRAT_02's parameters (Thermostat 23.5°C, Lighting Dim 35%, Battery 40 kW) and NOT STRAT_03's!")

if __name__ == "__main__":
    demo_alternative_strategy_selection()
