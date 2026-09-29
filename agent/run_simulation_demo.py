import sys
import json
from pathlib import Path

# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from agent.graph import create_energy_agent_graph


def run_simulation_demo():
    app = create_energy_agent_graph()

    initial_state = {
        "current_building_state": None,
        "forecast": None,
        "occupancy_trend": None,
        "risk_detected": False,
        "risk_details": None,
        "candidate_strategies": [],
        "selected_strategy": None,
        "simulation_results": {},
        "requires_human_approval": False,
        "approval_status": "APPROVED",
        "executed_actions": [],
        "cost_savings_estimate": {},
        "replan_count": 0,
        "should_continue": False,
        "messages": [],
    }

    print("=== EXECUTING ENERGY AGENT GRAPH (MONITOR -> FORECAST -> DIAGNOSE -> GENERATE -> SIMULATE) ===\n")
    final_state = app.invoke(initial_state)

    print("\n==========================================================================================")
    print("                 PROPOSED CANDIDATE STRATEGIES & WHAT-IF SIMULATION RESULTS              ")
    print("==========================================================================================")

    for i, cand in enumerate(final_state["candidate_strategies"], 1):
        sim = cand.get("simulation_result", {})
        print(f"\n[STRATEGY {i}: {cand['strategy_id']} - {cand['strategy_name']}]")
        print(f"  Rationale            : {cand['rationale']}")
        print(f"  Actions Proposed     : {json.dumps(cand['actions'])}")
        print(f"  Expected Reduction   : {cand['expected_kw_reduction']} kW/kWh")
        print("  --- What-If Simulation Outcome (2:00 PM - 5:00 PM Window) ---")
        print(f"    Baseline 2-5 PM kWh   : {sim.get('baseline_2to5pm_kwh')} kWh  (Cost: ${sim.get('baseline_2to5pm_cost_usd'):.2f})")
        print(f"    Strategy 2-5 PM kWh   : {sim.get('strategy_2to5pm_kwh')} kWh  (Cost: ${sim.get('strategy_2to5pm_cost_usd'):.2f})")
        print(f"    PROJECTED SAVINGS     : ${sim.get('projected_cost_savings_usd'):.2f} ({sim.get('projected_kwh_reduction')} kWh saved, {sim.get('window_energy_reduction_pct')}% energy reduction)")
        print(f"    Instantaneous Peak Red: {sim.get('instantaneous_peak_kw_reduction_pct')}% peak interval load reduction")
        print(f"    Max Indoor Temp       : {sim.get('max_indoor_temp_c')}C")
        print(f"    Comfort Assessment    : {sim.get('comfort_impact')}")
        print("-" * 90)

    selected = final_state.get("selected_strategy", {})
    print(f"\nTop Selected Strategy : {selected.get('strategy_id')} - {selected.get('strategy_name')}")
    print("==========================================================================================")


if __name__ == "__main__":
    run_simulation_demo()
