import sys
import json
from pathlib import Path

# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from agent.graph import create_energy_agent_graph
from agent.approval_cli import prompt_facility_manager_decision, format_approval_proposal


def run_trace_1_approve_top_strategy():
    print("\n" + "=" * 100)
    print("      TRACE 1: END-TO-END EXECUTION WITH HUMAN APPROVAL -> APPROVE TOP-RANKED STRATEGY     ")
    print("=" * 100)

    app = create_energy_agent_graph(checkpointer=True)
    config = {"configurable": {"thread_id": "trace_1"}}

    initial_state = {
        "current_building_state": None,
        "forecast": None,
        "occupancy_trend": None,
        "risk_detected": False,
        "risk_details": None,
        "candidate_strategies": [],
        "ranked_strategies": [],
        "selected_strategy": None,
        "simulation_results": {},
        "requires_human_approval": False,
        "approval_status": "PENDING",
        "approval_proposal": None,
        "executed_actions": [],
        "cost_savings_estimate": {},
        "replan_count": 0,
        "should_continue": False,
        "messages": [],
    }

    print("\n--- PHASE 1: Monitoring -> Forecasting -> Diagnosis -> Strategy Generation -> Simulation -> Scoring ---")
    state = app.invoke(initial_state, config=config)

    proposal = state.get("approval_proposal")
    print("\n--- PHASE 2: Human Approval Gate Prompt ---")
    
    # Operator approves Option 1 (Top Strategy)
    status, approved_strat = prompt_facility_manager_decision(proposal, choice_override="1")

    print("\n--- PHASE 3: Dispatching Approved Control Actions & Finalizing ---")
    resume_state = {
        "approval_status": status,
        "selected_strategy": approved_strat,
    }

    final_state = app.invoke(resume_state, config=config)

    print("\n--- TRACE 1 SUMMARY RESULT ---")
    selected = final_state.get("selected_strategy", {})
    sim = selected.get("simulation_result", {})
    print(f"Executed Strategy : {selected.get('strategy_id')} ('{selected.get('strategy_name')}')")
    print(f"Projected Savings : ${sim.get('projected_cost_savings_usd', 0.0):.2f} ({sim.get('projected_kwh_reduction', 0.0)} kWh saved, {sim.get('window_energy_reduction_pct', 0.0)}% energy red.)")
    print(f"Actions Executed  : {len(final_state.get('executed_actions', []))} tool calls completed successfully.")
    print("=" * 100)


def run_trace_2_pick_alternative_strategy():
    print("\n" + "=" * 100)
    print("  TRACE 2: END-TO-END EXECUTION WITH HUMAN APPROVAL -> PICK RANKED #2 ALTERNATIVE STRATEGY ")
    print("=" * 100)

    app = create_energy_agent_graph(checkpointer=True)
    config = {"configurable": {"thread_id": "trace_2"}}

    initial_state = {
        "current_building_state": None,
        "forecast": None,
        "occupancy_trend": None,
        "risk_detected": False,
        "risk_details": None,
        "candidate_strategies": [],
        "ranked_strategies": [],
        "selected_strategy": None,
        "simulation_results": {},
        "requires_human_approval": False,
        "approval_status": "PENDING",
        "approval_proposal": None,
        "executed_actions": [],
        "cost_savings_estimate": {},
        "replan_count": 0,
        "should_continue": False,
        "messages": [],
    }

    print("\n--- PHASE 1: Monitoring -> Forecasting -> Diagnosis -> Strategy Generation -> Simulation -> Scoring ---")
    state = app.invoke(initial_state, config=config)

    proposal = state.get("approval_proposal")
    print("\n--- PHASE 2: Human Approval Gate Prompt ---")
    
    # Operator selects Option 2 (Alternative Strategy #2)
    status, approved_strat = prompt_facility_manager_decision(proposal, choice_override="2")

    print("\n--- PHASE 3: Dispatching Alternative Control Actions & Finalizing ---")
    resume_state = {
        "approval_status": status,
        "selected_strategy": approved_strat,
    }

    final_state = app.invoke(resume_state, config=config)

    print("\n--- TRACE 2 SUMMARY RESULT ---")
    selected = final_state.get("selected_strategy", {})
    sim = selected.get("simulation_result", {})
    print(f"Executed Strategy : {selected.get('strategy_id')} ('{selected.get('strategy_name')}')")
    print(f"Projected Savings : ${sim.get('projected_cost_savings_usd', 0.0):.2f} ({sim.get('projected_kwh_reduction', 0.0)} kWh saved, {sim.get('window_energy_reduction_pct', 0.0)}% energy red.)")
    print(f"Actions Executed  : {len(final_state.get('executed_actions', []))} tool calls completed successfully.")
    print("=" * 100)


def run_trace_3_reject_all_safe_cancellation():
    print("\n" + "=" * 100)
    print("   TRACE 3: REJECT FIRST BATCH -> REGENERATE NEW BATCH -> REJECT AGAIN -> SAFE TERMINATION   ")
    print("=" * 100)

    app = create_energy_agent_graph(checkpointer=True)
    config = {"configurable": {"thread_id": "trace_3"}}

    initial_state = {
        "current_building_state": None,
        "forecast": None,
        "occupancy_trend": None,
        "risk_detected": False,
        "risk_details": None,
        "candidate_strategies": [],
        "ranked_strategies": [],
        "selected_strategy": None,
        "simulation_results": {},
        "requires_human_approval": False,
        "approval_status": "PENDING",
        "approval_proposal": None,
        "executed_actions": [],
        "cost_savings_estimate": {},
        "replan_count": 0,
        "should_continue": False,
        "messages": [],
    }

    # --- ATTEMPT 1: First Batch Proposal ---
    print("\n--- PHASE 1: Generating Initial Batch 1 of Strategies ---")
    state_1 = app.invoke(initial_state, config=config)

    proposal_1 = state_1.get("approval_proposal")
    print("\n--- PHASE 2: Human Approval Gate Prompt (Batch 1) ---")
    status_1, _ = prompt_facility_manager_decision(proposal_1, choice_override="3")  # Reject All

    # --- RETRY 1: Regenerating New Batch ---
    print("\n--- PHASE 3: Re-generating Strategy Batch 2 (Retry 1) ---")
    resume_state_1 = {
        "approval_status": status_1,  # "REJECTED"
        "replan_count": 1,
    }

    state_2 = app.invoke(resume_state_1, config=config)

    proposal_2 = state_2.get("approval_proposal")
    print("\n--- PHASE 4: Human Approval Gate Prompt (Batch 2 - Regenerated Candidates) ---")
    status_2, _ = prompt_facility_manager_decision(proposal_2, choice_override="3")  # Reject All Again

    # --- SAFE TERMINATION ---
    print("\n--- PHASE 5: Second Rejection Handled - Routing to Safe Termination ---")
    resume_state_2 = {
        "approval_status": "REJECTED_FINAL",
    }

    final_state = app.invoke(resume_state_2, config=config)

    print("\n--- TRACE 3 SUMMARY RESULT ---")
    print(f"Workflow Status   : Completed safely (should_continue={final_state.get('should_continue')})")
    print(f"Executed Actions  : {len(final_state.get('executed_actions', []))} actions taken (SAFE CANCELLED).")
    print("Audit Log Entry   : 'REJECTED: Operator refused all proposed demand response plans. Run terminated safely with 0 actions taken.'")
    print("=" * 100)


if __name__ == "__main__":
    run_trace_1_approve_top_strategy()
    run_trace_2_pick_alternative_strategy()
    run_trace_3_reject_all_safe_cancellation()
