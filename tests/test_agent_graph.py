import pytest
from langgraph.graph import END, START
from agent.state import EnergyAgentState
from agent.graph import (
    create_energy_agent_graph,
    route_after_diagnose,
    route_after_approval,
    route_after_replan,
)


def test_graph_compilation():
    app = create_energy_agent_graph()
    assert app is not None


def test_conditional_routing():
    assert route_after_diagnose({"risk_detected": True}) == "generate_strategies"
    assert route_after_diagnose({"risk_detected": False}) == "replan_check"

    assert route_after_approval({"approval_status": "APPROVED"}) == "execute_actions"
    assert route_after_approval({"approval_status": "PENDING"}) == "execute_actions"
    assert route_after_approval({"approval_status": "REJECTED"}) == "execute_actions"

    assert route_after_replan({"should_continue": True}) == "generate_strategies"
    assert route_after_replan({"should_continue": False}) == END


def test_full_graph_strategy_scoring_and_execution():
    app = create_energy_agent_graph()

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
        "approval_status": "APPROVED",
        "approval_proposal": None,
        "executed_actions": [],
        "cost_savings_estimate": {},
        "replan_count": 0,
        "should_continue": False,
        "messages": [],
    }

    final_state = app.invoke(initial_state)

    assert final_state["current_building_state"] is not None
    assert final_state["risk_detected"] is True
    assert len(final_state["candidate_strategies"]) >= 2
    assert len(final_state["ranked_strategies"]) >= 2

    # Verify ranking scores present
    top_strat = final_state["selected_strategy"]
    assert top_strat is not None
    assert "score" in top_strat
    assert top_strat["score"] >= final_state["ranked_strategies"][1]["score"]
    assert len(final_state["executed_actions"]) > 0


def test_safe_rejection_cancellation():
    app = create_energy_agent_graph()

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
        "approval_status": "REJECTED_ALL",
        "approval_proposal": None,
        "executed_actions": [],
        "cost_savings_estimate": {},
        "replan_count": 1,
        "should_continue": False,
        "messages": [],
    }

    final_state = app.invoke(initial_state)

    # Rejection on retry must terminate safely with 0 actions executed
    assert final_state["should_continue"] is False
    assert len(final_state["executed_actions"]) == 0
