from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class EnergyAgentState(TypedDict):
    """
    Shared graph state schema for the Energy Agent optimization system.
    Tracks live telemetry, time-series forecasting, risk diagnostics, candidate strategies,
    multi-objective strategy rankings, HITL approval gates, action execution logs, and conversational history.
    """
    current_building_state: Optional[Dict[str, Any]]
    forecast: Optional[Dict[str, Any]]
    occupancy_trend: Optional[str]
    risk_detected: bool
    risk_details: Optional[Dict[str, Any]]
    candidate_strategies: List[Dict[str, Any]]
    ranked_strategies: List[Dict[str, Any]]
    selected_strategy: Optional[Dict[str, Any]]
    simulation_results: Dict[str, Any]
    requires_human_approval: bool
    approval_status: Optional[str]  # "PENDING", "APPROVED", "REJECTED"
    approval_proposal: Optional[Dict[str, Any]]
    executed_actions: List[Dict[str, Any]]
    cost_savings_estimate: Dict[str, Any]
    replan_count: int
    should_continue: bool
    messages: Annotated[List[Dict[str, Any]], add_messages]
