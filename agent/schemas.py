from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ProposedAction(BaseModel):
    """Represents a single control action mapping to tools/building_tools.py."""
    action_type: str = Field(
        ...,
        description="Name of building tool action: adjust_thermostat, dim_lighting, dispatch_battery, or shift_equipment_load"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Key-value arguments required by the target building tool"
    )


class CandidateStrategy(BaseModel):
    """Candidate demand-response strategy proposed by the LLM."""
    strategy_id: str = Field(..., description="Unique strategy identifier (e.g. STRAT_01)")
    strategy_name: str = Field(..., description="Descriptive strategy name (e.g., Pre-Cooling & Setpoint Drift)")
    rationale: str = Field(..., description="Engineering rationale explaining peak shaving approach and comfort protection")
    actions: List[ProposedAction] = Field(..., description="Sequence of discrete building control actions")
    expected_kw_reduction: float = Field(..., description="Estimated peak demand reduction in kW or kWh")
    simulation_result: Optional[Dict[str, Any]] = Field(None, description="What-if simulation outcome attached after simulation")


class StrategyGenerationOutput(BaseModel):
    """Structured Pydantic container for LLM strategy generation response."""
    candidates: List[CandidateStrategy] = Field(..., description="List of 2 to 4 candidate demand-response strategies")
