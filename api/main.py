import os
import uuid
import json
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tools.building_tools import (
    get_building_sensor,
    reset_building_sensor,
    get_current_building_state,
    BuildingState,
)
from simulator.live_feed import LiveBuildingSensor
from forecasting.model import forecast_next_hours
from agent.graph import create_energy_agent_graph

# Initialize FastAPI App
app = FastAPI(
    title="Energy Agent System API",
    description="REST API backend for real-time commercial building telemetry, demand forecasting, agentic peak shaving, and HITL approval workflow.",
    version="1.0.0",
)

# Persistent In-Memory Checkpointer & Run Store
_memory_checkpointer = MemorySaver()
_agent_graph_app = create_energy_agent_graph(checkpointer=_memory_checkpointer)
in_flight_runs: Dict[str, Dict[str, Any]] = {}


# Pydantic Schemas for Requests & Responses

class RunAgentRequest(BaseModel):
    initial_time: Optional[str] = Field(None, description="Optional ISO timestamp to position sensor simulator (default: current sensor time)")
    auto_approve_below_threshold: bool = Field(True, description="Whether to auto-approve strategies below approval thresholds")


class ApproveRunRequest(BaseModel):
    run_id: str = Field(..., description="Active in-flight run ID returned by /agent/run")
    approved: bool = Field(..., description="True to approve, False to reject")
    chosen_strategy_id: Optional[str] = Field(None, description="Optional ID of chosen candidate (e.g., 'STRAT_02' to pick alternative)")
    notes: Optional[str] = Field(None, description="Optional operator notes")


# API & Dashboard Endpoints

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard_page():
    """
    Serve the Commercial Building Energy Management & Demand Response Live Monitoring Dashboard UI.
    """
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "dashboard.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard UI file not found</h1>"


@app.get("/building/state", response_model=BuildingState)
def get_building_state_endpoint():
    """
    Retrieve real-time IoT building sensor telemetry and thermal state.
    """
    return get_current_building_state()


@app.get("/forecast")
def get_forecast_endpoint(hours: int = Query(3, ge=1, le=12, description="Hours ahead to forecast (1 to 12 hours)")):
    """
    Retrieve short-term time-series energy demand forecast for the next N hours.
    """
    b_state = get_current_building_state()
    ts_str = b_state.timestamp
    sensor = get_building_sensor(initial_time=ts_str)

    history_ts = pd.date_range(pd.Timestamp(ts_str) - pd.Timedelta(hours=24), pd.Timestamp(ts_str), freq="15min")
    from simulator.generators import generate_occupancy, generate_outdoor_temperature, generate_electricity_price, generate_baseline_consumption
    occ = generate_occupancy(history_ts)
    temp = generate_outdoor_temperature(history_ts)
    price = generate_electricity_price(history_ts)
    cons = generate_baseline_consumption(history_ts, occ, temp)

    history_df = pd.DataFrame({
        "occupancy": occ,
        "outdoor_temperature": temp,
        "electricity_price": price,
        "total_consumption_kwh": cons,
    }, index=history_ts)
    history_df.index.name = "timestamp"

    model_path = "forecasting/saved_model.pkl"
    if os.path.exists(model_path):
        forecast_df = forecast_next_hours(history_df, hours_ahead=hours, model_path=model_path)
    else:
        future_ts = pd.date_range(pd.Timestamp(ts_str) + pd.Timedelta(minutes=15), periods=hours * 4, freq="15min")
        preds = [125.0 if 14 <= t.hour < 17 else 85.0 for t in future_ts]
        forecast_df = pd.DataFrame({"predicted_consumption_kwh": preds}, index=future_ts)

    records = []
    for ts, row in forecast_df.iterrows():
        records.append({
            "timestamp": ts.isoformat(),
            "predicted_consumption_kwh": round(float(row["predicted_consumption_kwh"]), 2),
            "occupancy": round(float(row.get("occupancy", 85.0)), 1),
            "outdoor_temperature": round(float(row.get("outdoor_temperature", 29.5)), 1),
            "electricity_price": round(float(row.get("electricity_price", 0.45 if 14 <= ts.hour < 17 else 0.22)), 2),
        })

    return {
        "forecast_horizon_hours": hours,
        "intervals": len(records),
        "peak_predicted_kwh": round(float(forecast_df["predicted_consumption_kwh"].max()), 2),
        "forecast_items": records,
    }


@app.post("/agent/run")
def run_agent_workflow(req: RunAgentRequest = RunAgentRequest()):
    """
    Trigger one full agent graph cycle (monitoring -> demand forecast -> risk diagnosis -> strategy generation -> simulation -> scoring -> approval gate).
    Returns either a completed execution summary or a PENDING_APPROVAL proposal payload.
    """
    run_id = uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": run_id}}

    init_time = req.initial_time or "2026-06-01 13:00:00"
    reset_building_sensor(LiveBuildingSensor(initial_time=init_time))

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

    # Execute graph up to approval gate or completion
    state = _agent_graph_app.invoke(initial_state, config=config)

    # Store run state
    in_flight_runs[run_id] = {
        "run_id": run_id,
        "state": state,
        "config": config,
    }

    requires_approval = state.get("requires_human_approval", False)
    approval_status = state.get("approval_status", "PENDING")

    if requires_approval and approval_status == "PENDING":
        in_flight_runs[run_id] = {
            "run_id": run_id,
            "state": state,
            "config": config,
        }
        return {
            "run_id": run_id,
            "status": "PENDING_APPROVAL",
            "message": "Peak demand risk detected. High-impact strategy proposal requires facility manager sign-off.",
            "approval_proposal": state.get("approval_proposal"),
        }
    else:
        # Auto-approve path: update state and resume execution so execute_actions node runs
        _agent_graph_app.update_state(config, {"approval_status": "APPROVED"})
        state = _agent_graph_app.invoke(None, config=config)
        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "message": "Agent cycle completed successfully.",
            "selected_strategy": state.get("selected_strategy"),
            "executed_actions": state.get("executed_actions", []),
            "cost_savings_estimate": state.get("cost_savings_estimate", {}),
        }


@app.post("/agent/approve")
def approve_agent_run(req: ApproveRunRequest):
    """
    Process facility manager approval decision for a paused in-flight agent run.
    Accepts approval, rejection, or alternative strategy selection.
    """
    if req.run_id not in in_flight_runs:
        raise HTTPException(status_code=404, detail=f"Run ID '{req.run_id}' not found or already completed.")

    run_info = in_flight_runs[req.run_id]
    config = run_info["config"]
    curr_state = run_info["state"]

    proposal = curr_state.get("approval_proposal", {})
    ranked = curr_state.get("ranked_strategies", [])

    if req.approved:
        approval_status = "APPROVED"
        # Determine strategy to execute
        selected_strat = proposal.get("top_strategy")
        if req.chosen_strategy_id:
            matching = next((c for c in ranked if c["strategy_id"] == req.chosen_strategy_id), None)
            if matching:
                selected_strat = matching

        resume_state = {
            "approval_status": "APPROVED",
            "selected_strategy": selected_strat,
            "cost_savings_estimate": selected_strat.get("simulation_result", {}),
        }

        _agent_graph_app.update_state(config, resume_state)
        final_state = _agent_graph_app.invoke(None, config=config)
        del in_flight_runs[req.run_id]

        return {
            "run_id": req.run_id,
            "status": "COMPLETED",
            "message": "Strategy approved and control actions executed successfully.",
            "selected_strategy": final_state.get("selected_strategy"),
            "executed_actions": final_state.get("executed_actions", []),
            "cost_savings_estimate": final_state.get("cost_savings_estimate", {}),
        }
    else:
        replan_cnt = curr_state.get("replan_count", 0)
        resume_state = {
            "approval_status": "REJECTED",
            "replan_count": replan_cnt,
        }

        _agent_graph_app.update_state(config, resume_state)
        final_state = _agent_graph_app.invoke(None, config=config)
        
        # If interrupted again for second batch, keep in flight; else remove
        if final_state.get("approval_status") == "PENDING" and final_state.get("requires_human_approval"):
            in_flight_runs[req.run_id]["state"] = final_state
            return {
                "run_id": req.run_id,
                "status": "PENDING_APPROVAL_RETRY",
                "message": "[RETRY 1/1] Previous proposal rejected. Regenerated new batch of conservative candidate strategies.",
                "approval_proposal": final_state.get("approval_proposal"),
            }

        if req.run_id in in_flight_runs:
            del in_flight_runs[req.run_id]

        return {
            "run_id": req.run_id,
            "status": "REJECTED_CANCELLED",
            "message": "Demand response proposal rejected by facility manager. Safe cancellation executed with 0 actions taken.",
            "executed_actions": [],
            "cost_savings_estimate": {},
        }


@app.get("/actions/log")
def get_action_log_endpoint(limit: int = Query(50, ge=1, le=500)):
    """
    Retrieve recent control action execution log entries from data/action_log.csv.
    """
    log_path = "data/action_log.csv"
    if not os.path.exists(log_path):
        return {"total_logs": 0, "logs": []}

    df_log = pd.read_csv(log_path)
    df_recent = df_log.tail(limit).iloc[::-1]  # Reverse chronological

    records = df_recent.to_dict(orient="records")
    return {
        "total_logs": len(df_log),
        "returned_logs": len(records),
        "logs": records,
    }


@app.get("/savings/summary")
def get_savings_summary_endpoint():
    """
    Retrieve aggregated cost savings and demand reduction summary vs unmitigated baseline.
    """
    log_path = "data/action_log.csv"
    total_actions = 0
    total_savings_usd = 0.0

    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        successful_actions = df_log[df_log["success"] == True]
        total_actions = len(successful_actions)
        total_savings_usd = float(successful_actions["cost_impact_usd"].abs().sum())

    return {
        "total_actions_executed": total_actions,
        "total_cost_savings_usd": round(total_savings_usd, 2),
        "estimated_kwh_reduced": round(total_savings_usd / 0.45, 1) if total_savings_usd > 0 else 0.0,
        "average_savings_per_action_usd": round(total_savings_usd / max(1, total_actions), 2),
        "baseline_comparison_period": "2:00 PM - 5:00 PM Peak Window",
    }
