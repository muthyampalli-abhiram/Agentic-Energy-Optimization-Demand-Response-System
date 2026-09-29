import sys
import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import EnergyAgentState
from agent.schemas import ProposedAction, CandidateStrategy, StrategyGenerationOutput
from agent.prompts import STRATEGY_GENERATION_SYSTEM_PROMPT, STRATEGY_GENERATION_USER_PROMPT
from agent.config import get_strategy_weights, get_approval_thresholds
from agent.approval_cli import prompt_facility_manager_decision

from tools.building_tools import (
    get_building_sensor,
    get_current_building_state,
    adjust_thermostat,
    dim_lighting,
    dispatch_battery,
    shift_equipment_load,
    _log_action,
)
from simulator.live_feed import LiveBuildingSensor
from forecasting.model import forecast_next_hours


# Node Implementation Logic

def monitor_node(state: EnergyAgentState) -> Dict[str, Any]:
    """1. Monitor Node: Pulls real-time IoT building sensor state."""
    b_state = get_current_building_state()
    print(f"  [NODE: monitor] Fetched state at {b_state.timestamp}: Occupancy={b_state.occupancy:.1f}%, Temp={b_state.outdoor_temperature:.1f}°C, Price=${b_state.electricity_price:.2f}/kWh")

    return {
        "current_building_state": b_state.model_dump(),
        "occupancy_trend": "HIGH" if b_state.occupancy > 50 else "LOW",
    }


def forecast_demand_node(state: EnergyAgentState) -> Dict[str, Any]:
    """2. Forecast Demand Node: Generates 3-hour iterative demand forecast curve."""
    b_state = state.get("current_building_state", {})
    ts_str = b_state.get("timestamp", "2026-06-01 13:00:00")
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
        forecast_df = forecast_next_hours(history_df, hours_ahead=3, model_path=model_path)
    else:
        future_ts = pd.date_range(pd.Timestamp(ts_str) + pd.Timedelta(minutes=15), periods=12, freq="15min")
        preds = [125.0 if 14 <= t.hour < 17 else 85.0 for t in future_ts]
        forecast_df = pd.DataFrame({"predicted_consumption_kwh": preds}, index=future_ts)

    print(f"  [NODE: forecast_demand] Generated forecast for next 3 hours (12 steps). Peak: {forecast_df['predicted_consumption_kwh'].max():.1f} kWh")

    forecast_data = {
        "timestamps": [t.isoformat() for t in forecast_df.index],
        "predicted_kwh": forecast_df["predicted_consumption_kwh"].values.tolist(),
        "summary": f"Peak forecasted consumption is {forecast_df['predicted_consumption_kwh'].max():.1f} kWh at {forecast_df['predicted_consumption_kwh'].idxmax().strftime('%H:%M')}",
    }

    return {"forecast": forecast_data}


def diagnose_node(state: EnergyAgentState) -> Dict[str, Any]:
    """3. Diagnose Node: Analyzes forecast and detects peak-demand risks during high-tariff windows."""
    forecast = state.get("forecast", {})
    predicted_kwh = forecast.get("predicted_kwh", [])

    max_forecast_kwh = max(predicted_kwh) if predicted_kwh else 0.0
    thresholds = get_approval_thresholds()
    threshold = thresholds["kw_reduction"]  # 30.0 / 110.0
    risk_found = max_forecast_kwh >= 110.0

    print(f"  [NODE: diagnose] Max forecasted load: {max_forecast_kwh:.1f} kWh -> Risk Detected: {risk_found}")

    return {
        "risk_detected": risk_found,
        "risk_details": {
            "max_load_kwh": max_forecast_kwh,
            "threshold_kwh": 110.0,
            "peak_window": "14:00 - 17:00",
        },
    }


def generate_strategies_node(state: EnergyAgentState) -> Dict[str, Any]:
    """
    4. Generate Strategies Node: Uses LLM reasoning engine to propose candidate strategies.
    Supports structured output Pydantic schemas with fallback generator if no API key is set.
    When replan_count > 0 (retry after operator rejection), generates a genuinely different batch of conservative strategies.
    """
    replan_cnt = state.get("replan_count", 0)
    is_retry = replan_cnt > 0 or state.get("approval_status") in ["REJECTED", "REJECTED_ALL"]

    if is_retry:
        print(f"  [RETRY 1/1] Regenerating strategies after operator rejection (Replan Count: {replan_cnt})...")
    else:
        print("  [NODE: generate_strategies] Invoking LLM reasoning engine to propose candidate strategies...")

    b_state = state.get("current_building_state", {})
    forecast = state.get("forecast", {})
    risk = state.get("risk_details", {})

    prompt_user = STRATEGY_GENERATION_USER_PROMPT.format(
        current_time=b_state.get("timestamp", "13:00"),
        occupancy=b_state.get("occupancy", 85.0),
        outdoor_temp=b_state.get("outdoor_temperature", 29.5),
        electricity_price=b_state.get("electricity_price", 0.22),
        indoor_temps=b_state.get("indoor_temperatures", {}),
        hvac_setpoints=b_state.get("hvac_setpoints", {}),
        max_forecast_kwh=risk.get("max_load_kwh", 129.5),
        threshold_kwh=risk.get("threshold_kwh", 110.0),
        forecast_summary=forecast.get("summary", "Peak demand between 14:00 and 17:00"),
    )

    if is_retry:
        prompt_user += "\n\nCRITICAL OPERATOR FEEDBACK: The facility manager REJECTED the previous strategy batch due to comfort concerns. Propose a completely DIFFERENT, softer set of conservative strategies with minimal thermostat drift (max 22.5°C), low lighting dimming (10-15%), or pure battery/load shift actions."

    strategy_output: Optional[StrategyGenerationOutput] = None
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if api_key:
        try:
            if "ANTHROPIC_API_KEY" in os.environ:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3 if is_retry else 0.2)
            else:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o", temperature=0.3 if is_retry else 0.2)

            structured_llm = llm.with_structured_output(StrategyGenerationOutput)
            messages = [
                {"role": "system", "content": STRATEGY_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ]
            strategy_output = structured_llm.invoke(messages)
            print("  [NODE: generate_strategies] Successfully generated candidate strategies via LLM.")
        except Exception as e:
            print(f"  [NODE: generate_strategies] LLM invocation warning: {e}. Switching to structured fallback generator.")

    if strategy_output is None:
        if is_retry:
            # Genuinely DIFFERENT batch of conservative strategies for retry 1
            strategy_output = StrategyGenerationOutput(
                candidates=[
                    CandidateStrategy(
                        strategy_id="STRAT_RETRY_01",
                        strategy_name="Conservative Low-Drift Thermostat (22.5°C) & Battery Support",
                        rationale="Conservative 1.5°C thermal setpoint drift to 22.5°C combined with 20 kW battery discharge to minimize occupant comfort impact.",
                        actions=[
                            ProposedAction(action_type="adjust_thermostat", parameters={"zone": "all", "new_setpoint_c": 22.5}),
                            ProposedAction(action_type="dispatch_battery", parameters={"kw": 20.0}),
                        ],
                        expected_kw_reduction=22.0,
                    ),
                    CandidateStrategy(
                        strategy_id="STRAT_RETRY_02",
                        strategy_name="Minimal Lighting Dimming (15%) & Gentle Battery Discharge",
                        rationale="Gentle 15% lighting dimming in common areas combined with 25 kW battery dispatch and 22.0°C thermostat setpoint.",
                        actions=[
                            ProposedAction(action_type="adjust_thermostat", parameters={"zone": "all", "new_setpoint_c": 22.0}),
                            ProposedAction(action_type="dim_lighting", parameters={"zone": "Zone_1", "percent": 15}),
                            ProposedAction(action_type="dispatch_battery", parameters={"kw": 25.0}),
                        ],
                        expected_kw_reduction=24.5,
                    ),
                    CandidateStrategy(
                        strategy_id="STRAT_RETRY_03",
                        strategy_name="Non-Intrusive Plug Load Deferral & Battery Dispatch",
                        rationale="Focus strictly on behind-the-scenes water heater load deferral (120 mins) and 30 kW battery discharge with zero lighting dimming.",
                        actions=[
                            ProposedAction(action_type="adjust_thermostat", parameters={"zone": "all", "new_setpoint_c": 22.0}),
                            ProposedAction(action_type="shift_equipment_load", parameters={"equipment_name": "water_heaters", "delay_minutes": 120}),
                            ProposedAction(action_type="dispatch_battery", parameters={"kw": 30.0}),
                        ],
                        expected_kw_reduction=28.0,
                    ),
                ]
            )
            print(f"  [NODE: generate_strategies] Formulated REGENERATED batch of {len(strategy_output.candidates)} conservative candidate strategies.")
        else:
            # Initial batch of candidate strategies
            strategy_output = StrategyGenerationOutput(
                candidates=[
                    CandidateStrategy(
                        strategy_id="STRAT_01",
                        strategy_name="Pre-Cooling & Setpoint Drift",
                        rationale="Pre-cool building zones to 20.0°C between 1:00-2:00 PM, then allow setpoints to drift to 24.0°C during 2-5 PM.",
                        actions=[
                            ProposedAction(action_type="adjust_thermostat", parameters={"zone": "all", "new_setpoint_c": 24.0}),
                            ProposedAction(action_type="dim_lighting", parameters={"zone": "Zone_1", "percent": 25}),
                        ],
                        expected_kw_reduction=32.0,
                    ),
                    CandidateStrategy(
                        strategy_id="STRAT_02",
                        strategy_name="Aggressive Peak Shaving (Battery + Lighting Dimming)",
                        rationale="Discharge 40 kW battery power during 2-5 PM combined with 35% lighting dimming and 23.5°C setpoint drift.",
                        actions=[
                            ProposedAction(action_type="adjust_thermostat", parameters={"zone": "all", "new_setpoint_c": 23.5}),
                            ProposedAction(action_type="dim_lighting", parameters={"zone": "Zone_1", "percent": 35}),
                            ProposedAction(action_type="dispatch_battery", parameters={"kw": 40.0}),
                        ],
                        expected_kw_reduction=48.5,
                    ),
                    CandidateStrategy(
                        strategy_id="STRAT_03",
                        strategy_name="Flexible Load Shifting & Moderate Thermal Drift",
                        rationale="Defer non-critical EV chargers for 180 minutes, dispatch 25 kW battery power, and set thermostats to 23.0°C.",
                        actions=[
                            ProposedAction(action_type="adjust_thermostat", parameters={"zone": "all", "new_setpoint_c": 23.0}),
                            ProposedAction(action_type="shift_equipment_load", parameters={"equipment_name": "ev_chargers", "delay_minutes": 180}),
                            ProposedAction(action_type="dispatch_battery", parameters={"kw": 25.0}),
                        ],
                        expected_kw_reduction=27.0,
                    ),
                ]
            )
            print(f"  [NODE: generate_strategies] Formulated INITIAL batch of {len(strategy_output.candidates)} candidate strategies.")

    candidates_dict = [c.model_dump() for c in strategy_output.candidates]
    curr_status = state.get("approval_status", "PENDING")
    new_status = curr_status if curr_status in ["REJECTED_ALL", "REJECTED_FINAL"] else "PENDING"
    return {
        "candidate_strategies": candidates_dict,
        "approval_status": new_status,
    }


def simulate_strategies_node(state: EnergyAgentState) -> Dict[str, Any]:
    """
    5. Simulate Strategies Node: Runs what-if physics simulations for candidate strategies.
    Compares 2-5 PM cost & comfort against unmitigated baseline.
    """
    candidates = state.get("candidate_strategies", [])
    b_state = state.get("current_building_state", {})
    init_time = b_state.get("timestamp", "2026-06-01 13:00:00")

    print(f"  [NODE: simulate_strategies] Running what-if physics simulations for {len(candidates)} candidates...")

    sim_timeline = pd.date_range("2026-06-01 13:00:00", "2026-06-01 17:00:00", freq="15min")

    baseline_sensor = LiveBuildingSensor(initial_time=init_time, seed=42)
    baseline_kwh_list = []
    baseline_cost_list = []

    for ts in sim_timeline:
        rd = baseline_sensor.tick(ts)
        if 14 <= ts.hour < 17:
            baseline_kwh_list.append(rd.total_consumption_kwh)
            baseline_cost_list.append(rd.total_consumption_kwh * rd.electricity_price)

    tot_baseline_kwh = sum(baseline_kwh_list)
    tot_baseline_cost = sum(baseline_cost_list)
    max_baseline_kw = max(baseline_kwh_list) if baseline_kwh_list else 130.0

    simulation_results = {}
    updated_candidates = []

    for cand in candidates:
        strat_id = cand["strategy_id"]
        strat_name = cand["strategy_name"]
        actions = cand.get("actions", [])

        sim_sensor = LiveBuildingSensor(initial_time=init_time, seed=42)
        strat_kwh_list = []
        strat_cost_list = []
        max_indoor_temp = 21.0

        for ts in sim_timeline:
            if ts == pd.Timestamp("2026-06-01 13:45:00"):
                for act in actions:
                    atype = act["action_type"]
                    params = act["parameters"]
                    if atype == "adjust_thermostat":
                        sim_sensor.set_thermostat(params.get("zone", "all"), params.get("new_setpoint_c", 24.0))
                    elif atype == "dim_lighting":
                        sim_sensor.shift_load(f"lighting_dim_{params.get('zone','all')}", 180)
                    elif atype == "dispatch_battery":
                        sim_sensor.dispatch_battery(params.get("kw", 40.0))
                    elif atype == "shift_equipment_load":
                        sim_sensor.shift_load(params.get("equipment_name", "ev_chargers"), params.get("delay_minutes", 180))

            rd = sim_sensor.tick(ts)

            if 14 <= ts.hour < 17:
                strat_kwh_list.append(rd.total_consumption_kwh)
                strat_cost_list.append(rd.total_consumption_kwh * rd.electricity_price)
                max_t = max(rd.indoor_temperatures.values())
                if max_t > max_indoor_temp:
                    max_indoor_temp = max_t

        tot_strat_kwh = sum(strat_kwh_list)
        tot_strat_cost = sum(strat_cost_list)
        max_strat_kw = max(strat_kwh_list) if strat_kwh_list else 90.0

        kwh_saved = tot_baseline_kwh - tot_strat_kwh
        cost_saved = tot_baseline_cost - tot_strat_cost
        window_energy_red_pct = (kwh_saved / tot_baseline_kwh) * 100.0 if tot_baseline_kwh > 0 else 0.0
        peak_kw_red_pct = (1.0 - max_strat_kw / max_baseline_kw) * 100.0 if max_baseline_kw > 0 else 0.0

        if max_indoor_temp <= 23.0:
            comfort_rating = "Optimal Thermal Comfort (PMV +0.1)"
        elif max_indoor_temp <= 24.2:
            comfort_rating = "Good Thermal Comfort (PMV +0.3)"
        else:
            comfort_rating = "Acceptable Warm Drift (PMV +0.5)"

        sim_outcome = {
            "strategy_id": strat_id,
            "strategy_name": strat_name,
            "baseline_2to5pm_kwh": round(tot_baseline_kwh, 2),
            "strategy_2to5pm_kwh": round(tot_strat_kwh, 2),
            "projected_kwh_reduction": round(kwh_saved, 2),
            "baseline_2to5pm_cost_usd": round(tot_baseline_cost, 2),
            "strategy_2to5pm_cost_usd": round(tot_strat_cost, 2),
            "projected_cost_savings_usd": round(cost_saved, 2),
            "window_energy_reduction_pct": round(window_energy_red_pct, 1),
            "instantaneous_peak_kw_reduction_pct": round(peak_kw_red_pct, 1),
            "max_indoor_temp_c": round(max_indoor_temp, 2),
            "comfort_impact": comfort_rating,
        }

        cand["simulation_result"] = sim_outcome
        simulation_results[strat_id] = sim_outcome
        updated_candidates.append(cand)

        print(f"    -> [{strat_id}: {strat_name}] Savings: ${cost_saved:.2f} ({kwh_saved:.1f} kWh, {window_energy_red_pct:.1f}% energy red.) | Instantaneous Peak Red: {peak_kw_red_pct:.1f}% | Comfort: {comfort_rating}")

    return {
        "candidate_strategies": updated_candidates,
        "simulation_results": simulation_results,
    }


def select_strategy_node(state: EnergyAgentState) -> Dict[str, Any]:
    """
    6. Select Strategy Node: Multi-objective scoring of simulated candidates.
    score = (cost_weight * normalized_savings) - (comfort_weight * normalized_comfort_deviation)
    Saves full audit ranking list to state.
    """
    candidates = state.get("candidate_strategies", [])
    if not candidates:
        print("  [NODE: select_strategy] Warning: No candidate strategies found.")
        return {"selected_strategy": None, "ranked_strategies": []}

    cost_w, comfort_w = get_strategy_weights()

    # Extract metrics for normalization
    savings_list = [c.get("simulation_result", {}).get("projected_cost_savings_usd", 0.0) for c in candidates]
    comfort_dev_list = [abs(c.get("simulation_result", {}).get("max_indoor_temp_c", 21.0) - 21.0) for c in candidates]

    min_savings, max_savings = min(savings_list), max(savings_list)
    min_dev, max_dev = min(comfort_dev_list), max(comfort_dev_list)

    print(f"  [NODE: select_strategy] Scoring candidates with weights: Cost={cost_w:.2f}, Comfort={comfort_w:.2f}...")

    scored_candidates = []
    for c, savings, dev in zip(candidates, savings_list, comfort_dev_list):
        # Normalize savings (0 to 1, higher is better)
        if max_savings > min_savings:
            norm_savings = (savings - min_savings) / (max_savings - min_savings)
        else:
            norm_savings = 1.0

        # Normalize comfort deviation (0 to 1, higher means worse comfort deviation)
        if max_dev > min_dev:
            norm_dev = (dev - min_dev) / (max_dev - min_dev)
        else:
            norm_dev = 0.0

        score = (cost_w * norm_savings) - (comfort_w * norm_dev)

        c_copy = dict(c)
        c_copy["score"] = round(float(score), 4)
        c_copy["norm_savings"] = round(float(norm_savings), 3)
        c_copy["norm_comfort_deviation"] = round(float(norm_dev), 3)
        scored_candidates.append(c_copy)

    # Sort descending by multi-objective score
    ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
    selected = ranked_candidates[0]

    print("\n  --- Full Auditable Strategy Ranking List ---")
    for rank, rc in enumerate(ranked_candidates, 1):
        sim = rc.get("simulation_result", {})
        print(f"    Rank #{rank}: {rc['strategy_id']} ('{rc['strategy_name']}') -> Score: {rc['score']:.4f} (Savings: ${sim.get('projected_cost_savings_usd'):.2f}, MaxTemp: {sim.get('max_indoor_temp_c')}°C)")

    return {
        "ranked_strategies": ranked_candidates,
        "selected_strategy": selected,
        "cost_savings_estimate": selected.get("simulation_result", {}),
    }


def human_approval_gate_node(state: EnergyAgentState) -> Dict[str, Any]:
    """
    7. Human Approval Gate Node: Evaluates approval thresholds.
    Triggers approval requirement if kW reduction, cost savings, or temperature drift cross configured thresholds.
    Formulates proposal summary including top strategy AND alternative strategy #2.
    """
    selected = state.get("selected_strategy")
    ranked = state.get("ranked_strategies", [])
    status = state.get("approval_status", "PENDING")

    if not selected:
        return {"requires_human_approval": False, "approval_status": "APPROVED"}

    sim = selected.get("simulation_result", {})
    thresholds = get_approval_thresholds()

    kw_red = selected.get("expected_kw_reduction", 0.0)
    cost_sav = sim.get("projected_cost_savings_usd", 0.0)
    max_temp = sim.get("max_indoor_temp_c", 21.0)

    triggers = []
    if kw_red >= thresholds["kw_reduction"]:
        triggers.append(f"Expected kW reduction ({kw_red:.1f} kW) >= threshold ({thresholds['kw_reduction']:.1f} kW)")
    if cost_sav >= thresholds["cost_savings"]:
        triggers.append(f"Projected cost savings (${cost_sav:.2f}) >= threshold (${thresholds['cost_savings']:.2f})")
    if max_temp >= thresholds["max_temp"]:
        triggers.append(f"Max indoor zone temperature ({max_temp:.1f}°C) >= threshold ({thresholds['max_temp']:.1f}°C)")

    requires_approval = len(triggers) > 0

    top_strategy = selected
    alt_strategy = ranked[1] if len(ranked) > 1 else None

    proposal = {
        "top_strategy": top_strategy,
        "alternative_strategy": alt_strategy,
        "threshold_triggers": triggers,
    }

    if status == "APPROVED":
        return {
            "requires_human_approval": False,
            "approval_status": "APPROVED",
            "approval_proposal": proposal,
        }

    if requires_approval and status == "PENDING":
        print(f"  [NODE: human_approval_gate] Approval Required! {len(triggers)} threshold trigger(s) detected.")
        return {
            "requires_human_approval": True,
            "approval_status": "PENDING",
            "approval_proposal": proposal,
        }

    return {
        "requires_human_approval": requires_approval,
        "approval_status": status,
        "approval_proposal": proposal,
    }


def execute_actions_node(state: EnergyAgentState) -> Dict[str, Any]:
    """
    8. Execute Actions Node: Dispatches control actions for approved strategy to building tools.
    Handles partial failures gracefully, continuing remaining actions and recording audit log.
    """
    status = state.get("approval_status", "APPROVED")
    if status.startswith("REJECTED"):
        print("  [NODE: execute_actions] Run rejected by human operator. Skipping action execution.")
        return {"executed_actions": []}

    strategy = state.get("selected_strategy", {})
    actions = strategy.get("actions", [])
    executed = []

    print(f"  [NODE: execute_actions] Dispatching {len(actions)} control actions to building IoT tools...")

    for act in actions:
        action_name = act["action_type"]
        params = act["parameters"]
        res = None

        if action_name == "adjust_thermostat":
            res = adjust_thermostat(**params)
        elif action_name == "dim_lighting":
            res = dim_lighting(**params)
        elif action_name == "dispatch_battery":
            res = dispatch_battery(**params)
        elif action_name == "shift_equipment_load":
            res = shift_equipment_load(**params)

        if res:
            executed.append(res.model_dump())
            if res.success:
                print(f"    -> [SUCCESS] Action '{action_name}': {res.message}")
            else:
                print(f"    -> [WARNING] Action '{action_name}' REJECTED/FAILED: {res.message}. Continuing remaining actions.")

    return {"executed_actions": executed}


def replan_check_node(state: EnergyAgentState) -> Dict[str, Any]:
    """
    9. Replan Check Node: Evaluates loop continuation or safe termination.
    Handles operator refusal safely (no infinite spinning) and max iteration guards.
    """
    replan_count = state.get("replan_count", 0)
    status = state.get("approval_status", "APPROVED")

    if status in ["REJECTED_ALL", "REJECTED_FINAL"]:
        print("  [NODE: replan_check] Operator refused all strategies on retry. Terminating safely with 0 actions taken.")
        _log_action(
            "replan_check",
            {"status": status},
            False,
            "REJECTED: Operator refused all proposed demand response plans. Run terminated safely with 0 actions taken.",
        )
        return {"replan_count": replan_count + 1, "should_continue": False}

    if status == "REJECTED":
        if replan_count < 1:
            print("  [NODE: replan_check] Rejection received. Initiating strategy regeneration retry...")
            return {"replan_count": replan_count + 1, "should_continue": True}
        else:
            print("  [NODE: replan_check] Max replan attempts reached. Terminating safely.")
            return {"replan_count": replan_count + 1, "should_continue": False}

    print(f"  [NODE: replan_check] Run complete. Continue loop: False")
    return {
        "replan_count": replan_count,
        "should_continue": False,
    }


# Router Functions for Conditional Edges

def route_after_diagnose(state: EnergyAgentState) -> Literal["generate_strategies", "replan_check"]:
    status = state.get("approval_status", "PENDING")
    if status in ["REJECTED_ALL", "REJECTED_FINAL"]:
        return "replan_check"
    if state.get("risk_detected", False):
        return "generate_strategies"
    return "replan_check"


def route_after_approval(state: EnergyAgentState) -> Literal["execute_actions", "replan_check"]:
    status = state.get("approval_status", "PENDING")
    requires_approval = state.get("requires_human_approval", False)

    if status in ["APPROVED", "PENDING", "REJECTED"] or not requires_approval:
        return "execute_actions"
    else:
        return "replan_check"


def route_after_replan(state: EnergyAgentState) -> Literal["generate_strategies", "__end__"]:
    if state.get("should_continue", False):
        return "generate_strategies"
    return END


# Create and Compile StateGraph with Checkpointer

def create_energy_agent_graph(checkpointer: Optional[Any] = None) -> StateGraph:
    workflow = StateGraph(EnergyAgentState)

    workflow.add_node("monitor", monitor_node)
    workflow.add_node("forecast_demand", forecast_demand_node)
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("generate_strategies", generate_strategies_node)
    workflow.add_node("simulate_strategies", simulate_strategies_node)
    workflow.add_node("select_strategy", select_strategy_node)
    workflow.add_node("human_approval_gate", human_approval_gate_node)
    workflow.add_node("execute_actions", execute_actions_node)
    workflow.add_node("replan_check", replan_check_node)

    workflow.add_edge(START, "monitor")
    workflow.add_edge("monitor", "forecast_demand")
    workflow.add_edge("forecast_demand", "diagnose")

    workflow.add_edge("generate_strategies", "simulate_strategies")
    workflow.add_edge("simulate_strategies", "select_strategy")
    workflow.add_edge("select_strategy", "human_approval_gate")

    workflow.add_edge("execute_actions", "replan_check")

    workflow.add_conditional_edges("diagnose", route_after_diagnose, {"generate_strategies": "generate_strategies", "replan_check": "replan_check"})
    workflow.add_conditional_edges("human_approval_gate", route_after_approval, {"execute_actions": "execute_actions", "replan_check": "replan_check"})
    workflow.add_conditional_edges("replan_check", route_after_replan, {"generate_strategies": "generate_strategies", END: END})

    if checkpointer is True:
        checkpointer = MemorySaver()

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_actions"] if checkpointer else None,
    )


if __name__ == "__main__":
    app = create_energy_agent_graph()
    print("Graph compiled successfully.")
