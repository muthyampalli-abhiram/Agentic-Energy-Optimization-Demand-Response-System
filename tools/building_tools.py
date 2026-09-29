import os
import json
import datetime
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from simulator.live_feed import LiveBuildingSensor, TelemetryReading
from forecasting.model import forecast_next_hours

# Define Pydantic Models for Structured Tool Inputs & Outputs

class ActionResult(BaseModel):
    """Structured result returned by control action tools."""
    success: bool = Field(..., description="Whether the action was safely executed or rejected")
    action_type: str = Field(..., description="Name of the control action")
    parameters: Dict[str, Any] = Field(..., description="Parameters passed to the action")
    message: str = Field(..., description="Detailed execution message or rejection reason")
    timestamp: str = Field(..., description="ISO timestamp of execution")
    cost_impact_usd: float = Field(0.0, description="Estimated 1-hour cost impact in USD (negative = savings)")


class BuildingState(BaseModel):
    """Read-only snapshot of current building IoT state."""
    timestamp: str
    occupancy: float
    outdoor_temperature: float
    electricity_price: float
    indoor_temperatures: Dict[str, float]
    hvac_setpoints: Dict[str, float]
    total_consumption_kwh: float
    hvac_kwh: float
    lighting_kwh: float
    plug_loads_kwh: float
    other_kwh: float
    battery_dispatch_kw: float
    battery_soc_kwh: float
    equipment_status: Dict[str, Any]


class CostEstimate(BaseModel):
    """Estimated financial and energy impact of a proposed intervention over the next 1 hour."""
    action_type: str = Field(..., description="Action evaluated")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")
    estimated_kwh_delta: float = Field(..., description="Estimated change in consumption (kWh) over 1 hour. Negative means energy saved.")
    estimated_cost_delta_usd: float = Field(..., description="Estimated change in energy cost (USD) over 1 hour. Negative means money saved.")
    current_electricity_price: float = Field(..., description="Current electricity tariff ($/kWh)")
    explanation: str = Field(..., description="Detailed breakdown of how the cost estimate was calculated")


# Critical non-deferrable equipment registry
CRITICAL_EQUIPMENT = [
    "life_safety",
    "server_cooling",
    "elevators",
    "emergency_lighting",
    "fire_suppression",
    "critical_medical",
]

# Singleton live building sensor reference for active session
_sensor_instance: Optional[LiveBuildingSensor] = None


def get_building_sensor(initial_time: str = "2026-06-01 13:00:00") -> LiveBuildingSensor:
    """Retrieve or initialize the active LiveBuildingSensor instance."""
    global _sensor_instance
    if _sensor_instance is None:
        _sensor_instance = LiveBuildingSensor(initial_time=initial_time)
    return _sensor_instance


def reset_building_sensor(sensor: Optional[LiveBuildingSensor] = None) -> LiveBuildingSensor:
    """Reset or replace the active LiveBuildingSensor instance."""
    global _sensor_instance
    _sensor_instance = sensor
    return get_building_sensor()


def _log_action(
    action_type: str,
    params: Dict[str, Any],
    success: bool,
    reason: str,
    cost_impact_usd: float = 0.0,
    log_path: str = "data/action_log.csv",
) -> None:
    """Log all attempted actions (both accepted and rejected) to data/action_log.csv."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)

    timestamp = datetime.datetime.now().isoformat()
    row = {
        "timestamp": timestamp,
        "action_type": action_type,
        "params": json.dumps(params),
        "success": success,
        "reason": reason,
        "cost_impact_usd": round(cost_impact_usd, 4),
    }

    df_row = pd.DataFrame([row])
    df_row.to_csv(log_path, mode="a", header=not file_exists, index=False)


# Tool Implementation Functions

def adjust_thermostat(zone: str, new_setpoint_c: float) -> ActionResult:
    """
    Adjust the thermostat setpoint for a specified zone or 'all' zones.

    Safety Boundary: Setpoint must strictly be between 19.0°C and 26.0°C to preserve occupant health & thermal comfort.
    """
    sensor = get_building_sensor()
    params = {"zone": zone, "new_setpoint_c": new_setpoint_c}
    ts = sensor.timestamp.isoformat()

    # Safety Validation Check
    if not (19.0 <= new_setpoint_c <= 26.0):
        reason = f"REJECTED: Setpoint {new_setpoint_c}°C violates safety boundary (allowed range: 19.0°C - 26.0°C)."
        _log_action("adjust_thermostat", params, False, reason)
        return ActionResult(
            success=False,
            action_type="adjust_thermostat",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )

    # Estimate 1-hour cost impact
    cost_est = estimate_cost_impact("adjust_thermostat", params)

    # Execute control action on simulator
    try:
        sensor.set_thermostat(zone, new_setpoint_c)
        reason = f"SUCCESS: Thermostat for zone '{zone}' set to {new_setpoint_c}°C."
        _log_action("adjust_thermostat", params, True, reason, cost_est.estimated_cost_delta_usd)
        return ActionResult(
            success=True,
            action_type="adjust_thermostat",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=cost_est.estimated_cost_delta_usd,
        )
    except Exception as e:
        reason = f"ERROR: Failed to adjust thermostat: {str(e)}"
        _log_action("adjust_thermostat", params, False, reason)
        return ActionResult(
            success=False,
            action_type="adjust_thermostat",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )


def shift_equipment_load(equipment_name: str, delay_minutes: int) -> ActionResult:
    """
    Defer electrical load for specified equipment by delay_minutes.

    Safety Boundary: Cannot shift critical equipment (e.g. life safety, server cooling, elevators). Delay must be 1 - 360 mins.
    """
    sensor = get_building_sensor()
    params = {"equipment_name": equipment_name, "delay_minutes": delay_minutes}
    ts = sensor.timestamp.isoformat()

    # Safety Validation: Critical Equipment
    eq_clean = equipment_name.lower().strip()
    if eq_clean in CRITICAL_EQUIPMENT or any(c in eq_clean for c in ["server", "life_safety", "elevator"]):
        reason = f"REJECTED: Equipment '{equipment_name}' is registered as CRITICAL and non-deferrable."
        _log_action("shift_equipment_load", params, False, reason)
        return ActionResult(
            success=False,
            action_type="shift_equipment_load",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )

    # Safety Validation: Delay Duration
    if delay_minutes <= 0 or delay_minutes > 360:
        reason = f"REJECTED: Delay minutes ({delay_minutes}) must be between 1 and 360 minutes."
        _log_action("shift_equipment_load", params, False, reason)
        return ActionResult(
            success=False,
            action_type="shift_equipment_load",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )

    cost_est = estimate_cost_impact("shift_equipment_load", params)

    try:
        sensor.shift_load(equipment_name, delay_minutes)
        reason = f"SUCCESS: Equipment '{equipment_name}' load deferred by {delay_minutes} minutes."
        _log_action("shift_equipment_load", params, True, reason, cost_est.estimated_cost_delta_usd)
        return ActionResult(
            success=True,
            action_type="shift_equipment_load",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=cost_est.estimated_cost_delta_usd,
        )
    except Exception as e:
        reason = f"ERROR: Failed to shift equipment load: {str(e)}"
        _log_action("shift_equipment_load", params, False, reason)
        return ActionResult(
            success=False,
            action_type="shift_equipment_load",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )


def dispatch_battery(kw: float) -> ActionResult:
    """
    Dispatch battery power in kW (positive = discharge to reduce grid load, negative = charge).

    Safety Boundary: Rate must be within +/- 50 kW max power limit, and SoC must be > 5.0 kWh for discharging.
    """
    sensor = get_building_sensor()
    params = {"kw": kw}
    ts = sensor.timestamp.isoformat()

    # Safety Validation: Discharge/Charge Power Limit
    if abs(kw) > 50.0:
        reason = f"REJECTED: Battery dispatch rate {kw} kW exceeds maximum safe power rating (+/- 50.0 kW)."
        _log_action("dispatch_battery", params, False, reason)
        return ActionResult(
            success=False,
            action_type="dispatch_battery",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )

    # Safety Validation: Battery State of Charge (SoC)
    if kw > 0 and sensor.battery_soc_kwh <= 5.0:
        reason = f"REJECTED: Battery State of Charge ({sensor.battery_soc_kwh:.1f} kWh) is too low for discharge."
        _log_action("dispatch_battery", params, False, reason)
        return ActionResult(
            success=False,
            action_type="dispatch_battery",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )

    cost_est = estimate_cost_impact("dispatch_battery", params)

    try:
        sensor.dispatch_battery(kw)
        mode = "discharge" if kw > 0 else "charge"
        reason = f"SUCCESS: Battery set to {mode} at {abs(kw)} kW."
        _log_action("dispatch_battery", params, True, reason, cost_est.estimated_cost_delta_usd)
        return ActionResult(
            success=True,
            action_type="dispatch_battery",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=cost_est.estimated_cost_delta_usd,
        )
    except Exception as e:
        reason = f"ERROR: Failed to dispatch battery: {str(e)}"
        _log_action("dispatch_battery", params, False, reason)
        return ActionResult(
            success=False,
            action_type="dispatch_battery",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )


def dim_lighting(zone: str, percent: int) -> ActionResult:
    """
    Dim common-area or office lighting in a specified zone by a given percentage (0% to 60%).

    Safety Boundary: Cannot dim by more than 60% (retaining minimum 40% illumination for workplace safety).
    """
    sensor = get_building_sensor()
    params = {"zone": zone, "percent": percent}
    ts = sensor.timestamp.isoformat()

    # Safety Validation: Lighting Dimming Limit
    if percent < 0 or percent > 60:
        reason = f"REJECTED: Lighting dimming level {percent}% violates safety limits (allowed range: 0% to 60%)."
        _log_action("dim_lighting", params, False, reason)
        return ActionResult(
            success=False,
            action_type="dim_lighting",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )

    cost_est = estimate_cost_impact("dim_lighting", params)

    try:
        sensor.shift_load(f"lighting_dim_{zone}", 180)
        sensor.equipment_status["lighting_dim_pct"] = percent
        reason = f"SUCCESS: Lighting in zone '{zone}' dimmed by {percent}%."
        _log_action("dim_lighting", params, True, reason, cost_est.estimated_cost_delta_usd)
        return ActionResult(
            success=True,
            action_type="dim_lighting",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=cost_est.estimated_cost_delta_usd,
        )
    except Exception as e:
        reason = f"ERROR: Failed to dim lighting: {str(e)}"
        _log_action("dim_lighting", params, False, reason)
        return ActionResult(
            success=False,
            action_type="dim_lighting",
            parameters=params,
            message=reason,
            timestamp=ts,
            cost_impact_usd=0.0,
        )


def get_current_building_state() -> BuildingState:
    """
    Retrieve a read-only snapshot of the current building IoT sensor telemetry and control state.
    """
    sensor = get_building_sensor()
    rd = sensor.get_current_reading()
    return BuildingState(**rd.model_dump())


def estimate_cost_impact(action_type: str, parameters: Dict[str, Any]) -> CostEstimate:
    """
    Estimate energy (kWh) and financial ($) impact over the next 1 hour for a proposed action.
    Negative delta values indicate savings.
    """
    sensor = get_building_sensor()
    price = sensor.electricity_price

    kwh_delta = 0.0
    explanation = ""

    if action_type == "adjust_thermostat":
        new_sp = parameters.get("new_setpoint_c", 21.0)
        current_sp = sensor.hvac_setpoints.get("Zone_1", 21.0)
        sp_diff = new_sp - current_sp

        # Each +1°C setpoint increase reduces HVAC consumption by approx ~5.0 kWh/hr (~1.25 kWh per 15-min)
        kwh_delta = -5.0 * sp_diff
        explanation = f"Adjusting setpoint by {sp_diff:+.1f}°C alters 1-hour HVAC energy by approximately {kwh_delta:+.2f} kWh."

    elif action_type == "dispatch_battery":
        kw = parameters.get("kw", 0.0)
        # Discharging kw kW for 1 hour saves kw kWh of grid energy
        kwh_delta = -kw * 1.0
        explanation = f"Discharging battery at {kw} kW for 1 hour offsets {kw} kWh of grid consumption."

    elif action_type == "dim_lighting":
        pct = parameters.get("percent", 0)
        # 1-hour baseline lighting is approx 4 * current 15-min lighting kWh
        baseline_lighting_1h = sensor.lighting_kwh * 4.0
        kwh_delta = -baseline_lighting_1h * (pct / 100.0)
        explanation = f"Dimming lighting by {pct}% saves {abs(kwh_delta):.2f} kWh over 1 hour."

    elif action_type == "shift_equipment_load":
        mins = parameters.get("delay_minutes", 60)
        kwh_delta = -4.0 * (min(mins, 60) / 60.0)
        explanation = f"Deferring non-critical load for {mins} mins reduces peak hour consumption by {abs(kwh_delta):.2f} kWh."

    else:
        explanation = f"Unknown action '{action_type}'. Estimated zero impact."

    cost_delta = kwh_delta * price

    return CostEstimate(
        action_type=action_type,
        parameters=parameters,
        estimated_kwh_delta=round(kwh_delta, 2),
        estimated_cost_delta_usd=round(cost_delta, 2),
        current_electricity_price=round(price, 4),
        explanation=explanation,
    )
