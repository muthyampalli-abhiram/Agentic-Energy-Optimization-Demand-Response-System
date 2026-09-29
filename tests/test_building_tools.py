import os
import pytest
import pandas as pd

from tools.building_tools import (
    adjust_thermostat,
    shift_equipment_load,
    dispatch_battery,
    dim_lighting,
    get_current_building_state,
    estimate_cost_impact,
    reset_building_sensor,
    ActionResult,
    BuildingState,
    CostEstimate,
)
from simulator.live_feed import LiveBuildingSensor


@pytest.fixture(autouse=True)
def fresh_building_sensor(tmp_path):
    log_file = tmp_path / "action_log.csv"
    sensor = LiveBuildingSensor(initial_time="2026-06-01 13:30:00")
    reset_building_sensor(sensor)
    yield
    # Cleanup log file if created in default location
    if os.path.exists("data/action_log.csv"):
        pass


def test_get_current_building_state():
    state = get_current_building_state()
    assert isinstance(state, BuildingState)
    assert state.occupancy >= 0.0
    assert state.electricity_price > 0.0
    assert len(state.indoor_temperatures) == 3


def test_adjust_thermostat_success_and_rejection():
    # Valid action (24.0°C within 19-26°C range)
    res_ok = adjust_thermostat("Zone_1", 24.0)
    assert isinstance(res_ok, ActionResult)
    assert res_ok.success is True
    assert "SUCCESS" in res_ok.message

    # Rejection action (28.0°C exceeds safety max of 26°C)
    res_reject_high = adjust_thermostat("Zone_1", 28.0)
    assert res_reject_high.success is False
    assert "REJECTED" in res_reject_high.message

    # Rejection action (17.0°C below safety min of 19°C)
    res_reject_low = adjust_thermostat("Zone_2", 17.0)
    assert res_reject_low.success is False
    assert "REJECTED" in res_reject_low.message


def test_shift_equipment_load_success_and_rejection():
    # Valid action (non-critical equipment)
    res_ok = shift_equipment_load("ev_chargers", 60)
    assert res_ok.success is True
    assert "SUCCESS" in res_ok.message

    # Rejection action (critical equipment: server_cooling, elevators, life_safety)
    res_reject_critical = shift_equipment_load("server_cooling", 30)
    assert res_reject_critical.success is False
    assert "REJECTED" in res_reject_critical.message
    assert "CRITICAL" in res_reject_critical.message

    # Rejection action (invalid duration: 0 or >360)
    res_reject_time = shift_equipment_load("water_heaters", 500)
    assert res_reject_time.success is False
    assert "REJECTED" in res_reject_time.message


def test_dispatch_battery_success_and_rejection():
    # Valid action (30 kW within +/- 50 kW limit)
    res_ok = dispatch_battery(30.0)
    assert res_ok.success is True
    assert "SUCCESS" in res_ok.message

    # Rejection action (80 kW exceeds 50 kW max limit)
    res_reject_rate = dispatch_battery(80.0)
    assert res_reject_rate.success is False
    assert "REJECTED" in res_reject_rate.message


def test_dim_lighting_success_and_rejection():
    # Valid action (30% dimming within 0-60% range)
    res_ok = dim_lighting("Zone_1", 30)
    assert res_ok.success is True
    assert "SUCCESS" in res_ok.message

    # Rejection action (80% dimming exceeds 60% max safety limit)
    res_reject = dim_lighting("Zone_2", 80)
    assert res_reject.success is False
    assert "REJECTED" in res_reject.message


def test_estimate_cost_impact():
    cost_est = estimate_cost_impact("adjust_thermostat", {"new_setpoint_c": 24.0})
    assert isinstance(cost_est, CostEstimate)
    assert cost_est.estimated_kwh_delta < 0.0  # Raising setpoint saves energy
    assert cost_est.estimated_cost_delta_usd < 0.0  # Saves cost


def test_action_logging():
    log_path = "data/action_log.csv"
    if os.path.exists(log_path):
        os.remove(log_path)

    # Perform action
    adjust_thermostat("Zone_1", 24.0)
    adjust_thermostat("Zone_1", 30.0)  # Rejection

    assert os.path.exists(log_path)
    df_log = pd.read_csv(log_path)
    assert len(df_log) >= 2
    assert "action_type" in df_log.columns
    assert "success" in df_log.columns
    assert "reason" in df_log.columns
