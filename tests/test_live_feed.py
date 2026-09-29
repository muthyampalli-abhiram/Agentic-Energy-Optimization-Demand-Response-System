import pytest
import pandas as pd
from simulator.live_feed import LiveBuildingSensor, TelemetryReading


def test_live_sensor_initialization():
    sensor = LiveBuildingSensor(initial_time="2026-06-01 12:00:00")
    reading = sensor.get_current_reading()

    assert isinstance(reading, TelemetryReading)
    assert len(reading.indoor_temperatures) == 3
    assert reading.hvac_setpoints["Zone_1"] == 21.0
    assert reading.total_consumption_kwh > 0.0


def test_thermostat_control_action():
    sensor = LiveBuildingSensor(initial_time="2026-06-01 13:00:00")

    # Baseline tick at 21°C setpoint
    r1 = sensor.tick("2026-06-01 13:15:00")
    hvac_load_before = r1.hvac_kwh

    # Raise setpoint to 25°C across all zones
    sensor.set_thermostat("all", 25.0)
    r2 = sensor.tick("2026-06-01 13:30:00")
    hvac_load_after = r2.hvac_kwh

    # Higher setpoint should reduce cooling demand immediately next tick
    assert hvac_load_after < hvac_load_before, (
        f"HVAC load after setpoint increase ({hvac_load_after:.2f} kWh) should be lower "
        f"than before ({hvac_load_before:.2f} kWh)"
    )


def test_battery_dispatch_control_action():
    sensor = LiveBuildingSensor(initial_time="2026-06-01 14:00:00", initial_battery_soc_kwh=50.0)

    # Tick without battery
    r1 = sensor.tick("2026-06-01 14:15:00")
    net_load_no_battery = r1.total_consumption_kwh

    # Dispatch battery at 40 kW (10 kWh per 15 min interval)
    sensor.dispatch_battery(40.0)
    r2 = sensor.tick("2026-06-01 14:30:00")
    net_load_with_battery = r2.total_consumption_kwh

    # Battery discharge should reduce net grid consumption
    assert net_load_with_battery < net_load_no_battery
    assert r2.battery_soc_kwh < 50.0  # SoC reduced


def test_load_shift_control_action():
    sensor = LiveBuildingSensor(initial_time="2026-06-01 14:00:00")

    r1 = sensor.tick("2026-06-01 14:15:00")
    lighting_before = r1.lighting_kwh

    # Shift/dim lighting load
    sensor.shift_load("lighting_dim", delay_minutes=60)
    r2 = sensor.tick("2026-06-01 14:30:00")
    lighting_after = r2.lighting_kwh

    assert lighting_after < lighting_before
    assert "lighting_dim" in r2.equipment_status["deferred_loads"]
