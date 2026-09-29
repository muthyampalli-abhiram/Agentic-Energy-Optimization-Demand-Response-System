from typing import Dict, Any, Union, List
import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from simulator.generators import (
    generate_occupancy,
    generate_outdoor_temperature,
    generate_electricity_price,
)


class TelemetryReading(BaseModel):
    """Pydantic model representing a live snapshot of IoT sensor readings."""
    timestamp: str = Field(..., description="Timestamp in ISO format")
    occupancy: float = Field(..., description="Occupancy percentage (0-100%)")
    outdoor_temperature: float = Field(..., description="Outdoor ambient temperature (°C)")
    electricity_price: float = Field(..., description="Current electricity tariff rate ($/kWh)")
    indoor_temperatures: Dict[str, float] = Field(..., description="Indoor temperature per zone (°C)")
    hvac_setpoints: Dict[str, float] = Field(..., description="Active HVAC setpoint per zone (°C)")
    total_consumption_kwh: float = Field(..., description="Net building grid load for 15-min interval (kWh)")
    hvac_kwh: float = Field(..., description="HVAC load for 15-min interval (kWh)")
    lighting_kwh: float = Field(..., description="Lighting load for 15-min interval (kWh)")
    plug_loads_kwh: float = Field(..., description="Plug loads for 15-min interval (kWh)")
    other_kwh: float = Field(..., description="Other base load for 15-min interval (kWh)")
    battery_dispatch_kw: float = Field(..., description="Active battery discharge rate (kW)")
    battery_soc_kwh: float = Field(..., description="Battery state of charge (kWh)")
    equipment_status: Dict[str, Any] = Field(..., description="Status of equipment and load shifts")


class LiveBuildingSensor:
    """
    Live IoT sensor feed simulator for a commercial office building with 3 thermal zones.
    Simulates real-time telemetry polling and control responses (thermostat, load shedding, battery).
    """

    def __init__(
        self,
        zones: List[str] = None,
        initial_time: Union[str, pd.Timestamp, datetime.datetime] = "2026-06-01 12:00:00",
        battery_capacity_kwh: float = 100.0,
        initial_battery_soc_kwh: float = 80.0,
        seed: int = 42,
    ):
        if zones is None:
            zones = ["Zone_1", "Zone_2", "Zone_3"]
        self.zones = zones
        self.timestamp = pd.Timestamp(initial_time)
        self.rng = np.random.default_rng(seed)

        # Thermal state per zone
        self.indoor_temperatures = {z: 21.0 for z in self.zones}
        self.hvac_setpoints = {z: 21.0 for z in self.zones}

        # Battery storage state
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_soc_kwh = min(initial_battery_soc_kwh, battery_capacity_kwh)
        self.battery_dispatch_kw = 0.0

        # Load shifting and equipment state
        self.deferred_equipment: Dict[str, int] = {}  # equipment_name -> remaining delay minutes
        self.equipment_status = {
            "hvac_mode": "COOLING",
            "lighting_dim_pct": 0,  # 0% dimmed (100% full power)
            "deferred_loads": [],
        }

        # Current telemetry values
        self.occupancy = 0.0
        self.outdoor_temp = 20.0
        self.electricity_price = 0.12
        self.hvac_kwh = 0.0
        self.lighting_kwh = 0.0
        self.plug_loads_kwh = 0.0
        self.other_kwh = 0.0
        self.total_consumption_kwh = 0.0

        # Perform initial tick
        self.tick(self.timestamp)

    def set_thermostat(self, zone: str, setpoint: float) -> None:
        """
        Set thermostat setpoint for a specific zone or 'all' zones.
        """
        if zone.lower() == "all":
            for z in self.zones:
                self.hvac_setpoints[z] = float(setpoint)
        elif zone in self.hvac_setpoints:
            self.hvac_setpoints[zone] = float(setpoint)
        else:
            raise ValueError(f"Unknown zone '{zone}'. Available zones: {self.zones}")

    def shift_load(self, equipment_name: str, delay_minutes: int) -> None:
        """
        Defer an equipment load (e.g. 'ev_chargers', 'water_heaters', 'lighting_dim') for specified minutes.
        """
        self.deferred_equipment[equipment_name] = max(0, delay_minutes)
        if equipment_name not in self.equipment_status["deferred_loads"]:
            self.equipment_status["deferred_loads"].append(equipment_name)

    def dispatch_battery(self, kw: float) -> None:
        """
        Dispatch battery power (kW). Positive values discharge battery (reducing building load).
        Negative values charge battery.
        """
        self.battery_dispatch_kw = float(kw)

    def tick(self, current_time: Union[str, pd.Timestamp, datetime.datetime]) -> TelemetryReading:
        """
        Advance the sensor simulator state to `current_time` (typically 15-minute intervals).
        Updates thermal physics, equipment loads, battery storage, and net grid consumption.
        """
        self.timestamp = pd.Timestamp(current_time)
        ts_idx = pd.DatetimeIndex([self.timestamp])

        # 1. Update environmental context using consistent generator logic
        self.occupancy = float(generate_occupancy(ts_idx, seed=int(self.timestamp.timestamp()) % 1000).iloc[0])
        self.outdoor_temp = float(generate_outdoor_temperature(ts_idx, seed=int(self.timestamp.timestamp()) % 1000).iloc[0])
        self.electricity_price = float(generate_electricity_price(ts_idx).iloc[0])

        # 2. Update deferred equipment timers
        active_deferred = []
        for eq, rem_mins in list(self.deferred_equipment.items()):
            if rem_mins > 0:
                self.deferred_equipment[eq] = rem_mins - 15
                active_deferred.append(eq)
            else:
                del self.deferred_equipment[eq]
        self.equipment_status["deferred_loads"] = active_deferred

        # 3. Simulate Zone Thermal Physics and HVAC Consumption
        total_hvac_kwh = 0.0
        for z in self.zones:
            setpoint = self.hvac_setpoints[z]
            t_in = self.indoor_temperatures[z]

            # Heat transfers (°C change over 15 mins)
            # Ambient heat gain from outside
            outdoor_heat_gain = 0.15 * (self.outdoor_temp - t_in)
            # Internal heat gain from occupancy
            occupancy_heat_gain = 0.10 * (self.occupancy / 100.0)

            # Unconditioned indoor temp float
            t_float = t_in + outdoor_heat_gain + occupancy_heat_gain

            # Cooling effort required to reach setpoint
            cooling_needed = max(0.0, t_float - setpoint)

            # HVAC cooling capacity per interval (max 1.5 °C cooling per 15 min per zone)
            cooling_applied = min(cooling_needed, 1.5)

            # HVAC energy needed per zone for this interval (kWh)
            # Baseline ~4.0 kWh + ~6.0 kWh per degree of active cooling
            zone_hvac_kwh = 4.0 + 6.0 * cooling_applied + self.rng.normal(0.0, 0.2)
            zone_hvac_kwh = max(1.5, zone_hvac_kwh)
            total_hvac_kwh += zone_hvac_kwh

            # Update zone indoor temperature for next reading
            self.indoor_temperatures[z] = round(t_float - cooling_applied, 2)

        self.hvac_kwh = round(total_hvac_kwh, 4)

        # 4. Lighting & Plug Loads
        # Lighting load (reduced if 'lighting_dim' is in deferred loads)
        lighting_reduction = 0.35 if "lighting_dim" in active_deferred else 0.0
        base_lighting = (8.0 + 15.0 * (self.occupancy / 100.0)) * (1.0 - lighting_reduction)
        self.lighting_kwh = round(max(2.0, base_lighting + self.rng.normal(0.0, 0.3)), 4)

        # Plug loads (reduced if load shifted)
        plug_reduction = 0.30 if "heavy_equipment" in active_deferred else 0.0
        base_plug = (10.0 + 20.0 * (self.occupancy / 100.0)) * (1.0 - plug_reduction)
        self.plug_loads_kwh = round(max(3.0, base_plug + self.rng.normal(0.0, 0.4)), 4)

        # Other base load (servers, security, elevators)
        self.other_kwh = round(9.0 + self.rng.normal(0.0, 0.2), 4)

        # Gross building load
        gross_kwh = self.hvac_kwh + self.lighting_kwh + self.plug_loads_kwh + self.other_kwh

        # 5. Battery Dispatch Logic
        interval_hours = 0.25  # 15 minutes
        actual_battery_kwh_delta = 0.0

        if self.battery_dispatch_kw > 0:
            # Discharging battery
            max_possible_discharge_kwh = self.battery_soc_kwh
            requested_discharge_kwh = self.battery_dispatch_kw * interval_hours
            actual_discharge_kwh = min(requested_discharge_kwh, max_possible_discharge_kwh)
            self.battery_soc_kwh -= actual_discharge_kwh
            actual_battery_kwh_delta = actual_discharge_kwh
        elif self.battery_dispatch_kw < 0:
            # Charging battery
            max_possible_charge_kwh = self.battery_capacity_kwh - self.battery_soc_kwh
            requested_charge_kwh = abs(self.battery_dispatch_kw) * interval_hours
            actual_charge_kwh = min(requested_charge_kwh, max_possible_charge_kwh)
            self.battery_soc_kwh += actual_charge_kwh
            actual_battery_kwh_delta = -actual_charge_kwh

        # Net grid consumption
        net_consumption = gross_kwh - actual_battery_kwh_delta
        self.total_consumption_kwh = round(max(0.0, net_consumption), 4)

        return self.get_current_reading()

    def get_current_reading(self) -> TelemetryReading:
        """
        Return a validated Pydantic model of the current sensor readings.
        """
        return TelemetryReading(
            timestamp=self.timestamp.isoformat(),
            occupancy=round(self.occupancy, 2),
            outdoor_temperature=round(self.outdoor_temp, 2),
            electricity_price=round(self.electricity_price, 4),
            indoor_temperatures={k: round(v, 2) for k, v in self.indoor_temperatures.items()},
            hvac_setpoints={k: round(v, 2) for k, v in self.hvac_setpoints.items()},
            total_consumption_kwh=self.total_consumption_kwh,
            hvac_kwh=self.hvac_kwh,
            lighting_kwh=self.lighting_kwh,
            plug_loads_kwh=self.plug_loads_kwh,
            other_kwh=self.other_kwh,
            battery_dispatch_kw=round(self.battery_dispatch_kw, 2),
            battery_soc_kwh=round(self.battery_soc_kwh, 2),
            equipment_status=dict(self.equipment_status),
        )
