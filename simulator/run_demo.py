import sys
from pathlib import Path
import pandas as pd

# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from simulator.live_feed import LiveBuildingSensor


def run_live_feed_demo():
    """
    Demonstrate real-time IoT polling across the critical 2:00 PM - 5:00 PM peak demand window.
    Simulates applying demand-response control interventions (thermostat setpoint adjustment,
    lighting dimming, and battery dispatch) to visually confirm their effect on subsequent readings.
    """
    print("==========================================================================================")
    print("         DEMAND RESPONSE & LIVE IoT SENSOR SIMULATION DEMO (2:00 PM - 5:00 PM PEAK)       ")
    print("==========================================================================================")

    # Initialize live sensor on a Monday afternoon at 1:00 PM (13:00)
    start_time = pd.Timestamp("2026-06-01 13:00:00")
    sensor = LiveBuildingSensor(initial_time=start_time)

    # Simulation timeline: 1:00 PM to 5:30 PM in 15-minute intervals
    timestamps = pd.date_range("2026-06-01 13:00:00", "2026-06-01 17:30:00", freq="15min")

    print(f"\nInitial State at {start_time.strftime('%H:%M')}:")
    reading = sensor.get_current_reading()
    print(f"  Electricity Price : ${reading.electricity_price:.2f}/kWh")
    print(f"  Indoor Setpoints  : {reading.hvac_setpoints}")
    print(f"  Indoor Temps      : {reading.indoor_temperatures}")
    print(f"  Total Consumption : {reading.total_consumption_kwh:.2f} kWh (HVAC: {reading.hvac_kwh:.2f} kWh)")

    header = (
        f"{'Time':<8} | {'Price':<7} | {'Occ %':<6} | {'Out T':<6} | {'In Temp(Z1/Z2/Z3)':<20} | "
        f"{'Setpoint':<9} | {'HVAC kWh':<9} | {'Battery kW':<10} | {'Net Load kWh':<12} | {'Intervention Notes':<25}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))

    for ts in timestamps:
        notes = []

        # At 1:45 PM (13:45): Proactive Pre-Peak Interventions applied by Energy Agent!
        if ts == pd.Timestamp("2026-06-01 13:45:00"):
            sensor.set_thermostat("all", 24.0)
            sensor.shift_load("lighting_dim", 180)  # 3 hours dimming during peak window
            sensor.dispatch_battery(40.0)  # Discharge 40 kW (10 kWh / 15-min)
            notes.append("[DR ACTIVATED]: Setpoint 24C, Dim Lights, Battery 40kW")

        # Tick the live sensor forward to current timestamp
        rd = sensor.tick(ts)

        in_temps_str = f"{rd.indoor_temperatures['Zone_1']:.1f}/{rd.indoor_temperatures['Zone_2']:.1f}/{rd.indoor_temperatures['Zone_3']:.1f}"
        setpoint_str = f"{rd.hvac_setpoints['Zone_1']:.1f}C"
        note_str = " | ".join(notes) if notes else ""

        print(
            f"{ts.strftime('%H:%M'):<8} | "
            f"${rd.electricity_price:<6.2f} | "
            f"{rd.occupancy:<6.1f} | "
            f"{rd.outdoor_temperature:<6.1f} | "
            f"{in_temps_str:<20} | "
            f"{setpoint_str:<9} | "
            f"{rd.hvac_kwh:<9.2f} | "
            f"{rd.battery_dispatch_kw:<10.1f} | "
            f"{rd.total_consumption_kwh:<12.2f} | "
            f"{note_str}"
        )

    print("-" * len(header))
    print("\nSimulation complete. Key Observations:")
    print("1. Electricity price jumps to $0.45/kWh between 14:00 and 17:00 (Peak Window).")
    print("2. DR intervention at 13:45 immediately drops HVAC load and net grid consumption.")
    print("3. Battery discharge (40 kW) offsets 10 kWh of grid consumption per 15-min interval.")
    print("4. Zone indoor temperatures drift safely toward 24°C while keeping occupants comfortable.")
    print("==========================================================================================")


if __name__ == "__main__":
    run_live_feed_demo()
