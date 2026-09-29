import os
import pandas as pd
from simulator.generators import (
    generate_occupancy,
    generate_outdoor_temperature,
    generate_electricity_price,
    generate_baseline_consumption,
    generate_equipment_breakdown,
)


def build_historical_dataset(
    start_date: str = "2026-06-01 00:00:00",
    days: int = 14,
    output_path: str = "data/historical_consumption.csv"
) -> pd.DataFrame:
    """
    Generate a 2-week simulated IoT telemetry dataset for an office building at 15-minute resolution,
    and save it to CSV format.
    """
    timestamps = pd.date_range(
        start=start_date,
        periods=days * 24 * 4,  # 15-min intervals over specified days
        freq="15min"
    )

    print(f"Generating simulated IoT telemetry for {days} days ({len(timestamps)} intervals)...")

    # Generate synthetic telemetry feeds
    occupancy = generate_occupancy(timestamps)
    outdoor_temp = generate_outdoor_temperature(timestamps)
    electricity_price = generate_electricity_price(timestamps)
    baseline_consumption = generate_baseline_consumption(timestamps, occupancy, outdoor_temp)
    equipment_breakdown = generate_equipment_breakdown(baseline_consumption)

    # Assemble into unified DataFrame
    df = pd.DataFrame({
        "occupancy": occupancy,
        "outdoor_temperature": outdoor_temp,
        "electricity_price": electricity_price,
        "total_consumption_kwh": baseline_consumption,
        "hvac_kwh": equipment_breakdown["hvac_kwh"],
        "lighting_kwh": equipment_breakdown["lighting_kwh"],
        "plug_loads_kwh": equipment_breakdown["plug_loads_kwh"],
        "other_kwh": equipment_breakdown["other_kwh"],
    }, index=timestamps)

    df.index.name = "timestamp"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path)
    print(f"Dataset successfully saved to '{output_path}'. Shape: {df.shape}")

    return df


if __name__ == "__main__":
    build_historical_dataset()
