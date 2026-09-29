import numpy as np
import pandas as pd


def generate_occupancy(timestamps: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    """
    Generate realistic synthetic building occupancy percentage (0-100%) at 15-minute resolution.
    Follows a typical weekday office pattern:
    - Near 0% overnight and on weekends
    - Ramping up between ~8-9 AM on weekdays
    - Peak midday plateau (~85-95%) with a slight lunch dip
    - Tapering off by ~6 PM down to near 0%
    """
    rng = np.random.default_rng(seed)
    occupancy = []

    for ts in timestamps:
        # Weekend check: 5 = Saturday, 6 = Sunday
        if ts.weekday() >= 5:
            # Low background occupancy (e.g. security/janitorial/minimal staff)
            val = rng.uniform(0.0, 3.0)
        else:
            hour = ts.hour + ts.minute / 60.0

            if hour < 7.0:
                # Overnight: 0-2%
                val = rng.uniform(0.0, 2.0)
            elif 7.0 <= hour < 9.0:
                # Morning arrival ramp (7 AM to 9 AM)
                progress = (hour - 7.0) / 2.0
                # Smooth sigmoid-like ramp from ~2% to ~85%
                val = 2.0 + 83.0 * (1.0 / (1.0 + np.exp(-6 * (progress - 0.5))))
            elif 9.0 <= hour < 12.0:
                # Morning peak work hours
                val = rng.uniform(85.0, 95.0)
            elif 12.0 <= hour < 13.0:
                # Lunch dip
                val = rng.uniform(70.0, 80.0)
            elif 13.0 <= hour < 17.0:
                # Afternoon peak work hours
                val = rng.uniform(85.0, 95.0)
            elif 17.0 <= hour < 19.5:
                # Evening departure taper (5 PM to 7:30 PM)
                progress = (hour - 17.0) / 2.5
                val = 90.0 * (1.0 - (1.0 / (1.0 + np.exp(-6 * (progress - 0.5)))))
            else:
                # Late evening / overnight
                val = rng.uniform(0.0, 2.0)

            # Add minor noise
            val += rng.normal(0.0, 1.0)

        occupancy.append(val)

    occupancy_series = pd.Series(np.clip(occupancy, 0.0, 100.0), index=timestamps, name="occupancy")
    return occupancy_series


def generate_outdoor_temperature(timestamps: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    """
    Generate daily sinusoidal outdoor temperature curve (°C) peaking mid-afternoon (~3 PM / 15:00).
    Includes random noise and subtle day-to-day weather variation.
    """
    rng = np.random.default_rng(seed)
    temps = []

    # Map timestamps to days for day-to-day weather fluctuation
    day_indices = (timestamps - timestamps[0]).days

    for ts, day_idx in zip(timestamps, day_indices):
        hour = ts.hour + ts.minute / 60.0

        # Sinusoidal diurnal wave peaking at 15:00 (3 PM), minimum at 03:00 (3 AM)
        # sin((15 - 9)/24 * 2pi) = sin(pi/2) = 1 (Peak)
        # sin((3 - 9)/24 * 2pi) = sin(-pi/2) = -1 (Min)
        diurnal_sine = np.sin((hour - 9.0) / 24.0 * 2.0 * np.pi)

        # Multi-day weather trend (varies by +/- 2.5 °C over several days)
        day_trend = 2.5 * np.sin(day_idx / 2.5)

        base_mean_temp = 23.0 + day_trend
        amplitude = 6.5  # Temp range: base_mean - 6.5 to base_mean + 6.5

        temp_val = base_mean_temp + amplitude * diurnal_sine + rng.normal(0.0, 0.4)
        temps.append(temp_val)

    return pd.Series(temps, index=timestamps, name="outdoor_temperature")


def generate_electricity_price(timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Generate Time-of-Use (TOU) electricity pricing ($/kWh):
    - Weekdays:
        - Critical Peak Rate: $0.45 / kWh from 2:00 PM to 5:00 PM (14:00 - 17:00)
        - Mid-Peak Rate: $0.22 / kWh from 7:00 AM - 2:00 PM and 5:00 PM - 10:00 PM
        - Off-Peak Rate: $0.12 / kWh overnight (10:00 PM - 7:00 AM)
    - Weekends:
        - Flat Off-Peak Rate: $0.12 / kWh
    """
    prices = []

    for ts in timestamps:
        if ts.weekday() >= 5:
            # Weekend off-peak rate
            price = 0.12
        else:
            hour = ts.hour + ts.minute / 60.0
            if 14.0 <= hour < 17.0:
                # Critical Peak Window (2 PM - 5 PM)
                price = 0.45
            elif (7.0 <= hour < 14.0) or (17.0 <= hour < 22.0):
                # Mid-peak window
                price = 0.22
            else:
                # Off-peak window
                price = 0.12

        prices.append(price)

    return pd.Series(prices, index=timestamps, name="electricity_price")


def generate_baseline_consumption(
    timestamps: pd.DatetimeIndex,
    occupancy: pd.Series,
    outdoor_temp: pd.Series,
    seed: int = 42
) -> pd.Series:
    """
    Generate total building energy consumption (kWh per 15-minute interval).
    Modeled as a function of:
    - Constant base load (standby power, servers, emergency systems)
    - Occupancy-driven load (plug loads, lighting, human activity)
    - HVAC thermal load (proportional to outdoor temp deviation from comfort target ~21°C)
    - Peak load driver: ensures consumption clearly peaks between 2-5 PM on weekdays.
    """
    rng = np.random.default_rng(seed)

    base_load = 45.0  # Base kWh / 15-min interval

    # Occupancy contribution (up to ~35 kWh at 100% occupancy)
    occupancy_load = (occupancy / 100.0) * 35.0

    # HVAC load: increases non-linearly with outdoor temperature above ~21°C
    temp_diff = np.maximum(0.0, outdoor_temp.values - 21.0)
    hvac_load = 3.5 * (temp_diff ** 1.25)

    # Thermal inertia & solar radiation peak boost during 2-5 PM weekdays
    weekday_mask = (timestamps.weekday < 5).astype(int)
    hour_decimal = timestamps.hour + timestamps.minute / 60.0
    peak_mask = ((hour_decimal >= 14.0) & (hour_decimal < 17.0)).astype(int)
    afternoon_boost = 15.0 * weekday_mask * peak_mask

    # Random operational fluctuation
    noise = rng.normal(0.0, 1.2, size=len(timestamps))

    total_consumption = base_load + occupancy_load + hvac_load + afternoon_boost + noise
    total_consumption = np.maximum(0.0, total_consumption)

    return pd.Series(total_consumption, index=timestamps, name="baseline_consumption")


def generate_equipment_breakdown(consumption: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Split total building consumption (kWh) into equipment components:
    - HVAC %
    - Lighting %
    - Plug loads %
    - Other %

    Returns a DataFrame with columns: ['hvac_kwh', 'lighting_kwh', 'plug_loads_kwh', 'other_kwh']
    """
    rng = np.random.default_rng(seed)
    
    # Calculate baseline proportions
    # Higher overall building consumption typically implies higher HVAC intensity
    c_vals = consumption.values
    c_norm = (c_vals - c_vals.min()) / (c_vals.max() - c_vals.min() + 1e-6)

    # Dynamic HVAC share: 38% at min load up to 52% at peak load
    hvac_pct = 0.38 + 0.14 * c_norm + rng.normal(0.0, 0.01, size=len(consumption))
    hvac_pct = np.clip(hvac_pct, 0.30, 0.60)

    # Lighting share: ~18%
    lighting_pct = 0.18 + rng.normal(0.0, 0.005, size=len(consumption))
    lighting_pct = np.clip(lighting_pct, 0.12, 0.25)

    # Plug loads share: ~24%
    plug_pct = 0.24 + rng.normal(0.0, 0.005, size=len(consumption))
    plug_pct = np.clip(plug_pct, 0.18, 0.30)

    # Other share: remainder
    other_pct = 1.0 - (hvac_pct + lighting_pct + plug_pct)
    other_pct = np.maximum(0.03, other_pct)

    # Re-normalize to sum to 1.0
    total_pct = hvac_pct + lighting_pct + plug_pct + other_pct
    hvac_pct /= total_pct
    lighting_pct /= total_pct
    plug_pct /= total_pct
    other_pct /= total_pct

    # Calculate kWh for each component
    hvac_kwh = np.round(c_vals * hvac_pct, 4)
    lighting_kwh = np.round(c_vals * lighting_pct, 4)
    plug_loads_kwh = np.round(c_vals * plug_pct, 4)
    
    # Assign remainder to 'other_kwh' to guarantee exact sum equal to total consumption
    other_kwh = np.round(c_vals - (hvac_kwh + lighting_kwh + plug_loads_kwh), 4)

    df = pd.DataFrame({
        'hvac_kwh': hvac_kwh,
        'lighting_kwh': lighting_kwh,
        'plug_loads_kwh': plug_loads_kwh,
        'other_kwh': other_kwh
    }, index=consumption.index)

    return df
