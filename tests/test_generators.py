import pytest
import pandas as pd
import numpy as np
from simulator.generators import (
    generate_occupancy,
    generate_outdoor_temperature,
    generate_electricity_price,
    generate_baseline_consumption,
    generate_equipment_breakdown,
)
from simulator.build_dataset import build_historical_dataset


@pytest.fixture
def one_week_timestamps():
    return pd.date_range("2026-06-01 00:00:00", periods=7 * 24 * 4, freq="15min")


@pytest.fixture
def sample_dataset(tmp_path):
    csv_file = tmp_path / "test_consumption.csv"
    df = build_historical_dataset(start_date="2026-06-01 00:00:00", days=14, output_path=str(csv_file))
    return df


def test_row_count(one_week_timestamps, sample_dataset):
    # 7 days * 24 hrs/day * 4 intervals/hr = 672
    assert len(one_week_timestamps) == 672
    # 14 days dataset should have 1344 rows
    assert len(sample_dataset) == 1344


def test_no_negative_values(sample_dataset):
    # Verify no negative values exist across all numeric columns
    numeric_cols = [
        "occupancy",
        "outdoor_temperature",
        "electricity_price",
        "total_consumption_kwh",
        "hvac_kwh",
        "lighting_kwh",
        "plug_loads_kwh",
        "other_kwh",
    ]
    for col in numeric_cols:
        assert (sample_dataset[col] >= 0).all(), f"Found negative values in column {col}"


def test_occupancy_range(one_week_timestamps):
    occupancy = generate_occupancy(one_week_timestamps)
    assert (occupancy >= 0.0).all() and (occupancy <= 100.0).all()


def test_peak_consumption_higher_than_offpeak(sample_dataset):
    # Ensure weekday 2-5 PM (14:00 - 17:00) average consumption is higher than off-peak hours
    df = sample_dataset.copy()
    df["weekday"] = df.index.weekday
    df["hour"] = df.index.hour

    # Weekday peak mask (2 PM to 5 PM)
    weekday_peak = df[(df["weekday"] < 5) & (df["hour"] >= 14) & (df["hour"] < 17)]

    # Weekday off-peak mask (e.g. overnight 00:00 to 07:00)
    weekday_offpeak = df[(df["weekday"] < 5) & (df["hour"] < 7)]

    peak_avg_consumption = weekday_peak["total_consumption_kwh"].mean()
    offpeak_avg_consumption = weekday_offpeak["total_consumption_kwh"].mean()

    assert peak_avg_consumption > offpeak_avg_consumption, (
        f"Peak average consumption ({peak_avg_consumption:.2f} kWh) should be strictly greater than "
        f"off-peak average consumption ({offpeak_avg_consumption:.2f} kWh)"
    )


def test_equipment_breakdown_sum(sample_dataset):
    # Verify sum of component kWh matches total consumption kWh
    component_sum = (
        sample_dataset["hvac_kwh"]
        + sample_dataset["lighting_kwh"]
        + sample_dataset["plug_loads_kwh"]
        + sample_dataset["other_kwh"]
    )
    np.testing.assert_allclose(component_sum, sample_dataset["total_consumption_kwh"], rtol=1e-4)
