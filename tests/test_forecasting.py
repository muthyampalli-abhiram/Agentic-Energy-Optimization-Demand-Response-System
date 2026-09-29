import pytest
import os
import pandas as pd
import numpy as np
from forecasting.model import prepare_features, train_and_save_model, forecast_next_hours
from forecasting.evaluate_multistep import evaluate_multistep_forecasting


@pytest.fixture
def sample_telemetry_df():
    timestamps = pd.date_range("2026-06-01 00:00:00", periods=500, freq="15min")
    from simulator.generators import generate_occupancy, generate_outdoor_temperature, generate_electricity_price, generate_baseline_consumption
    occ = generate_occupancy(timestamps)
    temp = generate_outdoor_temperature(timestamps)
    price = generate_electricity_price(timestamps)
    cons = generate_baseline_consumption(timestamps, occ, temp)

    df = pd.DataFrame({
        "occupancy": occ,
        "outdoor_temperature": temp,
        "electricity_price": price,
        "total_consumption_kwh": cons,
    }, index=timestamps)
    df.index.name = "timestamp"
    return df


def test_prepare_features(sample_telemetry_df):
    X, y = prepare_features(sample_telemetry_df)

    assert not X.empty
    assert len(X) == len(y)
    expected_cols = [
        "hour", "minute", "dayofweek", "is_weekend",
        "occupancy", "outdoor_temperature", "temp_deviation_21",
        "electricity_price", "lag_1", "lag_2", "lag_4", "lag_96",
        "rolling_mean_4", "rolling_std_4", "rolling_mean_96"
    ]
    for col in expected_cols:
        assert col in X.columns
    assert not X.isnull().values.any()


def test_train_and_forecast_pipeline(sample_telemetry_df, tmp_path):
    csv_file = tmp_path / "test_historical.csv"
    model_file = tmp_path / "test_model.pkl"
    sample_telemetry_df.to_csv(csv_file)

    bundle = train_and_save_model(data_path=str(csv_file), model_path=str(model_file), test_days=2)

    assert os.path.exists(model_file)
    assert bundle["metrics"]["mae"] >= 0.0
    assert bundle["metrics"]["r2"] > 0.8  # Strong fit on simulated building physics

    # Test multi-step forecast for next 3 hours (12 steps)
    forecast_df = forecast_next_hours(sample_telemetry_df, hours_ahead=3, model_path=str(model_file))

    assert len(forecast_df) == 12
    assert "predicted_consumption_kwh" in forecast_df.columns
    assert (forecast_df["predicted_consumption_kwh"] > 0).all()


def test_multistep_eval_function(sample_telemetry_df, tmp_path):
    csv_file = tmp_path / "test_historical.csv"
    model_file = tmp_path / "test_model.pkl"
    sample_telemetry_df.to_csv(csv_file)
    train_and_save_model(data_path=str(csv_file), model_path=str(model_file), test_days=2)

    # Should run without error
    evaluate_multistep_forecasting(data_path=str(csv_file), model_path=str(model_file), test_days=2)
