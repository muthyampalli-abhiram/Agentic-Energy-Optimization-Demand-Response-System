import os
import pickle
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct time-series regressor features and target for energy load forecasting.

    Features generated:
    - Time components: hour, minute, dayofweek, is_weekend
    - Exogenous telemetry: occupancy, outdoor_temperature, temp_deviation_21, electricity_price
    - Lagged consumption: lag_1 (15m), lag_2 (30m), lag_4 (1h), lag_96 (24h)
    - Rolling window statistics: rolling_mean_4, rolling_std_4, rolling_mean_96
    """
    data = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.set_index("timestamp")

    # Time features
    data["hour"] = data.index.hour
    data["minute"] = data.index.minute
    data["dayofweek"] = data.index.dayofweek
    data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

    # Exogenous thermal deviation
    data["temp_deviation_21"] = np.maximum(0.0, data["outdoor_temperature"] - 21.0)

    # Lag features of total consumption
    data["lag_1"] = data["total_consumption_kwh"].shift(1)
    data["lag_2"] = data["total_consumption_kwh"].shift(2)
    data["lag_4"] = data["total_consumption_kwh"].shift(4)
    data["lag_96"] = data["total_consumption_kwh"].shift(96)

    # Rolling window statistics
    data["rolling_mean_4"] = data["total_consumption_kwh"].shift(1).rolling(4).mean()
    data["rolling_std_4"] = data["total_consumption_kwh"].shift(1).rolling(4).std().fillna(0.0)
    data["rolling_mean_96"] = data["total_consumption_kwh"].shift(1).rolling(96).mean()

    # Drop rows with NaN resulting from initial lag windows
    data = data.dropna()

    feature_cols = [
        "hour",
        "minute",
        "dayofweek",
        "is_weekend",
        "occupancy",
        "outdoor_temperature",
        "temp_deviation_21",
        "electricity_price",
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_96",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_96",
    ]

    X = data[feature_cols]
    y = data["total_consumption_kwh"]

    return X, y


def train_and_save_model(
    data_path: str = "data/historical_consumption.csv",
    model_path: str = "forecasting/saved_model.pkl",
    use_prophet: bool = False,
    test_days: int = 2,
) -> Dict[str, Any]:
    """
    Train demand forecasting model on historical telemetry data, evaluate on held-out test split,
    and save trained model bundle to disk.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Historical dataset not found at '{data_path}'. Build dataset first.")

    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")

    print(f"Loaded dataset from '{data_path}' with {len(df)} rows.")

    if use_prophet:
        return _train_prophet_model(df, model_path, test_days)
    else:
        return _train_gbr_model(df, model_path, test_days)


def _train_gbr_model(df: pd.DataFrame, model_path: str, test_days: int) -> Dict[str, Any]:
    X, y = prepare_features(df)

    test_intervals = test_days * 24 * 4  # 192 intervals for 2 days
    if len(X) <= test_intervals:
        raise ValueError("Dataset is too short for the requested test split.")

    X_train, X_test = X.iloc[:-test_intervals], X.iloc[-test_intervals:]
    y_train, y_test = y.iloc[:-test_intervals], y.iloc[-test_intervals:]

    print(f"Training GBR model (Train size: {len(X_train)}, Test size: {len(X_test)})...")

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("\n--- GBR Model Test Evaluation ---")
    print(f"  MAE  : {mae:.3f} kWh")
    print(f"  RMSE : {rmse:.3f} kWh")
    print(f"  R²   : {r2:.4f}")

    bundle = {
        "model": model,
        "feature_names": list(X_train.columns),
        "use_prophet": False,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "last_historical_data": df.tail(150),  # Save recent context for multi-step lag calculation
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Model successfully saved to '{model_path}'.")
    return bundle


def _train_prophet_model(df: pd.DataFrame, model_path: str, test_days: int) -> Dict[str, Any]:
    try:
        from prophet import Prophet
    except ImportError as e:
        raise ImportError("Prophet package is not installed. Install prophet or set use_prophet=False.") from e

    print("Training Prophet forecasting model...")
    df_p = df.reset_index().rename(columns={"timestamp": "ds", "total_consumption_kwh": "y"})

    test_intervals = test_days * 24 * 4
    train_p = df_p.iloc[:-test_intervals]
    test_p = df_p.iloc[-test_intervals:]

    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.add_regressor("occupancy")
    m.add_regressor("outdoor_temperature")
    m.add_regressor("electricity_price")

    m.fit(train_p)

    forecast = m.predict(test_p)
    y_test = test_p["y"].values
    y_pred = forecast["yhat"].values

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("\n--- Prophet Model Test Evaluation ---")
    print(f"  MAE  : {mae:.3f} kWh")
    print(f"  RMSE : {rmse:.3f} kWh")
    print(f"  R²   : {r2:.4f}")

    bundle = {
        "model": m,
        "use_prophet": True,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "last_historical_data": df.tail(150),
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Prophet model saved to '{model_path}'.")
    return bundle


def forecast_next_hours(
    history_df: pd.DataFrame,
    hours_ahead: int = 3,
    model_path: str = "forecasting/saved_model.pkl"
) -> pd.DataFrame:
    """
    Generate iterative multi-step predictions for the next N hours (15-min intervals).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at '{model_path}'. Train model first.")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    use_prophet = bundle.get("use_prophet", False)

    steps = hours_ahead * 4  # 4 intervals per hour
    last_timestamp = history_df.index[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(minutes=15),
        periods=steps,
        freq="15min"
    )

    if use_prophet:
        # Prophet multi-step forecast
        future_df = pd.DataFrame({"ds": future_timestamps})
        # Use generators or last values for regressors
        from simulator.generators import generate_occupancy, generate_outdoor_temperature, generate_electricity_price
        future_df["occupancy"] = generate_occupancy(future_timestamps).values
        future_df["outdoor_temperature"] = generate_outdoor_temperature(future_timestamps).values
        future_df["electricity_price"] = generate_electricity_price(future_timestamps).values

        forecast = model.predict(future_df)
        forecast_df = pd.DataFrame({
            "timestamp": future_timestamps,
            "predicted_consumption_kwh": forecast["yhat"].values,
            "occupancy": future_df["occupancy"].values,
            "outdoor_temperature": future_df["outdoor_temperature"].values,
            "electricity_price": future_df["electricity_price"].values,
        }).set_index("timestamp")
        return forecast_df

    # Standard GBR multi-step recursive forecasting
    from simulator.generators import generate_occupancy, generate_outdoor_temperature, generate_electricity_price

    future_occ = generate_occupancy(future_timestamps)
    future_temp = generate_outdoor_temperature(future_timestamps)
    future_price = generate_electricity_price(future_timestamps)

    # Maintain running history buffer for computing lags
    history_buffer = history_df.copy()

    predictions = []

    for i, ts in enumerate(future_timestamps):
        ts_idx = pd.DatetimeIndex([ts])
        occ_val = float(future_occ.iloc[i])
        temp_val = float(future_temp.iloc[i])
        price_val = float(future_price.iloc[i])

        # Compute lag features from history buffer
        lag_1 = float(history_buffer["total_consumption_kwh"].iloc[-1])
        lag_2 = float(history_buffer["total_consumption_kwh"].iloc[-2]) if len(history_buffer) >= 2 else lag_1
        lag_4 = float(history_buffer["total_consumption_kwh"].iloc[-4]) if len(history_buffer) >= 4 else lag_1
        lag_96 = float(history_buffer["total_consumption_kwh"].iloc[-96]) if len(history_buffer) >= 96 else lag_1

        rolling_mean_4 = float(history_buffer["total_consumption_kwh"].iloc[-4:].mean())
        rolling_std_4 = float(history_buffer["total_consumption_kwh"].iloc[-4:].std())
        if np.isnan(rolling_std_4):
            rolling_std_4 = 0.0
        rolling_mean_96 = float(history_buffer["total_consumption_kwh"].iloc[-96:].mean()) if len(history_buffer) >= 96 else rolling_mean_4

        feature_row = pd.DataFrame([{
            "hour": ts.hour,
            "minute": ts.minute,
            "dayofweek": ts.dayofweek,
            "is_weekend": int(ts.dayofweek >= 5),
            "occupancy": occ_val,
            "outdoor_temperature": temp_val,
            "temp_deviation_21": max(0.0, temp_val - 21.0),
            "electricity_price": price_val,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_4": lag_4,
            "lag_96": lag_96,
            "rolling_mean_4": rolling_mean_4,
            "rolling_std_4": rolling_std_4,
            "rolling_mean_96": rolling_mean_96,
        }])

        pred_val = float(model.predict(feature_row[bundle["feature_names"]])[0])
        pred_val = max(0.0, pred_val)
        predictions.append(pred_val)

        # Append prediction to history buffer for next lag steps
        new_row = pd.DataFrame([{
            "occupancy": occ_val,
            "outdoor_temperature": temp_val,
            "electricity_price": price_val,
            "total_consumption_kwh": pred_val,
        }], index=[ts])
        history_buffer = pd.concat([history_buffer, new_row])

    forecast_df = pd.DataFrame({
        "predicted_consumption_kwh": predictions,
        "occupancy": future_occ.values,
        "outdoor_temperature": future_temp.values,
        "electricity_price": future_price.values,
    }, index=future_timestamps)

    forecast_df.index.name = "timestamp"
    return forecast_df


if __name__ == "__main__":
    train_and_save_model()
