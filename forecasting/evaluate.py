import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from forecasting.model import prepare_features


def evaluate_saved_model(
    data_path: str = "data/historical_consumption.csv",
    model_path: str = "forecasting/saved_model.pkl",
    test_days: int = 5,
):
    """
    Evaluate trained forecasting model on held-out test dataset.
    Produces detailed performance analysis overall and specifically during 2-5 PM peak pricing windows.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at '{model_path}'. Please train the model first.")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    use_prophet = bundle.get("use_prophet", False)

    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    X, y = prepare_features(df)

    test_intervals = test_days * 24 * 4  # e.g., 480 intervals for 5 days
    X_test = X.iloc[-test_intervals:]
    y_test = y.iloc[-test_intervals:]

    if use_prophet:
        df_p = df.reset_index().rename(columns={"timestamp": "ds", "total_consumption_kwh": "y"})
        test_p = df_p.iloc[-test_intervals:]
        forecast = model.predict(test_p)
        y_pred = forecast["yhat"].values
    else:
        y_pred = model.predict(X_test)

    eval_df = pd.DataFrame({
        "actual_kwh": y_test.values,
        "predicted_kwh": y_pred,
        "error_kwh": y_pred - y_test.values,
        "abs_error_kwh": np.abs(y_pred - y_test.values),
        "pct_error": np.abs((y_pred - y_test.values) / y_test.values) * 100.0,
    }, index=y_test.index)

    eval_df["hour"] = eval_df.index.hour
    eval_df["weekday"] = eval_df.index.weekday

    # Overall metrics
    overall_mae = mean_absolute_error(eval_df["actual_kwh"], eval_df["predicted_kwh"])
    overall_rmse = root_mean_squared_error(eval_df["actual_kwh"], eval_df["predicted_kwh"])
    overall_r2 = r2_score(eval_df["actual_kwh"], eval_df["predicted_kwh"])
    overall_mape = eval_df["pct_error"].mean()

    # Peak window metrics (14:00 - 17:00 on weekdays)
    peak_mask = (eval_df["weekday"] < 5) & (eval_df["hour"] >= 14) & (eval_df["hour"] < 17)
    peak_df = eval_df[peak_mask]

    if len(peak_df) > 0:
        peak_mae = mean_absolute_error(peak_df["actual_kwh"], peak_df["predicted_kwh"])
        peak_rmse = root_mean_squared_error(peak_df["actual_kwh"], peak_df["predicted_kwh"])
        peak_mape = peak_df["pct_error"].mean()
        peak_actual_avg = peak_df["actual_kwh"].mean()
        peak_pred_avg = peak_df["predicted_kwh"].mean()
    else:
        peak_mae = peak_rmse = peak_mape = peak_actual_avg = peak_pred_avg = 0.0

    print("==========================================================================================")
    print("                    ENERGY DEMAND FORECASTING MODEL EVALUATION REPORT                     ")
    print("==========================================================================================")
    print(f"Dataset             : {data_path} ({len(df)} total intervals)")
    print(f"Test Split          : Last {test_days} days ({len(y_test)} intervals, {y_test.index[0].strftime('%Y-%m-%d %H:%M')} to {y_test.index[-1].strftime('%Y-%m-%d %H:%M')})")
    print(f"Model Type          : {'Prophet' if use_prophet else 'GradientBoostingRegressor'}")
    print("------------------------------------------------------------------------------------------")
    print("\n[1] OVERALL TEST SET PERFORMANCE:")
    print(f"  - Mean Absolute Error (MAE)   : {overall_mae:.3f} kWh")
    print(f"  - Root Mean Sq Error (RMSE)   : {overall_rmse:.3f} kWh")
    print(f"  - Mean Abs Pct Error (MAPE)   : {overall_mape:.2f}%")
    print(f"  - R² Accuracy Score           : {overall_r2:.4f}")

    print("\n[2] 2:00 PM - 5:00 PM PEAK DEMAND WINDOW PERFORMANCE (WEEKDAYS):")
    print(f"  - Peak Window Intervals Evaluated : {len(peak_df)}")
    print(f"  - Actual Average Peak Load        : {peak_actual_avg:.2f} kWh")
    print(f"  - Predicted Average Peak Load     : {peak_pred_avg:.2f} kWh")
    print(f"  - Peak Window MAE                 : {peak_mae:.3f} kWh")
    print(f"  - Peak Window RMSE                : {peak_rmse:.3f} kWh")
    print(f"  - Peak Window MAPE                : {peak_mape:.2f}%")
    print("------------------------------------------------------------------------------------------")

    print("\n[3] SAMPLE ACTUAL VS PREDICTED SNAPSHOT (PEAK WINDOW):")
    if len(peak_df) > 0:
        sample_peak = peak_df[["actual_kwh", "predicted_kwh", "error_kwh", "pct_error"]].head(10)
        print(sample_peak.round(2).to_string())
    else:
        print("No weekday peak intervals in test split.")

    print("==========================================================================================")


if __name__ == "__main__":
    evaluate_saved_model()
