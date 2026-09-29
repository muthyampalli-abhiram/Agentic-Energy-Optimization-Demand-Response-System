import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from forecasting.model import forecast_next_hours


def evaluate_multistep_forecasting(
    data_path: str = "data/historical_consumption.csv",
    model_path: str = "forecasting/saved_model.pkl",
    peak_threshold_kwh: float = 110.0,
    test_days: int = 5,
):
    """
    Perform multi-step iterative forecasting evaluation anchored at 1:00 PM on weekdays.
    
    Evaluates:
    1. Horizon-by-horizon error degradation (30m, 1h, 2h, 3h, 4h ahead) without peeking at actuals.
    2. Early-warning detection capability: Whether the 1:00 PM forecast successfully predicts
       peak load exceeding the threshold during the 2:00 PM - 5:00 PM window.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Historical dataset not found at '{data_path}'.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at '{model_path}'.")

    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")

    # Select test set cutoff based on test_days
    cutoff_time = df.index[-1] - pd.Timedelta(days=test_days)
    test_df = df[df.index >= cutoff_time]

    # Find distinct weekdays in test dataset
    unique_dates = sorted(list(set(test_df.index.date)))
    weekday_dates = [d for d in unique_dates if d.weekday() < 5]

    if not weekday_dates:
        print("No weekdays found in test split. Using all available weekdays in dataset...")
        all_dates = sorted(list(set(df.index.date)))
        weekday_dates = [d for d in all_dates if d.weekday() < 5]

    results_by_horizon = {
        "30m_ahead": {"step": 2, "time": "13:30", "actuals": [], "preds": []},
        "1h_ahead":  {"step": 4, "time": "14:00", "actuals": [], "preds": []},
        "2h_ahead":  {"step": 8, "time": "15:00", "actuals": [], "preds": []},
        "3h_ahead":  {"step": 12, "time": "16:00", "actuals": [], "preds": []},
        "4h_ahead":  {"step": 16, "time": "17:00", "actuals": [], "preds": []},
    }

    daily_early_warning_reports = []
    all_multi_step_actuals = []
    all_multi_step_preds = []

    for date_val in weekday_dates:
        # Anchor at 1:00 PM (13:00) on date_val
        anchor_timestamp = pd.Timestamp(f"{date_val} 13:00:00")
        
        # Verify anchor exists in dataset
        if anchor_timestamp not in df.index:
            continue

        # History strictly up to 1:00 PM (no future peeking)
        history_up_to_1pm = df.loc[:anchor_timestamp]

        # Forecast 4 hours ahead (16 steps: 13:15 to 17:00)
        forecast_df = forecast_next_hours(history_up_to_1pm, hours_ahead=4, model_path=model_path)

        # Actuals from 13:15 to 17:00
        forecast_end_ts = pd.Timestamp(f"{date_val} 17:00:00")
        actual_subset = df.loc[anchor_timestamp + pd.Timedelta(minutes=15) : forecast_end_ts]

        if len(actual_subset) < 16:
            continue

        pred_vals = forecast_df["predicted_consumption_kwh"].values[:16]
        actual_vals = actual_subset["total_consumption_kwh"].values[:16]

        all_multi_step_actuals.extend(actual_vals)
        all_multi_step_preds.extend(pred_vals)

        # Record horizon benchmark predictions
        # Horizon steps (1-indexed step in 16-step forecast):
        # 13:30 -> step 2
        # 14:00 -> step 4
        # 15:00 -> step 8
        # 16:00 -> step 12
        # 17:00 -> step 16
        for key, info in results_by_horizon.items():
            step_idx = info["step"] - 1  # 0-indexed array
            info["actuals"].append(actual_vals[step_idx])
            info["preds"].append(pred_vals[step_idx])

        # Early Warning Evaluation for 2:00 PM - 5:00 PM window (steps 4 through 16)
        peak_actual_window = actual_vals[3:16]  # 14:00 to 17:00 inclusive
        peak_pred_window = pred_vals[3:16]

        max_actual_peak = float(np.max(peak_actual_window))
        max_pred_peak = float(np.max(peak_pred_window))

        actual_exceeded = max_actual_peak >= peak_threshold_kwh
        pred_exceeded = max_pred_peak >= peak_threshold_kwh

        if actual_exceeded and pred_exceeded:
            status = "TRUE POSITIVE (Success)"
        elif not actual_exceeded and not pred_exceeded:
            status = "TRUE NEGATIVE (No Alarm)"
        elif pred_exceeded and not actual_exceeded:
            status = "FALSE POSITIVE (False Alarm)"
        else:
            status = "FALSE NEGATIVE (Missed Warning)"

        daily_early_warning_reports.append({
            "date": str(date_val),
            "actual_peak_kwh": max_actual_peak,
            "forecasted_peak_kwh": max_pred_peak,
            "threshold_kwh": peak_threshold_kwh,
            "actual_exceeded": actual_exceeded,
            "warning_triggered": pred_exceeded,
            "status": status,
        })

    print("==========================================================================================")
    print("         ITERATIVE MULTI-STEP DEMAND FORECAST EVALUATION (ANCHORED AT 1:00 PM)            ")
    print("==========================================================================================")
    print(f"Evaluated Weekdays : {len(weekday_dates)} days")
    print(f"Forecast Horizon   : 4 Hours ahead (1:15 PM - 5:00 PM, 16 steps of 15 min)")
    print(f"Early Warning Thresh: {peak_threshold_kwh} kWh peak demand")
    print("------------------------------------------------------------------------------------------")

    print("\n[1] HORIZON-BY-HORIZON ERROR DEGRADATION TABLE:")
    horizon_rows = []
    for key, info in results_by_horizon.items():
        acts = np.array(info["actuals"])
        preds = np.array(info["preds"])
        mae = mean_absolute_error(acts, preds)
        rmse = root_mean_squared_error(acts, preds)
        mape = np.mean(np.abs((preds - acts) / acts)) * 100.0
        horizon_rows.append({
            "Forecast Horizon": key.replace("_", " ").title(),
            "Target Time": info["time"],
            "MAE (kWh)": round(mae, 3),
            "RMSE (kWh)": round(rmse, 3),
            "MAPE (%)": round(mape, 2),
        })

    horizon_df = pd.DataFrame(horizon_rows)
    print(horizon_df.to_string(index=False))

    # Overall 4-hour iterative forecast error
    tot_mae = mean_absolute_error(all_multi_step_actuals, all_multi_step_preds)
    tot_rmse = root_mean_squared_error(all_multi_step_actuals, all_multi_step_preds)
    tot_mape = np.mean(np.abs((np.array(all_multi_step_preds) - np.array(all_multi_step_actuals)) / np.array(all_multi_step_actuals))) * 100.0

    print(f"\nOverall 4-Hour Multi-Step Iterative MAE : {tot_mae:.3f} kWh")
    print(f"Overall 4-Hour Multi-Step Iterative RMSE: {tot_rmse:.3f} kWh")
    print(f"Overall 4-Hour Multi-Step Iterative MAPE: {tot_mape:.2f}%")

    print("------------------------------------------------------------------------------------------")
    print("\n[2] 1:00 PM EARLY-WARNING PEAK DETECTION RESULTS (2:00 PM - 5:00 PM WINDOW):")
    ew_df = pd.DataFrame(daily_early_warning_reports)
    print(ew_df[["date", "actual_peak_kwh", "forecasted_peak_kwh", "actual_exceeded", "warning_triggered", "status"]].to_string(index=False))

    # Confusion matrix summary
    tp = sum(1 for r in daily_early_warning_reports if r["status"] == "TRUE POSITIVE (Success)")
    tn = sum(1 for r in daily_early_warning_reports if r["status"] == "TRUE NEGATIVE (No Alarm)")
    fp = sum(1 for r in daily_early_warning_reports if r["status"] == "FALSE POSITIVE (False Alarm)")
    fn = sum(1 for r in daily_early_warning_reports if r["status"] == "FALSE NEGATIVE (Missed Warning)")
    acc = (tp + tn) / max(1, len(daily_early_warning_reports)) * 100.0

    print("\n--- Early Warning Summary Metrics ---")
    print(f"  - Total Weekdays Evaluated : {len(daily_early_warning_reports)}")
    print(f"  - True Positives (Success) : {tp}")
    print(f"  - True Negatives (No Alarm): {tn}")
    print(f"  - False Positives (False)  : {fp}")
    print(f"  - False Negatives (Missed) : {fn}")
    print(f"  - Early Warning Accuracy   : {acc:.1f}%")
    print("==========================================================================================")


if __name__ == "__main__":
    evaluate_multistep_forecasting()
