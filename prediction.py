import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
import joblib

from globals import *
from plot_ts_prediction import *
from utils import *


def ts_daily_classification(csv_path, output_csv_path, real_tag=False, model_path=XGB_BEST_MODEL_SUMMER):
    """
    Performs daily classification of time-series data using a pre-trained model.
    
    The function reads input CSV data, generates daily features, applies the model,
    and outputs daily class probabilities. Each day's probabilities are stored in 
    a DataFrame, which is then concatenated and exported to a CSV.

    Args:
        csv_path (str): Path to the CSV file containing time-series data.
        model_path (str): Path to the pre-trained model (joblib file).
        output_csv_path (str): Path to save the resulting daily class probabilities CSV.

    Returns:
        None: Saves the daily probabilities CSV to the specified path.
    """
    
    print(f"\nPerforming time series daily classification for {csv_path}...")
    
    # Load the pre-trained model
    model_file_path = Path(model_path)
    if not model_file_path.exists():
        print(f"Model file not found: {model_file_path}\n\tPlease, train a summer model first.")
        return  # Stop the function if model is not found
    classifier = joblib.load(model_file_path)
    
    # Load CSV normally first
    df = pd.read_csv(csv_path)

    # Detect and normalize the time column to 'date'
    if 'date' in df.columns:
        pass  # nothing to change
    elif 'collectTime' in df.columns:
        df = df.rename(columns={'collectTime': 'date'})
    else:
        raise ValueError(f"No valid time column found in {csv_path}. Expected 'date' or 'collectTime'.")

    # Parse 'date'
    df['date'] = pd.to_datetime(df['date'])

    # Group data by day
    df['day'] = df['date'].dt.date
    daily_groups = df.groupby('day')

    # Initialize list to store daily probabilities
    daily_prob_list = []
    
    # Iterate over each day
    for day, group in daily_groups:
        print(f"\tDay: {day}")
        
        # Set 'date' as index for time-based operations
        group = group.set_index('date')
        
        # Smooth numeric features with moving average if real data
        if real_tag:
            # Select only numeric columns except 'inverter_state' and weather features
            cols_to_smooth = [
                col for col in df.select_dtypes(include='number').columns
                if col not in ["inverter_state", "diffuse_radiation", "global_tilted_irradiance", "wind_speed_10m",
                                    "precipitation"]
            ]

            # Apply moving average smoothing only to those columns
            group[cols_to_smooth] = (
                group[cols_to_smooth]
                .rolling(window=48, min_periods=1)
                .mean()
            )

        # Compute daily features for the current day
        daily_df = compute_store_daily_comprehensive_features(group, day)
        
        # Select features expected by the model
        X_day = daily_df[classifier.feature_names_in_].to_frame().T
        
        # Predict class probabilities for the day
        probs = classifier.predict_proba(X_day)
        
        # Create DataFrame for predicted probabilities
        columns_names = [LABELS_MAP[c][0] for c in classifier.classes_]
        day_probs = pd.DataFrame(
            probs,
            columns=columns_names
        )
        day_probs.insert(0, 'date', day)
    
        # Append daily probabilities to the list
        daily_prob_list.append(day_probs)
    
    # Concatenate all daily probabilities into a single DataFrame
    df_daily_probs = pd.concat(daily_prob_list, ignore_index=True)
    
    # Export final daily probabilities CSV
    df_daily_probs.to_csv(output_csv_path, index=False)
    print(f"Daily class probabilities exported to: {output_csv_path}")
    

def ts_predict_days(input_csv_path, output_csv_path, 
                    threshold_start=0.5, threshold_target=0.8, threshold_class=0.2, window=30):
    """
    Predicts, for each date and each non-'Normal' class, how many days it will take 
    for the probability to reach a target threshold using linear regression 
    over a rolling window of past data.
    
    This function iterates over a daily probability time series for each class,
    applies a rolling-window linear regression, and estimates when the probability 
    will reach the target threshold. It also determines the actual class at the 
    predicted day using a threshold-based tie-breaking rule.

    Args:
        input_csv_path (str): Path to the CSV file containing daily probabilities.
        output_csv_path (str): Path to save the predictions CSV.
        threshold_start (float): Minimum probability to start regression (default=0.5).
        threshold_target (float): Target probability to predict (default=0.8).
        window (int): Number of previous days to use for regression (default=30).

    Returns:
        pd.DataFrame: Predictions with columns:
            ['date', 'class', 'predicted_days_to_X', 'predicted_date', 'actual_class_at_predicted_day'].
    """
    
    # Load daily probability CSV
    df_daily_probs = pd.read_csv(input_csv_path, parse_dates=['date'])
    df_daily_probs = df_daily_probs.sort_values('date').reset_index(drop=True)

    # Identify class probability columns (exclude 'date' and 'Normal')
    classes = [c for c in df_daily_probs.columns if c not in ['date', 'Normal']]
    predictions = []

    last_day_index = len(df_daily_probs) - 1 # Last valid index for prediction

    # Iterate over rolling window
    print(f"\nPerforming predictions on daily probabilities with "
          f"threshold_start={threshold_start}, threshold_target={threshold_target}, "
          f"threshold_class={threshold_class}, window={window}...")
    for i in range(window, len(df_daily_probs)):
        current_day = df_daily_probs.at[i, 'date']
        past_window = df_daily_probs.iloc[i - window:i]

        for cls in classes:
            current_prob = df_daily_probs.at[i, cls]

            # Skip regression if current probability is below start threshold
            if current_prob < threshold_start:
                continue

            # Fit linear regression
            X = np.arange(window).reshape(-1, 1)
            y = past_window[cls].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_

            # Skip if no upward trend
            if slope <= 0:
                continue

            # Predict day when target probability is reached
            day_to_target = (threshold_target - intercept) / slope
            relative_days = int(round(day_to_target - (window - 1)))
            predicted_index = i + relative_days

            # Skip if prediction is invalid or in the past
            if relative_days <= 0 or predicted_index > last_day_index:
                continue

            predicted_date = df_daily_probs.at[predicted_index, 'date']
            actual_probs = df_daily_probs.iloc[predicted_index].drop('date')

            # Determine actual class at predicted day using threshold-based tie-breaking rule
            sorted_probs = actual_probs.sort_values(ascending=False)
            top_class = sorted_probs.index[0]
            top_prob = sorted_probs.iloc[0]

            # Apply tie-breaking if second-highest class is close
            if len(sorted_probs) > 1:
                second_class = sorted_probs.index[1]
                second_prob = sorted_probs.iloc[1]

                if (top_prob - second_prob) <= threshold_class and second_class == cls:
                    actual_class = cls
                else:
                    actual_class = top_class
                    
            else:
                actual_class = top_class

            # Skip trivial predictions (less than 1 day)
            if relative_days <= 1:
                continue
            
            # Append prediction
            predictions.append({
                'date': current_day,
                'class': cls,
                'predicted_days_to_0.8': relative_days,
                'predicted_date': predicted_date,
                'actual_class_at_predicted_day': actual_class,
                'slope': slope,
                'intercept': intercept
            })

    # Export predictions
    if not predictions:
        print("No valid predictions were generated.")
        return pd.DataFrame(columns=[
            'date', 'class', 'predicted_days_to_0.8', 'predicted_date',
            'actual_class_at_predicted_day', 'slope', 'intercept'
        ])

    df_predictions = pd.DataFrame(predictions)
    df_predictions.sort_values(by=['date', 'class'], inplace=True)

    df_predictions.to_csv(output_csv_path, index=False)
    print(f"Predictions exported to: {output_csv_path}")
    
    # Print success rate
    if not df_predictions.empty:
        correct_preds = (df_predictions['class'] == df_predictions['actual_class_at_predicted_day']).sum()
        total_preds = len(df_predictions)
        success_pct = correct_preds / total_preds * 100
        print(f"Prediction success: {correct_preds}/{total_preds} ({success_pct:.2f}%)")

    return df_predictions


def multiple_ts_daily_classification():
    """
    Applies daily time-series classification to raw CSV files with different anomalies.
    """
    
    # Iterate over each subfolder in the base folder
    
    print("Starting multiple time-series daily classifications...")
    
    base_folder = Path(TS_SAMPLES_FOLDER)
    for subfolder in [f for f in base_folder.iterdir() if f.is_dir()]:
        real_tag = subfolder.name.startswith(('real_data'))
        if not real_tag:
            continue
        print(f"Processing subfolder: {subfolder.name}")

        # Iterate over all CSV files in the current subfolder
        for file_path in subfolder.glob("*.csv"):
            # Extract file name without extension
            name = file_path.stem

            # Create corresponding output subfolder
            subfolder_output = Path(PREDICTIONS_FOLDER) / (subfolder.name + "_probabilities")
            os.makedirs(subfolder_output, exist_ok=True)

            # Construct output CSV path
            output_csv_path = subfolder_output / f"{name}_daily_probabilities.csv"

            print(f"\tProcessing file: {file_path.name}")
            # Call the daily classification function
            ts_daily_classification(str(file_path), str(output_csv_path), real_tag=real_tag)
            
            output_image_prob = f"{PLOT_FOLDER}/TS_probabilities/{name}_daily_probabilities.png"
            plot_daily_class_probabilities(output_csv_path, output_image_prob)
            
            # exit()

multiple_ts_daily_classification()

def multiple_ts_predict_days(threshold_start=0.5, threshold_target=0.8, threshold_class=0.2, window=30):
    """
    Applies daily time-series prediction to probability CSV files with different anomalies.
    """
    
    print("Starting multiple time-series daily predictions...")
    
    # Iterate over each subfolder in the base folder
    base_folder = Path(PREDICTIONS_FOLDER)
    for subfolder in [f for f in base_folder.iterdir() if f.is_dir() and f.name.endswith('_probabilities')]:
        if not subfolder.name.startswith(('real_data')):
            continue
        print(f"\nProcessing subfolder: {subfolder.name}")

        # Iterate over all CSV files in the current subfolder
        for file_path in subfolder.glob("*.csv"):
            # Extract file name without extension
            name = file_path.stem

            # Create corresponding output subfolder
            subfolder_name = subfolder.name.replace('probabilities', 'predictions')
            subfolder_output = Path(PREDICTIONS_FOLDER) / subfolder_name
            os.makedirs(subfolder_output, exist_ok=True)

            # Construct output CSV path
            output_csv_path = subfolder_output / f"{name}_daily_predictions.csv"

            print(f"\tProcessing file: {file_path.name}")
            # Call the daily classification function
            df_predictions = ts_predict_days(
                str(file_path), 
                str(output_csv_path), 
                threshold_start=threshold_start, 
                threshold_target=threshold_target, 
                threshold_class=threshold_class, 
                window=window,
            )
            
            output_image_prob = f"{PLOT_FOLDER}/TS_predictions/{name}_predictions_cleveland.png"
            plot_predictions_cleveland(df_predictions, output_image_prob)
            
    
multiple_ts_predict_days(threshold_start=0.5, threshold_target=0.8, threshold_class=0.2, window=30)
    