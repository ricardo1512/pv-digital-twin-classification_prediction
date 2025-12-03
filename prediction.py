import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
import joblib

from globals import *
from plot_ts_prediction import *
from utils import *


def ts_daily_classification(input_file, output_csv_path=None, smooth=False, model_path=XGB_BEST_MODEL_SUMMER):
    """
    Performs daily classification of time-series data using a pre-trained model
    and plots the resulting daily class probabilities.

    Args:
        input_file (str or Path): Path to the CSV file containing time-series data.
        output_csv_path (str or Path, optional): Path to save the resulting daily probabilities CSV.
        smooth (bool): If True, applies moving average smoothing to numeric features.
        model_path (str): Path to the pre-trained model (joblib file).

    Returns:
        None
    """
    
    input_file = Path(input_file)
    print(f"Performing time series daily classification for {input_file}...")

    # Generate default output CSV path if not provided
    if output_csv_path is None:
        output_csv_path = Path(PREDICTIONS_FOLDER) / "real_data_probabilities" / f"{input_file.stem}_daily_probabilities.csv"
    output_csv_path = Path(output_csv_path)
    
    # Load the pre-trained model
    model_file_path = Path(model_path)
    if not model_file_path.exists():
        print(f"Model file not found: {model_file_path}\n\tPlease, train a summer model first.")
        return
    classifier = joblib.load(model_file_path)
    
    # Load CSV
    df = pd.read_csv(input_file)

    # Normalize time column
    if 'date' in df.columns:
        pass
    elif 'collectTime' in df.columns:
        df = df.rename(columns={'collectTime': 'date'})
    else:
        raise ValueError(f"No valid time column found in {input_file}. Expected 'date' or 'collectTime'.")

    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.date
    daily_groups = df.groupby('day')

    daily_prob_list = []
    
    for day, group in daily_groups:
        print(f"\tDay: {day}")
        group = group.set_index('date')
        
        # Smooth numeric features if needed
        if smooth:
            cols_to_smooth = [
                col for col in df.select_dtypes(include='number').columns
                if col not in ["inverter_state", "diffuse_radiation", "global_tilted_irradiance", 
                               "wind_speed_10m", "precipitation"]
            ]
            group[cols_to_smooth] = group[cols_to_smooth].rolling(window=48, min_periods=1).mean()

        daily_df = compute_store_daily_comprehensive_features(group, day)
        X_day = daily_df[classifier.feature_names_in_].to_frame().T
        probs = classifier.predict_proba(X_day)
        
        columns_names = [LABELS_MAP[c][0] for c in classifier.classes_]
        day_probs = pd.DataFrame(probs, columns=columns_names)
        day_probs.insert(0, 'date', day)
        daily_prob_list.append(day_probs)
    
    # Concatenate all daily probabilities
    df_daily_probs = pd.concat(daily_prob_list, ignore_index=True)
    df_daily_probs.to_csv(output_csv_path, index=False)
    print(f"Daily class probabilities exported to: {output_csv_path}")

    # Plot daily probabilities
    output_image_prob = Path(PLOT_FOLDER) / "TS_probabilities" / f"{output_csv_path.stem}.png"
    output_image_prob.parent.mkdir(parents=True, exist_ok=True)
    plot_daily_class_probabilities(output_csv_path, output_image_prob)
    print(f"Daily probabilities plot saved to: {output_image_prob}")

    
def synthetic_ts_daily_classification():
    """
    Applies daily time-series classification to all raw synthetic CSV files in TS_SAMPLES_FOLDER.
    """
    print("\nStarting multiple time-series daily classifications...")

    base_folder = Path(TS_SAMPLES_FOLDER)
    for subfolder in [f for f in base_folder.iterdir() if f.is_dir() and not f.name.startswith("real_data")]:
        print(f"\tProcessing subfolder: {subfolder.name}")
        
        for file_path in subfolder.glob("*.csv"):
            print(f"\t\tProcessing file: {file_path.name}")
            ts_daily_classification(file_path, subfolder.name + "_probabilities")
            

def ts_predict_days(input_csv_path, output_csv_path=None, 
                    threshold_start=0.5, threshold_target=0.8, threshold_class=0.2, window=30):
    """
    Predicts, for each date and each non-'Normal' class, how many days it will take 
    for the probability to reach a target threshold using linear regression 
    over a rolling window of past data, and plots the results.
    
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
    df_daily_probs = pd.read_csv(input_csv_path)
    
    # Generate default output CSV path if not provided
    if output_csv_path is None:
        output_csv_path = Path(PREDICTIONS_FOLDER) / "real_data_predictions" / f"{input_csv_path.stem}_daily_predictions.csv"
    output_csv_path = Path(output_csv_path)
    
    # Detect and normalize the time column to 'date'
    if 'date' in df_daily_probs.columns:
        pass 
    elif 'collectTime' in df_daily_probs.columns:
        df_daily_probs = df_daily_probs.rename(columns={'collectTime': 'date'})
    else:
        raise ValueError(f"No valid time column found in {input_csv_path}. Expected 'date' or 'collectTime'.")
    
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

    # Plot predictions
    output_image_prob = Path(PLOT_FOLDER) / "TS_predictions" / f"{input_csv_path.stem}_predictions_cleveland.png"
    plot_predictions_cleveland(df_predictions, output_image_prob)
    print(f"Prediction plot saved to: {output_image_prob}")
    
    return df_predictions
            


def single_ts_predict_days(
    input_file,
    output_subfolder_name=None,
    threshold_start=0.5,
    threshold_target=0.8,
    threshold_class=0.2,
    window=30
):
    """
    Applies daily time-series prediction to a single probability CSV file and
    generates the corresponding prediction CSV and plot.

    Args:
        input_file (str or Path): Path to the probability CSV file.
        output_subfolder_name (str): Optional subfolder name inside PREDICTIONS_FOLDER
                                     where the predictions will be saved.
        threshold_start (float): Start threshold for anomaly prediction.
        threshold_target (float): Target threshold for anomaly prediction.
        threshold_class (float): Class threshold for anomaly prediction.
        window (int): Rolling window size.
    """

    # Convert to Path object
    input_file = Path(input_file)
    name = input_file.stem

    # Determine output subfolder
    if output_subfolder_name is None:
        subfolder_output = Path(PREDICTIONS_FOLDER) / "real_data_predictions"
    else:
        subfolder_output = Path(PREDICTIONS_FOLDER) / output_subfolder_name

    # Construct output CSV path for predictions
    output_csv_path = subfolder_output / f"{name}_daily_predictions.csv"

    print(f"Processing file: {input_file.name}")

    # Run prediction
    df_predictions = ts_predict_days(
        str(input_file),
        str(output_csv_path),
        threshold_start=threshold_start,
        threshold_target=threshold_target,
        threshold_class=threshold_class,
        window=window,
    )

    # Create output image path
    output_image_prob = Path(PLOT_FOLDER) / "TS_predictions" / f"{name}_predictions_cleveland.png"

    # Plot results
    plot_predictions_cleveland(df_predictions, output_image_prob)


def synthetic_ts_predict_days(
        threshold_start=0.5,
        threshold_target=0.8,
        threshold_class=0.2,
        window=30
    ):
    """
    Applies daily time-series prediction across all probability CSV files
    located inside subfolders ending with '_probabilities'.
    """
    
    print("\nStarting multiple time-series daily predictions...")

    # Base folder where probability subfolders are stored
    base_folder = Path(PREDICTIONS_FOLDER)

    # Iterate over subfolders that contain probability CSV files
    for subfolder in [f for f in base_folder.iterdir() if f.is_dir() and f.name.endswith('_probabilities') and not f.name.startswith("real_data")]:

        print(f"\tProcessing subfolder: {subfolder.name}")

        # Replace 'probabilities' with 'predictions' to create the output subfolder
        output_subfolder_name = subfolder.name.replace("probabilities", "predictions")

        # Iterate over all probability CSV files inside the subfolder
        for file_path in subfolder.glob("*.csv"):
            print(f"\t\tProcessing file: {file_path.name}")
            
            
            print(f"\t\t\tThresholds: start={threshold_start}, target={threshold_target}, class={threshold_class}, window={window}")
            exit() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # Call the single-file prediction function
            single_ts_predict_days(
                file_path,
                output_subfolder_name=output_subfolder_name,
                threshold_start=threshold_start,
                threshold_target=threshold_target,
                threshold_class=threshold_class,
                window=window
            )