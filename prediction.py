import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
import joblib

from create_ts_samples_1_soiling import *
from create_ts_samples_2_shading import *
from create_ts_samples_3_cracks import *
from plot_ts_prediction import *
from utils import *


def ts_daily_classification(csv_path, model_path, output_csv_path):
    """

    """
    print(f"Performing time series daily classification for {csv_path}...")
    
    # Carrega o modelo
    xgb_model_file_path = Path(model_path)
    if not xgb_model_file_path.exists():
        print(f"Model file not found: {xgb_model_file_path}\n\tPlease, train the model first.")
        return  # Stop the function if model is not found
    xgb_classifier = joblib.load(xgb_model_file_path)
    
    # Carrega os dados
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # Agrupa por dia
    df['day'] = df['date'].dt.date
    daily_groups = df.groupby('day')
    
    # Lista para armazenar resultados diários
    daily_prob_list = []
    
    # Itera sobre cada dia
    for day, group in daily_groups:
        print(f"\tDay: {day}")
        # Define 'date' como índice para between_time funcionar
        group = group.set_index('date')

        # Gera features diárias
        daily_df = compute_store_daily_comprehensive_features(group, day)
        
        # Seleciona features que o modelo espera e transforma em DataFrame
        X_day = daily_df[xgb_classifier.feature_names_in_].to_frame().T
        
        # Calcula probabilidades
        probs = xgb_classifier.predict_proba(X_day)
        
        # Create a DataFrame to store predicted probabilities
        columns_names = [LABELS_MAP[c][0] for c in xgb_classifier.classes_]
        day_probs = pd.DataFrame(
            probs,
            columns=columns_names
        )
        day_probs.insert(0, 'date', day)
    
        # Add to list
        daily_prob_list.append(day_probs)
    
    # Concatenação final
    df_daily_probs = pd.concat(daily_prob_list, ignore_index=True)
    
    # Salva CSV
    df_daily_probs.to_csv(output_csv_path, index=False)
    print(f"Daily class probabilities exported to: {output_csv_path}")
    

def ts_predict_days(input_csv_path, output_csv_path, 
                    threshold_start=0.5, threshold_target=0.8, threshold_class=0.2, window=30):
    """
    Predicts, for each date and each non-'Normal' class, how many days it will take 
    for the probability to reach a target threshold using linear regression 
    over a rolling window of past data.

    Args:
        input_csv_path (str): Path to the CSV file containing daily probabilities.
        output_csv_path (str): Path to save the predictions CSV.
        threshold_start (float): Minimum probability to start regression (default=0.5).
        threshold_target (float): Target probability to predict (default=0.8).
        window (int): Number of previous days to use for regression (default=30).

    Returns:
        pd.DataFrame: Predictions with columns:
            ['date', 'class', 'predicted_days_to_0.8', 'predicted_date', 'actual_class_at_predicted_day'].
    """
    # --- Load CSV ---
    df_daily_probs = pd.read_csv(input_csv_path, parse_dates=['date'])
    df_daily_probs = df_daily_probs.sort_values('date').reset_index(drop=True)

    # Identify class probability columns (exclude 'date' and 'Normal')
    classes = [c for c in df_daily_probs.columns if c not in ['date', 'Normal']]
    predictions = []

    last_day_index = len(df_daily_probs) - 1

    # --- Rolling window prediction loop ---
    for i in range(window, len(df_daily_probs)):
        current_day = df_daily_probs.at[i, 'date']
        past_window = df_daily_probs.iloc[i - window:i]

        for cls in classes:
            current_prob = df_daily_probs.at[i, cls]

            # Skip if current probability below threshold_start
            if current_prob < threshold_start:
                continue

            # Linear regression over windowed data
            X = np.arange(window).reshape(-1, 1)
            y = past_window[cls].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_

            # Skip if no upward trend
            if slope <= 0:
                continue

            # Predict day index where probability reaches target threshold
            day_to_target = (threshold_target - intercept) / slope
            relative_days = int(round(day_to_target - (window - 1)))
            predicted_index = i + relative_days

            # Ensure prediction is valid (future day within range)
            if relative_days <= 0 or predicted_index > last_day_index:
                continue

            predicted_date = df_daily_probs.at[predicted_index, 'date']
            actual_probs = df_daily_probs.iloc[predicted_index].drop('date')

            # --- Determine actual class with your 20% rule ---
            sorted_probs = actual_probs.sort_values(ascending=False)
            top_class = sorted_probs.index[0]
            top_prob = sorted_probs.iloc[0]

            if len(sorted_probs) > 1:
                second_class = sorted_probs.index[1]
                second_prob = sorted_probs.iloc[1]

                # If the second class is close (<20% diff) and equals the predicted cls
                if (top_prob - second_prob) <= threshold_class and second_class == cls:
                    actual_class = cls
                else:
                    actual_class = top_class
            else:
                actual_class = top_class

            if relative_days <= 1:
                continue

            predictions.append({
                'date': current_day,
                'class': cls,
                'predicted_days_to_0.8': relative_days,
                'predicted_date': predicted_date,
                'actual_class_at_predicted_day': actual_class,
                'slope': slope,
                'intercept': intercept
            })

    # --- Output ---
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


create_ts_samples_1_soiling(plot_samples=True)
create_ts_samples_2_shading(plot_samples=True)
create_ts_samples_3_cracks(plot_samples=True)

model_path = "Models/xgb_best_model_summer.joblib"

csv_path_1 =  "TS_samples/1_soiling/1_soiling_2023_Albergaria-a-Velha_ts_samples.csv"
csv_path_2 = "TS_samples/2_shading/2_shading_2023_Albergaria-a-Velha_ts_samples.csv"
csv_path_3 = "TS_samples/3_cracks/3_cracks_2023_Albergaria-a-Velha_ts_samples.csv"

output_csv_prob_1 = "Predictions/1_soiling_2023_Albergaria-a-Velha_ts_samples_daily_probatilities.csv"
# ts_daily_classification(csv_path_1, model_path, output_csv_prob_1)
output_csv_prob_2 = "Predictions/2_shading_2023_Albergaria-a-Velha_ts_samples_daily_probatilities.csv"
# ts_daily_classification(csv_path_2, model_path, output_csv_prob_2)
output_csv_prob_3 = "Predictions/3_cracks_2023_Albergaria-a-Velha_ts_samples_daily_probatilities.csv"
# ts_daily_classification(csv_path_3, model_path, output_csv_prob_3)

output_image_prob_1 = "Plots/TS_probabilities/1_soiling_2023_Albergaria-a-Velha_ts_samples_daily_probatilities.png"
# plot_daily_class_probabilities(output_csv_prob_1, output_image_prob_1)
output_image_prob_2 = "Plots/TS_probabilities/2_shading_2023_Albergaria-a-Velha_ts_samples_daily_probatilities.png"
# plot_daily_class_probabilities(output_csv_prob_2, output_image_prob_2)
output_image_prob_3 = "Plots/TS_probabilities/3_cracks_2023_Albergaria-a-Velha_ts_samples_daily_probatilities.png"
plot_daily_class_probabilities(output_csv_prob_3, output_image_prob_3)


output_csv_pred_1 = "Predictions/1_soiling_2023_Albergaria-a-Velha_ts_samples_daily_predictions.csv"
output_image_slope_1 = "Plots/TS_probabilities/1_soiling_2023_Albergaria-a-Velha_ts_samples_daily_probatilities_slope.png"
# df_predictions_1 = ts_predict_days(output_csv_prob_1, output_csv_pred_1, threshold_start=0.2, threshold_target=0.6, threshold_class=0.2, window=60)
"""
plot_daily_class_probabilities(output_csv_prob_1, output_image_slope_1,
                                   threshold_start=0.2, threshold_target=0.6,
                                   regressions_df=df_predictions_1, nth_regression=11)
"""
output_csv_pred_2 = "Predictions/2_shading_2023_Albergaria-a-Velha_ts_samples_daily_predictions.csv"
# df_predictions_2 = ts_predict_days(output_csv_prob_2, output_csv_pred_2, threshold_start=0.2, threshold_target=0.5, threshold_class=0.2, window=60)
output_csv_pred_3 = "Predictions/3_cracks_2023_Albergaria-a-Velha_ts_samples_daily_predictions.csv"
df_predictions_3 = ts_predict_days(output_csv_prob_3, output_csv_pred_3, threshold_start=0.2, threshold_target=0.6, threshold_class=0.2, window=80)

output_image_1 = "Predictions/1_soiling_2023_Albergaria-a-Velha_ts_samples_daily_predictions_cleveland.png"
# plot_predictions_cleveland(df_predictions_1, output_image_1)
output_image_2 = "Predictions/2_shading_2023_Albergaria-a-Velha_ts_samples_daily_predictions_cleveland.png"
# plot_predictions_cleveland(df_predictions_2, output_image_2)
output_image_3 = "Predictions/3_cracks_2023_Albergaria-a-Velha_ts_samples_daily_predictions_cleveland.png"
plot_predictions_cleveland(df_predictions_3, output_image_3)
