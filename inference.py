import pandas as pd
import joblib
from pathlib import Path

from utils import *
from plot_inference import *

def inference(all_year=False, winter=False):
    """
    Run inference on a real dataset using a pre-trained XGBoost classifier.

    This function performs the following steps:
        1. Loads a pre-trained XGBoost model from disk.
        2. Reads the inference dataset CSV containing features and metadata.
        3. Selects the features required by the model.
        4. Performs classification to predict condition labels for each sample.
        5. Saves the predictions to a CSV file, including 'ID', 'date', and 'predicted_condition'.
        6. Counts occurrences of each predicted condition type and exports a summary report CSV.
        7. Generates a bar plot of the distribution of predicted conditions.
        8. Computes class probabilities for each sample and saves them to a CSV.
        9. Generates individual bar plots for each sample, showing the probability of being in each class.
        10. Selects and scales probabilities for decision making.
        11. Generates a report of selected and scaled probability combinations and exports it to CSV.
        12. Generates bar plots for each sample showing selected and scaled probabilities for further analysis.

    Args:
        all_year (bool, optional): If True, includes data from all months.
        winter (bool, optional): If True, filters the dataset for winter months only.
            
    Inputs:
        - `inference_test_file`: CSV file with test data.
        - `MODELS_FOLDER`: Folder to pre-trained XGBoost model.
        - Other output paths: CSV, image, and plot directories.

    Outputs:
        - CSV file with predicted condition labels per sample.
        - CSV file with predicted probabilities per sample.
        - Summary report CSV with counts of each predicted class.
        - Distribution plot of predicted conditions.
        - Individual bar plots of predicted probabilities for each sample.
        - Selected and scaled probabilities CSV per sample.
        - Report CSV of selected and scaled probability combinations sorted by combination size and count.
        - Bar plots of selected and scaled probabilities for each sample.
    """
    
    # Determine the active season, its corresponding months, and a formatted name for file usage
    season_name, _, season_name_file = determine_season(all_year, winter)
    
    # ===============================================
    # INITIALIZATION
    # ===============================================
    # INPUT FILE
    # Pretrained Model
    xgb_model_path = f"{MODELS_FOLDER}/xgb_best_model_{season_name_file}.joblib"
    # Preprocessed inference test set
    inference_test_file = f"{DATASETS_FOLDER}/inference_test_set_before_classification_{season_name_file}.csv"
    
    # OUTPUT FILES AND FOLDERS
    # Results
    output_results_path = f"{REPORT_FOLDER}/xgb_inference_results_{season_name_file}.csv"
    # Inference classifications
    output_inference_path = f"{DATASETS_FOLDER}/xgb_inference_test_set_with_classification_{season_name_file}.csv"
    output_inference_image = f"{IMAGE_FOLDER}/xgb_inference_classification_distribution_{season_name_file}.png"
    # Inference probabilities
    output_inference_prob_path = f"{REPORT_FOLDER}/xgb_inference_test_set_with_prob_classification_{season_name_file}.csv"
    output_inference_prob_folder = f"{PLOT_FOLDER}/Inference_probabilities/Plots_inference_probabilities_{season_name_file}"
    output_inference_scaled_prob_path = f"{REPORT_FOLDER}/xgb_inference_adjusted_probabilities_report_{season_name_file}.csv"
    output_inference_scaled_prob_folder = f"{PLOT_FOLDER}/Inference_probabilities_scaled/Plots_inference_probabilities_scaled_{season_name_file}"

    print("\n" + "=" * 60)
    print(f"PERFORMING INFERENCE, TRAINING SEASON: {season_name.upper()} ...")
    print("=" * 60)
    
    # ===============================================
    # LOAD PRE-TRAINED XGBOOST MODEL
    # ===============================================
    # Raise an exception if the model file does not exist
    xgb_model_file_path = Path(xgb_model_path)
    if not os.path.exists(xgb_model_file_path):
        print(
            f"\nXGBoost model file not found: {xgb_model_path}\n"
            f"\tPlease train the model first for {season_name} (--{season_name_file}).\n"
        )
        exit()

    # Load pre-trained XGBoost model
    xgb_classifier = joblib.load(xgb_model_path)

    # Raise an exception if the file does not exist, stopping the program
    inference_test_file_path = Path(inference_test_file)
    if not inference_test_file_path.exists():
        print(
            f"\nInference test file not found: {inference_test_file}\n"
            f"\tPlease create the inference test set first for {season_name} (--{season_name_file}).\n"
        )
        exit()

    # ===============================================
    # INFERENCE
    # ===============================================
    # Load input dataset
    df_inference = pd.read_csv(inference_test_file)

    # Select features used by the model
    X_inference = df_inference[xgb_classifier.feature_names_in_]
    
    print("\nPerforming inference on the dataset...")
    # Predict condition labels for each sample in the inference set
    y_inference_pred = xgb_classifier.predict(X_inference)
    df_inference['predicted_condition'] = y_inference_pred.astype(int)

    # Reorder columns for readability
    cols = ['ID', 'date', 'predicted_condition'] + [c for c in df_inference.columns if c not in ['ID', 'date', 'predicted_condition']]
    df_inference = df_inference[cols]

    # Save classifications to CSV
    df_inference.to_csv(output_inference_path, index=False)
    print(f"Inference completed.\nResults saved to {output_inference_path}")

    # Ensure column is int before counting
    df_inference['predicted_condition'] = df_inference['predicted_condition'].astype(int)

    # Prepare class names and IDs
    all_states = list(LABELS_MAP.keys())

    # Count occurrences for each condition type
    state_counts = (
        df_inference['predicted_condition']
        .value_counts()
        .reindex(all_states, fill_value=0)
    )

    # Create a DataFrame with class names and counts
    state_counts_df = pd.DataFrame({
        'Predicted Condition': [LABELS_MAP[i][0] for i in all_states],
        'Count': state_counts.values
    })

    # Print and export
    print("\nNumber of entries per predicted condition:")
    print(state_counts_df)

    state_counts_df.to_csv(output_results_path, index=False)
    print(f"\nInference Report exported to: output_{output_results_path}")

    # Plot distribution
    plot_inference_condition_distribution(season_name, state_counts, output_inference_image)
    
    # ===============================================
    # INFERENCE CLASS PROBABILITIES
    # ===============================================
    print("Calculating class probabilities for inference dataset...")
    # Predict class probabilities for each sample in the inference set
    # Returns an array of shape (n_samples, n_classes) with probabilities for each class
    proba_inference = xgb_classifier.predict_proba(X_inference)

    # Create a DataFrame to store predicted probabilities
    columns_names = [LABELS_MAP[c][0] for c in xgb_classifier.classes_]
    proba_inference_df = pd.DataFrame(
        proba_inference,
        columns=columns_names
    )

    # Insert additional metadata columns
    proba_inference_df.insert(0, 'ID', df_inference['ID'])
    proba_inference_df.insert(1, 'date', df_inference['date'])
    proba_inference_df.insert(2, 'predicted_condition', y_inference_pred.astype(int))
    
    # RAW PROBABILITIES
    # Save the probabilities and metadata to a CSV file for further analysis
    proba_inference_df.to_csv(output_inference_prob_path, index=False)
    print(f"Class probabilities for inference dataset saved to {output_inference_prob_path}")
    # Generate bar plots for each sample's condition probabilities
    plot_inference_condition_probabilities(proba_inference_df, season_name_file, output_inference_prob_folder)
    print(f"Class probabilities plots saved to {output_inference_prob_folder}")
    
    # SELECTED AND SCALED PROBABILITIES FOR DECISION MAKING
    # Select and scale probabilities for decision making
    adjusted_proba_df = adjust_and_scale_probabilities(proba_inference_df, delta=0.2, top=2)
    # Create a report of adjusted probabilities
    report_adjusted_proba_df = generate_adjusted_probabilities_report(adjusted_proba_df)
    # Save the selected and scaled probabilities report to CSV
    report_adjusted_proba_df.to_csv(output_inference_scaled_prob_path, index=False)
    print(f"\nAdjusted probabilities report saved to: {output_inference_scaled_prob_path}")
    # Generate bar plots for each sample's selected and scaled condition probabilities    
    plot_inference_condition_probabilities(adjusted_proba_df, season_name_file, output_inference_scaled_prob_folder, adjusted=True)
    print(f"Scaled class probabilities plots saved to {output_inference_scaled_prob_folder}")
