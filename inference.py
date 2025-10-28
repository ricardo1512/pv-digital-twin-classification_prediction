import pandas as pd
import joblib
from pathlib import Path
from plot import *
from utils import *

def inference(all_year=False, winter=False):
    """
        Run inference on a real dataset using a pre-trained Random Forest classifier.

        This function performs the following steps:
            1. Loads a pre-trained Random Forest model from disk.
            2. Reads the inference dataset CSV containing features and metadata.
            3. Selects the features required by the model.
            4. Performs classification to predict anomaly and fault labels for each sample.
            5. Saves the predictions to a CSV file, including 'ID', 'date', and 'predicted_fault'.
            6. Counts occurrences of each predicted anomaly and fault type and exports a summary report CSV.
            7. Generates a bar plot of the distribution of predicted anomalies and faults.
            8. Computes class probabilities for each sample and saves them to a CSV.
            9. Generates individual bar plots for each sample, showing the probability of being in each class.

        Args:
            all_year (bool, optional): If True, includes data from all months.
            winter (bool, optional): If True, filters the dataset for winter months only.
            
        Inputs:
            - `inference_test_file`: CSV file with test data.
            - `MODELS_FOLDER`: Folder to pre-trained Random Forest model.
            - Other output paths: CSV, image, and plot directories.

        Outputs:
            - CSV file with predicted anomaly and fault labels per sample.
            - CSV file with predicted probabilities per sample.
            - Summary report CSV with counts of each predicted class.
            - Distribution plot of predicted faults.
            - Individual bar plots of predicted probabilities for each sample.
    """
    # Determine the active season, its corresponding months, and a formatted name for file usage
    season_name, _, season_name_file = determine_season(all_year, winter)
    
    # Input file path
    inference_test_file = f"{DATASETS_FOLDER}/inference_test_set_before_classification_{season_name_file}.csv"
    
    # Output file paths
    output_results_path = f"{REPORT_FOLDER}/inference_results_{season_name_file}.csv"
    output_inference_file = f"{DATASETS_FOLDER}/inference_test_set_with_classification_{season_name_file}.csv"
    output_inference_prob_file = f"{DATASETS_FOLDER}/inference_test_set_prob_with_classification_{season_name_file}.csv"
    output_inference_image = f"{IMAGE_FOLDER}/inference_classification_distribution_{season_name_file}.png"
    output_inference_prob_folder = f"{PLOT_FOLDER}/Probabilities/Plots_inference_probabilities_{season_name_file}"
    output_inference_scaled_prob_folder = f"{PLOT_FOLDER}/Probabilities_scaled/Plots_inference_probabilities_scaled_{season_name_file}"
    output_inference_scaled_prob_report = f"{REPORT_FOLDER}/inference_adjusted_probabilities_report_{season_name_file}.csv"

    print("\n" + "=" * 60)
    print(f"PERFORMING INFERENCE, TRAINING SEASON: {season_name.upper()} ...")
    print("=" * 60)

    # Define the path to the pre-trained Random Forest model
    rf_model_file = os.path.join(MODELS_FOLDER, f"rf_best_model_{season_name_file}.joblib")
    
    # Raise an exception if the model file does not exist
    rf_model_file_path = Path(rf_model_file)
    if not os.path.exists(rf_model_file_path):
        print(
            f"\nRandom Forest model file not found: {rf_model_file}\n"
            f"\tPlease train the model first for {season_name} (--{season_name_file}).\n"
        )
        exit()

    # Load pre-trained Random Forest model
    rf_classifier = joblib.load(rf_model_file)

    # Raise an exception if the file does not exist, stopping the program
    inference_test_file_path = Path(inference_test_file)
    if not inference_test_file_path.exists():
        print(
            f"\nInference test file not found: {inference_test_file}\n"
            f"\tPlease create the inference test set first for {season_name} (--{season_name_file}).\n"
        )
        exit()
        
    # Load input dataset
    df_inference = pd.read_csv(inference_test_file)

    # Select features used by the model
    X_inference = df_inference[rf_classifier.feature_names_in_]

    # ===============================================
    # INFERENCE
    # ===============================================
    print("\nPerforming inference on the dataset...")
    y_inference_pred = rf_classifier.predict(X_inference)
    df_inference['predicted_fault'] = y_inference_pred.astype(int)

    # Reorder columns for readability
    cols = ['ID', 'date', 'predicted_fault'] + [c for c in df_inference.columns if c not in ['ID', 'date', 'predicted_fault']]
    df_inference = df_inference[cols]

    # Save classifications to CSV
    df_inference.to_csv(output_inference_file, index=False)
    print(f"Inference completed.\nResults saved to {output_inference_file}")

    # Ensure column is int before counting
    df_inference['predicted_fault'] = df_inference['predicted_fault'].astype(int)

    # Prepare class names and IDs
    all_states = list(LABELS_MAP.keys())

    # Count occurrences for each fault type
    state_counts = (
        df_inference['predicted_fault']
        .value_counts()
        .reindex(all_states, fill_value=0)
    )

    # Create a DataFrame with class names and counts
    state_counts_df = pd.DataFrame({
        'Predicted Fault': [LABELS_MAP[i][0] for i in all_states],
        'Count': state_counts.values
    })

    # Print and export
    print("\nNumber of entries per predicted fault:")
    print(state_counts_df)

    state_counts_df.to_csv(output_results_path, index=False)
    print(f"\nInference Report exported to: output_{output_results_path}")

    # Plot distribution
    plot_inference_fault_distribution(season_name, state_counts, output_inference_image)
    
    # ===============================================
    # INFERENCE CLASS PROBABILITIES
    # ===============================================
    print("Calculating class probabilities for inference dataset...")
    # Predict class probabilities for each sample in the inference set
    # Returns an array of shape (n_samples, n_classes) with probabilities for each class
    proba_inference = rf_classifier.predict_proba(X_inference)

    # Create a DataFrame to store predicted probabilities
    columns_names = [f'{LABELS_MAP[c][0]}' for c in rf_classifier.classes_]
    proba_inference_df = pd.DataFrame(
        proba_inference,
        columns=columns_names
    )

    # Insert additional metadata columns
    proba_inference_df.insert(0, 'ID', df_inference['ID'])
    proba_inference_df.insert(1, 'date', df_inference['date'])
    proba_inference_df.insert(2, 'predicted_fault', y_inference_pred.astype(int))

    # Save the probabilities and metadata to a CSV file for further analysis
    proba_inference_df.to_csv(output_inference_prob_file, index=False)
    print(f"Class probabilities for inference dataset saved to {output_inference_prob_file}")
    # Generate bar plots for each sample's fault probabilities
    # plot_inference_fault_probabilities(proba_inference_df, season_name_file, output_inference_prob_folder) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(f"Class probabilities plots saved to {output_inference_prob_folder}")
    
    # Adjust and scale probabilities for decision making
    adjusted_proba_df = adjust_and_scale_probabilities(proba_inference_df, delta=0.2, top=2)
    # Create a report of adjusted probabilities
    report_adjusted_proba_df = generate_adjusted_probabilities_report(adjusted_proba_df)
    # Save the adjusted probabilities report to CSV
    report_adjusted_proba_df.to_csv(output_inference_scaled_prob_report, index=False)
    print(f"\nAdjusted probabilities report saved to: {output_inference_scaled_prob_report}")
    # Generate bar plots for each sample's scaled fault probabilities    
    # plot_inference_fault_probabilities(adjusted_proba_df, season_name_file, output_inference_scaled_prob_folder) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(f"Scaled class probabilities plots saved to {output_inference_scaled_prob_folder}")
