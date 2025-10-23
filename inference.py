import pandas as pd
import joblib
from plot import *

def inference():
    """
        Run inference on a new dataset using a pre-trained Random Forest model.

        Loads the model, predicts fault labels, saves results to CSV,
        exports a summary of class counts, and generates a distribution plot.

        Args:
            inference_test_file (str): Path to the CSV file with test data.

        Outputs:
            CSV with predictions, summary report, and class distribution plot.
    """

    # Input file path
    inference_test_file = "Datasets/inference_test_set_before_classification.csv"
    
    # Output file paths
    output_inference_file = "Datasets/inference_test_set_with_predictions.csv"
    output_inference_image = "Images/inference_predictions_distribution.png"
    results_path = "Reports/inference_results.csv"

    print("\n" + "=" * 40)
    print("PERFORMING INFERENCE...")
    print("=" * 40)

    # Load pre-trained Random Forest model
    rf_classifier = joblib.load(MODEL_PATH)

    # Load input dataset
    df_inference = pd.read_csv(inference_test_file)

    # Select features used by the model
    X_inference = df_inference[rf_classifier.feature_names_in_]

    # Predict fault labels
    y_inference_pred = rf_classifier.predict(X_inference)
    df_inference['fault'] = y_inference_pred.astype(int)

    # Reorder columns for readability
    cols = ['ID', 'date', 'fault'] + [c for c in df_inference.columns if c not in ['ID', 'date', 'fault']]
    df_inference = df_inference[cols]

    # Save predictions to CSV
    df_inference.to_csv(output_inference_file, index=False)
    print(f"Inference completed.\nResults saved to {output_inference_file}")

    # Ensure column is int before counting
    df_inference['fault'] = df_inference['fault'].astype(int)

    # Prepare class names and IDs
    all_states = list(LABELS_MAP.keys())

    # Count occurrences for each fault type
    state_counts = (
        df_inference['fault']
        .value_counts()
        .reindex(all_states, fill_value=0)
    )

    # Create a DataFrame with class names and counts
    state_counts_df = pd.DataFrame({
        'Fault': [LABELS_MAP[i][0] for i in all_states],
        'Count': state_counts.values
    })

    # Print and export
    print("\nNumber of entries per fault:")
    print(state_counts_df)

    state_counts_df.to_csv(results_path, index=False)
    print(f"\nInference Report exported to: {results_path}")

    # Plot distribution
    plot_inference_fault_distribution(state_counts, output_inference_image)
