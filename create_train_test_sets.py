import os
import pandas as pd

from globals import *
from plot_day_samples import *

# ==============================================================
# Dataset Consolidation
# ==============================================================
def create_train_test_sets():    
    """
    Consolidate and generate the final training and testing datasets for PV anomaly/fault classification.

    This function loads all individual CSV sample files generated for each anomaly/fault scenario
    (including the no-anomaly/fault condition) from their respective folders — 2023 (training) and
    2024 (testing). It then concatenates them into unified CSV files that represent the
    complete datasets used for model training and evaluation.

    Args:
        None

    Workflow:
        1. Define input folders for 2023 (training) and 2024 (testing) samples.
        2. Load all CSV files within each folder and store them in separate DataFrame lists.
        3. Concatenate all DataFrames for each year into a single dataset.
        4. Save the final consolidated training and testing datasets to disk.

    Output:
        - Combined training dataset saved to a TRAIN_VALID_SET_FILE
        - Combined testing dataset saved to a TEST_SET_FILE

    Note:
        This function should be executed after all individual sample generation processes
        have completed successfully to ensure all fault types are included in the datasets.
    """

    # Output folder and files for plots
    output_folder = f"{PLOT_FOLDER}/Day_samples"
    train_file = "trainset_2023_scatter_iv.png"
    test_file = "testset_2024_scatter_iv.png"
    
    print("\n" + "=" * 40)
    print("CREATING TRAIN AND TEST SETS...")
    print("=" * 40)

    # Lists to store DataFrames for each year
    dfs_train, dfs_test = [], []
    
    # Initialize empty DataFrames
    df_plot_train = pd.DataFrame()
    df_plot_test = pd.DataFrame()

    # Load and store all CSV files for 2023 (training)
    for filename in os.listdir(FOLDER_TRAIN_SAMPLES):
        if filename.endswith(".csv"):
            filepath = os.path.join(FOLDER_TRAIN_SAMPLES, filename)
            df = pd.read_csv(filepath)
            dfs_train.append(df)

            # Select only required columns
            df_part = df[['pv1_i_mean', 'pv1_u_mean', 'inverter_state']].copy()

            # Append directly to cumulative DataFrame
            df_plot_train = pd.concat([df_plot_train, df_part], ignore_index=True)

    # Load and store all CSV files for 2024 (testing)
    for filename in os.listdir(FOLDER_TEST_SAMPLES):
        if filename.endswith(".csv"):
            filepath = os.path.join(FOLDER_TEST_SAMPLES, filename)
            df = pd.read_csv(filepath)
            dfs_test.append(df)
            
            # Select only required columns
            df_part = df[['pv1_i_mean', 'pv1_u_mean', 'inverter_state']].copy()

            # Append directly to cumulative DataFrame
            df_plot_test = pd.concat([df_plot_test, df_part], ignore_index=True)

    # Plot combined scatter I-V for training and test data
    plot_scatter_iv(df_plot_train, output_folder, train_file)
    plot_scatter_iv(df_plot_test, output_folder, test_file)
    
    # Concatenate all DataFrames into single train/test datasets
    combined_df_train = pd.concat(dfs_train, ignore_index=True)
    combined_df_test = pd.concat(dfs_test, ignore_index=True)

    # Save the combined DataFrames to disk
    combined_df_train.to_csv(TRAIN_VALID_SET_FILE, index=False)
    print(f"Combined Train CSV saved as {TRAIN_VALID_SET_FILE}")
    combined_df_test.to_csv(TEST_SET_FILE, index=False)
    print(f"Combined Test CSV saved as {TEST_SET_FILE}")
    