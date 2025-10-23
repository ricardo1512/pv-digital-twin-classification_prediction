import os
import pandas as pd
from globals import *

# ==============================================================
# Dataset Consolidation
# ==============================================================
def create_train_test_sets():
    """
    Consolidate and generate the final training and testing datasets for PV fault classification.

    This function loads all individual CSV sample files generated for each fault scenario
    (including the no-fault condition) from their respective folders — 2023 (training) and
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

    # Folder and output path setup
    folder_2023 = FOLDER_TRAIN_SAMPLES
    folder_2024 = FOLDER_TEST_SAMPLES
    output_file_2023 = TRAIN_VALID_SET_FILE
    output_file_2024 = TEST_SET_FILE

    print("\n" + "=" * 40)
    print("CREATING TRAIN AND TEST SETS...")
    print("=" * 40)

    # Lists to store DataFrames for each year
    dfs_2023, dfs_2024 = [], []

    # Load and store all CSV files for 2023 (training)
    for filename in os.listdir(folder_2023):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_2023, filename)
            df = pd.read_csv(filepath)
            dfs_2023.append(df)

    # Load and store all CSV files for 2024 (testing)
    for filename in os.listdir(folder_2024):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_2024, filename)
            df = pd.read_csv(filepath)
            dfs_2024.append(df)

    # Concatenate all DataFrames into single train/test datasets
    combined_df_2023 = pd.concat(dfs_2023, ignore_index=True)
    combined_df_2024 = pd.concat(dfs_2024, ignore_index=True)

    # Save the combined DataFrames to disk
    combined_df_2023.to_csv(output_file_2023, index=False)
    print(f"Combined Train CSV saved as {output_file_2023}")
    combined_df_2024.to_csv(output_file_2024, index=False)
    print(f"Combined Test CSV saved as {output_file_2024}")