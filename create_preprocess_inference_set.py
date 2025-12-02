import os

from utils import *

def create_preprocess_inference_set(smoothing=False, all_year=False, winter=False):
    """
     Preprocesses inverter CSV data for inference by extracting daily features.

    Steps performed:
        1. Reads all CSV files from the 'Inverters' folder.
        2. Converts 'collectTime' to datetime format.
        3. Extracts the inverter ID from the filename.
        4. Groups data by day and checks if the inverter had an inverter_state of 768 on that day.
        5. Optionally applies moving average smoothing to numeric features (excluding inverter_state and weather features).
        6. Extracts comprehensive features for each day using `extract_comprehensive_features`.
        7. Combines all daily features into a single DataFrame.
        8. Exports the processed DataFrame to the specified output path.

    Args:
        smoothing (bool, optional): Whether to apply moving average smoothing to numeric features. Defaults to False.
        all_year (bool, optional): If True, includes data from all months.
        winter (bool, optional): If True, filters the dataset for winter months only.
    """

    # Determine the active season, its corresponding months, and a formatted name for file usage
    season_name, _, season_name_file = determine_season(all_year, winter)
    
    # Input folder
    input_folder = "Inverters"
    
    # Output file path
    output_path = f"{DATASETS_FOLDER}/inference_test_set_before_classification_{season_name_file}.csv"

    # Get all CSV files in the input folder
    csv_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith('.csv')
    ]

    results = []

    print("\n" + "=" * 60)
    print(f"PREPROCESSING TEST SET FOR INFERENCE, TRAINING SEASON: {season_name.upper()} ...")
    print("=" * 60)

    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        print("Processing file:", file)
        df = pd.read_csv(file_path)
        df['collectTime'] = pd.to_datetime(df['collectTime'])

        # Extract inverter ID from the filename
        filename_no_ext = os.path.splitext(file)[0]
        inverter_id = filename_no_ext.replace("inverter_", "")

        # Process data grouped by each day
        for day, group in df.groupby(df['collectTime'].dt.date):
            # Only consider days when the inverter was active (state 768)
            if (group['inverter_state'] == 768).any():

                # Select numeric columns to smooth, excluding inverter_state and weather features
                cols_to_smooth = [
                    col for col in df.select_dtypes(include='number').columns
                    if col not in ["inverter_state", "diffuse_radiation", "global_tilted_irradiance", "wind_speed_10m",
                                   "precipitation"]
                ]

                if smoothing:
                    # Apply moving average smoothing to the selected columns
                    group[cols_to_smooth] = (
                        group[cols_to_smooth]
                        .rolling(window=24, min_periods=1)
                        .mean()
                    )

                # Extract comprehensive features from the (smoothed) daily data
                features_dict = extract_comprehensive_features(group)

                if features_dict is not None:
                    # Convert features dictionary to a Series and add metadata
                    feature_series = pd.Series(features_dict)
                    feature_series['date'] = day
                    feature_series['ID'] = inverter_id
                    results.append(feature_series)

    # Combine all daily feature Series into a single DataFrame
    final_df = pd.DataFrame(results)

    # Reorder columns: date + ID first, then all extracted features
    cols = ['date', 'ID'] + [col for col in final_df.columns if col not in ['date', 'ID']]
    final_df = final_df[cols]
    final_df = final_df.sort_values(by=['date', 'ID'])

    # Save the final processed DataFrame to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Final results saved to {output_path}")
