import os
import pandas as pd
from globals import *
from plot_day_samples import *

def real_data_visualisation(smoothing=False):
    
    # Input folder
    input_folder = "Inverters"
    
    # Output file path
    plot_folder = f"{PLOT_FOLDER}/Real_data"
    
    # Get all CSV files in the input folder
    csv_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith('.csv')
    ]
    
    print("\n" + "=" * 60)
    print(f"CREATING PLOTS FOR REAL DATA VISUALISATION ...")
    print("=" * 60)
    
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        print("\nProcessing file:", file)
        df = pd.read_csv(file_path)
        df['collectTime'] = pd.to_datetime(df['collectTime'])

        # Extract inverter ID from the filename
        filename_no_ext = os.path.splitext(file)[0]
        inverter_id = filename_no_ext.replace("inverter_", "")

        # Process data grouped by each day
        for date, group in df.groupby(df['collectTime'].dt.date):
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
                
                # Fix: ensure index is datetime
                group = group.set_index('collectTime')
                smoothed = "_smoothed" if smoothing else ""
                condition_name = "Real Data"
                output_image = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_{inverter_id}_inverter{smoothed}"
                plot_mppt(group, date, condition_name, plot_folder, output_image, soiling=True)
                plot_currents(group, date,condition_name, plot_folder, output_image, soiling=True)
                plot_voltage(group, date, condition_name, plot_folder, output_image)
                exit()

real_data_visualisation(smoothing=True)