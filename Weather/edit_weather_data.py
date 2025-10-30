import pandas as pd

# Define input and output file paths for the weather datasets
input_file = "Classification/original_weather_Vila_do_Conde.csv"
output_file_2023 = "Classification/Vila_do_Conde_weather_2023.csv"
output_file_2024 = "Classification/Vila_do_Conde_weather_2024.csv"

# Load the original CSV file containing the raw meteorological dataset
df = pd.read_csv(input_file)

# Define a dictionary to rename columns for consistent naming and easier reference
rename_dict = {
        'timestamp': 'date',
        'temperature_2m (°C)': 'temperature_2m',
        'wind_speed_10m (km/h)': 'wind_speed_10m',
        'global_tilted_irradiance (W/m²)': 'global_tilted_irradiance',
        'diffuse_radiation (W/m²)': 'diffuse_radiation',
        'precipitation (mm)': 'precipitation'
    }

# Rename only the columns that exist in the DataFrame to avoid potential KeyErrors
df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

# Convert the 'date' column to datetime format, localize it to UTC, and set it as the index
df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
df.set_index('date', inplace=True)

# Define the list of relevant columns to retain for further analysis
desired_columns = [
        'temperature_2m', 'wind_speed_10m',
        'global_tilted_irradiance', 'diffuse_radiation', 'precipitation'
    ]

# Retain only columns that actually exist in the DataFrame (in case some are missing)
existing_columns = [col for col in desired_columns if col in df.columns]
df = df[existing_columns]

# Ensure all selected columns are of type float64 for numerical consistency
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

# Resample the hourly dataset into 5-minute intervals
# Linear interpolation fills in intermediate values between hourly measurements
df_5min = df.resample('5min').interpolate(method='linear')

# Adjust precipitation values: evenly distribute each hourly total across twelve 5-minute intervals
df_5min['precipitation'] = df_5min['precipitation'] / 3

# Reset the index so that the datetime index becomes a regular column again
df_5min.reset_index(inplace=True)

# Separate the resampled dataset into two DataFrames corresponding to 2023 and 2024
df_2023 = df_5min[df_5min['date'].dt.year == 2023]
df_2024 = df_5min[df_5min['date'].dt.year == 2024]

# Display data type information and basic verification outputs
print("Final data types:")
print(df_5min.dtypes)
print(f"\nOriginal data shape: {df.shape}")
print(f"Resampled data shape: {df_5min.shape}")
print("\nFirst few rows of resampled data:")
print(df_5min.head(5))  # Display the first few rows to confirm the 5-minute pattern

# Save the processed 5-minute datasets into separate CSV files for each year
df_2023.to_csv(output_file_2023, index=False)
df_2024.to_csv(output_file_2024, index=False)
print(f"\nFiles saved successfully:\n"
      f"\t{output_file_2023}\n"
      f"\t{output_file_2024}")