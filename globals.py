# ==================================================================================
# Global Configuration for PV System Anomaly and Fault Classification and Prediction
# ==================================================================================

# ==============================
# Location Settings
# ==============================
LOCAL = "Vila do Conde"                         # Location of the PV plant
LATITUDE = 41.337579                            # Latitude of the LOCAL
LONGITUDE = -8.689926                           # Longitude of the LOCAL

# ==============================
# System Configuration
# ==============================
MODULES_PER_STRING = 10                         # Number of modules per string

# ==============================
# Variables and Folders
# ==============================
LABEL = 'inverter_state'                        # Target label for classification
YEARS = [2023, 2024]                            # Available years with data
WEATHER_FOLDER_CLASS = "Weather/Classification" # Folder to the weather data for classification
WEATHER_FOLDER_PRED = "Weather/Prediction"      # Folder to the weather data for prediction
DATASETS_FOLDER = "Datasets"                    # Folder to the datasets
FOLDER_TRAIN_SAMPLES = "Day_samples_train"      # Folder with training samples
FOLDER_TEST_SAMPLES = "Day_samples_test"        # Folder with test samples
MODELS_FOLDER = "Models"                        # Folder to the trained models
IMAGE_FOLDER = "Images"                         # Folder for saving images
PLOT_FOLDER = "Plots"                           # Folder for saving plots
REPORT_FOLDER = "Reports"                       # Folder for saving reports
TS_SAMPLES_FOLDER = "TS_samples"                # Folder for saving time series samples
PREDICTIONS_FOLDER = "Predictions"              # Folder for saving predictions

# ==============================
# Data File Paths
# ==============================
FILE_YEAR_TRAIN = (
    f"{WEATHER_FOLDER_CLASS}/{LOCAL.replace(' ', '_')}_weather_2023.csv", 
    FOLDER_TRAIN_SAMPLES,
    "train"
)                                               # Weather 2023
FILE_YEAR_TEST = (
    f"{WEATHER_FOLDER_CLASS}/{LOCAL.replace(' ', '_')}_weather_2024.csv", 
    FOLDER_TEST_SAMPLES, 
    "test"
)                                               # Weather 2024
TRAIN_VALID_SET_FILE = f"{DATASETS_FOLDER}/trainset_2023.csv"  # Training + validation dataset
TEST_SET_FILE = f"{DATASETS_FOLDER}/testset_2024.csv"          # Test dataset

# ==============================
# Meteorological Data Columns
# ==============================
GLOBAL_TILTED_IRRADIANCE = 'global_tilted_irradiance'
DIFFUSE_RADIATION = 'diffuse_radiation'
TEMPERATURE_2M = 'temperature_2m'
WIND_SPEED_10M = 'wind_speed_10m'
PRECIPITATION = 'precipitation'

METEOROLOGICAL_COLUMNS = [
    GLOBAL_TILTED_IRRADIANCE,
    DIFFUSE_RADIATION,
    TEMPERATURE_2M,
    WIND_SPEED_10M,
    PRECIPITATION,
]

# ==============================
# Export Columns
# ==============================
EXPORT_COLUMNS = [
    'inverter_state',
    'pv1_i',
    'pv1_u',
    'mppt_power',
    'active_power',
    'efficiency',
    'a_i',
    'b_i',
    'c_i',
    'a_u',
    'b_u',
    'c_u',
    'ab_u',
    'bc_u',
    'ca_u',
    'inv_temperature',
]

# =======================================
# Classification and Prediction Settings
# =======================================
CLASSIFICATION_HOUR_INIT = '04:00'  # Start hour for classification
CLASSIFICATION_HOUR_END = '22:00'   # End hour for classification
TOP_FEATURES = 20                   # Number of top features to consider for classification
TIME_INIT = "05:00"                 # Start time for classification and prediction plots
TIME_END = "21:00"                  # End time for classification and prediction plots
PREDICTION_HOUR_INIT = '04:00'      # Start hour for prediction
PREDICTION_HOUR_END = '22:00'       # End hour for prediction

# Define realistic crack degradation scenarios. 
# Each tuple: (current_degradation, voltage_degradation, power_degradation)
CRACKS_DEGRADATION_SCENARIOS = [   # For random
    (0.980, 0.990, 0.970),  # light: -2% current, -1% voltage, -3% power
    (0.960, 0.980, 0.940),  # moderate: -4% current, -2% voltage, -6% power
    (0.949, 0.956, 0.850),  # reference case: -5.1% current, -4.4% voltage, -15% power
    (0.970, 0.960, 0.930),  # -3% current, -4% voltage, -7% power
    (0.930, 0.950, 0.800),  # -7% current, -5% voltage, -20% power
    (0.990, 0.970, 0.960),  # -1% current, -3% voltage, -4% power
    (0.910, 0.940, 0.750),  # -9% current, -6% voltage, -25% power
    (0.940, 0.930, 0.810),  # -6% current, -7% voltage, -19% power
    (0.925, 0.965, 0.780),  # -7.5% current, -3.5% voltage, -22% power
    (0.970, 0.985, 0.950),  # -3% current, -1.5% voltage, -5% power
]

# =================================
# Labels Mapping for Visualization
# =================================
LABELS_MAP = {
    0: ('Normal', '#00cc00'),        # Green
    1: ('Soiling', '#996600'),       # Light brown
    2: ('Shading', '#ff9900'),       # Orange
    3: ('Cracks', '#ffcc00'),        # Yellow
    4: ('Ground Fault', '#9966ff'),  # Lilac
    5: ('Arc Fault', '#ff66ff'),     # Dark pink
    6: ('Diode Fault', '#ff99cc'),   # Light pink
}

# ================================
# Plots Default colors
# ================================
MPPT_PALETTE = {
    'mppt_power': '#33cc33',               # Light cyan / pale cyan
    'mppt_power_clean': '#33cc33',         # Light cyan / pale cyan
    'global_tilted_irradiance': '#b3b300', # Light yellow
    'diffuse_radiation': '#e65c00',        # Orange
    'temperature_2m': '#cc0088',           # Light pink / soft pink
    'wind_speed_10m': '#3333cc',           # Medium blue / cornflower blue
    'precipitation': '#00b3b3',            # Blueish white
}

CURR_VOLT_PALETTE = {
    'pv1_i': '#ff3300',         # Red
    'pv1_i_clean': '#ff3300',   # Red
    'pv1_u': '#cc5200',         # Orange
    'pv1_u_clean': '#cc5200',   # Orange
    'precipitation': '#00b3b3', # Blueish white
}

# ====================================
# Coordinates for Different Locations
# ====================================
COORDINATES = {
    'Albergaria-a-Velha': (41.168936, -8.590617),
}