# ===================================================================
# Global Configuration for PV System Anomaly and Fault Classification
# ===================================================================

# ==============================
# Location Settings
# ==============================
LOCAL = "Vila do Conde"                     # Location of the PV plant
LATITUDE = 41.337579                        # Latitude of the LOCAL
LONGITUDE = -8.689926                       # Longitude of the LOCAL

# ==============================
# System Configuration
# ==============================
MODULES_PER_STRING = 10                      # Number of modules per string

# ==============================
# Variables and Folders
# ==============================
LABEL = 'inverter_state'                        # Target label for classification
YEARS = [2023, 2024]                            # Available years with data
WEATHER_FOLDER_CLASS = "Weather/Classification" # Folder to the weather data for classification
WEATHER_FOLDER_PRED = "Weather/Prediction"      # Folder to the weather data for prediction
DATASETS_FOLDER = "Datasets"                    # Folder to the datasets
MODELS_FOLDER = "Models"                        # Folder to the trained models
IMAGE_FOLDER = "Images"                         # Folder for saving images
PLOT_FOLDER = "Plots"                           # Folder for saving plots
REPORT_FOLDER = "Reports"                       # Folder for saving reports

# ==============================
# Data File Paths
# ==============================
FOLDER_TRAIN_SAMPLES = "Day_samples_train"                # Folder with training samples
FOLDER_TEST_SAMPLES = "Day_samples_test"                  # Folder with test samples
TS_SAMPLES = "TS_samples"
FILE_YEAR_TRAIN = (f"{WEATHER_FOLDER_CLASS}/{LOCAL.replace(' ', '_')}_weather_2023.csv", FOLDER_TRAIN_SAMPLES, "train")  # Weather 2023
FILE_YEAR_TEST = (f"{WEATHER_FOLDER_CLASS}/{LOCAL.replace(' ', '_')}_weather_2024.csv", FOLDER_TEST_SAMPLES, "train")  # Weather 2024
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

# ==============================
# Classification and Prediction Settings
# ==============================
CLASSIFICATION_HOUR_INIT = '04:00'  # Start hour for classification
CLASSIFICATION_HOUR_END = '22:00'   # End hour for classification
TOP_FEATURES = 20                   # Number of top features to consider for classification
PREDICTION_HOUR_INIT = '04:00'      # Start hour for prediction
PREDICTION_HOUR_END = '22:00'       # End hour for prediction

# ================================
# Labels Mapping for Visualization
# ================================
LABELS_MAP = {
    0: ('Normal', '#00cc00'),      # green
    1: ('Soling', '#996600'),        # light brown
    2: ('Shading', '#ff9900'),       # orange
    3: ('Cracks', '#ffcc00'),        # yellow
    4: ('Ground Fault', '#9966ff'),  # lilac
    5: ('Arc Fault', '#ff66ff'),     # dark pink
    6: ('Diode Fault', '#ff99cc'),   # light pink
}