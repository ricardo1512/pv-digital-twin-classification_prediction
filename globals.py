
# ==============================
# Seasons and Location Settings
# ==============================
WINTER = ("Winter", [1, 2, 3, 10, 11, 12])  # Winter months
SUMMER = ("Summer", [4, 5, 6, 7, 8, 9])     # Summer months

SEASON = SUMMER                             # Currently selected season
LOCAL = "Vila do Conde"                     # Location of the PV plant
LATITUDE = 41.337579                        # Latitude of the LOCAL
LONGITUDE = -8.689926                       # Longitude of the LOCAL

# ==============================
# System Configuration
# ==============================
MODULES_PER_STRING = 10                      # Number of modules per string

# ==============================
# Data File Paths
# ==============================
FILES_YEAR_2023 = (f"Weather/{LOCAL.replace(' ', '_')}_weather_2023.csv", "Samples_2023")  # Weather 2023
FILES_YEAR_2024 = (f"Weather/{LOCAL.replace(' ', '_')}_weather_2024.csv", "Samples_2024")  # Weather 2024
FOLDER_TRAIN_SAMPLES = "Samples_2023"                # Folder with training samples
FOLDER_TEST_SAMPLES = "Samples_2024"                 # Folder with test samples
TRAIN_VALID_SET_FILE = "Datasets/trainset_2023.csv"  # Training + validation dataset
TEST_SET_FILE = "Datasets/testset_2024.csv"          # Test dataset

# ==============================
# Model and Outputs
# ==============================
MODEL_PATH = "Model/rf_best_model.joblib"    # Path to the trained model
LABEL = 'inverter_state'                     # Target label for classification
IMAGE_FOLDER = "Images"                      # Folder for saving images
PLOT_FOLDER = "Plots"                        # Folder for saving plots
REPORT_FOLDER = "Reports"                    # Folder for saving reports

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
    'mppt_power',
    'active_power',
    'efficiency',
    'pv1_i',
    'a_i',
    'b_i',
    'c_i',
    'pv1_u',
    'a_u',
    'b_u',
    'c_u',
    'ab_u',
    'bc_u',
    'ca_u',
    'inv_temperature',
]

# ==============================
# Classification Settings
# ==============================
CLASSIFICATION_HOUR_INIT = '04:00'  # Start hour for classification
CLASSIFICATION_HOUR_END = '22:00'   # End hour for classification
TOP_FEATURES = 20                            # Number of top features to consider

# ==============================
# Labels Mapping for Visualization
# ==============================
LABELS_MAP = {
    0: ('No Fault', '#00cc00'),      # green
    1: ('Soling', '#996600'),        # light brown
    2: ('Shading', '#ff9900'),       # orange
    3: ('Cracks', '#ffcc00'),        # yellow
    4: ('Ground Fault', '#9966ff'), # lilac
    5: ('Arc Fault', '#ff66ff'),    # dark pink
    6: ('Diode Fault', '#ff99cc'),  # light pink
}