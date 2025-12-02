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
MODELS_FOLDER = "Models"                        # Folder to the trained models
IMAGE_FOLDER = "Images"                         # Folder for saving images
PLOT_FOLDER = "Plots"                           # Folder for saving plots
REPORT_FOLDER = "Reports"                       # Folder for saving reports

# ==============================
# Data File Paths
# ==============================
FOLDER_TRAIN_SAMPLES = "Day_samples_train"      # Folder with training samples
FOLDER_TEST_SAMPLES = "Day_samples_test"        # Folder with test samples
TS_SAMPLES = "TS_samples"
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

# ==============================
# Classification and Prediction Settings
# ==============================
CLASSIFICATION_HOUR_INIT = '04:00'  # Start hour for classification
CLASSIFICATION_HOUR_END = '22:00'   # End hour for classification
TOP_FEATURES = 20                   # Number of top features to consider for classification
CLASS_PLOT_TIME_INIT = "05:00"      # Start time for plots
CLASS_PLOT_TIME_END = "21:00"       # End time for plots
PREDICTION_HOUR_INIT = '04:00'      # Start hour for prediction
PREDICTION_HOUR_END = '22:00'       # End hour for prediction

# ================================
# Labels Mapping for Visualization
# ================================
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
# Default colors
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

# ================================
# Coordinates for Different Locations
# ================================
COORDINATES = {
    'Albergaria-a-Velha': (41.168936, -8.590617),
    'Alcobaça': (39.633944, -8.919936),
    'Algoz': (37.182202, -8.28171),
    'Arraiolos A': (38.724523, -7.992924),
    'Arraiolos B': (38.724131, -7.996119),
    'Arrifana': (40.918532, -8.492255),
    'Braga': (41.556869, -8.433374),
    'Cambres': (41.15377, -7.792128),
    'Chaves': (41.722846, -7.486212),
    'Cortes do Meio': (40.033263, -7.889626),
    'Esposende': (41.606827, -8.80519),
    'Famalicão': (41.415644, -8.507602),
    'Felgueiras': (41.315754, -8.135029),
    'Folgosa': (41.272655, -8.571356),
    'Gandra A': (41.188737, -8.443189),
    'Guarda': (40.547439, -7.25182),
    'Guarda B': (40.549004, -7.238805),
    'Hamburg': (53.622344, 10.094405),
    'Lages de Ranhados': (41.067956, -8.47051),
    'Loulé A': (37.134902, -8.020329),
    'Loulé B': (37.096112, -8.12379),
    'Macieira da Maia': (41.337778, -8.689722),
    'Maia': (41.205448, -8.599787),
    'Margaride': (41.365711, -8.208229),
    'Milheirós de Poiares': (40.92693, -8.47872),
    'Ovar': (40.866119, -8.622233),
    'Panoias Cima': (40.502555, -7.235411),
    'Paranhos da Beira': (40.471071, -7.771463),
    'Póvoa de Varzim A': (41.390556, -8.76),
    'Póvoa de Varzim B': (41.382997, -8.757877),
    'Ribeirão A': (41.362061, -8.545315),
    'Ribeirão B': (41.361454, -8.553326),
    'Ribeirão C': (41.35195, -8.585138),
    'Rio Tinto': (41.172046, -8.543919),
    'Romariz': (40.933335, -8.460201),
    'Serzedelo': (41.396293, -8.372483),
    'Sobreda': (38.648978, -9.184123),
    'São Bartolomeu de Messines A': (37.26605, -8.271416),
    'São Bartolomeu de Messines B': (37.258692, -8.290258),
    'São João da Madeira': (40.919646, -8.444928),
    'São Mamede de Infesta': (41.192311, -8.608794),
    'Touguinha': (41.372222, -8.715278),
    'Trofa A': (41.31673, -8.586815),
    'Trofa B': (41.285331, -8.567417),
    'Veade': (41.426136, -7.973019),
    'Vila Chã': (41.300211, -8.726056),
    'Vila Cova de Perrinho': (40.899167, -8.383901),
    'Vila Franca Das Naves A': (40.725212, -7.264546),
    'Vila Franca Das Naves B': (40.72491, -7.263937),
    'Vila Franca Das Naves C': (40.724687, -7.263821),
    'Vila Nova de Cacela': (37.162976, -7.563605),
    'Vila Nova de Gaia': (41.127879, -8.627558),
    'Vila Verde': (41.650833, -8.433333),
    'Vila do Conde': (41.337579, -8.689926),
    'Vilar': (41.28058, -8.680119),
    'Árvore': (41.339444, -8.720833),
    'Évora': (38.56485, -7.912691),
}