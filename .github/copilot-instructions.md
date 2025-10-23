# AI Agent Instructions for DT Classification Project

## Project Overview
This is a Digital Twin (DT) Classification project for PV systems fault detection. The project uses a Random Forest model to classify different types of faults in photovoltaic systems based on simulated and real data.

## Core Components

### Digital Twin Implementation
- Core simulation logic is in `classes.py`
- Three main classes:
  - `PVPlant`: Represents physical PV system configuration
  - `InverterTwin`: Models DC-to-AC power conversion
  - `DigitalTwin`: Handles system simulation with weather data

### Data Pipeline
The workflow is organized into sequential blocks (see `main.py`):
1. Sample Creation (`create_samples.py`)
2. Train/Test Set Generation (`create_train_test_sets.py`)
3. Random Forest Model Training (`random_forest.py`)
4. Inference Set Preparation (`create_preprocess_inference_set.py`)
5. Real Data Inference (`inference.py`)

## Key Conventions

### Data Organization
- Raw inverter data: `Inverters/*.csv`
- Weather data: `Weather/`
- Generated samples: `Samples_2023/`, `Samples_2024/`
- Model outputs: `Model/`
- Datasets: `Datasets/`
  - Training: `trainset_2023.csv`
  - Testing: `testset_2024.csv`
  - Inference: `inference_test_set.csv`

### PV System Configuration
- Uses Jinko Solar modules and Huawei inverters
- Fixed parameters in `globals.py`:
  - Location: Lisbon, Portugal
  - Surface tilt: 30°
  - Surface azimuth: 180° (South-facing)
  - Albedo: 0.2

## Development Workflow

### Running the Pipeline
Use command-line arguments to control workflow:
```bash
python main.py --create_samples            # Generate sample data
python main.py --create_train_test        # Prepare datasets
python main.py --random_forest_run        # Train model
python main.py --create_preprocess_inference_test  # Prepare inference data
python main.py --inference                # Run inference
```

### Common Parameters
- `--create_samples_with_plot`: Visualize sample generation
- `--smooth_inference_test`: Apply smoothing to inference data
- `--inference_test_file`: Specify custom inference input file

## Integration Points
- Weather data integration via pvlib's ModelChain
- Uses SAM (System Advisor Model) database for PV components
- Inverter data processing follows standard CSV format with timestamps

## Project-Specific Patterns
1. Fault Classification:
   - 0: No Fault
   - 1: Soiling
   - 2: Shading
   - 3: Crack
   - 4: Ground Fault
   - 5: Arc Fault
   - 6: Diode Fault

2. Digital Twin Data Flow:
   - Weather data → PVPlant simulation → Inverter model → Classification