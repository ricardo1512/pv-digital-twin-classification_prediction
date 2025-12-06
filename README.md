# Photovoltaic Digital Twin Simulation Framework for Anomaly and Fault Classification and Prediction
#### Ricardo Vicente, Lisbon, 2025

## Description / Overview
This project implements a **Photovoltaic (PV) Digital Twin** for the simulation, monitoring, and analysis of a PV system under normal and multiple anomaly and fault conditions.
The framework models the electrical, thermal, and environmental behaviour of a solar plant consisting of **Jinko JKM410M-72HL-V modules** and a **Huawei SUN2000-10KTL-USL0-240V inverter**.  

It uses **pvlib** for irradiance, temperature, wind speed, precipitation, and power flow modeling, and integrates anomaly and fault-specific degradation models (e.g., cracks, shading, ground faults, arc faults). The system supports data aggregation, anomaly and fault classification dataset generation, XGBoosting training, and inference on real data.

---

## Features
### Classification
- Digital twin for classification.
- Realistic modelling of seven anomaly and fault scenarios:
  1. Normal  
  2. Soiling  
  3. Shading  
  4. Cracks / Microcracks  
  5. Ground Fault  
  6. Arc Fault  
  7. Bypass Diode Fault
- Parallel processing of anomaly and fault simulation.
- ML-ready daily dataset generation with daily feature extraction.
- XGBoost classifier for automated anomaly and fault classification.
- Command-line execution with modular pipeline control.

### Prediction
- Digital twin for prediction.
- Realistic modelling of three anomaly scenarios:
  1. Soiling  
  2. Shading  
  3. Cracks / Microcracks  
- Processing of anomaly simulation.
- Time series dataset generation.
- Time series daily classification with pre-trained XGBoost model using feature extraction.
- Anomaly prediction with Linear Regression.
- Command-line execution with modular pipeline control.
---

## Installation / Requirements
**Dependencies:**
```bash
python >= 3.10
pvlib
numpy
pandas
matplotlib
joblib
argparse
scikit-learn
xgboost
```
**Install requirements:**
```bash
pip install -r requirements.txt
```

---

# Machine Learning Pipeline Overview

This project includes all required datasets and scripts to execute each stage of the complete machine learning workflow, either individually or in sequence.

The end-to-end pipeline is divided into **Classification** and **Prediction** workflows and covers the following steps:

---

## A. Classification Workflow (Daily Samples)

### 1. Training Workflow

#### 1.1. Create Samples for Classification
- Simulates photovoltaic system operation under normal conditions and various anomaly and fault scenarios using a digital twin.
- Optional plots can be generated to visualize each scenario and verify simulation quality.
- **Script:** `create_day_samples.py`
- **CLI Options:** `--create_samples`, `--create_samples_with_plots`

#### 1.2. Create Train and Test Sets
- Aggregates simulated samples into unified **training** and **testing** datasets.
- Ensures proper temporal and seasonal coverage.
- Handles splitting and formatting for supervised learning.
- **Script:** `create_train_test_sets.py`
- **CLI Option:** `--create_train_test`

#### 1.3. Run the XGBoost Model
- Trains and evaluates a **XGBoost classifier** for anomaly and fault detection.
- Metrics include accuracy, precision, recall, F1-score.
- Performance plots are generated.
- **Script:** `xgboost_classifier.py`
- **CLI Options:** `--xgboost_run`, `--all_year`, `--winter`

### 2. Inference Workflow

#### 2.1. Create and Preprocess the Inference Test Set
- Preprocesses real inverter data for model inference.
- Recommended to apply smoothing to reduce noise from cloud transients or measurement fluctuations.
- Supports configuration of smoothing window (12 for each hour).
- **Script:** `create_preprocess_inference_set.py`
- **CLI Options:** `--create_inference_set`, `--create_inference_set_smooth`, `--all_year`, `--winter`

#### 2.2. Perform Inference on Real Data
- Performs model inference on the prepared dataset.
- Generates diagnostic reports including probability distributions and anomaly recommendations.
- **Delta (`--delta`) and Top (`--top`) options** are used for **probabilistic recommendation analysis**:
  - `--top` selects the N most important class probabilities for inspection.
  - `--delta` defines a tolerance value: if a class probability is within `delta` of the highest class probability, it will also be considered as a potential recommendation.
- **Script:** `inference.py`
- **CLI Options:** `--inference_run_user`, `--inference_smooth`,`--inference_run_user`, `--delta`, `--top`, `--all_year`, `--winter`


---

## B. Prediction Workflow (Time-Series)

### 1. Create Anomaly Samples for Prediction, with Plots
- Generates synthetic time series representing anomalies (soiling, shading, and cracks) for the summer season.
- Plots visualize anomaly patterns.
- **Script:** `create_ts_samples.py`
- **CLI Option:** `--create_ts_samples`

### 2. Synthetic Time Series Prediction, with Plots

#### 2.1. Perform Daily Classification
- Performs daily classification on synthetic summer time series samples.
- Generates plots showing classification results.
- **Script:** `prediction.py`
- **CLI Option:** `--synthetic_ts_daily_classification`

#### 2.2. Predict Anomalies
- Predicts future anomalies in synthetic time series using configurable thresholds and window sizes.
- Generates visualizations of predicted anomaly days.
- **Script:** `prediction.py`
- **CLI Options:** `--synthetic_ts_predict_days`,`--synt_threshold_start`, `--synt_threshold_target`, `--synt_threshold_class`, `--synt_window`


### 3. Perform Daily Classification and Prediction in Real Time Series, with Plots
- Performs daily classification on real inverter time series data.
- Optional smoothing can be applied to reduce measurement noise.
- **Script:** `prediction.py`
- **CLI Options:** `--real_ts_prediction`, `--ts_smooth`, `real_threshold_start`, `--real_threshold_target`, `--real_threshold_class`, `--real_window`, `--all_year`, `--winter`

### Note: 
**Time-Series Prediction sensitivity analysis parameters:**
  - `start` (`--*_threshold_start`): start percentage from which regression begins.
  - `target` (`--*_threshold_target`): target percentage in the future that the regression aims to exceed.
  - `class` (`--*_threshold_class`): class tolerance, allowing prediction to be considered correct if it is within this delta below the majority class probability.
  - `window` (`--*_window`): number of past samples considered for regression.
---

## C. Options
- CLI options can be combined to run multiple workflow stages in sequence.
- Smoothing parameters are optional but recommended for real data.
- Default behaviour assumes summer months; use `--all_year` or `--winter` to adjust seasonal coverage.


---
## File Inputs and Outputs

Each function receives specific inputs and generates standardized outputs used in subsequent stages.

### A. Classification Workflow (Daily Samples)

#### **1. Training Workflow**
##### **1.1. Create Samples for Classification:** `create_samples()`

Inputs:
- Meteorological data
  - `Weather/Classification/[LOCAL]_weather_2023.csv`
  - `Weather/Classification/[LOCAL]_weather_2024.csv`

Outputs:
- Simulated sample data stored in `Samples_2023/` and `Samples_2024/`
- Optional plots saved in `Plots/`


##### **1.2. Create Train and Test Sets:** `create_train_test_sets()`

Inputs:
- Simulated sample data stored in `Samples_2023/` and `Samples_2024/`

Outputs:
- Consolidated datasets in `Datasets/`  
  - `train_set.csv`  
  - `test_set.csv`
- I-V Scatter plots in `Day_samples/`  
  - `trainset_2023_scatter_iv.png`  
  - `testset_2024_scatter_iv.png`


##### **1.3. Run the XGBoost Model:** `xgboost_classifier()`

Inputs:
- Consolidated datasets in `Datasets/`  
  - `train_set.csv`  
  - `test_set.csv`

Outputs:
- Trained XGBoost model saved in `Models/xgb_best_model_*.joblib`
- Classification reports and accuracy metrics in `Reports/`:
  - `xgb_validation_classification_report_*.csv`
  - `xgb_test_classification_report_*.csv`
  - `xgb_class_accuracies_*.csv`
  - `xgb_top_*_features_*.csv`
  - `xgb_cross_validation_raw_scores_*.csv`
  - `xgb_cross_validation_summary_*.csv`
  - `xgb_overall_performance.csv`

- Plots in `Images/`:
  - `xgb_confusion_matrix_validation_*.png`
  - `xgb_confusion_matrix_test_*.png`
  - `xgb_val_class_accuracy_*.png`
  - `xgb_test_class_accuracy_*.png`
  - `xgb_feature_importance_*.png`
  - `xgb_auc_val_precision_vs_recall_*.png`
  - `xgb_auc_test_precision_vs_recall_*.png`
  - `xgb_fp_tp_curve_validation_*.png`
  - `xgb_fp_tp_curve_test_*.png`


#### **2. Inference Workflow**

##### **2.1. Create and Preprocess the Inference Test Set:** `create_inference_set()`

Inputs:
- Real inverter measurements: `Inverters/*.csv`

Outputs:
- Preprocessed inference dataset: `Datasets/inference_test_set_before_classification_*.csv`

##### **2.2. Perform Inference on Real Data:** `inference()`

Inputs:
- Trained model: `Models/xgb_best_model_*.joblib`
- Inference dataset: `Datasets/inference_test_set_before_classification_*.csv`

Outputs::
- Classified data in `Datasets`: 
  - `inference_test_set_with_classification_*.csv`
- Results in `Reports`:
  - `inference_results_*.csv`
  - `inference_test_set_with_prob_classification_*.csv`
  - `inference_adjusted_probabilities_report_*.csv`

- Plots in `Images/`:
  - `inference_classification_distribution_*.png`

- Singular plots in:
  - `Plots/Probabilities/Plots_inference_probabilities_*/`
  - `Plots/Probabilities_scaled/Plots_inference_probabilities_*/`

### B. Prediction Workflow (Time-Series)

#### **1. Create Anomaly Samples for Prediction, with Plots:** `create_ts_samples()`

Inputs:
- Meteorological data
  - `Weather/Prediction/weather_*.csv`

Outputs:
- Simulated sample data stored in `TS_samples/*/`
- Optional plots saved in `Plots/TS_samples/*/`

#### **2. Synthetic Time Series Prediction, with Plots**

##### **2.1. Perform Daily Classification:** `ts_daily_classification()` (synthetic or real, single files) and `synthetic_ts_daily_classification()` (synthetic, multiple files)

Inputs:
- Single time-series CSV file:  
  - `TS_samples/*/*.csv`    

Outputs:
- Daily classification probabilities CSV:  
  - `Predictions/real_data_probabilities/*_daily_probabilities.csv`  
- Daily probabilities plot:  
  - `Plots/TS_probabilities/*_daily_probabilities.png`  

##### **2.2. Predict Anomalies:** `ts_predict_days()` (synthetic or real, single files) and `synthetic_ts_predict_days()` (synthetic, multiple files)

Inputs:
- Daily probability CSV file:  
  - `Predictions/real_data_probabilities/*.csv`   

Outputs:
- Predictions CSV with estimated days to reach target probability:  
  - `Predictions/real_data_predictions/*_daily_predictions.csv`  
- Prediction plot (Cleveland-style) for visualizing estimated days:  
  - `Plots/TS_predictions/*_predictions_cleveland.png`  

---
## Usage / How to Run
The complete pipeline can be executed using the command-line interface (`argparse`).

By default, for classification and prediction the training season is set to **Summer** (April to September) since the inverter data used for inference, stored in the `Inverters` (Classification) and `TS_samples/real_data` (Prediction) folders, was collected during this period. However, if other datasets are used for inference or prediction, the season can be changed to **Winter** (October to March) (`--winter`) or **All Year** (`--all_year`).

### 1. Examples of full, recommended workflows

#### 1. Classification using default real data for inference

```bash
python main.py --create_samples --create_train_test --xgboost_run --create_inference_set_smooth --inference_run
```

#### 2. Classification using user real data for inference

```bash
python main.py --create_samples --create_train_test --xgboost_run --inference_run_user
```
#### 3. Prediction with synthetic time series:

```bash
python main.py --create_ts_samples --synthetic_ts_daily_classification --synthetic_ts_predict_days
```
#### 4. Prediction with user real time series:
```bash
python main.py --real_ts_prediction "TS_samples/real_data/inverter_Aveiro_060.csv" --ts_smooth 36
```

### 2. Examples of individual blocks

#### 2.1. Classification
| Stage | Command |
|--------|----------|
| A.1.1. Create synthetic day samples | `python main.py --create_samples` |
| A.1.1. Create synthetic day samples with plots | `python main.py --create_samples_with_plots` |
| A.1.2. Generate train and test datasets | `python main.py --create_train_test` |
| A.1.3. Train and evaluate the XGBoost classifier | `python main.py --xgboost_run` |
| A.2.1. Create and preprocess inference dataset | `python main.py --create_inference_set` |
| A.2.1. Create and preprocess inference dataset with smoothing | `python main.py --create_inference_set --inference_set_smooth <N>` |
| A.2.2. Run inference on provided real data | `python main.py --inference_run` |
| A.2.2. Run inference on provided real data with delta and top | `python main.py --inference_run --delta <float> --top <N>` |
| A.2.2. Run inference on user real data files | `python main.py --inference_run_user` |
| A.2.2. Run inference on user smoothed real data with delta and top | `python main.py --inference_run_user --inference_smooth <N> --delta <float> --top <N>` |


#### 2.2. Prediction

| Stage | Command |
|--------|----------|
| B.1. Create anomaly time-series samples | `python main.py --create_ts_samples` |
| B.2.1. Daily classification in synthetic time series | `python main.py --synthetic_ts_daily_classification` |
| B.2.2. Predict anomalies in synthetic time series | `python main.py --synthetic_ts_predict_days` |
| B.2.2. Predict anomalies in synthetic time series with some parameters | `python main.py --synthetic_ts_predict_days --synt_threshold_start <v> --synt_window <N>` |
| B.3.  Daily classification and prediction in real time series with some parameters | `python main.py --real_ts_prediction <path> --ts_smooth <N> --real_threshold_class <v>` |

---

## Real Data
### 1. Classification
To run inference on specific file(s) using `--inference_run_user`, the file(s) in `.csv` format must be placed in the `Datasets/user` folder.

### 2. Prediction
To run prediction on a specific file using `--real_ts_prediction`, the `.csv` file must be placed in the `TS_samples/real_data` folder.

### 3. Features
For both classification and prediction, the dataset is expected to represent a PV system with a single string and must follow the required input format, consisting of a series of rows with 5-minute intervals and the following features:


| Name                     | Meaning                                               | Datatype | Unit        |
|--------------------------|-------------------------------------------------------|----------|-------------|
| `collectTime`              | Timestamp of the measurement (e.g., 2025-05-09 06:00:00)                          | datetime | str           |
|
| `pv1_u`                    | PV string voltage                                   | float    | V           |
| `pv1_i`                    | PV string current                                   | float    | A           |
| `a_u`                      | Phase A voltage                                       | float    | V           |
| `b_u`                      | Phase B voltage                                       | float    | V           |
| `c_u`                      | Phase C voltage                                       | float    | V           |
| `ab_u`                     | Line voltage between phases A-B                       | float    | V           |
| `bc_u`                     | Line voltage between phases B-C                       | float    | V           |
| `ca_u`                     | Line voltage between phases C-A                       | float    | V           |
| `a_i`                      | Phase A current                                       | float    | A           |
| `b_i`                      | Phase B current                                       | float    | A           |
| `c_i`                      | Phase C current                                       | float    | A           |
| `mppt_power`               | MPPT output power                               | float    | kW           |
| `active_power`             | Inverter active power                                 | float    | kW           |
| `efficiency`               | Inverter instantaneous efficiency                     | float    | %           |
| `inv_temperature`          | Inverter internal temperature                         | float    | °C          |
| `temperature_2m`           | Ambient temperature at 2 meters                       | float    | °C          |
| `diffuse_radiation`        | DHI: Diffuse solar radiation                               | float    | W/m²        |
| `global_tilted_irradiance` | GTI: Global tilted solar irradiance                       | float    | W/m²        |
| `wind_speed_10m`           | Wind speed at 10 meters                               | float    | m/s         |
| `precipitation`            | Precipitation                                         | float    | mm/h  |

---

## Project Structure
```
📦 PV_Digital_Twin_Classification
├── main.py                                    # CLI entry point (argparse workflow)
|
├── globals.py                                 # Global constants and paths
├── utils.py                                   # Feature extraction and helpers
├── preprocess_weather_for_classification.py   # Row weather data for classification preprocessing
├── real_data_visualisation.py                 # Visualizing real-world data using plots
|
├── classes.py                                 # Core system and inverter models
├── create_day_samples_*.py                    # Normal, anomaly and fault-specific simulation modules
├── create_day_samples.py                      # Parallel orchestration of all scenarios
├── create_preprocess_inference_set.py         # Inverter data preprocessing
├── create_ts_samples_*.py                     # Anomaly-specific time series simulation modules
├── create_train_test_sets.py                  # Simulation data aggregation
├── create_ts_samples.py                       # Time series anomaly sample generation
|
├── xgboost_classifier.py                      # ML model training and evaluation
├── inference.py                               # Anomaly and fault classification inference
├── prediction.py                              # Time series daily classification and prediction
|
├── plot_day_samples.py                        # Plotting utilities for sample creation
├── plot_inference.py                          # Plotting utilities for inference
├── plot_training.py                           # Plotting utilities for training
├── plot_ts_samples.py                         # Plotting utilities for time series samples
├── plot_ts_predictions.py                     # Plotting utilities for time series predictions
|
├── Inverters/                                 # Real inverter data for inference
├── Datasets/                                  # Aggregated datasets for training and inference
├── Day_samples_train/                         # Training samples
├── Day_samples_test/                          # Testing samples
├── TS_samples/                                # Time series samples
|
├── Models/                                    # Trained Random Forest models
├── Images/                                    # Training and testing plots
├── Reports/                                   # Performance reports
├── Predictions/                               # Time series prediction outputs
├── Plots/                                     # Sample plots
└── Weather/                                   # Meteorological input data

```

---

## Configuration / Settings

All adjustable parameters are in **`globals.py`**:

- **Training Location**: `LOCAL`, `LATITUDE`, `LONGITUDE`  
- **PV System**: `MODULES_PER_STRING`, `LABEL`, `YEARS`  
- **Folders**: Weather, datasets, samples, models, images, plots, reports, predictions  
- **Data Files**: Paths for train and test  
- **Meteorological Inputs**: `global_tilted_irradiance`, `diffuse_radiation`, `temperature_2m`, `wind_speed_10m`, `precipitation`  
- **Classification & Prediction**: Plotting, classification and prediction time range, `TOP_FEATURES`,  
- **Anomalies**: Crack degradation scenarios
- **Plotting Defaults**: `LABELS_MAP`, `MPPT_PALETTE`, `CURR_VOLT_PALETTE`  
- **Optional Coordinates**: `COORDINATES` for time series generation in multi-site setups 


---

## Contributing
Contributions are welcome!  
To contribute:
1. Fork this repository  
2. Create a feature branch (`git checkout -b feature/new-fault-model`)  
3. Commit your changes (`git commit -m "Add partial shading model"`)  
4. Push and open a Pull Request.

---

## Institutional Context
This project was developed as part of the research activities of [INESC-ID](https://www.inesc-id.pt/).

---

## Authors / Credits
**Author:** Ricardo de Jesus Vicente Tavares  
**Advisors:** Hugo Gabriel Valente Morais, Amâncio Lucas de Sousa Pereira  
**Institution:** INESC-ID & Instituto Superior Técnico, University of Lisbon 
**Email:** ricardo.j.vicente@tecnico.ulisboa.pt

Core frameworks:
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [pvlib](https://pvlib-python.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [xgboost](https://xgboost.readthedocs.io/)


---

## Contact / Support
For technical issues or inquiries, please email the author.

---

## Version / Changelog
| Version | Date | Description |
|----------|------|-------------|
| 1.0.0 | 2025-10-29 | Initial release with full CLI workflow, 7 anomaly and fault modes, XGBoost integration, and inference on real data. |
| 2.0.0 | 2025-12-03 | Added functionality for creating time series with 3 types of anomalies, and prediction capabilities for synthetic and real data. |
