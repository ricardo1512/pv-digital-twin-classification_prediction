# Photovoltaic Digital Twin Simulation Framework for Anomaly and Fault Classification
#### Ricardo Vicente, Lisbon, 2025

## Description / Overview
This project implements a **Photovoltaic (PV) Digital Twin** for the simulation, monitoring, and analysis of a PV system under normal and multiple anomaly and fault conditions.  
The framework models the electrical, thermal, and environmental behaviour of a solar plant consisting of **Jinko JKM410M-72HL-V modules** and a **Huawei SUN2000-10KTL-USL0-240V inverter**.  

It uses **pvlib** for irradiance, temperature, wind speed, precipitation, and power flow modeling, and integrates anomaly and fault-specific degradation models (e.g., cracks, shading, ground faults, arc faults). The system supports data aggregation, anomaly and fault classification dataset generation, Random Forest training, and inference on real data.

---

## Features
- Digital twin of a PV inverter system.
- Realistic modeling of seven scenarios:
  1. Normal  
  2. Soiling  
  3. Shading  
  4. Cracks / Microcracks  
  5. Ground Fault  
  6. Arc Fault  
  7. Bypass Diode Fault
- Parallel processing for anomaly and fault simulation.
- ML-ready dataset generation with daily feature extraction.
- Random Forest classifier for automated anomaly and fault classification.
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
```
**Install requirements:**
```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

This project includes all required datasets and scripts to execute each stage of the complete machine learning workflow, either individually or in sequence.  
  
The end-to-end pipeline covers the following steps:

### 1. Sample Generation
Simulates photovoltaic operation, and anomaly and fault scenarios using the digital twin. Optionally generates plots for each scenario.

### 2. Dataset Consolidation
Aggregates simulated data into unified training and testing sets.

### 3. Model Training and Evaluation
Trains and evaluates a Random Forest classifier for anomaly and fault classification.

### 4. Inference Set Preparation
Preprocesses inverter real data for model inference. It is recommended to apply smoothing to the input data, since real-world measurements may contain noise, like cloud transient effects.

### 5. Inference
Performs final inference and generates diagnostic reports.

---
## File Inputs and Outputs

Each function receives specific inputs and generates standardized outputs used in subsequent stages.


### 1. `create_samples()`

**Inputs:**
- Meteorological data (`Weather/*.csv`)

**Outputs:**
- Simulated sample data stored in `Samples_2023/` and `Samples_2024/`
- Optional plots saved in `Plots/`


### 2. `create_train_test_sets()`

**Inputs:**
- Simulated sample data stored in `Samples_2023/` and `Samples_2024/`

**Outputs:**
- Consolidated datasets in `Datasets/`  
  - `train_set.csv`  
  - `test_set.csv`


### 3. `random_forest()`

**Inputs:**
- Consolidated datasets in `Datasets/`  
  - `train_set.csv`  
  - `test_set.csv`

**Outputs:**
- Trained Random Forest model saved in `Models/rf_best_model_*.joblib`
- Classification reports and accuracy metrics in `Reports/`
  - `class_accuracies_*.csv`
  - `cross_validation_raw_scores_*.csv`
  - `cross_validation_summary_*.csv`
  - `validation_classification_report_*.csv`
  - `test_classification_report_*.csv`
- Feature importance plots in `Images/`
  - `val_class_accuracy_*.png`
  - `test_class_accuracy_*.png`
  - `feature_importance_*.png`
  - `confusion_matrix_validation_*.png`
  - `confusion_matrix_test_*.png`


### 4. `create_inference_set()`

**Inputs:**
- Real inverter measurements (`Inverters/*.csv`)

**Outputs:**
- Preprocessed inference dataset (`Datasets/inference_test_set_before_classification_*.csv`)

### 5. `inference()`

**Inputs:**
- Trained model (`Models/rf_best_model_*.joblib`)
- Inference dataset (`Datasets/inference_test_set_before_classification_*.csv`)

**Outputs:**
- Classified anomaly and fault class for each data record (`Datasets/inference_test_set_with_classification_*.csv`)
- Classified anomaly and fault class probabilities for each data record (`Datasets/inference_test_set_prob_with_classification_*.csv`)
- Classification report (`Reports/inference_results_*.csv`)
- Plot illustrating model classification (`Images/inference_classification_distribution.png`)
- Plots illustrating class probabilities for each anomaly and fault (`Plots/PLots_inference_probabilities_*/Classification_prob_*.png`)


---
## Usage / How to Run
The complete pipeline can be executed using the command-line interface (`argparse`).

By default, the training season is set to **Summer** (April to September) since the inverter data used for inference, stored in the `Inverters` folder, was collected during this period. However, if other datasets are used, the season can be changed to **Winter** (October to March) (`--winter`) or **All Year** (`--all_year`).
### Example of full, recommended workflow

```bash
python main.py --create_samples --create_train_test --random_forest_run --create_inference_set_smooth --inference_run
```

### Individual blocks
| Stage | Command |
|--------|----------|
| 1. Create synthetic samples | `python main.py --create_samples` |
| 1. Create synthetic samples with plots | `python main.py --create_samples_with_plot` |
| 2. Generate train/test datasets | `python main.py --create_train_test` |
| 3. Train and evaluate Random Forest | `python main.py --random_forest_run` |
| 4. Create and preprocess inference dataset | `python main.py --create_inference_set` |
| 4. Create and preprocess inference dataset with smoothing | `python main.py --create_inference_set_smooth` |
| 5. Perform inference on real data | `python main.py --inference_run` |

---

## Project Structure
```
📦 PV_Digital_Twin_Classification
├── main.py                              # CLI entry point (argparse workflow)
├── classes.py                           # Core system and inverter models
├── create_samples_*.py                  # Normal, anomaly and fault-specific simulation modules
├── create_samples.py                    # Parallel orchestration of all scenarios
├── create_train_test_sets.py            # Aggregates simulation data
├── random_forest.py                     # ML model training and evaluation
├── create_preprocess_inference_set.py   # Preprocesses inverter data
├── inference.py                         # Anomaly and fault classification inference
├── plot.py                              # Plotting utilities
├── utils.py                             # Feature extraction and helpers
├── globals.py                           # Global constants and paths
├── Datasets/                            # Aggregated datasets for ML and inference sets
├── Images/                              # Training and testing plots
├── Inverters/                           # Real inverter data for inference
├── Model/                               # Trained Random Forest model
├── Plots/                               # Sample generation plots
├── Reports/                             # Model performance reports
├── Samples_2023/                        # Training samples
├── Samples_2024/                        # Testing samples
└── Weather/                             # Meteorological input data
    └──edit_weather_data.py              # Edits row weather data for simulations
```

---

## Configuration / Settings
All adjustable parameters are located in **`globals.py`**:
- Geographic coordinates (LATITUDE, LONGITUDE)
- Module and inverter specifications
- Anomaly and fault mappings and dataset paths
- Output directories for samples and plots

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
- [pvlib](https://pvlib-python.readthedocs.io/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

---

## Contact / Support
For technical issues or inquiries, please email the author.

---

## Version / Changelog
| Version | Date | Description |
|----------|------|-------------|
| 1.0.0 | 2025-10-23 | Initial release with full CLI workflow, 7 anomaly and fault modes, and Random Forest integration. |
| 1.0.1 | 2025-10-24 | Function improvements: added computation and visualization of the probabilities that guide the model’s class decisions during training |

---
