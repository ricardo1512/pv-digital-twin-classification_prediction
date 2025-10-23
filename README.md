# Photovoltaic Digital Twin Simulation Framework for Anomaly and Fault Classification
#### Ricardo Vicente, Lisbon, 2025

## Description / Overview
This project implements a **Photovoltaic (PV) Digital Twin** for the simulation, monitoring, and analysis of a PV system under normal and multiple anomaly and fault conditions.  
The framework models the electrical, thermal, and environmental behaviour of a solar plant consisting of **Jinko JKM410M-72HL-V modules** and a **Huawei SUN2000-10KTL-USL0-240V inverter**.  

It uses **pvlib** for irradiance, temperature, wind speed, precipitation, and power flow modeling, and integrates anomaly and fault-specific degradation models (e.g., cracks, shading, ground faults, arc faults). The system supports data aggregation, fault classification dataset generation, Random Forest training, and inference on real data.

---

## Features
- Digital twin of a PV inverter system.
- Realistic modeling of seven scenarios:
  1. No Fault  
  2. Soiling  
  3. Shading  
  4. Cracks / Microcracks  
  5. Ground Fault  
  6. Arc Fault  
  7. Bypass Diode Fault
- Parallel processing for fault simulation.
- ML-ready dataset generation with daily feature extraction.
- Random Forest classifier for automated anomaly and fault detection.
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

#### 1. Sample Generation – `--create_samples`
Simulates photovoltaic operation and fault scenarios using the digital twin.

#### 2. Dataset Consolidation – `--create_train_test`
Aggregates simulated data into unified training and testing sets.

#### 3. Model Training – `--random_forest_run`
Trains and evaluates a Random Forest classifier for fault detection.

#### 4. Inference Set Preparation – `--create_inference_set`
Preprocesses inverter real data for model inference.

#### 5. Inference – `--inference_run`
Performs final inference and generates diagnostic reports.



## Usage / How to Run
The complete pipeline can be executed using the command-line interface (`argparse`).

### Example of full workflow

```bash
python main.py --create_samples --create_train_test --random_forest_run --create_inference_set --inference_run
```

### Individual blocks
| Stage | Command |
|--------|----------|
| Create synthetic samples | `python main.py --create_samples` |
| Create synthetic samples with plots | `python main.py --create_samples_with_plot` |
| Generate train/test datasets | `python main.py --create_train_test` |
| Train and evaluate Random Forest | `python main.py --random_forest_run` |
| Create and preprocess inference dataset | `python main.py --create_inference_set` |
| Create and preprocess inference dataset with smoothing | `python main.py --create_inference_set_smooth` |
| Perform inference on real data | `python main.py --inference_run` |

---

## Project Structure
```
📦 PV_Digital_Twin_Classification
├── main.py                              # CLI entry point (argparse workflow)
├── classes.py                           # Core system and inverter models
├── create_samples_*.py                  # No Fault, Anomaly, and Fault-specific simulation modules
├── create_samples.py                    # Parallel orchestration of all scenarios
├── create_train_test_sets.py            # Aggregates simulation data
├── random_forest.py                     # ML model training and evaluation
├── create_preprocess_inference_set.py   # Preprocesses inverter data
├── inference.py                         # Fault classification inference
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
- Fault mappings and dataset paths
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
| 1.0.0 | 2025-10-23 | Initial release with full CLI workflow, 7 fault modes, and Random Forest integration. |

---
