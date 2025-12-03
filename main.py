import sys
import argparse

# Classification
from create_day_samples import *
from create_train_test_sets import *
from xgboost_classifier import *
from create_preprocess_inference_set import *
from inference import *

# Prediction
from create_ts_samples import *


def run(args):
    # ------------------------------------------------------
    # Block A.1.1: Create Samples for Classification
    # ------------------------------------------------------
    if args.create_samples or args.create_samples_with_plots:
        create_day_samples(
            plot_samples=args.create_samples_with_plots,
        )

    # ------------------------------------------------------
    # Block A.1.2: Create Train and Test Sets
    # ------------------------------------------------------
    if args.create_train_test:
        create_train_test_sets()

    # ------------------------------------------------------
    # Block A.1.3: Run the XGBoost Model
    # ------------------------------------------------------
    if args.xgboost_run:
        xgboost_classifier(
            all_year=args.all_year, 
            winter=args.winter,
        )
        
    # ------------------------------------------------------
    # Block A.2.1: Create and Preprocess the Inference Test Set
    # ------------------------------------------------------
    if (args.create_inference_set
        or args.create_inference_set_smooth):
        create_preprocess_inference_set(
            smoothing=args.create_inference_set_smooth,
            all_year=args.all_year, 
            winter=args.winter,
        )

    # ------------------------------------------------------
    # Block A.2.2: Perform Inference on Real Data
    # ------------------------------------------------------
    if (args.inference_run or args.inference_run_file or args.delta or args.top):
        kwargs = {}

        if args.inference_run_file is not None:
            kwargs['inference_test_file'] = args.inference_run_file

        if args.delta is not None:
            kwargs['delta'] = args.delta

        if args.top is not None:
            kwargs['top'] = args.top

        inference(
            all_year=args.all_year,
            winter=args.winter,
            **kwargs
        )

    # ------------------------------------------------------
    # Block B.1: Create Anomaly Samples for Prediction
    # ------------------------------------------------------
    if args.create_ts_samples:
        # Soiling, Shading, Cracks
        create_ts_samples()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full ML workflow.")
    
    # ==========================================================
    # A. CLASSIFICATION WORKFLOW OPTIONS
    # ==========================================================

    # ----------------------------------------------------------
    # Block A.1.1: Create Samples for Classification
    # ----------------------------------------------------------
    parser.add_argument('--create_samples', action='store_true', help="Create sample dataset.")
    parser.add_argument('--create_samples_with_plots', action='store_true', help="Plot sample data.")

    # ----------------------------------------------------------
    # Block A.1.2: Create Train and Test Sets
    # ----------------------------------------------------------
    parser.add_argument('--create_train_test', action='store_true', help="Create train/test sets.")

    # ----------------------------------------------------------
    # Block A.1.3: Run the XGBoost Model
    # ----------------------------------------------------------
    parser.add_argument('--xgboost_run', action='store_true', help="Run XGBoost model.")

    # ----------------------------------------------------------
    # Block A.2: Inference Workflow Options
    # ----------------------------------------------------------
    parser.add_argument('--all_year', action='store_true', help="Select all months [Default: Summer].")
    parser.add_argument('--winter', action='store_true', help="Select winter months [Default: Summer].")
    # ----------------------------------------------------------
    # Block A.2.1: Create and Preprocess the Inference Test Set
    # ----------------------------------------------------------
    parser.add_argument('--create_inference_set', action='store_true', help="Create and preprocess inference test set.")
    parser.add_argument('--create_inference_set_smooth', action='store_true', help="Apply smoothing to inference test set.")

    # ----------------------------------------------------------
    # Block A.2.2: Perform Inference on Real Data
    # ----------------------------------------------------------
    parser.add_argument('--inference_run', action='store_true', help="Run inference.")
    parser.add_argument('--inference_run_file', type=str, help="Path to the inference test set CSV file.")
    parser.add_argument('--delta', type=float, help="Delta value for adjusting probabilities [Default: 0.2].")
    parser.add_argument('--top', type=int, help="Top N probabilities to consider for adjustment [Default: 2].")
    
    # ==========================================================
    # B. PREDICTION WORKFLOW OPTIONS
    # ==========================================================
    
    # ----------------------------------------------------------
    # Block B.1: Create Anomaly Samples for Prediction
    # ----------------------------------------------------------
    parser.add_argument('--create_ts_samples', action='store_true', help="Create anomaly time series samples.")
    
    # ----------------------------------------------------------
    # Block B.2: Perform Daily Classification in Time Series
    # ----------------------------------------------------------
    
    # ----------------------------------------------------------
    # Block B.3: Predict Anomalies in Time Series
    # ----------------------------------------------------------
    
    args = parser.parse_args()
    run(args)
