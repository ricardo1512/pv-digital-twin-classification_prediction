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
from prediction import *


def run(args):
    # =======================================================================================
    # A. CLASSIFICATION WORKFLOW (DAILY SAMPLES)
    # =======================================================================================
    
    # ---------------------------------------------------------------------------------------
    # A.1. Training Workflow
    # ---------------------------------------------------------------------------------------
    
    # A.1.1. Create Samples for Classification
    if args.create_samples or args.create_samples_with_plots:
        create_day_samples(
            plot_samples=args.create_samples_with_plots,
        )

    # A.1.2. Create Train and Test Sets
    if args.create_train_test:
        create_train_test_sets()

    # A.1.3. Run the XGBoost Model
    if args.xgboost_run:
        xgboost_classifier(
            all_year=args.all_year, 
            winter=args.winter,
        )
    
    # ---------------------------------------------------------------------------------------
    # A.2. Inference Workflow
    # ---------------------------------------------------------------------------------------
    
    # A.2.1. Create and Preprocess the Inference Test Set
    if args.create_inference_set or args.create_inference_set_smooth:
        create_preprocess_inference_set(
            smoothing=args.create_inference_set_smooth,
            all_year=args.all_year, 
            winter=args.winter,
        )

    # A.2.2. Perform Inference on Real Data
    if args.inference_run or args.inference_smooth \
        or args.inference_run_user or args.delta or args.top:
        inference(
            all_year=args.all_year,
            winter=args.winter,
            inference_user=args.inference_run_user,
            smoothing=args.inference_smooth,
            delta=args.delta,
            top=args.top,
        )

    # =======================================================================================
    # B. PREDICTION WORKFLOW (TIME SERIES)
    # =======================================================================================
    
    # ---------------------------------------------------------------------------------------
    # B.1. Create Anomaly Samples for Prediction, with Plots
    # ---------------------------------------------------------------------------------------
    if args.create_ts_samples:
        # Soiling, Shading, Cracks
        create_ts_samples()
        
    # ---------------------------------------------------------------------------------------
    # B.2. Synthetic Time Series Prediction, with Plots
    # ---------------------------------------------------------------------------------------
    
    # B.2.1. Perform Daily Classification
    if args.synthetic_ts_daily_classification:
        synthetic_ts_daily_classification()

    # B.2.2. Predict Anomalies
    if args.synthetic_ts_predict_days or args.synt_threshold_start \
        or args.synt_threshold_target or args.synt_threshold_class or args.synt_window:
      
        synthetic_ts_predict_days(
            threshold_start=args.synt_threshold_start,
            threshold_target=args.synt_threshold_target,
            threshold_class=args.synt_threshold_class,
            window=args.synt_window,
        )

    # ----------------------------------------------------------------------------------------
    # B.3. Perform Daily Classification and Prediction in Real Time Series, with Plots
    # ----------------------------------------------------------------------------------------
    if args.real_ts_prediction or args.ts_smooth or args.real_threshold_start \
        or args.real_threshold_target or args.real_threshold_class or args.real_window:
        # Daily Classification
        output_path_classification = ts_daily_classification(
            input_file=args.real_ts_prediction,
            all_year=args.all_year,
            winter=args.winter,
            smoothing=args.ts_smooth
        )
        
        # Prediction
        ts_predict_days(
            input_csv_path=output_path_classification,
            threshold_start=args.real_threshold_start,
            threshold_target=args.real_threshold_target,
            threshold_class=args.real_threshold_class,
            window=args.real_window 
        )
    
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Run the full ML workflow.")
    
    # ---------------------------------------------------------------------------------------
    # Classification, Inference and Prediction Season Options
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--all_year', action='store_true', help="Select all months [Default: Summer].")
    parser.add_argument('--winter', action='store_true', help="Select winter months [Default: Summer].")
    
    # =======================================================================================
    # A. CLASSIFICATION WORKFLOW (DAILY SAMPLES)
    # =======================================================================================
    
    # ---------------------------------------------------------------------------------------
    # A.1. Training Workflow
    # ---------------------------------------------------------------------------------------
    
    # A.1.1. Create Samples for Classification
    parser.add_argument('--create_samples', action='store_true', help="Create sample dataset.")
    parser.add_argument('--create_samples_with_plots', action='store_true', help="Plot sample data.")

    # A.1.2. Create Train and Test Sets
    parser.add_argument('--create_train_test', action='store_true', help="Create train/test sets.")

    # A.1.3. Run the XGBoost Model
    parser.add_argument('--xgboost_run', action='store_true', help="Run XGBoost model.")

    # ---------------------------------------------------------------------------------------
    # A.2. Inference Workflow
    # ---------------------------------------------------------------------------------------
    # A.2.1. Create and Preprocess the Inference Test Set
    parser.add_argument('--create_inference_set', action='store_true', 
                        help="Create and preprocess inference test set.")
    parser.add_argument('--create_inference_set_smooth', type=int, default=24,
                        help="Create and preprocess inference test set applying smoothing [Default: 24 (2 hours)].")

    # A.2.2. Perform Inference on Real Data
    parser.add_argument('--inference_run', action='store_true', 
                        help="Run inference using the available real set.")
    parser.add_argument('--inference_run_user', action='store_true', 
                        help="Run inference providing new real data.")
    parser.add_argument('--inference_smooth', type=int, default=24,
                        help="Smoothing window for the new real data [Default: 24 (2 hours)].")
    parser.add_argument('--delta', type=float, default=0.2,
                        help="Delta value for adjusting probabilities [Default: 0.2].")
    parser.add_argument('--top', type=int, default=2,
                        help="Top N probabilities to consider for adjustment [Default: 2].")
    
    
    # =======================================================================================
    # B. PREDICTION WORKFLOW (TIME SERIES)
    # =======================================================================================
    
    # ---------------------------------------------------------------------------------------
    # B.1. Create Anomaly Samples for Prediction, with Plots
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--create_ts_samples', action='store_true', help="Create anomaly time series samples.")
    
    # ---------------------------------------------------------------------------------------
    # B.2. Synthetic Time Series Prediction, with Plots
    # ---------------------------------------------------------------------------------------
    # B.2.1. Perform Daily Classification
    parser.add_argument('--synthetic_ts_daily_classification', action='store_true', 
                        help="Perform daily classification in synthetic time series.")
    
    # B.2.2. Predict Anomalies
    parser.add_argument('--synthetic_ts_predict_days', action='store_true', 
                        help="Perform prediction in synthetic time series.")
    parser.add_argument('--synt_threshold_start', type=float, default=0.5,
                        help="Threshold to start predicting an anomaly [Default: 0.5].")
    parser.add_argument('--synt_threshold_target', type=float, default=0.8,
                        help="Target threshold for predicting an anomaly [Default: 0.8].")
    parser.add_argument('--synt_threshold_class', type=float, default=0.2,
                        help="Class threshold for predicting an anomaly [Default: 0.2].")
    parser.add_argument('--synt_window', type=int, default=30,
                        help="Window size for prediction [Default: 30].")
    
    # ---------------------------------------------------------------------------------------
    # B.3. Perform Daily Classification and Prediction in Real Time Series, with Plots
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--real_ts_prediction', type=str, 
                        help="Perform daily classification with path to real time series.")
    parser.add_argument('--ts_smooth', type=int, default=48,
                        help="Apply smoothing to real time series [Default: 48 (4 hours)].")
    parser.add_argument('--real_threshold_start', type=float, default=0.5,
                        help="Threshold to start predicting an anomaly [Default: 0.5].")
    parser.add_argument('--real_threshold_target', type=float, default=0.8,
                        help="Target threshold for predicting an anomaly [Default: 0.8].")
    parser.add_argument('--real_threshold_class', type=float, default=0.2,
                        help="Class threshold for predicting an anomaly [Default: 0.2].")
    parser.add_argument('--real_window', type=int, default=30,
                        help="Window size for prediction [Default: 30].")
    
    # Parse arguments and run
    args = parser.parse_args()
    run(args)
