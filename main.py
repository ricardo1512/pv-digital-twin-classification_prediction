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
    # -------------------------------------------------------------------------------
    # Block A.1.1: Create Samples for Classification
    # -------------------------------------------------------------------------------
    if args.create_samples or args.create_samples_with_plots:
        create_day_samples(
            plot_samples=args.create_samples_with_plots,
        )

    # -------------------------------------------------------------------------------
    # Block A.1.2: Create Train and Test Sets
    # -------------------------------------------------------------------------------
    if args.create_train_test:
        create_train_test_sets()

    # -------------------------------------------------------------------------------
    # Block A.1.3: Run the XGBoost Model
    # -------------------------------------------------------------------------------
    if args.xgboost_run:
        xgboost_classifier(
            all_year=args.all_year, 
            winter=args.winter,
        )
        
    # -------------------------------------------------------------------------------
    # Block A.2.1: Create and Preprocess the Inference Test Set
    # -------------------------------------------------------------------------------
    if args.create_inference_set or args.create_inference_set_smooth:
        create_preprocess_inference_set(
            smoothing=args.create_inference_set_smooth,
            all_year=args.all_year, 
            winter=args.winter,
        )

    # -------------------------------------------------------------------------------
    # Block A.2.2: Perform Inference on Real Data
    # -------------------------------------------------------------------------------
    if args.inference_run or args.delta or args.top:
        kwargs = {}

        if isinstance(args.inference_run, str):
            kwargs['inference_test_file'] = args.inference_run

        if args.delta is not None:
            kwargs['delta'] = args.delta

        if args.top is not None:
            kwargs['top'] = args.top

        inference(
            all_year=args.all_year,
            winter=args.winter,
            **kwargs
        )

    # -------------------------------------------------------------------------------
    # Block B.1: Create Anomaly Samples for Prediction, with Plots
    # -------------------------------------------------------------------------------
    if args.create_ts_samples:
        # Soiling, Shading, Cracks
        create_ts_samples()

    # -------------------------------------------------------------------------------
    # Block B.2.1: Perform Daily Classification in Synthetic Time Series, with Plots
    # -------------------------------------------------------------------------------
    if args.synthetic_ts_daily_classification:
        synthetic_ts_daily_classification()
    
    # -------------------------------------------------------------------------------
    # Block B.2.2: Perform Daily Classification in Real Time Series, with Plots
    # -------------------------------------------------------------------------------
    if args.real_ts_daily_classification or args.ts_smooth:
        ts_daily_classification(
            input_file=args.real_ts_daily_classification,
            smooth=args.ts_smooth
        )

    # -------------------------------------------------------------------------------
    # Block B.3.1: Predict Anomalies in Synthetic Time Series, with Plots
    # -------------------------------------------------------------------------------
    if args.synthetic_ts_predict_days or args.synt_threshold_start \
        or args.synt_threshold_target or args.synt_threshold_class or args.synt_window:
        kwargs = {}

        if args.synt_threshold_start is not None:
            kwargs['threshold_start'] = args.synt_threshold_start

        if args.synt_threshold_target is not None:
            kwargs['threshold_target'] = args.synt_threshold_target

        if args.synt_threshold_class is not None:
            kwargs['threshold_class'] = args.synt_threshold_class

        if args.synt_window is not None:
            kwargs['window'] = args.synt_window
            
        synthetic_ts_predict_days(**kwargs)

    # ------------------------------------------------------------------------------
    # Block B.3.2: Predict Anomalies in Real Time Series, with Plots
    # ------------------------------------------------------------------------------
    if args.ts_predict_days or args.real_threshold_start \
        or args.real_threshold_target or args.real_threshold_class or args.real_window: 
        kwargs = {}
        
        if args.real_threshold_start is not None:
            kwargs['threshold_start'] = args.real_threshold_start  
        
        if args.real_threshold_target is not None:
            kwargs['threshold_target'] = args.real_threshold_target
        
        if args.real_threshold_class is not None:
            kwargs['threshold_class'] = args.real_threshold_class
        
        if args.real_window is not None:
            kwargs['window'] = args.real_window
            
        ts_predict_days(
            input_csv_path=args.ts_predict_days,
            **kwargs
        )

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full ML workflow.")
    
    # ==============================================================================
    # A. CLASSIFICATION WORKFLOW OPTIONS
    # ==============================================================================

    # ------------------------------------------------------------------------------
    # Block A.1.1: Create Samples for Classification
    # ------------------------------------------------------------------------------
    parser.add_argument('--create_samples', action='store_true', help="Create sample dataset.")
    parser.add_argument('--create_samples_with_plots', action='store_true', help="Plot sample data.")

    # ------------------------------------------------------------------------------
    # Block A.1.2: Create Train and Test Sets
    # ------------------------------------------------------------------------------
    parser.add_argument('--create_train_test', action='store_true', help="Create train/test sets.")

    # ------------------------------------------------------------------------------
    # Block A.1.3: Run the XGBoost Model
    # ------------------------------------------------------------------------------
    parser.add_argument('--xgboost_run', action='store_true', help="Run XGBoost model.")

    # ------------------------------------------------------------------------------
    # Block A.2: Inference Workflow Options
    # ------------------------------------------------------------------------------
    parser.add_argument('--all_year', action='store_true', help="Select all months [Default: Summer].")
    parser.add_argument('--winter', action='store_true', help="Select winter months [Default: Summer].")
    # ------------------------------------------------------------------------------
    # Block A.2.1: Create and Preprocess the Inference Test Set
    # ------------------------------------------------------------------------------
    parser.add_argument('--create_inference_set', action='store_true', 
                        help="Create and preprocess inference test set.")
    parser.add_argument('--create_inference_set_smooth', type=int, default=48,
                        help="Create and preprocess inference test set applying smoothing [Default: 48 (4 hours)].")

    # ------------------------------------------------------------------------------
    # Block A.2.2: Perform Inference on Real Data    
    # ------------------------------------------------------------------------------
    parser.add_argument('--inference_run', type=str, 
                        help="Run inference with option of the path to the inference test set CSV file.")
    parser.add_argument('--delta', type=float, help="Delta value for adjusting probabilities [Default: 0.2].")
    parser.add_argument('--top', type=int, help="Top N probabilities to consider for adjustment [Default: 2].")
    
    # ==============================================================================
    # B. PREDICTION WORKFLOW OPTIONS
    # ==============================================================================
    
    # ------------------------------------------------------------------------------
    # Block B.1: Create Anomaly Samples for Prediction, with Plots
    # ------------------------------------------------------------------------------
    parser.add_argument('--create_ts_samples', action='store_true', help="Create anomaly time series samples.")
    
    # ------------------------------------------------------------------------------
    # Block B.2.1: Perform Daily Classification in Synthetic Time Series, with Plots
    # ------------------------------------------------------------------------------
    parser.add_argument('--synthetic_ts_daily_classification', action='store_true', 
                        help="Perform daily classification in synthetic time series.")
    # ------------------------------------------------------------------------------
    # Block B.2.2: Perform Daily Classification in Real Time Series, with Plots
    # ------------------------------------------------------------------------------
    parser.add_argument('--real_ts_daily_classification', type=str, 
                        help="Perform daily classification with path to real time series.")
    parser.add_argument('--ts_smooth', type=int, 
                        help="Apply smoothing to real time series [Default: 48 (4 hours)].")
    
    # ------------------------------------------------------------------------------
    # Block B.3.1: Predict Anomalies in Synthetic Time Series, with Plots
    # ------------------------------------------------------------------------------
    parser.add_argument('--synthetic_ts_predict_days', action='store_true', 
                        help="Perform prediction in synthetic time series.")
    parser.add_argument('--synt_threshold_start', type=float, 
                        help="Threshold to start predicting an anomaly [Default: 0.5].")
    parser.add_argument('--synt_threshold_target', type=float, 
                        help="Target threshold for predicting an anomaly [Default: 0.8].")
    parser.add_argument('--synt_threshold_class', type=float, 
                        help="Class threshold for predicting an anomaly [Default: 0.2].")
    parser.add_argument('--synt_window', type=int, 
                        help="Window size for prediction [Default: 30].")
    
    # ------------------------------------------------------------------------------
    # Block B.3.2: Predict Anomalies in Real Time Series, with Plots
    # ------------------------------------------------------------------------------
    parser.add_argument('--ts_predict_days', type=str, 
                        help="Perform prediction with path to real time series.")
    parser.add_argument('--real_threshold_start', type=float, 
                        help="Threshold to start predicting an anomaly [Default: 0.5].")
    parser.add_argument('--real_threshold_target', type=float, 
                        help="Target threshold for predicting an anomaly [Default: 0.8].")
    parser.add_argument('--real_threshold_class', type=float, 
                        help="Class threshold for predicting an anomaly [Default: 0.2].")
    parser.add_argument('--real_window', type=int, 
                        help="Window size for prediction [Default: 30].")
    
    args = parser.parse_args()
    run(args)
