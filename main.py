import argparse
from create_day_samples import *
from create_train_test_sets import *
from xgboost_classifier import *
from create_preprocess_inference_set import *
from inference import *


def run(args): 
    # ------------------------------------------------------
    # Block 1: Create Samples
    # ------------------------------------------------------
    if args.create_samples or args.create_samples_with_plots:
        create_day_samples(
            plot_samples=args.create_samples_with_plots,
        )

    # ------------------------------------------------------
    # Block 2: Create Train and Test Sets
    # ------------------------------------------------------
    if args.create_train_test:
        create_train_test_sets()

    # ------------------------------------------------------
    # Block 3: Run the XGBoost Model
    # ------------------------------------------------------
    if args.xgboost_run:
        xgboost_classifier(
            all_year=args.all_year, 
            winter=args.winter,
        )
        
    # ------------------------------------------------------
    # Block 4: Create and Preprocess the Inference Test Set
    # ------------------------------------------------------
    if (args.create_inference_set
        or args.create_inference_set_smooth):
        create_preprocess_inference_set(
            smoothing=args.create_inference_set_smooth,
            all_year=args.all_year, 
            winter=args.winter,
        )

    # ------------------------------------------------------
    # Block 5: Perform Inference on Real Data
    # ------------------------------------------------------
    if args.inference_run:
        inference(
            all_year=args.all_year, 
            winter=args.winter,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full ML workflow.")

    # ------------------------------------------------------
    # Block 1: Create Samples
    # ------------------------------------------------------
    parser.add_argument('--create_samples', action='store_true', help="Create sample dataset")
    parser.add_argument('--create_samples_with_plots', action='store_true', help="Plot sample data")

    # ------------------------------------------------------
    # Block 2: Create Train and Test Sets
    # ------------------------------------------------------
    parser.add_argument('--create_train_test', action='store_true', help="Create train/test sets")

    # ------------------------------------------------------
    # Block 3: Run the XGBoost Model
    # ------------------------------------------------------
    parser.add_argument('--xgboost_run', action='store_true', help="Run XGBoost model")

    # ------------------------------------------------------
    # Block 4: Create and Preprocess the Inference Test Set
    # ------------------------------------------------------
    parser.add_argument('--all_year', action='store_true', help="Select all months")
    parser.add_argument('--winter', action='store_true', help="Select winter months")
    parser.add_argument('--create_inference_set', action='store_true', help="Create and preprocess inference test set")
    parser.add_argument('--create_inference_set_smooth', action='store_true', help="Apply smoothing to inference test set")

    # ------------------------------------------------------
    # Block 5: Perform Inference on Real Data
    # ------------------------------------------------------
    parser.add_argument('--inference_run', action='store_true', help="Perform inference on real data")

    args = parser.parse_args()
    run(args)
