import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

from plot_training import *
from utils import *

# ==============================================================
# XGBoost Classifier for Anomaly/Fault Prediction
# ==============================================================
def xgboost_classifier(all_year=False, winter=False):
    """
    Run a complete XGBoost classification workflow for anomaly and fault prediction
    using synthetic PV system data.

    This function performs the following steps:
        1. Loads and filters the dataset by season.
        2. Prepares labels and features for training, validation, and test sets.
        3. Splits the training dataset into train and validation subsets with stratification.
        4. Performs Stratified K-Fold cross-validation on the training set.
        5. Trains the XGBoost classifier on the training set.
        6. Saves the trained model for future use.
        7. Evaluates model performance on validation and test sets:
            - Computes accuracy
            - Produces classification reports
            - Calculates per-class accuracies
        8. Plots and saves:
            - Confusion matrices
            - Class-wise accuracy bars for validation and test sets
            - Feature importance (top N features)
        9. Computes additional cross-validation metrics (accuracy, precision, recall, f1)
            and saves raw and summary CSV files.
        10. Generates Precision–Recall (AUC) and FP–TP curves for validation and test sets.
        11. Exports global performance metrics (accuracy, precision, recall, f1 for validation and test)
            to an accumulative CSV file.
        12. Calculates class-wise accuracy with bootstrap confidence intervals.
        13. Prints a final performance summary including CV, validation, and test accuracies.

    Args:
        all_year (bool, optional): If True, includes data from all months.
        winter (bool, optional): If True, filters the dataset for winter months only.

    Inputs (global variables):
        - `TRAIN_VALID_SET_FILE`, `TEST_SET_FILE`: Paths to training/validation and test datasets.
        - `LABEL`: Target column name.
        - `MODEL_FOLDER`: Folder to save the trained XGBoost model.
        - `IMAGE_FOLDER`, `REPORT_FOLDER`: Directories for output plots and reports.
        - `TOP_FEATURES`: Number of top features to display in importance plots.
        - `LABELS_MAP`: Mapping of class indices to descriptive class names.
        
    Outputs:
        - Saved XGBoost model.
        - CSVs with:
            - Validation and test classification reports
            - Per-class accuracies
            - Top N feature importances
            - Cross-validation raw scores and summary
            - Accumulative performance metrics (accuracy, precision, recall, f1)
        - Plots saved in `IMAGE_FOLDER`:
            - Confusion matrices
            - Class-wise accuracy bars
            - Feature importance
            - Precision vs Recall curves
            - FP vs TP curves
            - Class-wise accuracy with bootstrap CIs
        - Printed performance summary to console (CV, validation, and test accuracies).
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================
    # Determine the active season, its corresponding months, and a formatted name for file usage
    season_name, months, season_name_file = determine_season(all_year, winter)
    
    # OUTPUT FILES
    # Model
    model_path = f"{MODELS_FOLDER}/xgb_best_model_{season_name_file}.joblib"
    # Confusion Matrices
    cm_validation_image_path = f"{IMAGE_FOLDER}/xgb_confusion_matrix_validation_{season_name_file}.png"
    cm_test_image_path = f"{IMAGE_FOLDER}/xgb_confusion_matrix_test_{season_name_file}.png"
    # Several Metrics
    val_report_path = f"{REPORT_FOLDER}/xgb_validation_classification_report_{season_name_file}.csv"
    test_report_path = f"{REPORT_FOLDER}/xgb_test_classification_report_{season_name_file}.csv"
    combined_acc_path = f"{REPORT_FOLDER}/xgb_class_accuracies_{season_name_file}.csv"
    val_class_acc_image_path = f"{IMAGE_FOLDER}/xgb_val_class_accuracy_{season_name_file}.png"
    test_class_acc_image_path = f"{IMAGE_FOLDER}/xgb_test_class_accuracy_{season_name_file}.png"
    # Top Features
    top_features_file_path = f"{REPORT_FOLDER}/xgb_top_{TOP_FEATURES}_features_{season_name_file}.csv"
    top_features_image_path = f"{IMAGE_FOLDER}/xgb_feature_importance_{TOP_FEATURES}_{season_name_file}.png"
    # Metrics Reports
    cv_metrics_path = f"{REPORT_FOLDER}/xgb_cross_validation_raw_scores_{season_name_file}.csv"
    summary_metrics_path = f"{REPORT_FOLDER}/xgb_cross_validation_summary_{season_name_file}.csv"
    # AUC and FP-TP Plots
    auc_val_image_path = f"{IMAGE_FOLDER}/xgb_auc_val_precision_vs_recall_{season_name_file}.png"
    auc_test_image_path = f"{IMAGE_FOLDER}/xgb_auc_test_precision_vs_recall_{season_name_file}.png"
    ft_tp_val_image_path = f"{IMAGE_FOLDER}/xgb_fp_tp_curve_val_{season_name_file}.png"
    ft_tp_test_image_path = f"{IMAGE_FOLDER}/xgb_fp_tp_curve_test_{season_name_file}.png"
    # Bootstrap CI Plot
    ci_image_path = f"{IMAGE_FOLDER}/xgb_class_accuracy_ci_{season_name_file}.png"
    # Overall Performance
    performance_csv_path = f"{REPORT_FOLDER}/overall_performance.csv"

    print("\n" + "=" * 60)
    print(f"RUNNING XGBOOST CLASSIFIER, TRAINING SEASON: {season_name.upper()} ...")
    print("=" * 60)

    # ==========================================================
    # LOAD AND FILTER DATASETS
    # ==========================================================
    print("\nLoading datasets...")
    df_train_valid = pd.read_csv(TRAIN_VALID_SET_FILE)
    df_test = pd.read_csv(TEST_SET_FILE)

    # Convert 'date' column to datetime format
    df_train_valid['date'] = pd.to_datetime(df_train_valid['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])

    # Filter data by selected months (season)
    df_train_valid = df_train_valid[df_train_valid['date'].dt.month.isin(months)]
    df_test = df_test[df_test['date'].dt.month.isin(months)]

    print(f"\nDataset shape: {df_test.shape}")

    # ==========================================================
    # LABEL PREPARATION
    # ==========================================================
    # Convert labels to integers (removes .0 if present)
    df_train_valid[LABEL] = df_train_valid[LABEL].astype(int)
    df_test[LABEL] = df_test[LABEL].astype(int)
    print(f"\nLabel value counts:\n{df_test[LABEL].value_counts()}")

    # ==========================================================
    # FEATURE AND TARGET PREPARATION
    # ==========================================================
    # Remove 'date' column and separate target variable
    X_train_valid = df_train_valid.drop(columns=[LABEL, 'date'])
    y_train_valid = df_train_valid[LABEL]

    X_test = df_test.drop(columns=[LABEL, 'date'])
    y_test = df_test[LABEL]

    # ==========================================================
    # SPLIT DATASET INTO TRAIN (70%) AND VALIDATION (30%)
    # ==========================================================
    print("\nSplitting dataset into Train and Validation sets...")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid, y_train_valid,
        test_size=0.3,
        random_state=42,
        stratify=y_train_valid)

    print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_train_valid)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X_train_valid)*100:.1f}%)")

    # Check class distribution in each set
    print(f"\nClass distribution in Train set:")
    print(y_train.value_counts(normalize=True).sort_index())
    print(f"\nClass distribution in Validation set:")
    print(y_val.value_counts(normalize=True).sort_index())
    print(f"\nClass distribution in Test set:")
    print(y_test.value_counts(normalize=True).sort_index())

    # ==========================================================
    # CROSS-VALIDATION ON TRAINING SET
    # ==========================================================
    print("\nPerforming Cross-Validation on Training set...")

    # Initialize the classifier
    xgb_classifier = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    # Stratified K-Fold Cross-Validation ensures balanced class splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Compute cross-validation accuracy scores
    cv_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # ==========================================================
    # FINAL MODEL TRAINING
    # ==========================================================
    print("\nTraining XGBoost classifier...")
    xgb_classifier.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(xgb_classifier, model_path)
    print(f"\nModel saved to {model_path}")
    
    # ==========================================================
    # MODEL EVALUATION ON VALIDATION AND TEST SETS
    # ==========================================================
    print("\nEvaluating model performance...")

    # Generate predictions
    y_val_pred = xgb_classifier.predict(X_val)
    y_test_pred = xgb_classifier.predict(X_test)

    # Compute accuracies
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # ==========================================================
    # CONFUSION MATRIX PLOTS
    # ==========================================================
    # Plot and save validation confusion matrix
    val_cm, _ = plot_confusion_matrix_combined(
        "Validation",
        season_name,
        y_val,
        y_val_pred,
        cm_validation_image_path,
    )
    
    # Plot and save test confusion matrix
    test_cm, _ = plot_confusion_matrix_combined(
        "Test",
        season_name,
        y_test,
        y_test_pred,
        cm_test_image_path,
    )
    
    # ==========================================================
    # CLASSIFICATION REPORTS
    # ==========================================================
    classes = [int(cls) for cls in np.unique(y_test)]
    class_names = [LABELS_MAP[cls][0] for cls in classes]

    # Validation report
    val_report = classification_report(y_val, y_val_pred, target_names=class_names, digits=4, output_dict=True)
    val_report_df = pd.DataFrame(val_report).transpose()
    val_report_df.to_csv(val_report_path, index=True)

    # Test report
    test_report = classification_report(y_test, y_test_pred, target_names=class_names, digits=4, output_dict=True)
    test_report_df = pd.DataFrame(test_report).transpose()
    test_report_df.to_csv(test_report_path, index=True)

    print(f"\nValidation Set Classification Report saved to {val_report_path}")
    print(f"Test Set Classification Report saved to {test_report_path}")

    # ==========================================================
    # REPORT AND PLOT PER-CLASS ACCURACY
    # ==========================================================
    val_class_acc = class_accuracy(val_cm)
    test_class_acc = class_accuracy(test_cm)

    # Validation accuracies
    val_acc_df = pd.DataFrame({
        "Class": class_names,
        "Validation Accuracy (%)": [round(acc, 1) for acc in val_class_acc]
    })

    # Test accuracies
    test_acc_df = pd.DataFrame({
        "Class": class_names,
        "Test Accuracy (%)": [round(acc, 1) for acc in test_class_acc]
    })

    # Combine both in a single DataFrame (optional)
    combined_acc_df = pd.merge(val_acc_df, test_acc_df, on="Class")

    # Save to CSV
    combined_acc_df.to_csv(combined_acc_path, index=False)
    print(f"\nPer-class accuracies saved to {combined_acc_path}")

    # Plot class accuracy bars
    plot_class_accuracy(val_class_acc, classes,
                        f"Validation Accuracy per Class, {season_name.title()}", val_class_acc_image_path)
    plot_class_accuracy(test_class_acc, classes, 
                        f"Test Accuracy per Class, {season_name.title()}", test_class_acc_image_path)
    
    # ==========================================================
    # FEATURE IMPORTANCE EXPORT AND PLOT
    # ==========================================================
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train_valid.columns,
        'importance': xgb_classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    # Export to CSV
    top_features = feature_importance.head(TOP_FEATURES)
    top_features.to_csv(top_features_file_path, index=False)
    print(f"\nTop {TOP_FEATURES} most important features saved to {top_features_file_path}")
    
    # Plot feature importance  
    plot_feature_importance(top_features, season_name, top_features_image_path)
    
    # ==========================================================
    # ADDITIONAL CROSS-VALIDATION METRICS
    # ==========================================================
    print("\nComputing additional cross-validation metrics...")
    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    cv_results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(xgb_classifier, X_train, y_train, cv=cv, scoring=metric)
        cv_results[metric] = scores

    # Transform CV results into DataFrame
    cv_df = pd.DataFrame(cv_results)

    # Compute mean and 2*std for each metric
    summary_df = pd.DataFrame({
        "Metric": cv_df.columns,
        "Mean": cv_df.mean().values,
        "2*Std": (cv_df.std() * 2).values
    })

    # Save raw CV scores and summary to CSV
    cv_df.to_csv(cv_metrics_path, index=False)
    summary_df.to_csv(summary_metrics_path, index=False)

    print(f"Cross-validation raw scores saved to {cv_metrics_path}")
    print(f"Cross-validation summary saved to {summary_metrics_path}")

    # ==========================================================
    # COMPUTE PREDICTED PROBABILITIES FOR PRECISION VS RECALL
    # ==========================================================
    y_val_proba = xgb_classifier.predict_proba(X_val)
    y_test_proba = xgb_classifier.predict_proba(X_test)

    # Plot Precision vs Recall curves for validation set
    plot_auc_recall_vs_precision(
        y_val,
        y_val_proba,
        class_names,
        auc_val_image_path
    )

    # Plot Precision vs Recall curves for test set
    plot_auc_recall_vs_precision(
        y_test,
        y_test_proba,
        class_names,
        auc_test_image_path
    )
    
    # ==========================================================
    # COMPUTE PREDICTED PROBABILITIES FOR FP VS TP PLOTS
    # ==========================================================
    # Plot FP vs TP curves for validation and test sets
    plot_fp_tp_curve(
        y_val,
        y_val_proba,
        class_names,
        ft_tp_val_image_path
    )

    plot_fp_tp_curve(
        y_test,
        y_test_proba,
        class_names,
        ft_tp_test_image_path
    )
    
    # ==============================================================
    # EXPORT ACCURACY, PRECISION, RECALL, AND F1 TO ACCUMULATIVE CSV
    # ==============================================================
    # Compute overall metrics for validation and test sets
    val_precision = val_report_df.loc["weighted avg", "precision"]
    val_recall = val_report_df.loc["weighted avg", "recall"]
    val_f1 = val_report_df.loc["weighted avg", "f1-score"]

    test_precision = test_report_df.loc["weighted avg", "precision"]
    test_recall = test_report_df.loc["weighted avg", "recall"]
    test_f1 = test_report_df.loc["weighted avg", "f1-score"]

    # Create a DataFrame for the current run
    new_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Season": season_name,
        "Model": "XGBoost",
        "Validation_Accuracy": val_accuracy,
        "Validation_Precision": val_precision,
        "Validation_Recall": val_recall,
        "Validation_F1": val_f1,
        "Test_Accuracy": test_accuracy,
        "Test_Precision": test_precision,
        "Test_Recall": test_recall,
        "Test_F1": test_f1
    }])

    # If file exists, append; otherwise, create new file with header
    if os.path.exists(performance_csv_path):
        existing_df = pd.read_csv(performance_csv_path)
        combined_df = pd.concat([existing_df, new_entry], ignore_index=True)
        combined_df.to_csv(performance_csv_path, index=False)
    else:
        new_entry.to_csv(performance_csv_path, index=False)

    print(f"\nOverall performance metrics saved to {performance_csv_path}")

    # ==========================================================
    # RUN XGBOOST CLASSIFIER WITH BOOTSTRAP CONFIDENCE INTERVALS
    # ==========================================================
    print("\nCalculating class-wise accuracy with bootstrap confidence intervals...")
    
    # Parameters for bootstrapping
    n_bootstrap = 1000
    ci_level = 95
    
   # Bootstrapping for class-wise CI
    classes = sorted(y_test.unique())
    class_acc_samples = []

    for i in range(n_bootstrap):
        if i % 50 == 0:
            print(f"Bootstrap sample {i}/{n_bootstrap}...")
        X_res, y_res = resample(X_test, y_test, replace=True, stratify=y_test)
        y_pred_res = xgb_classifier.predict(X_res)
        cm_res = confusion_matrix(y_res, y_pred_res, labels=classes)
        acc_per_class = class_accuracy(cm_res)
        class_acc_samples.append(acc_per_class)

    class_acc_samples = np.array(class_acc_samples)
    mean_acc = np.mean(class_acc_samples, axis=0)
    lower_bounds = np.percentile(class_acc_samples, (100 - ci_level)/2, axis=0)
    upper_bounds = np.percentile(class_acc_samples, 100 - (100 - ci_level)/2, axis=0)

    # Print results
    for idx, cls in enumerate(classes):
        class_name = LABELS_MAP[cls][0]
        print(f"{class_name}: Accuracy = {mean_acc[idx]:.3f} [{lower_bounds[idx]:.3f} - {upper_bounds[idx]:.3f}] ({ci_level}% CI)")

    # Plot with CI
    plot_class_accuracy_ci(mean_acc, lower_bounds, upper_bounds, classes, f"Class-wise Accuracy with {ci_level}% CI", ci_image_path)
    
    # ==========================================================
    # FINAL PERFORMANCE SUMMARY
    # ==========================================================
    print("\n" + "-"*40)
    print("XGBOOST FINAL PERFORMANCE SUMMARY")
    print("-"*40)
    print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Validation Set Accuracy: {val_accuracy:.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("-"*40)
    