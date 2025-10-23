import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from plot import *
from utils import *

# ==============================================================
# Random Forest Classifier for Anomaly/Fault Prediction
# ==============================================================
def random_forest():
    """
    This function performs a complete Random Forest classification workflow
    for anomalies and fault using synthetic PV system data.
    It includes:
      - Data loading and filtering by season
      - Train/validation/test split
      - Stratified cross-validation
      - Model training, saving, and evaluation
      - Confusion matrix, class accuracy, and feature importance plots
    """

    # ==========================================================
    # Dataset Initialization
    # ==========================================================
    season_name = SEASON[0]
    months = SEASON[1]

    print("\n" + "=" * 40)
    print("RUNNING RANDOM FOREST CLASSIFIER...")
    print("=" * 40)

    # ==========================================================
    # Load and Filter Datasets
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
    # Label Preparation
    # ==========================================================
    # Convert labels to integers (removes .0 if present)
    df_train_valid[LABEL] = df_train_valid[LABEL].astype(int)
    df_test[LABEL] = df_test[LABEL].astype(int)
    print(f"\nLabel value counts:\n{df_test[LABEL].value_counts()}")

    # ==========================================================
    # Feature and Target Preparation
    # ==========================================================
    # Remove 'date' column and separate target variable
    X_train_valid = df_train_valid.drop(columns=[LABEL, 'date'])
    y_train_valid = df_train_valid[LABEL]

    X_test = df_test.drop(columns=[LABEL, 'date'])
    y_test = df_test[LABEL]

    print(f"\nLabel dtype: {y_train_valid.dtype}")
    print(f"Unique classes: {np.unique(y_train_valid)}")

    # ==========================================================
    # Split Dataset into Train (70%) and Validation (30%)
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
    # Cross-Validation on Training Set
    # ==========================================================
    print("\nPerforming Cross-Validation on Training set...")

    # Initialize the classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    # Stratified K-Fold Cross-Validation ensures balanced class splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Compute cross-validation accuracy scores
    cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # ==========================================================
    # Final Model Training
    # ==========================================================
    print("\nTraining Random Forest classifier...")
    rf_classifier.fit(X_train, y_train)

    # Save trained model for reuse
    joblib.dump(rf_classifier, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # ==========================================================
    # Model Evaluation on Validation and Test Sets
    # ==========================================================
    print("\nEvaluating model performance...")

    # Generate predictions
    y_val_pred = rf_classifier.predict(X_val)
    y_test_pred = rf_classifier.predict(X_test)

    # Compute accuracies
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    # ==========================================================
    # Confusion Matrix Plots
    # ==========================================================
    val_cm, val_cm_pct = plot_confusion_matrix_combined(
        season_name,
        IMAGE_FOLDER,
        y_val,
        y_val_pred,
        'Validation',
    )
    test_cm, test_cm_pct = plot_confusion_matrix_combined(
        season_name,
        IMAGE_FOLDER,
        y_test,
        y_test_pred,
        'Test',
    )

    # ==========================================================
    # Classification Reports
    # ==========================================================
    classes = [int(cls) for cls in np.unique(y_test)]
    class_names = [LABELS_MAP[cls][0] for cls in classes]

    # Validation report
    val_report = classification_report(y_val, y_val_pred, target_names=class_names, digits=4, output_dict=True)
    val_report_df = pd.DataFrame(val_report).transpose()
    val_report_df.to_csv(f"{REPORT_FOLDER}/validation_classification_report_{season_name.lower()}.csv", index=True)

    # Test report
    test_report = classification_report(y_test, y_test_pred, target_names=class_names, digits=4, output_dict=True)
    test_report_df = pd.DataFrame(test_report).transpose()
    test_report_df.to_csv(f"{REPORT_FOLDER}/test_classification_report_{season_name.lower()}.csv", index=True)

    print(f"\nValidation Set Classification Report saved to '{REPORT_FOLDER}/report_validation_classification_{season_name.lower()}.csv'")
    print(f"Test Set Classification Report saved to '{REPORT_FOLDER}/report_test_classification_{season_name.lower()}.csv'")

    # ==========================================================
    # Report and Plot Per-Class Accuracy
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
    combined_acc_df.to_csv(f"{REPORT_FOLDER}/class_accuracies_{season_name.lower()}.csv", index=False)

    print(f"\nPer-class accuracies saved to '{REPORT_FOLDER}/class_accuracies_{season_name.lower()}.csv'")

    # Plot class accuracy bars
    plot_class_accuracy(val_class_acc, classes, f"Validation Accuracy per Class, {season_name.title()}", IMAGE_FOLDER,
                        f"val_class_accuracy_{season_name.lower()}.png")
    plot_class_accuracy(test_class_acc, classes, f"Test Accuracy per Class, {season_name.title()}", IMAGE_FOLDER,
                        f"test_class_accuracy_{season_name.lower()}.png")

    # ==========================================================
    # Feature Importance Plot
    # ==========================================================
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train_valid.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    # Export to CSV
    top_features = feature_importance.head(20)
    top_features.to_csv(f"{REPORT_FOLDER}/top_{TOP_FEATURES}_features_{season_name.lower()}.csv", index=False)
    print(f"\nTop {TOP_FEATURES} most important features saved to '{REPORT_FOLDER}/top_{TOP_FEATURES}_features_{season_name.lower()}.csv'")

    plot_feature_importance(feature_importance, season_name, IMAGE_FOLDER, top_n=TOP_FEATURES)

    # ==========================================================
    # Additional Cross-Validation Metrics
    # ==========================================================
    print("\nComputing additional cross-validation metrics...")
    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    cv_results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring=metric)
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
    cv_df.to_csv(f"{REPORT_FOLDER}/cross_validation_raw_scores_{season_name.lower()}.csv", index=False)
    summary_df.to_csv(f"{REPORT_FOLDER}/cross_validation_summary_{season_name.lower()}.csv", index=False)

    print(f"Cross-validation raw scores saved to '{REPORT_FOLDER}/cross_validation_raw_scores_{season_name.lower()}.csv'")
    print(f"Cross-validation summary saved to '{REPORT_FOLDER}/cross_validation_summary_{season_name.lower()}.csv'")

    # ==========================================================
    # Final Performance Summary
    # ==========================================================
    print("\n" + "-"*40)
    print("RANDOM FOREST FINAL PERFORMANCE SUMMARY")
    print("-"*40)
    print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"Validation Set Accuracy: {val_accuracy:.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print("-"*40 + "\n")