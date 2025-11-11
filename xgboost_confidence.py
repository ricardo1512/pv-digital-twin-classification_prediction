import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier
from utils import class_accuracy 
from globals import *
from plot_training import *

# ==========================================================
# XGBoost training + bootstrap CI
# ==========================================================
def xgboost_classifier_bootstrap_ci(n_bootstrap=1000, ci_level=95):
    # Load datasets
    df_train = pd.read_csv(TRAIN_VALID_SET_FILE)
    df_test = pd.read_csv(TEST_SET_FILE)
    df_train[LABEL] = df_train[LABEL].astype(int)
    df_test[LABEL] = df_test[LABEL].astype(int)

    X_train = df_train.drop(columns=[LABEL, 'date'])
    y_train = df_train[LABEL]
    X_test = df_test.drop(columns=[LABEL, 'date'])
    y_test = df_test[LABEL]

    # Train classifier
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
    xgb_classifier.fit(X_train, y_train)

    # Bootstrapping for class-wise CI
    classes = sorted(y_test.unique())
    class_acc_samples = []

    for _ in range(n_bootstrap):
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
    plot_class_accuracy_ci(mean_acc, lower_bounds, upper_bounds, classes,
                           title="Class-wise Accuracy with 95% CI", 
                           output_file="Images/xgb_class_accuracy_ci.png")

