import numpy as np
import pandas as pd
from globals import *

def safe_std(arr):
    """
    Calculate the standard deviation safely.

    Returns 0.0 if the array has fewer than 2 elements or all values are identical,
    avoiding invalid or meaningless results.
    """

    if len(arr) < 2 or np.all(arr == arr[0]):
        return 0.0
    return np.std(arr)


def safe_var(arr):
    """
    Calculate the variance safely.

    Returns 0.0 if the array has fewer than 2 elements or all values are identical,
    preventing invalid variance computation.
    """

    if len(arr) < 2 or np.all(arr == arr[0]):
        return 0.0
    return np.var(arr)


def safe_mean(arr):
    """
    Calculate the mean safely.

    Returns np.nan if the array is empty or contains only NaN values,
    ensuring a meaningful result.
    """

    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.mean(arr)


def safe_corrcoef(arr1, arr2):
    """
    Calculate the Pearson correlation coefficient safely.

    Returns np.nan if arrays are too short, have constant values, or different lengths,
    avoiding errors or meaningless correlations.
    """

    if (len(arr1) < 2 or len(arr2) < 2 or
            np.all(arr1 == arr1[0]) or
            np.all(arr2 == arr2[0]) or
            len(arr1) != len(arr2)):
        return np.nan
    try:
        return np.corrcoef(arr1, arr2)[0, 1]
    except:
        return np.nan


def extract_comprehensive_features(filtered_results):
    """
        Extracts a comprehensive set of features from numerical columns of a DataFrame.

        For each numeric column, the function calculates:
        - Basic statistics (mean, std, min, max, percentiles, variance, etc.)
        - Differences between consecutive values (diff features)
        - Rolling statistics (mean, std, range, volatility, correlation)

        Returns a dictionary where keys are feature names and values are the computed metrics.
        If the input DataFrame has no numeric data, returns None.
    """

    # Select numeric columns
    numeric_data = filtered_results.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return None

    features = {}
    window_size = 5

    # Iterate over each numeric column
    for column in numeric_data.columns:
        window = numeric_data[column].dropna().values

        if len(window) > 3:  # Minimum number of values for meaningful features
            # ===============================
            # Basic statistical features
            # ===============================
            basic_feat = [
                np.max(window),                 # Maximum value
                np.ptp(window),                 # Range (max - min)
                safe_mean(window),              # Mean
                np.median(window),              # Median
                safe_std(window),               # Standard deviation
                safe_var(window),               # Variance
                np.percentile(window, 25),   # 25th percentile
                np.percentile(window, 75),   # 75th percentile
            ]

            # ===============================
            # Difference-based features
            # ===============================
            if len(window) > 1:
                diffs = np.diff(window)
                diff_feat = [
                    np.sum(diffs > 0),          # Count of positive differences
                    np.sum(diffs < 0),          # Count of negative differences
                    np.max(diffs),              # Maximum difference
                    safe_mean(diffs),           # Mean of differences
                    safe_mean(np.abs(diffs)),   # Mean absolute difference
                    safe_std(diffs),            # Std of differences
                ]
            else:
                diff_feat = [np.nan] * 6

            # ===============================
            # # Rolling window features
            # ===============================
            if len(window) >= window_size:
                # Rolling mean
                rolling_mean = np.convolve(window, np.ones(window_size) / float(window_size), mode='valid')

                # Safe rolling standard deviation
                rolling_std = []
                for i in range(len(window) - window_size + 1):
                    window_slice = window[i:i + window_size]
                    rolling_std.append(safe_std(window_slice))
                rolling_std = np.array(rolling_std)

                rolling_feat = [
                    safe_mean(rolling_mean),                                    # Average rolling mean
                    safe_std(rolling_mean),                                     # Std of rolling mean
                    np.ptp(rolling_mean) if len(rolling_mean) > 0 else np.nan,  # Range of rolling mean
                    safe_mean(rolling_std),                                     # Average rolling std (volatility)
                    safe_std(rolling_std),                                      # Std of rolling std (volatility)
                    safe_corrcoef(rolling_mean, rolling_std),                   # Correlation between mean and volatility
                ]
            else:
                rolling_feat = [np.nan] * 6

            # Combine all features
            all_feat = basic_feat + diff_feat + rolling_feat

        else:
            all_feat = [np.nan] * 20  # Total number of features

        # Map features to descriptive names
        feature_names = [
            # Basic features
            f"{column}_max",            # np.max(window)
            f"{column}_range",          # np.ptp(window)
            f"{column}_mean",           # safe_mean(window)
            f"{column}_median",         # np.median(window)
            f"{column}_std",            # safe_std(window)
            f"{column}_var",            # safe_var(window)
            f"{column}_q25",            # np.percentile(window, 25)
            f"{column}_q75",            # np.percentile(window, 75)

            # Difference features
            f"{column}_diff_pos_count", # np.sum(diffs > 0)
            f"{column}_diff_neg_count", # np.sum(diffs < 0)
            f"{column}_diff_max",       # np.max(diffs)
            f"{column}_diff_mean",      # safe_mean(diffs)
            f"{column}_diff_abs_mean",  # safe_mean(np.abs(diffs))
            f"{column}_diff_std",       # safe_std(diffs)

            # Rolling features
            f"{column}_roll_mean_mean", # safe_mean(rolling_mean)
            f"{column}_roll_mean_std",  # safe_std(rolling_mean)
            f"{column}_roll_mean_range",# np.ptp(rolling_mean)
            f"{column}_roll_vol_mean",  # safe_mean(rolling_std)
            f"{column}_roll_vol_std",   # safe_std(rolling_std)
            f"{column}_roll_corr"       # safe_corrcoef(rolling_mean, rolling_std)
        ]

        for name, value in zip(feature_names, all_feat):
            features[name] = value

    return features


def compute_store_daily_comprehensive_features(results_full, date, daily_features):
    """
        Compute daily statistical features for a store's numerical data.

        Steps:
        1. Filter the data to the classification hours.
        2. Extract the inverter state at the start of the period.
        3. Compute comprehensive statistical features for all numeric columns.
        4. Combine features and inverter state into a single Pandas Series.
        5. Append the result to the daily_features list.

        Parameters:
        - results_full: DataFrame containing all store data for the day
        - date: Date corresponding to the data
        - daily_features: List to append the computed daily feature Series
    """

    # Filter data within classification hours
    filtered_results = results_full.between_time(CLASSIFICATION_HOUR_INIT, CLASSIFICATION_HOUR_END)

    # Get inverter state at the start of the period
    inverter_state_value = filtered_results['inverter_state'].iloc[0]

    # Remove inverter state column before feature extraction
    filtered_results_features = filtered_results.drop(columns=['inverter_state'])

    # Extract comprehensive features from numeric columns
    features_array = extract_comprehensive_features(filtered_results_features)

    # Convert features to Pandas Series for easy combination and indexing
    feature_series = pd.Series(features_array)

    # Add inverter state as a feature
    feature_series['inverter_state'] = inverter_state_value

    # Assign the series a name corresponding to the current date
    feature_series.name = pd.to_datetime(date)

    # Append the series to the list of daily features
    daily_features.append(feature_series)

def class_accuracy(cm):
    """
        Calculate accuracy for each class from a confusion matrix.

        Parameters:
            - cm: 2D numpy array representing the confusion matrix, where
                  cm[i, j] is the number of instances of class i predicted as class j.

        Returns:
            - 1D numpy array containing per-class accuracy in percentage, rounded to 1 decimal place.
    """

    # Compute true positives per class (diagonal of confusion matrix) # =====
    true_positives = np.diag(cm)

    # Compute total samples per class (sum of each row)
    total_per_class = np.sum(cm, axis=1)

    # Compute class-wise accuracy in percentage
    class_acc = true_positives / total_per_class * 100

    # Round accuracies to 1 decimal place
    return np.round(class_acc, 1)