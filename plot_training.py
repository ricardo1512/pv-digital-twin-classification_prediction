import numpy as np
from itertools import cycle
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from globals import *


def plot_confusion_matrix_combined(set_name, season_name, y_true, y_pred, image_file):
    """
        Plot combined confusion matrix showing both absolute counts and row-wise percentages.

        Args:
            - set_name (str): Name of the dataset (e.g., 'Train', 'Test') for the plot title.
            - season_name (str): Name of the season or scenario to display in the plot title.
            - y_true (array-like): Ground truth labels.
            - y_pred (array-like): Predicted labels.
            - image_file (str): Path where the plot image will be saved.
            
            - classes (list of str, optional): Class names for axes labels. If None, integers are used.

        Returns:
            - cm (np.ndarray): Confusion matrix with absolute counts.
            - cm_percentage (np.ndarray): Confusion matrix converted to percentages by row.

        Notes:
            - Each cell shows both count and percentage (Count\n(Percentage%)).
            - Heatmap uses a white/grey/black colormap, with values ranging from 0% to 100%.
            - The plot is saved as a high-resolution PNG in the specified folder.
    """

    # Compute the confusion matrix from true and predicted labels
    cm = confusion_matrix(y_true, y_pred)

    # Convert counts to percentages by row to better compare class performance
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percentage = np.round(cm_percentage, 1) # Round percentages to 1 decimal place

    # Create annotations combining absolute counts and percentages for each cell
    annotations = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm)):
            row.append(f"{int(cm[i, j])}\n({cm_percentage[i, j]:.1f}%)")
        annotations.append(row)

    # Use label names from LABELS_MAP
    classes = [LABELS_MAP[i][0] for i in range(len(LABELS_MAP))]

    # Custom colormap: light blue → navy
    cmap = plt.cm.Blues

    # Create the figure for the heatmap
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

    # Plot heatmap
    sns.heatmap(
        cm_percentage,          # 2D array or DataFrame to visualize as a heatmap
        annot=annotations,      # Custom text annotations inside each cell (showing Count and Percentage)
        fmt='',                 # Format string for annotations
        cmap=cmap,              # Custom colormap defining color gradient (black → gray → white)
        xticklabels=classes,    # Labels for the x-axis (Predicted classes)
        yticklabels=classes,    # Labels for the x-axis (Predicted classes)
        cbar=True,              # Display the grayscale bar showing mapping between intensity and percentage values
        vmin=0,                 # Minimum value for color scale (0% → black)
        vmax=100,               # Maximum value for color scale (100% → white)
        linewidths=0.2,           # Width of the grid lines separating cells
        linecolor='black',      # Color of the grid lines separating cells
        ax=ax                   # Matplotlib Axes object to draw the heatmap on (ensures full control over styling)
    )

    # Customize the scale bar appearance
    cbar = ax.collections[0].colorbar
    # Set color of tick marks and numeric labels
    cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
    # Set the color and label text of the colorbar border
    cbar.outline.set_edgecolor('black')
    # Add label text to the colorbar 
    cbar.set_label('Percentage (%)', color='black')  # label da colorbar

    # Set plot title and axis labels
    # ax.set_title(f'Confusion Matrix, {season_name}, {set_name} Set\n(Values: Count over Percentage)',
    #              color='white', fontsize=14)
    ax.set_ylabel('True Label', color='black')
    ax.set_xlabel('Predicted Label', color='black')

    # Tick labels
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=90)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Adjust layout to avoid overlapping labels
    plt.tight_layout()

    # Save the figure
    plt.savefig(image_file, dpi=300, bbox_inches='tight')

    # Close the figure to free memory
    plt.close()

    # Confirm saving
    print(f"{set_name.title()} Confusion Matrix saved to {image_file}")

    return cm, cm_percentage


def plot_class_accuracy(class_acc, classes, title, output_file):
    """
        Plot per-class accuracy as a vertical bar chart.
        Includes a red dashed line for mean accuracy with an automatic legend and values on top of each bar.

        Args:
            - class_acc (list or np.ndarray): Accuracy values for each class (in %).
            - class_names (list of str): Names/labels of each class for x-axis.
            - title (str): Title of the plot.
            - output_file (str): Filename for saving the plot (including .png extension).

        Notes:
            - Each bar displays its corresponding accuracy value on top.
            - Mean accuracy is indicated with a red dashed horizontal line and automatic legend.
            - The resulting figure is saved as high-resolution PNG (dpi=300).
    """

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Map class_names to colors using LABELS_MAP
    colors = [LABELS_MAP[i][1] for i in range(len(classes))]
    # Map class indices to their names using LABELS_MAP
    class_labels = [LABELS_MAP[int(i)][0] for i in classes]

    # Plot bars for each class
    bars = ax.bar(class_labels, class_acc, color=colors, edgecolor='white')

    # Annotate bars with accuracy values on top
    for bar, acc in zip(bars, class_acc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{acc:.1f}%",
                ha='center', va='bottom', color='black', fontsize=10)

    # Calculate mean accuracy and plot a horizontal dashed line
    mean_acc = np.mean(class_acc)
    ax.axhline(mean_acc, color='red', linestyle='--', linewidth=1.2, label=f"Mean accuracy = {mean_acc:.1f}%")

    # Set axis labels and title
    ax.set_xlabel('Label', color='black')
    ax.set_ylabel('Accuracy (%)', color='black')
    # ax.set_title(title, color='black')

    # Customize tick labels
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=0)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Add horizontal dashed grid lines for reference
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

    # Add legend with automatic positioning
    legend = ax.legend(fontsize=10, loc='best')

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()

    # Save figure in high resolution
    plt.savefig(output_file, dpi=300, facecolor=fig.get_facecolor())

    # Close the figure to free memory
    plt.close()

    # Confirm saving
    print(f"Class Accuracy plot saved to {output_file}")
    
    
def plot_feature_importance(top_features, season_name, image_file):
    """
        Plot the top-N most important features from a trained model.

        Args:
            - feature_importance (pd.DataFrame): DataFrame containing TOP_FEATURES 'feature' and 'importance' columns.
            - season_name (str): Name of the season used in the plot title.
            - image_file (str): Path where the resulting plot image will be saved.

        Notes:
            - The function visualizes feature importance in descending order.
            - The image is saved as a high-resolution PNG file in the given folder.
    """

    # Create figure and axes
    plt.figure(figsize=(12, 7.2))
    ax = plt.gca()

    # Create horizontal barplot
    sns.barplot(
        x='importance',
        y='feature',
        data=top_features,
        color='#009999',
        edgecolor='white',
    )

    # Set title and axis labels
    # ax.set_title(f"Top {TOP_FEATURES} Most Important Features, {season_name}",
    #              color='white', fontsize=14)
    ax.set_xlabel('Importance', color='black', fontsize=12)
    ax.set_ylabel('Features', color='black', fontsize=12)
    
    # Configure tick labels
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='white', fontsize=10)

    # Customize tick colors
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Add vertical dashed grid lines for better reference
    ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.6)

    # Adjust layout for clarity
    plt.tight_layout()

    # Save figure in high resolution
    plt.savefig(image_file, dpi=300, bbox_inches='tight')

    # Close the figure to free memory
    plt.close()

    # Confirm saving
    print(f"Feature Importance plot saved to {image_file}")


def plot_auc_recall_vs_precision(y_true, y_scores, class_names, output_file):
    """
        Plot Precision vs Recall (Recall on X, Precision on Y) for multi-class classification.

        Args:
            y_true (array-like): True class labels.
            y_scores (array-like, shape=[n_samples, n_classes]): Predicted probabilities for each class.
            class_names (list of str): Names of classes for legend.
            output_file (str): Name for the saved filename.
            
        Notes:
            - Each class's curve is colored according to LABELS_MAP.
            - AUC for each class is displayed in the legend.
            - The resulting figure is saved as high-resolution PNG (dpi=300).
    """
    
    # Binarize the true labels for multi-class precision-recall calculation
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Use colors from LABELS_MAP
    from globals import LABELS_MAP
    colors = cycle([LABELS_MAP[i][1] for i in range(n_classes)])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Precision-Recall curve for each class
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        auc_score = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{class_names[i]} (AUC={auc_score:.2f})")

    # Set labels, title, limits, grid, and legend
    ax.set_xlabel('Recall', color='black')
    ax.set_ylabel('Precision', color='black')
    # ax.set_title('Recall vs Precision Curve Per Class', color='white', fontsize=14)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.set_xlim(1, 0)
    ax.set_ylim(0, 1.005)
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.legend(fontsize=10, loc='lower right')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    
    # Close figure to free memory
    plt.close()

    # Confirm saving
    print(f"Precision vs Recall plot saved to {output_file}")


def plot_fp_tp_curve(y_true, y_scores, class_names, output_file):
    """
    Plot False Positives (FP) vs True Positives (TP) per class for multi-class classification.
    This visualization helps understand how each class separates in terms of true and false detections.

    Args:
        y_true (array-like): True class labels.
        y_scores (array-like, shape=[n_samples, n_classes]): Predicted probabilities for each class.
        class_names (list of str): Names of classes for labeling.
        output_file (str): Path to save the output plot.

    Notes:
        - Prints a confirmation message with the save path.
    """

    # Convert true labels to binary format for multi-class case
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Color cycle for each class (using a global palette if available)
    colors = cycle([LABELS_MAP[i][1] for i in range(n_classes)])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute FP-TP pairs (equivalent to ROC without thresholds)
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        auc_score = auc(fpr, tpr)

        ax.plot(
            fpr, tpr, color=color, lw=2,
            label=f"{class_names[i]} (AUC={auc_score:.2f})"
        )

    # Plot formatting
    ax.plot([0, 1], [0, 1], 'w--', lw=1, label='Random Guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.005])
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    ax.set_xlabel('False Positive Rate (FP)', color='black', fontsize=12)
    ax.set_ylabel('True Positive Rate (TP)', color='black', fontsize=12)
    # ax.set_title('True Positive vs False Positive Per Class', color='white', fontsize=14)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)

    # Save and close
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close()

    # Confirm saving
    print(f"FP vs TP curve saved to {output_file}")


def plot_class_accuracy_ci(class_acc, ci_lower, ci_upper, classes, title, output_file):
    """
        Plot per-class accuracy as vertical bars with bootstrap confidence intervals.
        
        Args:
            class_acc (list or np.ndarray): Mean accuracy per class (%)
            ci_lower (list or np.ndarray): Lower bound of CI per class (%)
            ci_upper (list or np.ndarray): Upper bound of CI per class (%)
            classes (list): Class indices
            title (str): Plot title
            output_file (str): File path to save the plot
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar colors
    colors = [LABELS_MAP[i][1] for i in classes]
    class_labels = [LABELS_MAP[int(i)][0] for i in classes]
    
    # Compute CI ranges
    ci_diff = np.array(ci_upper) - np.array(ci_lower)

    # Plot bars with white error bars (CI)
    bars = ax.bar(
        class_labels,
        class_acc,
        color=colors,
        yerr=[class_acc - ci_lower, ci_upper - class_acc],
        capsize=5,
        ecolor='black'  # black error bars
    )
    
    # Add text showing CI width above bars
    for i, bar in enumerate(bars):
        upper = ci_upper[i]
        diff_text = f"{ci_diff[i]/2:.1f}"  
        ax.text(
            bar.get_x() + bar.get_width()/2,
            upper + 0.5,
            diff_text,
            ha='center',
            va='bottom',
            fontsize=8,
            color='black', 
        )


    # Labels, grid, and theme ticks
    ax.set_xlabel('Label', color='black')
    ax.set_ylabel('Accuracy (%)', color='black')
    # ax.set_title(title, color='black')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=0)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Class Accuracy plot saved to {output_file}")
    