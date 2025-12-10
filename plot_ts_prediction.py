import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from adjustText import adjust_text

from globals import *


def plot_daily_class_probabilities(csv_path, output_image_path,
                                   threshold_start=0.5, threshold_target=0.8,
                                   regressions_df=None, nth_regression=None):
    """
    Plots the daily evolution of predicted condition probabilities for each class,
    with threshold lines and (optionally) the Nth regression line.

    Args:
        csv_path (str): Path to the CSV containing 'date' and class probability columns.
        output_image_path (str): Path to save the final line plot.
        threshold_start (float): Threshold start value.
        threshold_target (float): Threshold target value.
        regressions_df (pd.DataFrame, optional): DataFrame with ['date', 'class', 'slope', 'intercept'].
        nth_regression (int, optional): Index (0-based) of which regression line to draw from regressions_df.
    """
    
    # Load the data
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot probabilities per class
    for _, (label, color) in LABELS_MAP.items():
        if label in df.columns:
            ax.plot(df['date'], df[label], label=label, color=color, linewidth=1)

    # Plot only the Nth regression line (if available)
    if regressions_df is not None and nth_regression is not None:
        Nth_regression = nth_regression - 1
        
        # Check if Nth_regression is valid
        if Nth_regression < len(regressions_df):
            row = regressions_df.iloc[Nth_regression]
            start_date = row['date']
            slope = row['slope']
            intercept = row['intercept']
            cls = row['class']
            pred_cls = row['actual_class_at_predicted_day']

            # Determine class color from LABELS_MAP
            class_color, class_pred_color = None, None
            for _, (label, color) in LABELS_MAP.items():
                if label == cls:
                    class_color = color
                if label == pred_cls:
                    class_pred_color = color

            # Generate regression line starting from that date forward
            if start_date in df['date'].values:
                start_idx = df.index[df['date'] == start_date][0]
                x_vals = np.arange(len(df) - start_idx)
                y_vals = slope * x_vals + intercept
                y_vals = np.clip(y_vals, 0, 1)

                ax.plot(df['date'].iloc[start_idx:], y_vals,
                        color=class_color, linewidth=1.8,
                        label=f'Regression #{Nth_regression + 1}, {cls}')

                # Colored threshold lines
                ax.axhline(y=threshold_start, color=class_color, linestyle='--', linewidth=1.3,
                           label=f'Threshold Start ({threshold_start}), {cls}')
                ax.axhline(y=threshold_target, color=class_pred_color, linestyle='--', linewidth=1.3,
                           label=f'Threshold Target ({threshold_target}), {pred_cls}')
        else:
            print(f"[Warning:] Nth_regression={Nth_regression} out of range (total regressions={len(regressions_df)})")
    else:
        # Default black thresholds if no regression selected
        ax.axhline(y=threshold_start, color='black', linestyle='--', linewidth=1.3,
                   label=f'Threshold Start ({threshold_start})')
        ax.axhline(y=threshold_target, color='black', linestyle='--', linewidth=1.3,
                   label=f'Threshold Target ({threshold_target})')

    # Axis formatting
    # ax.set_xlabel('Date', color='black', fontsize=10)
    ax.set_ylabel('Probability', color='black', fontsize=10)
    ax.tick_params(axis='x', colors='black', rotation=0)
    ax.tick_params(axis='y', colors='black')
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')
    ax.legend(fontsize=9)

    # Finalize and save plot
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"Daily condition probabilities plot saved to {output_image_path}")
    
    # Close plot to free memory
    plt.close()


def plot_predictions_cleveland(df_predictions, output_image_path):
    """
    Plots a compact horizontal Cleveland-style timeline of predictions:
        - Each line connects prediction (colored circle) to actual (colored circle).
        - Start ball = predicted class color.
        - End ball = actual class color.
        - Displays predicted_days_to_X near the prediction ball.
        - No Y-axis shown.
        - Legend for classes.
        
    Args:
        df_predictions (pd.DataFrame) with columns:
            date, class, predicted_days_to_X, predicted_date, actual_class_at_predicted_day, slope, intercept.
        output_image_path (str): Path to save the final plot.
    """

    if df_predictions is None or df_predictions.empty:
        print("Cleveland Plot: No predictions to plot.")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Parameters for layout
    bar_height = 0.2
    spacing_factor = 1.1
    spacing = bar_height * spacing_factor
    layers = []

    # Ensure datetime
    df_predictions['date'] = pd.to_datetime(df_predictions['date'])
    df_predictions['predicted_date'] = pd.to_datetime(df_predictions['predicted_date'])

    for _, row in df_predictions.iterrows():
        start_date = row['date']
        end_date = row['predicted_date']
        predicted_cls = row['class']
        actual_cls = row['actual_class_at_predicted_day']
        correct = predicted_cls == actual_cls
        days_ahead = int(row['predicted_days_to_target'])

        # Find free vertical layer (avoid overlap)
        placed = False
        for j, last_end in enumerate(layers):
            if start_date > last_end:
                layer_index = j
                layers[j] = end_date
                placed = True
                break
            
        if not placed:
            layer_index = len(layers)
            layers.append(end_date)

        # Predicted and actual colors
        predicted_color = LABELS_MAP[[k for k, v in LABELS_MAP.items() if v[0] == predicted_cls][0]][1]
        actual_color = LABELS_MAP[[k for k, v in LABELS_MAP.items() if v[0] == actual_cls][0]][1]

        # Position
        y = layer_index * spacing

        # Draw connecting thin line
        ax.plot(
            [start_date, end_date],
            [y, y],
            color='black',
            linewidth=0.5,
            zorder=1
        )

        # Draw predicted ball (start)
        ax.scatter(
            start_date,
            y,
            color=predicted_color,
            s=70,
            zorder=2
        )

        # Draw actual ball (end)
        ax.scatter(
            end_date,
            y,
            color=actual_color,
            s=70,
            zorder=2
        )

        # Label (days ahead)
        shift = timedelta(days=1)
        text_color = "black" if correct else actual_color
        ax.text(
            start_date + 0.7 * shift,
            y + 0.05,
            str(days_ahead),
            color=text_color,
            va='center',
            ha='left',
            fontsize=7,
        )

    # Hide Y-axis
    ax.get_yaxis().set_visible(False)

    # X-axis formatting
    # ax.set_xlabel('Date', color='black', fontsize=10)
    # ax.set_title('Prediction Timeline, Cleveland-style', color='black')

    # Grid and ticks
    ax.xaxis.grid(True, linestyle='--', linewidth=0.4, color='gray')
    ax.set_axisbelow(True)
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)

    # Adjust limits
    shift = timedelta(days=0.4)
    ax.set_xlim(df_predictions['date'].min() - 8 * shift, df_predictions['predicted_date'].max() + 3 * shift)
    
    # Dynamic Y-limits based on number of layers
    num_layers = len(layers)
    if num_layers == 0:
        # Fallback: no predictions
        ax.set_ylim(-bar_height / 2 - 0.1, bar_height / 2 + 0.1)
    else:
        # Maximum y for maximum index
        max_layer_y = (num_layers - 1) * spacing
        # Top and bottom with margins proportional to spacing/bar_height
        vertical_margin = spacing * 0.25
        bottom = -bar_height / 2 - vertical_margin
        top = max_layer_y + bar_height / 2 + vertical_margin
        # Offset to approximate the first line to zero
        offset = - spacing * 0.1
        ax.set_ylim(bottom - offset, top - offset)
    
    # Legend for classes
    class_patches = [
        Patch(facecolor=LABELS_MAP[k][1], edgecolor='none', label=LABELS_MAP[k][0])
        for k in LABELS_MAP
    ]
    ax.legend(
        handles=class_patches,
        loc='upper left',
        fontsize=8,
    )

    # Finalize and save plot
    fig.set_size_inches(14, max(4, num_layers * spacing))
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Saved Cleveland-style timeline to {output_image_path}")
    
    # Close plot to free memory
    plt.close()
    

def pareto_plot(df_pareto, df_all_results, input_csv_path, output_image_folder):
    """
    Plots the Pareto front of configurations based on mean predicted days to target vs accuracy.
    
    Args:
        df_pareto (pd.DataFrame): DataFrame containing Pareto-optimal configurations.
        df_all_results (pd.DataFrame): DataFrame containing all configurations.
        input_csv_path (Path): Path object of the input CSV file.
        output_image_folder (Path): Path object of the folder to save the output image.
        
    """
    # Create figure and axis
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    
    # Plot all configurations and Pareto front
    sns.scatterplot(data=df_all_results, x='mean_predicted_days', y='accuracy', label='All configurations')
    sns.lineplot(data=df_pareto, x='mean_predicted_days', y='accuracy', color='red', marker='o', label='Pareto front')

    # Annotate Pareto points with parameter settings
    texts = []
    for _, row in df_pareto.iterrows():
        param_text = f"ts={row['threshold_start']}, tt={row['threshold_target']}\n" \
                    f"tc={row['threshold_class']}, w={int(row['window'])}"
        texts.append(ax.text(row['mean_predicted_days'], row['accuracy'], param_text,
                            fontsize=8, color='red'))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    # Axis labels and title
    plt.xlabel("Mean Predicted Days to Target")
    plt.ylabel("Accuracy (%)")
    # plt.title("Pareto-optimal configurations")
    
    # Axis limits, grid, legend
    plt.ylim(df_all_results['accuracy'].min()*0.95, 100.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_image_folder / f"{input_csv_path.stem}_pareto_front.png")
    print(f"Pareto front plot saved to {output_image_folder / f'{input_csv_path.stem}_pareto_front.png'}")
    
    # Close plot to free memory
    plt.close()