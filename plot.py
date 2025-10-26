import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from globals import *

# Default colors
MPPT_PALETTE = {
    'mppt_power': '#66ffff',              # Light cyan / pale cyan
    'global_tilted_irradiance': '#ffff99',# Light yellow
    'diffuse_radiation': 'orange',        # Orange
    'temperature_2m': '#ff99bb',          # Light pink / soft pink
    'wind_speed_10m': '#4d94ff',          # Medium blue / cornflower blue
}

CURR_VOLT_PALETTE = {
    'pv1_i': '#ffff66',   # Light yellow
    'a_i': '#ff6666',     # Light red / coral red
    'b_i': '#66ff66',     # Light green / lime green
    'c_i': '#66ccff',     # Light blue / sky blue
    'pv1_u': '#ffff66',   # Light yellow
    'a_u': '#ff6666',     # Light red / coral red
    'b_u': '#66ff66',     # Light green / lime green
    'c_u': '#66ccff',     # Light blue / sky blue
    'ab_u': '#ff99cc',    # Pink / pastel pink
    'bc_u': '#99ffcc',    # Mint green / pale green
    'ca_u': '#ccccff',    # Lavender / light purple
}


def plot_mppt(df, date, output_folder, filename):
    """
        Plots MPPT power, irradiances (GTI & DHI), air temperature, and wind speed
        on the same graph using multiple Y-axes.

        Parameters:
            - df (pd.DataFrame): DataFrame containing the PV plant measurements with datetime index.
            - date (str): Date of the plot (used in title).
            - output_folder (str): Folder where the figure will be saved.
            - filename (str): Base filename for the saved figure.
    """

    # Filter data between 07:00 and 19:00 (daytime)
    df = df.between_time('07:00', '19:00')

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Use dark style for the plot
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # -------------------------
    # MPPT Power (primary axis)
    # -------------------------
    ax1.plot(df.index, df['mppt_power'], color=MPPT_PALETTE['mppt_power'], label='MPPT Power (kW)', linewidth=3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("MPPT (kW)", color=MPPT_PALETTE['mppt_power'])
    ax1.tick_params(axis='y', labelcolor=MPPT_PALETTE['mppt_power'])

    # -------------------------
    # Global Tilted Irradiance (secondary axis)
    # -------------------------
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['global_tilted_irradiance'], color=MPPT_PALETTE['global_tilted_irradiance'],
             label='Global Tilted Irradiance (W/m²)', linewidth=0.9)
    ax2.set_ylabel("GTI (W/m²)", color=MPPT_PALETTE['global_tilted_irradiance'])
    ax2.tick_params(axis='y', labelcolor=MPPT_PALETTE['global_tilted_irradiance'])

    # -------------------------
    # Diffuse Horizontal Irradiance (tertiary axis)
    # -------------------------
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(df.index, df['diffuse_radiation'], color=MPPT_PALETTE['diffuse_radiation'],
             label='Diffuse Radiation (W/m²)', linewidth=0.6)
    ax3.set_ylabel("DHI (W/m²)", color=MPPT_PALETTE['diffuse_radiation'])
    ax3.tick_params(axis='y', labelcolor=MPPT_PALETTE['diffuse_radiation'])

    # Adjust Y-axis limits for irradiances
    min_rad = min(df['diffuse_radiation'].min(), df['global_tilted_irradiance'].min())
    max_rad = max(df['diffuse_radiation'].max(), df['global_tilted_irradiance'].max())
    ax2.set_ylim(min_rad, max_rad)
    ax3.set_ylim(min_rad, max_rad)

    # -------------------------
    # Air Temperature (quaternary axis)
    # -------------------------
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.plot(df.index, df['temperature_2m'], color=MPPT_PALETTE['temperature_2m'], linewidth=0.5, label='Air Temperature (°C)')
    ax4.set_ylabel("Air Temp (°C)", color=MPPT_PALETTE['temperature_2m'])
    ax4.tick_params(axis='y', labelcolor=MPPT_PALETTE['temperature_2m'])

    # -------------------------
    # Wind Speed (quinary axis)
    # -------------------------
    ax5 = ax1.twinx()
    ax5.spines['right'].set_position(('outward', 180))
    ax5.plot(df.index, df['wind_speed_10m'], color=MPPT_PALETTE['wind_speed_10m'], linewidth=0.5, label='Wind Speed (m/s)')
    ax5.set_ylabel("Wind Speed (m/s)", color=MPPT_PALETTE['wind_speed_10m'])
    ax5.tick_params(axis='y', labelcolor=MPPT_PALETTE['wind_speed_10m'])

    # Format X-axis as hours:minutes
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Title of the plot
    plt.title(f"MPPT, DHI, GTI, Air Temperature & Wind Speed ({LOCAL}, {date})", fontsize=18, verticalalignment='bottom')

    # Enable grid for Y-axis
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_folder, f"{filename}_mppt_dhi_gti_temp_wind.png"), dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()


def plot_currents(df, date, output_folder, filename):
    """
        Plot combined inverter phase currents (a_i, b_i, c_i) and PV string current (pv1_i)
        in the same dark-themed graph.

        Parameters:
            - df (pd.DataFrame): DataFrame containing current measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
    """

    # Filter data to include only the time range between 07:00 and 19:00
    df = df.between_time('07:00', '19:00')

    # Ensure output folder exists; create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Set dark background style for the plot
    plt.style.use('dark_background')

    # Create figure and axis objects with defined size
    fig, ax = plt.subplots(figsize=(18, 6))

    # Plot each current with a predefined color and line width
    for col in ['pv1_i', 'a_i', 'b_i', 'c_i']:
        ax.plot(df.index, df[col], label=col, color=CURR_VOLT_PALETTE[col], linewidth=3)

    # Label axes and set tick colors to white for visibility
    ax.set_xlabel("Time")
    ax.set_ylabel("Current (A)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Add horizontal grid lines with light transparency
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Format x-axis as hours and minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Set plot title with location and date, white text for dark background
    plt.title(f"PV and Phase Currents ({LOCAL}, {date})", fontsize=18, verticalalignment='bottom', color='white')

    # Add legend with black background and white edges
    plt.legend(facecolor='black', edgecolor='white', fontsize=10, loc='upper right')

    # Adjust layout to avoid clipping of labels
    plt.tight_layout()

    # Construct full output path and save figure with high resolution
    output_path = os.path.join(output_folder, f"{filename}_phase_pv_currents.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()


def plot_voltages(df, date, output_folder, filename):
    """
        Plot combined inverter voltages (a_u, b_u, c_u, ab_u, bc_u, ca_u)
        and PV string voltage (pv1_u) in a dark-themed graph.

        Parameters:
            - df (pd.DataFrame): DataFrame containing voltage measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
    """

    # Filter data to include only the time range between 07:00 and 19:00
    df = df.between_time('07:00', '19:00')

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Set dark background style
    plt.style.use('dark_background')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(18, 6))

    # Plot each voltage if it exists in the DataFrame
    for col in ['pv1_u', 'a_u', 'b_u', 'c_u', 'ab_u', 'bc_u', 'ca_u']:
        ax.plot(df.index, df[col], label=col, color=CURR_VOLT_PALETTE[col], linewidth=3)

    # Axis labels and tick colors
    ax.set_xlabel("Time")
    ax.set_ylabel("Voltage (V)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Horizontal grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Format x-axis as HH:MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Title and legend
    plt.title(f"PV and Phase Voltages ({LOCAL}, {date})", fontsize=18, verticalalignment='bottom', color='white')
    plt.legend(facecolor='black', edgecolor='white', fontsize=10, loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_folder, f"{filename}_phase_pv_voltages.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()

def plot_confusion_matrix_combined(season_name, image_folder, y_true, y_pred, set_name):
    """
        Plot combined confusion matrix showing both absolute counts and row-wise percentages.

        Parameters:
            - season_name (str): Name of the season or scenario to display in the plot title.
            - image_folder (str): Folder path where the plot image will be saved.
            - y_true (array-like): Ground truth labels.
            - y_pred (array-like): Predicted labels.
            - set_name (str): Name of the dataset (e.g., 'Train', 'Test') for the plot title and filename.
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

    # Custom colormap: black -> gray -> white
    colors = [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("black_gray_white", colors)

    # Create the figure for the heatmap
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')

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
        linewidths=1,           # Width of the grid lines separating cells
        linecolor='white',      # Color of the grid lines (white for contrast on dark background)
        ax=ax                   # Matplotlib Axes object to draw the heatmap on (ensures full control over styling)
    )

    # Customize the grayscale bar appearance for dark theme
    cbar = ax.collections[0].colorbar
    # Set color of tick marks and numeric labels (for readability on dark background)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    # Set the color and label text of the colorbar border
    cbar.outline.set_edgecolor('white')
    # Add label text to the colorbar with white font color for visibility
    cbar.set_label('Percentage (%)', color='white')  # label da colorbar

    # Set plot title and axis labels
    ax.set_title(f'Confusion Matrix, {season_name}, {set_name} Set\n(Values: Count over Percentage)',
                 color='white', fontsize=14)
    ax.set_ylabel('True Label', color='white')
    ax.set_xlabel('Predicted Label', color='white')

    # Tick labels
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=90)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Adjust layout to avoid overlapping labels
    plt.tight_layout()

    # Build the filename and save the figure
    output_path = os.path.join(image_folder, f"confusion_matrix_{set_name.lower()}_{season_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')

    # Close the figure to free memory
    plt.close()

    # Confirm saving
    print(f"{set_name.title()} Confusion Matrix saved to {output_path}")

    return cm, cm_percentage


def plot_feature_importance(feature_importance, season_name, image_folder, top_n=15):
    """
        Plot the top-N most important features from a trained model using a dark-themed bar chart.

        Parameters:
            - feature_importance (pd.DataFrame): DataFrame containing 'feature' and 'importance' columns.
            - season_name (str): Name of the season or scenario for labeling the plot and output file.
            - image_folder (str): Folder path where the resulting plot image will be saved.
            - top_n (int, optional): Number of top features to display (default = 15).

        Notes:
            - The function visualizes feature importance in descending order.
            - The chart uses a dark background with white text and vertical dashed grid lines.
            - The image is saved as a high-resolution PNG file in the given folder.
    """

    # Select the top-N most important features
    top_features = feature_importance.head(top_n)

    # Create figure and axes with dark background
    plt.figure(figsize=(12, 8), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # Create horizontal barplot
    sns.barplot(
        x='importance',
        y='feature',
        data=top_features,
        # color='white',
        edgecolor='white',
    )

    # Set title and axis labels
    ax.set_title(f"Top {top_n} Most Important Features, {season_name}",
                 color='white', fontsize=14)
    ax.set_xlabel('Importance', color='white', fontsize=12)
    ax.set_ylabel('Feature', color='white', fontsize=12)

    # Customize tick colors for readability on dark background
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Add vertical dashed grid lines for better reference
    ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.6)

    # Adjust layout for clarity
    plt.tight_layout()

    # Save figure in high resolution
    output_path = os.path.join(image_folder, f"feature_importance_{season_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')

    # Close the figure to free memory
    plt.close()

    # Confirm saving
    print(f"Feature Importance plot saved to {output_path}")


def plot_class_accuracy(class_acc, classes, title, output_folder, filename):
    """
        Plot per-class accuracy as a vertical bar chart using a dark theme.
        Includes a red dashed line for mean accuracy with an automatic legend and values on top of each bar.

        Parameters:
            - class_acc (list or np.ndarray): Accuracy values for each class (in %).
            - class_names (list of str): Names/labels of each class for x-axis.
            - title (str): Title of the plot.
            - output_folder (str): Directory where the plot will be saved.
            - filename (str): Filename for saving the plot (including .png extension).

        Notes:
            - Dark theme: black background, white text, grid lines gray.
            - Each bar displays its corresponding accuracy value on top.
            - Mean accuracy is indicated with a red dashed horizontal line and automatic legend.
            - The resulting figure is saved as high-resolution PNG (dpi=300).
    """

    # Use dark background style
    plt.style.use('dark_background')

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

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
                ha='center', va='bottom', color='white', fontsize=10)

    # Calculate mean accuracy and plot a horizontal dashed line
    mean_acc = np.mean(class_acc)
    ax.axhline(mean_acc, color='red', linestyle='--', linewidth=1.2, label=f"Mean accuracy = {mean_acc:.1f}%")

    # Set axis labels and title in white for dark theme
    ax.set_xlabel('Label', color='white')
    ax.set_ylabel('Accuracy (%)', color='white')
    ax.set_title(title, color='white')

    # Customize tick labels
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), fontsize=8, rotation=0)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Add horizontal dashed grid lines for reference
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

    # Add legend with automatic positioning, styled for dark background
    legend = ax.legend(facecolor='black', edgecolor='white', fontsize=10, loc='best')
    for text in legend.get_texts():
        text.set_color('white')

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()

    # Save figure in high resolution
    output_path = f"{output_folder}/{filename}"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())

    # Close the figure to free memory
    plt.close()

    # Confirm saving
    print(f"Class Accuracy plot saved to {output_path}")


def plot_inference_fault_distribution(season_name, state_counts, output_inference_image):
    """
        Plot the distribution of predicted inverter states from the inference DataFrame.
        Ensures all classes appear on the X-axis, even if some have zero counts.

        Parameters:
            - state_counts (dict or pd.Series): Number of predictions per inverter state.
            - save_path (str, optional): File path to save the resulting plot.

        Notes:
            - Dark theme: black background, white text, gray grid lines.
            - Bars colored using a vibrant colormap with white borders for contrast.
            - X-axis shows all inverter states (0-5), Y-axis shows counts.
            - High-resolution PNG saved (dpi=300).
    """

    # Set dark background style for the plot
    plt.style.use('dark_background')

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Prepare labels and colors from LABELS_MAP
    labels = [LABELS_MAP[state][0] for state in state_counts.index]
    colors = [LABELS_MAP[state][1] for state in state_counts.index]
    counts = state_counts.values

    # Plot the bar chart
    bars = ax.bar(
        labels,             # X-axis labels: inverter state names
        counts,             # Y-axis: number of entries per state
        color=colors,       # Bar fill colors
        edgecolor='white',  # White borders for contrast on dark background
    )

    # Add counts on top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,              # X-position: center of the bar
            height + 1,  # Y-position: slightly above the bar
            f"{int(count)}",  # Text: integer count
            ha='center', va='bottom', color='white', fontsize=10
        )

    # Axis labels and title
    ax.set_xlabel('Label', color='white')
    ax.set_ylabel('Number of Entries', color='white')
    ax.set_title(f"Predicted Fault Distribution, {season_name}", color='white')

    # Customize tick labels color and rotation
    plt.xticks(rotation=0, ha='center', color='white', fontsize=8)
    plt.yticks(color='white', fontsize=8)

    # Add horizontal grid lines (gray, dashed)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

    # Adjust layout to avoid clipping labels
    plt.tight_layout()

    # Save the figure in high resolution
    plt.savefig(output_inference_image, dpi=300, facecolor=fig.get_facecolor())

    # Close figure to free memory
    plt.close()

    # Print confirmation
    print(f"Inference Fault Distribution plot saved to {output_inference_image}\n")


def plot_inference_fault_probabilities(input_file, season_name, output_folder):
    """
    Generate and save bar plots of predicted anomaly and fault probabilities for each inverter entry.

    Each row in the input CSV is visualized as a separate bar chart, with probability
    values for each anomaly or fault type.

    Parameters:
        - input_file (str): Path to the CSV file containing predicted probabilities.
          Expected columns include: 'ID', 'date', 'predicted_fault', and one column
          per fault type corresponding to LABELS_MAP keys.
        - output_folder (str): Directory where the generated plots will be saved.
    """

    # Load predictions CSV into a DataFrame
    df = pd.read_csv(input_file)

    # Iterate through each row to create an individual plot per inverter entry
    for _, row in df.iterrows():
        # Set dark background style for consistency
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Extract labels and colors for the bar chart from LABELS_MAP
        labels = [LABELS_MAP[c][0] for c in range(len(LABELS_MAP))]
        colors = [LABELS_MAP[c][1] for c in range(len(LABELS_MAP))]

        # Extract predicted probabilities for each anomaly and fault type and convert to percentage
        probs = [row[f"{LABELS_MAP[c][0]}"] * 100 for c in range(len(LABELS_MAP))]
        
        # Plot the probabilities as a colored bar chart
        bars = ax.bar(labels, probs, color=colors, edgecolor='white')

        # Annotate each bar with its probability value
        for bar, prob in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                prob + 2, # Position label slightly above the bar
                f"{int(prob)}",
                ha='center', va='bottom', color='white', fontsize=9
            )

        # Set axis labels and chart title
        ax.set_xlabel('Fault Type', color='white')
        ax.set_ylabel('Probability (%)', color='white')
        
        # Extract inverter location and number from ID for the title
        id = row['ID'].split('_')
        local = ' '.join(id[:-1])
        number = id[-1] 
        ax.set_title(f"Predicted Probabilities for Inverter {number}, {local}, {row['date']}", color='white')

        # Fix Y-axis limits from 0% to 100%
        ax.set_ylim(0, 100)

        # Configure tick labels
        plt.xticks(rotation=0, ha='center', color='white', fontsize=8)
        plt.yticks(color='white', fontsize=8)

        # Add horizontal grid lines for easier probability comparison
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Save the figure to the specified output folder
        save_path = os.path.join(output_folder, f"Classification_prob_{row['ID']}_{row['date']}_{season_name}.png")
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        
        # Close figure to free memory
        plt.close()
        
        # Confirmation message
        print(f"Saved Inference Probabilities plot for {row['ID']}, {row['date']}\n\tat {save_path}")