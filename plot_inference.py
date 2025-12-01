import os
import matplotlib.pyplot as plt
from globals import *

def plot_inference_condition_distribution(season_name, state_counts, output_inference_image):
    """
    Plot the distribution of predicted inverter states from the inference DataFrame.
    Ensures all classes appear on the X-axis, even if some have zero counts.

    Args:
        - season_name (str): Name of the season for the title.
        - state_counts (dict or pd.Series): Number of predictions per inverter state.
        - save_path (str, optional): File path to save the resulting plot.
        
    Notes:
        - Each bar displays its corresponding count on top.
        - The resulting figure is saved as high-resolution PNG (dpi=300).
    """

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare labels and colors from LABELS_MAP
    labels = [LABELS_MAP[state][0] for state in state_counts.index]
    colors = [LABELS_MAP[state][1] for state in state_counts.index]
    counts = state_counts.values

    # Plot the bar chart
    bars = ax.bar(
        labels,             # X-axis labels: inverter state names
        counts,             # Y-axis: number of entries per state
        color=colors,       # Bar fill colors
    )

    # Add counts on top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X-position: center of the bar
            height + 1,  # Y-position: slightly above the bar
            f"{int(count)}",  # Text: integer count
            ha='center', va='bottom', color='black', fontsize=10
        )

    # Axis labels and title
    ax.set_xlabel('Label', color='black')
    ax.set_ylabel('Number of Entries', color='black')
    #ax.set_title(f"Predicted Condition Distribution, {season_name}", color='black')

    # Customize tick labels color and rotation
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)

    # Add horizontal grid lines (gray, dashed)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

    # Adjust layout to avoid clipping labels
    plt.tight_layout()

    # Save the figure in high resolution
    plt.savefig(output_inference_image, dpi=300, facecolor=fig.get_facecolor())

    # Close figure to free memory
    plt.close()

    # Print confirmation
    print(f"Inference Condition Distribution plot saved to {output_inference_image}\n")


def plot_inference_condition_probabilities(df, season_name_file, output_folder, adjusted=False):
    """
    Generate and save bar plots of predicted condition probabilities for each inverter entry.

    Args:
        - df (pd.DataFrame): DataFrame containing predicted probabilities for each condition type, along with 'ID' and 'date' columns.
        - season_name_file (str): File name used to identify the saved plots.
        - output_folder (str): Directory where the plots will be saved.
            
    Notes:
        - Bars are colored according to LABELS_MAP.
        - Each bar displays its corresponding probability value on top.
        - The resulting figures are saved as high-resolution PNG files in the specified folder.
    """

    # Iterate through each row to create an individual plot per inverter entry
    for _, row in df.iterrows():
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 5))

        # Extract labels and colors for the bar chart from LABELS_MAP
        labels = [LABELS_MAP[c][0] for c in range(len(LABELS_MAP))]
        colors = [LABELS_MAP[c][1] for c in range(len(LABELS_MAP))]

        # Extract predicted probabilities for each condition type and convert to percentage
        probs = [row[f"{LABELS_MAP[c][0]}"] * 100 for c in range(len(LABELS_MAP))]
        
        # Plot the probabilities as a colored bar chart
        bars = ax.bar(labels, probs, color=colors)

        # Annotate each bar with its probability value
        for bar, prob in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                prob + 2, # Position label slightly above the bar
                f"{prob:.1f}",
                ha='center', va='bottom', color='black', fontsize=9
            )

        # Set axis labels
        ax.set_xlabel('Condition Type', color='black')
        ax.set_ylabel('Probability (%)', color='black')
        
        # Extract inverter location and number from ID for the title
        id = row['ID'].split('_')
        local = ' '.join(id[:-1])
        number = id[-1] 
        # ax.set_title(f"Predicted Probabilities for Inverter {number}, {local}, {row['date']}", color='black')

        # Fix Y-axis limits from 0% to 100%
        ax.set_ylim(0, 100)

        # Configure tick labels
        plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
        plt.yticks(color='black', fontsize=8)

        # Add horizontal grid lines for easier probability comparison
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, axis='y')

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Save the figure to the specified output folder
        adjusted_tag = "adjusted" if adjusted else ""
        save_path = os.path.join(output_folder, f"Classification_prob_{row['ID']}_{row['date']}_{season_name_file}_{adjusted_tag}.png")
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        
        # Close figure to free memory
        plt.close()
        
        # Confirmation message
        print(f"Saved Inference Probabilities plot for {row['ID']}, {row['date']}\n\tat {save_path}")