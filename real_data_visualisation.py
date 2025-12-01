import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from globals import *
from plot_day_samples import *


def correlation_matrix(df, plot_folder):
    """
    Generate and save a global correlation matrix heatmap for all inverters' data.
        
    Args:
        df (pd.DataFrame): DataFrame containing data from all inverters.
        plot_folder (str): Directory where the plot will be saved.
            
    Notes:
        - Only the lower triangle of the correlation matrix is displayed.
        - Annotation text color adapts automatically based on cell brightness.
        - Prints the top 20 positive and top 5 negative correlations to the console.
    """

    # Define feature order
    feature_order = EXPORT_COLUMNS + METEOROLOGICAL_COLUMNS

    # Select only numeric columns
    df_numeric = df.select_dtypes(include='number')
    df_numeric = df_numeric[[col for col in feature_order if col in df_numeric.columns and col != 'inverter_state']]
    corr = df_numeric.corr().round(2)
    
    # Reorder correlation matrix
    corr_pairs = corr.unstack().reset_index()
    corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # Remove self-correlations
    corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]

    # Remove duplicate pairs (A-B and B-A)
    corr_pairs = corr_pairs.drop_duplicates(subset=['Correlation'])

    # Top 20 positive correlations
    top_positive = corr_pairs.sort_values(by='Correlation', ascending=False).head(20)

    # Top 5 negative correlations
    top_negative = corr_pairs.sort_values(by='Correlation', ascending=True).head(5)

    print("\nTop 20 Positive Correlations:")
    print(top_positive.to_string(index=False))

    print("\nTop 5 Negative Correlations:")
    print(top_negative.to_string(index=False))

    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Custom colormap: light blue → navy
    cmap = plt.cm.Blues

    # Create figure
    _, ax = plt.subplots(figsize=(14, 12), facecolor='white')

    # Plot heatmap without annotations, applying the mask
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        fmt='',
        cmap=cmap,
        cbar=True,
        vmin=-1,
        vmax=1,
        linewidths=1,
        ax=ax
    )

    # Set the upper triangle cells to black explicitly
    # (ensures masked area matches the figure background)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i < j: # only upper triangle
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='white', lw=0))
                
    # Add annotations manually for lower triangle only
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i >= j:  # only lower triangle
                val = corr.iloc[i, j]
                normalized_val = (val + 1) / 2  # -1 -> 0, 1 -> 1
                color = 'white' if val > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', color=color)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black')
    cbar.outline.set_edgecolor('black')
    cbar.set_label('Correlation', color='black')

    # Titles and ticks
    # ax.set_title(f'Correlation Matrix, All Inverters (Values: Pearson r)', color='white', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=90, ha='right', color='black')
    plt.setp(ax.get_yticklabels(), fontsize=10, rotation=0, color='black')

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(plot_folder, "real_data_correlation_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Close plot to free memory
    plt.close()

    # Log the output file path
    print(f"Global Correlation Matrix saved to {output_file}")
    
    
def real_data_visualisation(smoothing=False):
    
    # Input folder
    input_folder = "Inverters"
    
    # Output file path
    plot_folder = f"{PLOT_FOLDER}/Real_data"
    
    # Initialize empty DataFrames
    all_dfs = []
    
    # Get all CSV files in the input folder
    csv_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith('.csv')
    ]
    
    print("\n" + "=" * 60)
    print(f"CREATING PLOTS FOR REAL DATA VISUALISATION ...")
    print("=" * 60)
    
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        print("\nProcessing file:", file)
        df = pd.read_csv(file_path)
        df['collectTime'] = pd.to_datetime(df['collectTime'])

        # Extract inverter ID from the filename
        filename_no_ext = os.path.splitext(file)[0]
        inverter_id = filename_no_ext.replace("inverter_", "")

        if smoothing:
            # Select numeric columns to smooth, excluding inverter_state and weather features
            cols_to_smooth = [
                col for col in df.select_dtypes(include='number').columns
                if col not in ["inverter_state", "diffuse_radiation", "global_tilted_irradiance", "wind_speed_10m",
                                   "precipitation"]
            ]
        
        # Append the DataFrame to the list
        all_dfs.append(df)
        
        # Process data grouped by each day
        for date, group in df.groupby(df['collectTime'].dt.date):
            # continue
            # Only consider days when the inverter was active (state 768)
            if (group['inverter_state'] == 768).any():
                if smoothing:
                    # Apply moving average smoothing to the selected columns
                    group[cols_to_smooth] = (
                        group[cols_to_smooth]
                        .rolling(window=24, min_periods=1)
                        .mean()
                    )
                
                # Fix: ensure index is datetime
                group = group.set_index('collectTime')
                smoothed = "_smoothed" if smoothing else ""
                condition_name = "Real Data"
                output_image = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_{inverter_id}_inverter{smoothed}"
                plot_mppt(group, date, condition_name, plot_folder, output_image, soiling=True)
                plot_currents(group, date,condition_name, plot_folder, output_image, soiling=True)
                plot_voltage(group, date, condition_name, plot_folder, output_image)
                
                # exit()

    # Concatenate all files
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Generate correlation matrix for all inverters' data
    correlation_matrix(combined_df, plot_folder)


real_data_visualisation(smoothing=True)