import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from globals import *

# Time limits
TIME_INIT = "05:00" 
TIME_END = "21:00"


def plot_mppt(df_original, date, condition_title, plot_folder, output_image, soiling=False):
    """
        Plots MPPT powers (reference and effective), irradiances (GTI & DHI), air temperature, wind speed and precipitation
        on the same graph using multiple Y-axes.

        Args:
            - df (pd.DataFrame): DataFrame containing the PV plant measurements with datetime index.
            - date (str): Date of the plot (used in title).
            - condition_title (str): Condition name to display in the plot title.
            - output_folder (str): Folder where the figure will be saved.
            - filename (str): Base filename for the saved figure.
            - soiling (bool): If True, adds precipitation axis.
            
        Notes:
            - MPPT powers is plotted on the primary Y-axis (left).
            - Global Tilted Irradiance (GTI) and Diffuse Horizontal Irradiance (DHI)
              are plotted on secondary and tertiary Y-axes respectively (right).
            - Air temperature and wind speed are plotted on quaternary and quinary Y-axes (right).
            - If soiling is True, Precipitation is plotted on senary Y-axes (right).
            - The figure is saved as a high-resolution PNG file in the specified folder.
    """

    # Filter data by time
    df = df_original.between_time(TIME_INIT, TIME_END)
    
    # Check if the DataFrame contains the column for reference MPPT power measurements
    mppt_clean = 'mppt_power_clean' in df.columns
    # Ensure output folder exists
    os.makedirs(plot_folder, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(18, 6))

    # -------------------------
    # MPPT Power (primary axis)
    # -------------------------
    color_mppt = MPPT_PALETTE['mppt_power']
    if mppt_clean:
        line1, = ax1.plot(df.index, df['mppt_power_clean'], color=MPPT_PALETTE['mppt_power_clean'], label='Reference MPPT', linewidth=2, linestyle=':')
    line2, = ax1.plot(df.index, df['mppt_power'], color=color_mppt, label='Effective MPPT', linewidth=3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("MPPT (kW)", color=color_mppt)

    # Configure tick labels
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    ax1.tick_params(axis='y', labelcolor=color_mppt)
        
    # Adjust Y-axis limits for MPPT
    ax1.set_ylim(0, df['mppt_power_clean'].max() * 1.1 if mppt_clean else df['mppt_power'].max() * 1.1)
    
    
    # Create legend only for MPPT lines
    if mppt_clean:
        ax1.legend(handles=[line1, line2], fontsize=10, loc='upper left')
        
    # -------------------------
    # Global Tilted Irradiance (secondary axis)
    # -------------------------
    ax2 = ax1.twinx()
    color_gti = MPPT_PALETTE['global_tilted_irradiance']
    ax2.plot(df.index, df['global_tilted_irradiance'], color=color_gti,
             label='Global Tilted Irradiance (W/m²)', linewidth=0.9)
    ax2.set_ylabel("GTI (W/m²)", color=color_gti)
    ax2.tick_params(axis='y', labelcolor=color_gti)

    # -------------------------
    # Diffuse Horizontal Irradiance (tertiary axis)
    # -------------------------
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 50))
    color_dr = MPPT_PALETTE['diffuse_radiation']
    ax3.plot(df.index, df['diffuse_radiation'], color=color_dr,
             label='Diffuse Radiation (W/m²)', linewidth=0.6)
    ax3.set_ylabel("DHI (W/m²)", color=color_dr)
    ax3.tick_params(axis='y', labelcolor=color_dr)

    # Adjust Y-axis limits for irradiances
    max_rad = max(df['diffuse_radiation'].max(), df['global_tilted_irradiance'].max()) * 1.1
    ax2.set_ylim(0, max_rad)
    ax3.set_ylim(0, max_rad)

    # -------------------------
    # Air Temperature (quaternary axis)
    # -------------------------
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 100))
    color_temp = MPPT_PALETTE['temperature_2m']
    ax4.plot(df.index, df['temperature_2m'], color=color_temp, linewidth=0.5, label='Air Temperature (°C)')
    ax4.set_ylabel("Air Temp (°C)", color=color_temp)
    ax4.tick_params(axis='y', labelcolor=color_temp)

    # -------------------------
    # Wind Speed (quinary axis)
    # -------------------------
    ax5 = ax1.twinx()
    ax5.spines['right'].set_position(('outward', 150))
    color_wind = MPPT_PALETTE['wind_speed_10m']
    ax5.plot(df.index, df['wind_speed_10m'], color=color_wind, linewidth=0.5, label='Wind Speed (m/s)')
    ax5.set_ylabel("Wind Speed (m/s)", color=color_wind)
    ax5.tick_params(axis='y', labelcolor=color_wind)
    
    # Adjust Y-axis limits for wind spead
    ax5.set_ylim(0, df['wind_speed_10m'].max() * 1.1)

    # -------------------------
    # Precipitation, if soiling (senary axis)
    # -------------------------
    if soiling:
        ax6 = ax1.twinx()
        ax6.spines['right'].set_position(('outward', 200))
        color_precip = MPPT_PALETTE['precipitation']
        ax6.plot(df.index, df['precipitation'], color=color_precip, linewidth=0.5, label='Precipitation (mm/h)')
        ax6.set_ylabel("Precipitation (mm/h)", color=color_precip)
        ax6.tick_params(axis='y', labelcolor=color_precip)
        # Normalize y-axis for small precipitation values
        if df['precipitation'].max() < 1:
            ax6.set_ylim(0, 1)  # normalize y-axis
        else:
            ax6.set_ylim(0, df['precipitation'].max() * 1.1)
        
    # Format x-axis as HH:MM
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Title of the plot
    # plt.title(f"MPPT, DHI, GTI, Air Temperature, Wind Speed and Precipitation, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom')

    # Enable grid for Y-axis
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Draw vertical dashed lines for a specific inverter_state value
    if condition_title == "Real Data":
        for i, state in enumerate(df[LABEL]):
            if state == 768:
                ax1.axvline(df.index[i], color='red', linestyle='--', linewidth=0.5)

        # Create the vertical line legend manually
        line_red = mlines.Line2D([], [], color='red', linestyle='--', linewidth=0.5, label='is 768')

        # Add the lines to the legend
        ax1.legend(handles=[line_red], loc='upper left')
    
    # Adjusts subplot parameters to ensure that all labels, titles, and legends fit within the figure area
    plt.tight_layout()

    # Save figure
    image_path = os.path.join(plot_folder, f"{output_image}_mppt_dhi_gti_temp_wind_precip.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()
    
    # Confirm saving
    print(f"MPPT plot saved to {image_path}")


def plot_currents(df_original, date, condition_title, output_folder, filename, soiling=False):
    """
        Plot reference and effective PV string DC (pv1_i), as well as Precipitation if soiling is True.

        Args:
            - df_original (pd.DataFrame): DataFrame containing current measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - condition_title (str): Condition name to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
            - soiling (bool): If True, adds precipitation axis.
            
        Notes:
            - The figure is saved as a high-resolution PNG file in the specified folder.
    """

    # Filter data by time
    df = df_original.between_time(TIME_INIT, TIME_END)
    
    # Check if the DataFrame contains the column for reference MPPT power measurements
    pv1_i_clean = 'pv1_i_clean' in df.columns

    # Ensure output folder exists; create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Create figure and axis objects with defined size
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot each current with a predefined color and line width
    color_curr = CURR_VOLT_PALETTE['pv1_i']
    if pv1_i_clean:
        line1, = ax1.plot(df.index, df['pv1_i_clean'], color=CURR_VOLT_PALETTE['pv1_i_clean'], label='Reference pv1_i', linewidth=1.5, linestyle=':')
    pv1_i_label = 'Effective pv1_i' if pv1_i_clean else 'pv1_i'
    line2, = ax1.plot(df.index, df['pv1_i'], color=color_curr, label=pv1_i_label, linewidth=2)
    
    # Adjust Y-axis limits for currents
    max_y = df['pv1_i_clean'].max() * 1.1 if pv1_i_clean else df['pv1_i'].max() * 1.1
    ax1.set_ylim(0, max_y)
    
    # Configure tick labels
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)

    # Label axes and set tick colors
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Current (A)", color=color_curr)
    ax1.tick_params(axis='x', colors='black', labelsize=8)
    ax1.tick_params(axis='y', labelcolor=color_curr, labelsize=8)

    # Add horizontal grid lines with light transparency
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Format x-axis as HH:MM
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Set plot title
    # plt.title(f"PV and Phase Currents, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom', color='black')
    
    # Initialize a list to store legend handles
    handles = []
    # -------------------------
    # Precipitation axis (right)
    # -------------------------
    if soiling:
        # Set plot title
        # plt.title(f"PV and Phase Currents, and Precipitation, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom', color='black')
        if condition_title != "Real Data":
            ax2 = ax1.twinx()
            ax2.plot(df.index, df['precipitation'], color=CURR_VOLT_PALETTE['precipitation'],
                    linewidth=0.5, label='Precipitation (mm/h)')
            ax2.set_ylabel("Precipitation (mm/h)", color=CURR_VOLT_PALETTE['precipitation'])
            ax2.tick_params(axis='y', labelcolor=CURR_VOLT_PALETTE['precipitation'])
            # Normalize y-axis for small precipitation values
            max_precip = df['precipitation'].max()
            if max_precip < 1:
                ax2.set_ylim(0, 1)
            else:
                ax2.set_ylim(0, max_precip * 1.1)

    # Add MPPT lines to the handles
    if pv1_i_clean:
        handles.extend([line1, line2])
    
    # Draw vertical dashed lines for a specific inverter_state value
    if condition_title == "Real Data":
        for i, state in enumerate(df[LABEL]):
            if state == 768:
                ax1.axvline(df.index[i], color='red', linestyle='--', linewidth=0.5)

        # Create the vertical line legend manually
        line_red = mlines.Line2D([], [], color='red', linestyle='--', linewidth=0.5, label='is 768')

        # Add the lines to the legend
        ax1.legend(handles=[line_red], loc='upper left')
        handles.append(line_red)
        
    # Create a single combined legend for all handles
    ax1.legend(handles=handles, fontsize=10, loc='upper left')
        
    # Adjust layout to avoid clipping of labels
    plt.tight_layout()

    # Construct full output path and save figure with high resolution
    image_path = os.path.join(output_folder, f"{filename}_pv_phase_currents.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()
    
    # Confirm saving
    print(f"Currents plot saved to {image_path}")


def plot_voltage(df_original, date, condition_title, output_folder, filename):
    """
        Plot PV string reference and effective DC (pv1_u).

        Args:
            - df_original (pd.DataFrame): DataFrame containing voltage measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - condition_title (str): Condition name to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
        
        Notes:
            - The figure is saved as a high-resolution PNG file in the specified folder.
    """

    # Filter data by time
    df = df_original.between_time(TIME_INIT, TIME_END)
    
    # Check if the DataFrame contains the column for reference MPPT power measurements
    pv1_u_clean = 'pv1_u_clean' in df.columns

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create figure and axis
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot voltage
    color_volt = CURR_VOLT_PALETTE['pv1_u']
    if pv1_u_clean:
        line1, = ax.plot(df.index, df['pv1_u_clean'], color=CURR_VOLT_PALETTE['pv1_u_clean'], label='Reference pv1_u', linewidth=1.5, linestyle=':')
    line2, = ax.plot(df.index, df['pv1_u'], label='Effective pv1_u', color=color_volt, linewidth=2)
    
    # Create legend only for MPPT lines
    if pv1_u_clean:
        ax.legend(handles=[line1, line2], fontsize=10, loc='upper left')

    # Horizontal grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Adjust Y-axis limits for voltage
    max_y = df['pv1_u_clean'].max() * 1.1 if pv1_u_clean else df['pv1_u'].max() * 1.1
    ax.set_ylim(0, max_y)
    
    # Configure tick labels
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    
    # Axis labels and tick colors
    ax.set_xlabel("Time")
    ax.set_ylabel("Voltage (V)", color=color_volt)
    ax.tick_params(axis='x', colors='black', labelsize=8)
    ax.tick_params(axis='y', labelcolor=color_volt, labelsize=8)

    # Format x-axis as HH:MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=0)
    
    # Draw vertical dashed lines for a specific inverter_state value
    if condition_title == "Real Data":
        for i, state in enumerate(df[LABEL]):
            if state == 768:
                ax.axvline(df.index[i], color='red', linestyle='--', linewidth=0.5)

        # Create the vertical line legend manually
        line_red = mlines.Line2D([], [], color='red', linestyle='--', linewidth=0.5, label='is 768')

        # Add the lines to the legend
        ax.legend(handles=[line_red], loc='upper left')

    # Title and legend
    # plt.title(f"PV and Phase Voltages, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom', color='black')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    image_path = os.path.join(output_folder, f"{filename}_pv_phase_voltages.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()
    
        # Confirm saving
    print(f"Voltages plot saved to {image_path}")
    
    
def plot_scatter_iv(df_plot, output_folder, filename):
    """
        Scatter plot of pv1_u_mean (x) vs pv1_i_mean (y), colored by inverter_state.

        Args:
            df_plot (pd.DataFrame): Must contain ['pv1_u_mean', 'pv1_i_mean', 'inverter_state'].
            output_folder (str): Folder where to save the figure.
            filename (str): Base name for the saved image (without extension).
    """
    
    # Ensure folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Shuffle dataframe to avoid overplotting patterns
    df_plot = df_plot.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort unique states for consistent color assignment
    unique_states = sorted(df_plot['inverter_state'].dropna().unique())

    # Plot by inverter_state
    handles = []
    for state in unique_states:
        if state in LABELS_MAP:
            label, color = LABELS_MAP[state]
        else:
            label, color = (f"State {state}", "#cccccc")  # fallback color

        subset = df_plot[df_plot['inverter_state'] == state]
        ax.scatter(subset['pv1_u_mean'], subset['pv1_i_mean'],
                   s=15, alpha=1.0, edgecolors='none', color=color, label=label)

        # Legend marker
        handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                               markersize=6, label=label)
        handles.append(handle)

    # Axes labels
    ax.set_xlabel("PV1 Voltage Mean (V)", color='black', fontsize=12)
    ax.set_ylabel("PV1 Current Mean (A)", color='black', fontsize=12)

    # Title
    # plt.title(f"PV1 I–V Scatter by Inverter State, {LOCAL}", fontsize=18, verticalalignment='bottom', color='black')

    # Ticks & grid
    ax.tick_params(axis='x', colors='black', labelsize=9)
    ax.tick_params(axis='y', colors='black', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend
    leg = ax.legend(handles=handles, loc='upper left', fontsize=9)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('gray')
    leg.get_title().set_color('black')

    # Layout & save
    plt.tight_layout()
    image_path = os.path.join(output_folder, filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"I–V scatter plot saved to {image_path}")
