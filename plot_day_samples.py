import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from globals import *

# Time limits
TIME_INIT = "05:00" 
TIME_END = "21:00"


# Default colors
MPPT_PALETTE = {
    'mppt_power': '#00ff00',               # Light cyan / pale cyan
    'mppt_power_clean': '#00ff00',         # Light cyan / pale cyan
    'global_tilted_irradiance': '#ffff99', # Light yellow
    'diffuse_radiation': '#ff6600',        # Orange
    'temperature_2m': '#ff99bb',           # Light pink / soft pink
    'wind_speed_10m': '#4d94ff',           # Medium blue / cornflower blue
    'precipitation': '#66ffff',                # White
}

CURR_VOLT_PALETTE = {
    'pv1_i': '#ff3300',         # Red
    'pv1_i_clean': '#ff3300',   # Red
    'a_i': '#3399ff',           # Blue
    'a_i_clean': '#3399ff',     # Blue
    'pv1_u': '#ff6600',         # Orange
    'pv1_u_clean': '#ff6600',   # Orange
    'precipitation': '#66ffff',     # White
}


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
            - The plot uses a dark theme for better visibility.
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

    # Use dark style for the plot
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # -------------------------
    # MPPT Power (primary axis)
    # -------------------------
    color_mppt = MPPT_PALETTE['mppt_power']
    if mppt_clean:
        line1, = ax1.plot(df.index, df['mppt_power_clean'], color=MPPT_PALETTE['mppt_power_clean'], label='Reference MPPT', linewidth=1.5, linestyle=':')
    line2, = ax1.plot(df.index, df['mppt_power'], color=color_mppt, label='Effective MPPT', linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("MPPT (kW)", color=color_mppt)
    ax1.tick_params(axis='y', labelcolor=color_mppt)

    # Adjust Y-axis limits for MPPT
    ax1.set_ylim(0, df['mppt_power_clean'].max() * 1.1 if mppt_clean else df['mppt_power'].max() * 1.1)
    
    # Create legend only for MPPT lines with white frame
    if mppt_clean:
        ax1.legend(handles=[line1, line2], fontsize=10, loc='upper left', facecolor='black', edgecolor='white')

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
    plt.title(f"MPPT, DHI, GTI, Air Temperature, Wind Speed and Precipitation, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom')

    # Enable grid for Y-axis
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Draw vertical dashed lines for a specific inverter_state value
    if condition_title == "Real Data":
        for i, state in enumerate(df[LABEL]):
            if state == 768:
                ax1.axvline(df.index[i], color='red', linestyle='--', linewidth=0.5)

        # Create the vertical line legend manually
        line_red = mlines.Line2D([], [], color='red', linestyle='--', linewidth=0.5, label='Fault')

        # Add the lines to the legend
        ax1.legend(handles=[line_red], loc='upper left')
    
    # Adjusts subplot parameters to ensure that all labels, titles, and legends fit within the figure area
    plt.tight_layout()

    # Save figure
    image_path = os.path.join(plot_folder, f"{output_image}_mppt_dhi_gti_temp_wind.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()
    
    # Confirm saving
    print(f"MPPT plot saved to {image_path}")


def plot_currents(df_original, date, condition_title, output_folder, filename, soiling=False):
    """
        Plot PV string DC (pv1_i) and AC phase current (a_i), both reference and effective, as well as Precipitation if soiling is True,
        in the same dark-themed graph.

        Args:
            - df_original (pd.DataFrame): DataFrame containing current measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - condition_title (str): Condition name to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
            - soiling (bool): If True, adds precipitation axis.
            
        Notes:
            - The plot uses a dark background for better visibility.
            - The figure is saved as a high-resolution PNG file in the specified folder.
    """

    # Filter data by time
    df = df_original.between_time(TIME_INIT, TIME_END)
    
    # Check if the DataFrame contains the column for reference MPPT power measurements
    pv1_i_clean = 'pv1_i_clean' in df.columns

    # Ensure output folder exists; create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Set dark background style for the plot
    plt.style.use('dark_background')

    # Create figure and axis objects with defined size
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot each current with a predefined color and line width
    if pv1_i_clean:
        line1, = ax1.plot(df.index, df['pv1_i_clean'], color=CURR_VOLT_PALETTE['pv1_i_clean'], label='Reference pv1_i', linewidth=1.5, linestyle=':')
        line2, = ax1.plot(df.index, df['a_i_clean'], color=CURR_VOLT_PALETTE['a_i_clean'], label='Reference a_i', linewidth=1.5, linestyle=':')
    pv1_i_label = 'Effective pv1_i' if pv1_i_clean else 'pv1_i'
    a_i_label = 'Effective pv1_i' if pv1_i_clean else 'a_i'
    line3, = ax1.plot(df.index, df['pv1_i'], color=CURR_VOLT_PALETTE['pv1_i'], label=pv1_i_label, linewidth=2)
    line4, = ax1.plot(df.index, df['a_i'], color=CURR_VOLT_PALETTE['a_i'], label=a_i_label, linewidth=2)
    
    # Adjust Y-axis limits for currents
    max_y = df['pv1_i_clean'].max() * 1.1 if pv1_i_clean else df['pv1_i'].max() * 1.1
    ax1.set_ylim(0, max_y)

    # Label axes and set tick colors to white for visibility
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Current (A)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    # Add horizontal grid lines with light transparency
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Format x-axis as HH:MM
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Set plot title
    plt.title(f"PV and Phase Currents, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom', color='white')
    
    # -------------------------
    # Precipitation axis (right)
    # -------------------------
    if soiling:
        # Set plot title
        plt.title(f"PV and Phase Currents, and Precipitation, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom', color='white')
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

        # Initialize a list to store legend handles
        handles = []

        # Add MPPT lines to the handles
        if pv1_i_clean:
            handles.extend([line1, line3, line2, line4])
        else:
            handles.extend([line3, line4])
    
    # Draw vertical dashed lines for a specific inverter_state value
    if condition_title == "Real Data":
        for i, state in enumerate(df[LABEL]):
            if state == 768:
                ax1.axvline(df.index[i], color='red', linestyle='--', linewidth=0.5)

        # Create the vertical line legend manually
        line_red = mlines.Line2D([], [], color='red', linestyle='--', linewidth=0.5, label='Fault')

        # Add the lines to the legend
        ax1.legend(handles=[line_red], loc='upper left')
        handles.append(line_red)
    
        
    # Create a single combined legend for all handles
    ax1.legend(handles=handles, fontsize=10, loc='upper left', facecolor='black', edgecolor='white')
        
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
        Plot PV string reference and effective DC (pv1_u) in a dark-themed graph.

        Args:
            - df_original (pd.DataFrame): DataFrame containing voltage measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - condition_title (str): Condition name to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
        
        Notes:
            - The plot uses a dark background for better visibility.
            - The figure is saved as a high-resolution PNG file in the specified folder.
    """

    # Filter data by time
    df = df_original.between_time(TIME_INIT, TIME_END)
    
    # Check if the DataFrame contains the column for reference MPPT power measurements
    pv1_u_clean = 'pv1_u_clean' in df.columns

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Set dark background style
    plt.style.use('dark_background')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot voltage
    if pv1_u_clean:
        line1, = ax.plot(df.index, df['pv1_u_clean'], color=CURR_VOLT_PALETTE['pv1_u_clean'], label='Reference pv1_u', linewidth=1.5, linestyle=':')
    line2, = ax.plot(df.index, df['pv1_u'], label='Effective pv1_u', color=CURR_VOLT_PALETTE['pv1_u'], linewidth=2)

    # Axis labels and tick colors
    ax.set_xlabel("Time")
    ax.set_ylabel("Voltage (V)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Create legend only for MPPT lines with white frame
    if pv1_u_clean:
        ax.legend(handles=[line1, line2], fontsize=10, loc='upper left', facecolor='black', edgecolor='white')

    # Horizontal grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Adjust Y-axis limits for voltage
    max_y = df['pv1_u_clean'].max() * 1.1 if pv1_u_clean else df['pv1_u'].max() * 1.1
    ax.set_ylim(0, max_y)

    # Format x-axis as HH:MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=0)
    
    # Draw vertical dashed lines for a specific inverter_state value
    if condition_title == "Real Data":
        for i, state in enumerate(df[LABEL]):
            if state == 768:
                ax.axvline(df.index[i], color='red', linestyle='--', linewidth=0.5)

        # Create the vertical line legend manually
        line_red = mlines.Line2D([], [], color='red', linestyle='--', linewidth=0.5, label='Fault')

        # Add the lines to the legend
        ax.legend(handles=[line_red], loc='upper left')

    # Title and legend
    plt.title(f"PV and Phase Voltages, {condition_title}, {LOCAL}, {date}", fontsize=18, verticalalignment='bottom', color='white')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    image_path = os.path.join(output_folder, f"{filename}_pv_phase_voltages.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()
    
        # Confirm saving
    print(f"Voltages plot saved to {image_path}")