import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from globals import *

# Default colors
MPPT_PALETTE = {
    'mppt_power': '#66ffff',              # Light cyan / pale cyan
    'global_tilted_irradiance': '#ffff99',# Light yellow
    'diffuse_radiation': 'orange',          # Orange
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
            
        Notes:
            - The plot uses a dark theme for better visibility.
            - MPPT power is plotted on the primary Y-axis (left).
            - Global Tilted Irradiance (GTI) and Diffuse Horizontal Irradiance (DHI)
              are plotted on secondary and tertiary Y-axes respectively (right).
            - Air temperature and wind speed are plotted on quaternary and quinary Y-axes (right).
            - The figure is saved as a high-resolution PNG file in the specified folder.
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

        Args:
            - df (pd.DataFrame): DataFrame containing current measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
            
        Notes:
            - The plot uses a dark background for better visibility.
            - The figure is saved as a high-resolution PNG file in the specified folder.
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

        Args:
            - df (pd.DataFrame): DataFrame containing voltage measurements with a datetime index.
            - date (str): Date string to display in the plot title.
            - output_folder (str): Folder path where the plot image will be saved.
            - filename (str): Base name for the saved plot image.
        
        Notes:
            - The plot uses a dark background for better visibility.
            - The figure is saved as a high-resolution PNG file in the specified folder.
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