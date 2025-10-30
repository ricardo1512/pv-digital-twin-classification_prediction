import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from globals import *

# Time limits
TIME_INIT = "06:00" 
TIME_END = "20:00"


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
    'pv1_i': '#ff3300',       # Red
    'pv1_i_clean': '#ff3300', # Red
    'a_i': '#3399ff',         # Blue
    'a_i_clean': '#3399ff',   # Blue
    'pv1_u': '#ff6600',       # Orange
    'pv1_u_clean': '#ff6600', # Orange
    'precipitation': '#66ffff',   # White
}


def plot_mppt_ts(ts, local, condition_title, plot_folder, output_image, soiling=False):
    """
        ?????
    """
    
    # Check if the DataFrame contains the column for reference MPPT power measurements
    mppt_clean = 'mppt_power_clean' in ts.columns

    # Ensure o
    # utput folder exists
    os.makedirs(plot_folder, exist_ok=True)

    # Use dark style for the plot
    plt.style.use('dark_background')
    _, ax1 = plt.subplots(figsize=(25, 6))

    # -------------------------
    # MPPT Power (primary axis)
    # -------------------------
    color_mppt = MPPT_PALETTE['mppt_power']
    if mppt_clean:
        line1, = ax1.plot(ts.index, ts['mppt_power_clean'], color=MPPT_PALETTE['mppt_power_clean'], label='Reference MPPT', linewidth=1.8, linestyle=':')
    line2, = ax1.plot(ts.index, ts['mppt_power'], color=color_mppt, label='Effective MPPT', linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("MPPT (kW)", color=color_mppt)
    ax1.tick_params(axis='y', labelcolor=color_mppt)
    
    # Adjust Y-axis limits for MPPT
    ax1.set_ylim(0, ts['mppt_power_clean'].max() * 1.1 if mppt_clean else ts['mppt_power'].max() * 1.1)
    
    # Create legend only for MPPT lines with white frame
    if mppt_clean:
        ax1.legend(handles=[line1, line2], fontsize=10, loc='upper left', facecolor='black', edgecolor='white')
    
    # -------------------------
    # Global Tilted Irradiance (secondary axis)
    # -------------------------
    ax2 = ax1.twinx()
    color_gti = MPPT_PALETTE['global_tilted_irradiance']
    ax2.plot(ts.index, ts['global_tilted_irradiance'], color=color_gti,
             label='Global Tilted Irradiance (W/m²)', linewidth=0.6)
    ax2.set_ylabel("GTI (W/m²)", color=color_gti)
    ax2.tick_params(axis='y', labelcolor=color_gti)

    # -------------------------
    # Diffuse Horizontal Irradiance (tertiary axis)
    # -------------------------
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 50))
    color_dr = MPPT_PALETTE['diffuse_radiation']
    ax3.plot(ts.index, ts['diffuse_radiation'], color=color_dr,
             label='Diffuse Radiation (W/m²)', linewidth=0.6)
    ax3.set_ylabel("DHI (W/m²)", color=color_dr)
    ax3.tick_params(axis='y', labelcolor=color_dr)

    # Adjust Y-axis limits for irradiances
    max_rad = max(ts['diffuse_radiation'].max(), ts['global_tilted_irradiance'].max()) * 1.1
    ax2.set_ylim(0, max_rad)
    ax3.set_ylim(0, max_rad)

    # -------------------------
    # Air Temperature (quaternary axis)
    # -------------------------
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 100))
    color_temp = MPPT_PALETTE['temperature_2m']
    ax4.plot(ts.index, ts['temperature_2m'], color=color_temp, linewidth=0.5, label='Air Temperature (°C)')
    ax4.set_ylabel("Air Temp (°C)", color=color_temp)
    ax4.tick_params(axis='y', labelcolor=color_temp)

    # -------------------------
    # Wind Speed (quinary axis)
    # -------------------------
    ax5 = ax1.twinx()
    ax5.spines['right'].set_position(('outward', 150))
    color_wind = MPPT_PALETTE['wind_speed_10m']
    ax5.plot(ts.index, ts['wind_speed_10m'], color=color_wind, linewidth=0.5, label='Wind Speed (m/s)')
    ax5.set_ylabel("Wind Speed (m/s)", color=color_wind)
    ax5.tick_params(axis='y', labelcolor=color_wind)
    
    # Adjust Y-axis limits for wind spead
    ax5.set_ylim(0, ts['wind_speed_10m'].max() * 1.1)
    
    # -------------------------
    # Precipitation, if soiling (sixth axis)
    # -------------------------
    if soiling:
        ax6 = ax1.twinx()
        ax6.spines['right'].set_position(('outward', 200))
        color_precip = MPPT_PALETTE['precipitation']
        ax6.plot(ts.index, ts['precipitation'], color=color_precip, linewidth=0.5, label='Precipitation (mm/h)')
        ax6.set_ylabel("Precipitation (mm/h)", color=color_precip)
        ax6.tick_params(axis='y', labelcolor=color_precip)
        # Normalize y-axis for small precipitation values
        if ts['precipitation'].max() < 1:
            ax6.set_ylim(0, 1)  # normalize y-axis
        else:
            ax6.set_ylim(0, ts['precipitation'].max() * 1.1)

    # Format X-axis as year-month-day
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Title of the plot
    if soiling:
        title = "MPPT, DHI, GTI, Air Temperature, Wind Speed and Precipitation"
    else:
        title = "MPPT, DHI, GTI, Air Temperature and Wind Speed"
    
    plt.title(f"{title}, {condition_title}, {local}", fontsize=18, verticalalignment='bottom')

    # Enable grid for Y-axis
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save figure
    if soiling:
        features = "mppt_dhi_gti_temp_wind_precipitation"
    else:
        features = "mppt_dhi_gti_temp_wind"
    plt.savefig(os.path.join(plot_folder, f"{output_image}_ts_{features}.png"), dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()


def plot_currents_ts(ts, local, condition_title, plot_folder, output_image, soiling=False):
    """

    """

    # Check if the DataFrame contains the column for reference MPPT power measurements
    pv1_i_clean = 'pv1_i_clean' in ts.columns

    # Ensure output folder exists; create it if necessary
    os.makedirs(plot_folder, exist_ok=True)

    # Set dark background style for the plot
    plt.style.use('dark_background')

    # Create figure and axis objects with defined size
    _, ax1 = plt.subplots(figsize=(20, 6))

    # Plot each current with a predefined color and line width
    if pv1_i_clean:
        line1, = ax1.plot(ts.index, ts['pv1_i_clean'], color=CURR_VOLT_PALETTE['pv1_i_clean'], label='Reference pv1_i', linewidth=1.5, linestyle=':')
        line2, = ax1.plot(ts.index, ts['a_i_clean'], color=CURR_VOLT_PALETTE['a_i_clean'], label='Reference a_i', linewidth=1.5, linestyle=':')
    line3, = ax1.plot(ts.index, ts['pv1_i'], color=CURR_VOLT_PALETTE['pv1_i'], label='Effective pv1_i', linewidth=2)
    line4, = ax1.plot(ts.index, ts['a_i'], color=CURR_VOLT_PALETTE['a_i'], label='Effective a_i', linewidth=2)

    # Adjust Y-axis limits for currents
    ax1.set_ylim(0, ts['pv1_i_clean'].max() * 1.1)
    
    # Label axes and set tick colors to white for visibility
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Current (A)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    
    # Create legend only for MPPT lines with white frame
    if pv1_i_clean:
        ax1.legend(handles=[line1, line3, line2, line4], fontsize=10, loc='upper left', facecolor='black', edgecolor='white')
    else:
        ax1.legend(handles=[line3, line4], fontsize=10, loc='upper left', facecolor='black', edgecolor='white')

    # Add horizontal grid lines with light transparency
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Format X-axis as year-month-day
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Set plot title with location and date, white text for dark background
    plt.title(f"PV and Phase Currents, {condition_title}, {local}", fontsize=18, verticalalignment='bottom', color='white')

        # -------------------------
    # Precipitation axis (right)
    # -------------------------
    if soiling:
        # Set plot title
        plt.title(f"PV and Phase Currents, and Precipitation, {condition_title}, {local}", fontsize=18, verticalalignment='bottom', color='white')
        ax2 = ax1.twinx()
        ax2.plot(ts.index, ts['precipitation'], color=CURR_VOLT_PALETTE['precipitation'],
                 linewidth=0.5, label='Precipitation (mm/h)')
        ax2.set_ylabel("Precipitation (mm/h)", color=CURR_VOLT_PALETTE['precipitation'])
        ax2.tick_params(axis='y', labelcolor=CURR_VOLT_PALETTE['precipitation'])
        # Normalize y-axis for small precipitation values
        max_precip = ts['precipitation'].max()
        if max_precip < 1:
            ax2.set_ylim(0, 1)
        else:
            ax2.set_ylim(0, max_precip * 1.1)
    # Adjust layout to avoid clipping of labels
    plt.tight_layout()

    # Construct full output path and save figure with high resolution
    output_path = os.path.join(plot_folder, f"{output_image}_pv_phase_currents.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()


def plot_voltage_ts(ts, local, condition_title, output_folder, filename):
    """

    """
    # For better visualisation, keep only the first third of the time series
    n = len(ts)
    ts = ts.iloc[:n // 3]

    # Check if the DataFrame contains the column for reference MPPT power measurements
    pv1_u_clean = 'pv1_u_clean' in ts.columns

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Set dark background style
    plt.style.use('dark_background')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot voltage
    if pv1_u_clean:
        line1, = ax.plot(ts.index, ts['pv1_u_clean'], color=CURR_VOLT_PALETTE['pv1_u_clean'], label='Reference pv1_u', linewidth=1, linestyle=':')
    line2, = ax.plot(ts.index, ts['pv1_u'], label='Effective pv1_u', color=CURR_VOLT_PALETTE['pv1_u'], linewidth=1)

    # Axis labels and tick colors
    ax.set_xlabel("Date")
    ax.set_ylabel("Voltage (V)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Create legend only for MPPT lines with white frame
    if pv1_u_clean:
        ax.legend(handles=[line1, line2], fontsize=10, loc='upper left', facecolor='black', edgecolor='white')

    # Horizontal grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Adjust Y-axis limits for voltage
    ax.set_ylim(300, ts['pv1_u_clean'].max() * 1.1)

    # Format X-axis as year-month-day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Title and legend
    plt.title(f"PV and Phase Voltages, {condition_title}, {local}", fontsize=18, verticalalignment='bottom', color='white')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_folder, f"{filename}_pv_phase_voltages.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Close figure to free memory
    plt.close()