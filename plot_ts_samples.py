import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from globals import *


def plot_voltage_ts(ts, local, output_folder, filename, condition_title,):
    """
    Generates a time-series plot for PV module voltage (effective and reference).

    This function visualizes PV system voltage measurements, comparing the effective
    voltage with the reference (clean) voltage if available. It configures axes, tick
    labels, gridlines, and saves the figure as PNG for later analysis.

    Args:
        ts (pd.DataFrame): Time-series DataFrame containing PV voltage data.
        local (str): Location identifier for plot titles and file names.
        output_folder (str): Path to save the generated figure.
        filename (str): Base name of the output PNG file.
        condition_title (str): Title describing the operating condition.
    """
    
    # Keep only the first third of the time series for better visualization
    n = len(ts)
    ts = ts.iloc[:n // 3]

    # Check if the DataFrame contains reference voltage column
    pv1_u_clean = 'pv1_u_clean' in ts.columns

    # Create figure and axis
    _, ax = plt.subplots(figsize=(20, 6))

    # Plot voltage
    if pv1_u_clean:
        line1, = ax.plot(
            ts.index, 
            ts['pv1_u_clean'], 
            color=CURR_VOLT_PALETTE['pv1_u_clean'], 
            label='Reference pv1_u', 
            linewidth=1, 
            linestyle=':',
        )
    color_volt = CURR_VOLT_PALETTE['pv1_u']
    line2, = ax.plot(ts.index, ts['pv1_u'], label='Effective pv1_u', color=color_volt, linewidth=1)

    # Configure Axis Labels and Ticks
    ax.set_xlabel("Time")
    ax.set_ylabel("Daily Mean Voltage (V)", color=color_volt)
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    ax.tick_params(axis='x', colors='black', labelsize=8)
    ax.tick_params(axis='y', labelcolor=color_volt, labelsize=8)
    
    # Configure legend only if reference voltage is present
    if pv1_u_clean:
        ax.legend(handles=[line1, line2], fontsize=10, loc='upper left')

    # Add horizontal grid lines with light transparency
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Adjust Y-axis limits for voltage
    ax.set_ylim(300, ts['pv1_u_clean'].max() * 1.1 if pv1_u_clean else ts['pv1_u'].max() * 1.1)

    # Format X-axis as year-month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Title
    # plt.title(f"PV and Phase Voltages, {condition_title}, {local.replace('_', ' ')}", fontsize=18, verticalalignment='bottom', color='black')
    
    # Finalize and save plot
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{filename}_pv_phase_voltages.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved voltage plot to {output_path}")

    # Close figure to free memory
    plt.close()


def plot_currents_ts(ts, local, plot_folder, output_image, condition_title, soiling=False):
    """
    Generates a time-series plot for PV module currents (effective and reference).

    This function visualizes PV system current measurements, comparing the effective
    current with the reference (clean) current if available. It can optionally include
    precipitation on a secondary Y-axis when soiling simulations are enabled. The plot
    includes proper axis labels, tick formatting, grid lines, and saves the figure as PNG.

    Args:
        ts (pd.DataFrame): Time-series DataFrame containing PV current data and optionally precipitation.
        local (str): Location identifier for plot titles and file names.
        plot_folder (str): Path to save the generated figure.
        output_image (str): Base name of the output PNG file.
        condition_title (str): Title describing the operating condition.
        soiling (bool): If True, include precipitation in the plot.
    """

    # Check if the DataFrame contains reference current column
    pv1_i_clean = 'pv1_i_clean' in ts.columns

    # Create figure and axis objects
    _, ax1 = plt.subplots(figsize=(20, 6))

    # Plot each current with a predefined color and line width
    if pv1_i_clean:
        line1, = ax1.plot(
            ts.index, 
            ts['pv1_i_clean'], 
            color=CURR_VOLT_PALETTE['pv1_i_clean'], 
            label='Reference pv1_i', 
            linewidth=1.5, 
            linestyle=':',
        )
    color_curr = CURR_VOLT_PALETTE['pv1_i']
    line2, = ax1.plot(ts.index, ts['pv1_i'], color=color_curr, label='Effective pv1_i', linewidth=2)

    # Adjust Y-axis limits for currents
    ax1.set_ylim(0, ts['pv1_i_clean'].max() * 1.1 if pv1_i_clean else ts['pv1_i'].max() * 1.1)
    
    # Configure Axis Labels and Ticks
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Mean Current (A)", color=color_curr)
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    ax1.tick_params(axis='x', colors='black', labelsize=8)
    ax1.tick_params(axis='y', labelcolor=color_curr, labelsize=8)

    # Create legend only for MPPT lines
    if pv1_i_clean:
        ax1.legend(handles=[line1, line2], fontsize=10, loc='upper left')
    else:
        ax1.legend(handles=[line2], fontsize=10, loc='upper left')

    # Add horizontal grid lines with light transparency
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Format X-axis as year-month
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Set plot title with location and date
    # plt.title(f"PV and Phase Currents, {condition_title}, {local.replace('_', ' ')}", fontsize=18, verticalalignment='bottom', color='black')

    # Precipitation axis (right)
    if soiling:
        # plt.title(f"PV and Phase Currents, and Precipitation, {condition_title}, {local.replace('_', ' ')}", fontsize=18, verticalalignment='bottom', color='black')
        color_precip = CURR_VOLT_PALETTE['precipitation']
        ax2 = ax1.twinx()
        ax2.plot(ts.index, ts['precipitation'], color=color_precip,
                 linewidth=0.5, label='Precipitation (mm/h)')
        ax2.set_ylabel("Daily Sum Precipitation (mm/h)", color=color_precip)
        ax2.tick_params(axis='y', labelcolor=color_precip)
        # Normalize y-axis for small precipitation values
        max_precip = ts['precipitation'].max()
        if max_precip < 1:
            ax2.set_ylim(0, 1)
        else:
            ax2.set_ylim(0, max_precip * 1.1)
    
    # Finalize and save plot
    plt.tight_layout()
    output_path = os.path.join(plot_folder, f"{output_image}_pv_phase_currents.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved current plot to {output_path}")

    # Close figure to free memory
    plt.close()


def plot_mppt_ts(ts, local, plot_folder, output_image, condition_title,soiling=False):
    """
    Generates a multi-axis time-series plot for MPPT power and environmental variables.
    
    This function visualizes photovoltaic system performance together with key
    meteorological variables. It can optionally include precipitation if soiling
    simulations are enabled. The plot combines several Y-axes to show:
        - MPPT power (actual and reference)
        - Global Tilted Irradiance (GTI)
        - Diffuse Horizontal Irradiance (DHI)
        - Air temperature
        - Wind speed
        - Precipitation (optional, only for soiling condition)

    Args:
        ts (pd.DataFrame): Time-series DataFrame containing MPPT power, meteorological
            variables, and optionally precipitation.
        local (str): Location identifier for plot titles and file names.
        plot_folder (str): Path to save the generated figure.
        output_image (str): Base name of the output PNG file.
        condition_title (str): Title describing the operating condition.
        soiling (bool): If True, include precipitation in the plot.
    """
    
    # Check for reference MPPT column
    mppt_clean = 'mppt_power_clean' in ts.columns

    # Create plot
    _, ax1 = plt.subplots(figsize=(25, 6))

    # --------------------------
    # MPPT Power (primary axis)
    # --------------------------
    # Reference MPPT Power
    if mppt_clean:
        line1, = ax1.plot(
            ts.index, 
            ts['mppt_power_clean'], 
            color=MPPT_PALETTE['mppt_power_clean'], 
            label='Reference MPPT', 
            linewidth=1.8, 
            linestyle=':',
        )
        
    # Effective MPPT Power
    color_mppt = MPPT_PALETTE['mppt_power']
    line2, = ax1.plot(ts.index, ts['mppt_power'], color=color_mppt, label='Effective MPPT', linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Mean MPPT (kW)", color=color_mppt)
    ax1.tick_params(axis='y', labelcolor=color_mppt)
    
    # Adjust Y-axis limits for MPPT
    ax1.set_ylim(0, ts['mppt_power_clean'].max() * 1.1 if mppt_clean else ts['mppt_power'].max() * 1.1)
    
    # Configure tick labels
    plt.xticks(rotation=0, ha='center', color='black', fontsize=8)
    plt.yticks(color='black', fontsize=8)
    ax1.tick_params(axis='y', labelcolor=color_mppt)
    
    # Create legend only for MPPT lines
    if mppt_clean:
        ax1.legend(handles=[line1, line2], fontsize=10, loc='upper left')
    
    # ------------------------------------------
    # Global Tilted Irradiance (secondary axis)
    # ------------------------------------------
    ax2 = ax1.twinx()
    color_gti = MPPT_PALETTE['global_tilted_irradiance']
    ax2.plot(ts.index, ts['global_tilted_irradiance'], color=color_gti, label='Global Tilted Irradiance (W/m²)', linewidth=0.6)
    ax2.set_ylabel("Daily Mean GTI (W/m²)", color=color_gti)
    ax2.tick_params(axis='y', labelcolor=color_gti)

    # ----------------------------------------------
    # Diffuse Horizontal Irradiance (tertiary axis)
    # ----------------------------------------------
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 50))
    color_dr = MPPT_PALETTE['diffuse_radiation']
    ax3.plot(ts.index, ts['diffuse_radiation'], color=color_dr, label='Diffuse Radiation (W/m²)', linewidth=0.6)
    ax3.set_ylabel("Daily Mean DHI (W/m²)", color=color_dr)
    ax3.tick_params(axis='y', labelcolor=color_dr)

    # Adjust Y-axis limits for irradiances
    max_rad = max(ts['diffuse_radiation'].max(), ts['global_tilted_irradiance'].max()) * 1.1
    ax2.set_ylim(0, max_rad)
    ax3.set_ylim(0, max_rad)

    # ----------------------------------
    # Air Temperature (quaternary axis)
    # ----------------------------------
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 100))
    color_temp = MPPT_PALETTE['temperature_2m']
    ax4.plot(ts.index, ts['temperature_2m'], color=color_temp, linewidth=0.5, label='Air Temperature (°C)')
    ax4.set_ylabel("Daily Mean Air Temp (°C)", color=color_temp)
    ax4.tick_params(axis='y', labelcolor=color_temp)

    # --------------------------
    # Wind Speed (quinary axis)
    # --------------------------
    ax5 = ax1.twinx()
    ax5.spines['right'].set_position(('outward', 150))
    color_wind = MPPT_PALETTE['wind_speed_10m']
    ax5.plot(ts.index, ts['wind_speed_10m'], color=color_wind, linewidth=0.5, label='Wind Speed (m/s)')
    ax5.set_ylabel("Daily Mean Wind Speed (m/s)", color=color_wind)
    ax5.tick_params(axis='y', labelcolor=color_wind)
    ax5.set_ylim(0, ts['wind_speed_10m'].max() * 1.1)
    
    # --------------------------------------
    # Precipitation, if soiling (sixth axis)
    # --------------------------------------
    if soiling:
        ax6 = ax1.twinx()
        ax6.spines['right'].set_position(('outward', 200))
        color_precip = MPPT_PALETTE['precipitation']
        ax6.plot(ts.index, ts['precipitation'], color=color_precip, linewidth=0.5, label='Precipitation (mm/h)')
        ax6.set_ylabel("Daily Sum Precipitation (mm/h)", color=color_precip)
        ax6.tick_params(axis='y', labelcolor=color_precip)
        # Normalize y-axis for small precipitation values
        if ts['precipitation'].max() < 1:
            ax6.set_ylim(0, 1)  # Normalize y-axis
        else:
            ax6.set_ylim(0, ts['precipitation'].max() * 1.1)

    # Format X-axis as year-month
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Title of the plot
    if soiling:
        title = "MPPT, DHI, GTI, Air Temperature, Wind Speed and Precipitation"
    else:
        title = "MPPT, DHI, GTI, Air Temperature and Wind Speed"
    # plt.title(f"{title}, {condition_title}, {local.replace('_', ' ')}", fontsize=18, verticalalignment='bottom')

    # Enable grid for Y-axis
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Finalize and save plot
    plt.tight_layout()
    if soiling:
        features = "mppt_dhi_gti_temp_wind_precipitation"
    else:
        features = "mppt_dhi_gti_temp_wind"
    plt.savefig(os.path.join(plot_folder, f"{output_image}_{features}.png"), dpi=300, bbox_inches='tight')
    print(f"Saved MPPT plot to {os.path.join(plot_folder, f'{output_image}_{features}.png')}")

    # Close figure to free memory
    plt.close()

