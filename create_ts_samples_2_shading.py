import os
import numpy as np
import pandas as pd
import random

from classes import *
from globals import *
from utils import *
from plot_ts_samples import *

# =================================================================
# Digital Twin Simulation - Shading Condition, Time Series Samples
# =================================================================
def create_ts_samples_2_shading(plot_samples=False):
    """
    Runs the digital twin simulation for photovoltaic systems under shading
    conditions using time-series meteorological data.

    This function processes multi-location weather datasets, interpolates
    measurements to a uniform 5-minute resolution, and simulates PV system
    behaviour while modeling gradually increasing shading fractions. The
    shading factor evolves over time, reducing photocurrent and slightly
    affecting module voltage.

    Args:
        plot_samples (bool): If True, generates time-series plots of the simulated electrical
            quantities. Default is False.

    Workflow:
        1. Initialize parameters and iterate through all available weather files.
        2. Load meteorological datasets and extract time-series for selected months.
        3. Interpolate measurements to a uniform 5-minute timestep.
        4. Instantiate PVPlant, InverterTwin, and a DigitalTwin subclass with shading.
        5. Simulate system behaviour while applying gradually increasing shading fractions over time.
        6. Merge simulated electrical quantities with meteorological variables.
        7. Optionally produce plots for power, currents, and voltage.
        8. Export final time-series results for machine learning analysis.

    Output:
        CSV files saved in the TS_SAMPLES directory, containing time-series features
        with simulated PV behavior under shading conditions for each location and year.
    """
    
    # ==============================================================
    # Initialization
    # ==============================================================
    condition_nr = 2
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    plot_folder = f"{PLOT_FOLDER}/TS_samples/Plots_{condition_nr}_{condition_name}_samples"
    
    # ==============================================================
    # Parameters for shading growth
    # ==============================================================
    shading_min, shading_max = 0.0, 0.4  # Minimum and maximum shading fraction
    shading_growth_rate = 0.000006       # Fraction increase per timestep

    # ==============================================================
    # Subclass Time Series Digital Twin with Shading
    # ==============================================================
    class DigitalTwinShadingTS(DigitalTwin):
        """
        Extends DigitalTwin core by including shading losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                location,
                initial_shading_fraction,
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
                location,
            )
            self.initial_shading_fraction = initial_shading_fraction
                
        def run(self):
            # Run PVLib ModelChain Simulation
            self.mc.run_model(self.weather)
            dc = self.mc.results.dc
            ac = self.mc.results.ac

            # Time-dependent shading simulation
            shading_fraction = self.initial_shading_fraction
            shading_levels = []
            
            # Incrementally increase shading at each timestep
            for _ in self.df.index:
                shading_levels.append(min(1.0, shading_fraction))
                shading_fraction += shading_growth_rate

            # Store shading fraction in dataframe
            self.df['shading_fraction'] = shading_levels

            # Apply shading to current and voltage
            current_factor = 1 - self.df['shading_fraction']
            voltage_factor = 1 - self.df['shading_fraction'] * 0.2  # smaller voltage drop

            # Initialize output DataFrame for storing simulation results
            output = pd.DataFrame(index=self.df.index)
            
            # Current, voltage, and power under normal (unaffected) conditions
            output['pv1_i_clean'] = dc['i_mp']
            output['pv1_u_clean'] = dc['v_mp']
            output['mppt_power_clean'] = dc['p_mp'] / 1000  # W to kW
            output['a_i_clean'] = ac / (400 * np.sqrt(3))
            
            # Set inverter_state to indicate anomaly condition
            output['inverter_state'] = self.condition_nr 
            
            # Compute DC current and voltage
            output['pv1_i'] = dc['i_mp'] * current_factor
            output['pv1_u'] = dc['v_mp'] * voltage_factor
            
            # Compute Power Quantities
            output['mppt_power'] = (dc['p_mp'] * current_factor * voltage_factor) / 1000 # W to kW
            output['active_power'] = (ac * current_factor * voltage_factor) / 1000 # W to kW
            output['efficiency'] = (output['active_power'] / output['mppt_power'].replace(0, np.nan)) * 100 # Fraction to percentage

            # Compute AC phase currents (balanced assumption)
            output['a_i'] = output['active_power'] * 1000 / (400 * np.sqrt(3))
            output = output.assign(b_i=output['a_i'], c_i=output['a_i'])
            
            # Compute AC Voltages per phase and line (balanced assumption)
            output.loc[:, ['a_u', 'b_u', 'c_u']] = 230
            output.loc[:, ['ab_u', 'bc_u', 'ca_u']] = 400

            # Estimate inverter temperature as ambient + 35°C
            output['inv_temperature'] = self.df['temp_air'] + 35

            # Prepare meteorological variables for merging
            df_merge = self.df.copy()
            df_merge['temperature_2m'] = df_merge['temp_air']
            df_merge['diffuse_radiation'] = df_merge['dhi']
            df_merge['global_tilted_irradiance'] = df_merge['gti']
            df_merge['wind_speed_10m'] = df_merge['wind_speed']
            df_merge = df_merge[METEOROLOGICAL_COLUMNS]

            # Merge simulation output and environmental data
            output_full = output.merge(df_merge, left_index=True, right_index=True)
            output_full = output_full.clip(lower=0).fillna(0)

            # Remove helper columns
            cols_to_export = [c for c in output_full.columns if c not in ['shading_fraction']]

            return output_full[cols_to_export]

    # ==============================================================
    # Main Simulation Loop (Per File/Location)
    # ==============================================================
    for file in os.listdir(WEATHER_FOLDER_PRED):
        print(f"\n{condition_name.title()} | {file}:")
        
        # Extract location identifier
        local = file.removeprefix("weather_").removesuffix(".csv")
        
        # Instantiate geographic location
        latitude, longitude = COORDINATES[local.replace('_', ' ')]
        location = Location(latitude=latitude, longitude=longitude, tz="Europe/Lisbon")
        
        # Read meteorological CSV
        df_input = pd.read_csv(os.path.join(WEATHER_FOLDER_PRED, file))
        
        # Convert timestamp to timezone-aware datetime
        df_input['date'] = pd.to_datetime(df_input['timestamp']).dt.tz_localize('UTC')
        
        # Ensure chronological order
        df_input = df_input.sort_values('date')
        
        for year in YEARS:
            print(f"\tYear {year}")
           
            # Filter data by year
            ts_df = df_input[df_input['date'].dt.year == year]
            
            # Restrict to April–September months (Summer period)
            ts_df = ts_df[ts_df['date'].dt.month.isin([4, 5, 6, 7, 8, 9])]

            # Time-of-day filtering
            ts_df = ts_df.set_index('date').between_time(PREDICTION_HOUR_INIT, PREDICTION_HOUR_END)
            if ts_df.empty:
                continue
            
            # Interpolate to uniform 5-minute resolution
            for col in ts_df.columns:
                ts_df[col] = pd.to_numeric(ts_df[col], errors='coerce').astype('float64')
            ts_df = ts_df.resample('5min').interpolate(method='linear').reset_index()

            # Instantiate plant and inverter
            plant = PVPlant(module_data, inverter_data)
            inverter = InverterTwin(inverter_data)
            initial_shading_fraction = random.uniform(shading_min, shading_max)
            print(f"\t\tInitial shading fraction: {initial_shading_fraction:.3f}")
            twin = DigitalTwinShadingTS(
                plant, 
                inverter, 
                ts_df, 
                condition_nr, 
                location, 
                initial_shading_fraction
            )

            # Run simulation
            results = twin.run()

            # Generate Plots with shading
            if plot_samples:
                results_plot = ts_resampling(results.between_time(TIME_INIT, TIME_END))
                condition_title = LABELS_MAP[condition_nr][0]
                output_image = f"{condition_nr}_{condition_name}_ts_samples_{year}_{local}"
                plot_mppt_ts(results_plot, local, plot_folder, output_image, condition_title=condition_title)
                plot_currents_ts(results_plot, local, plot_folder, output_image, condition_title=condition_title)
                plot_voltage_ts(results, local, plot_folder, output_image, condition_title=condition_title)

            # Export CSV without clean features columns
            cols_to_export = [c for c in results.columns if c not in ['pv1_i_clean', 'pv1_u_clean', 'mppt_power_clean', 'a_i_clean']]
            results_to_export = results[cols_to_export]
            results_to_export.index.name = 'date'
            output_file = f"{TS_SAMPLES_FOLDER}/{condition_nr}_{condition_name}/{condition_nr}_{condition_name}_{year}_{local}_ts_samples.csv"
            results_to_export.to_csv(output_file)
            print(f"Exported timeseries: {output_file}\n")