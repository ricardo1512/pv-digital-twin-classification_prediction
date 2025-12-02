import os
import numpy as np
import pandas as pd
import random
from classes import *
from globals import *
from utils import *
from plot_ts_samples import *

def create_ts_samples_1_soiling(plot_samples=False):
    """
        ?????????
    """
    
    # ==============================================================
    # Initialization
    # ==============================================================
    condition_nr = 1
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    plot_folder = f"{PLOT_FOLDER}/TS_samples/Plots_{condition_nr}_{condition_name}_samples"
    
    # ==============================================================
    # Parameters for soiling accumulation and cleaning
    # ==============================================================
    soiling_min, soiling_max = 0.05, 0.3  # Daily soiling range, for random
    rain_threshold_mm = 1                 # Minimum rainfall to partially clean PV modules
    cleaning_efficiency = 0.4             # Fraction of soiling removed per rain event

    # ==============================================================
    # Subclass Time Series Digital Twin with Soiling
    # ==============================================================
    class DigitalTwinSoilingTS(DigitalTwin):
        """
        Extends DigitalTwin core by including soiling losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                location,
                initial_soiling_factor
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
                location
            )
            self.soiling_factor = initial_soiling_factor
                
        def run(self):
            """
            Simulate the PV system operation under soling conditions for the given meteorological data.
            """
        
            # Run PVLib ModelChain Simulation
            self.mc.run_model(self.weather)
            dc = self.mc.results.dc
            ac = self.mc.results.ac

            # === Soiling simulation
            soiling_levels = []
            growth_rate = random.uniform(0.002, 0.006)  # fraction per day

            dt_seconds = (self.df.index[1] - self.df.index[0]).total_seconds()
            dt_days = dt_seconds / (24 * 3600)

            for i, _ in enumerate(self.df.index):
                rain = self.df['precipitation'].iloc[i]

                if rain > rain_threshold_mm:
                    self.soiling_factor *= (1 - cleaning_efficiency)
                else:
                    self.soiling_factor += growth_rate * dt_days * (soiling_max - self.soiling_factor)
                
                self.soiling_factor = np.clip(self.soiling_factor, 0.001, 1.0)

                soiling_levels.append(self.soiling_factor)
            
            self.df['soiling_fraction'] = soiling_levels
            self.df['soiling_factor'] = 1 - self.df['soiling_fraction']

            # Initialize output DataFrame for storing simulation results
            output = pd.DataFrame(index=self.df.index)
            
            # Set inverter_state to indicate anomaly condition
            output['inverter_state'] = self.condition_nr 
            
            # Compute DC current and voltage
            output['pv1_i'] = dc['i_mp'] * self.df['soiling_factor']
            output['pv1_u'] = dc['v_mp']
            
            # Compute Power Quantities
            output['mppt_power'] = (dc['p_mp'] * self.df['soiling_factor']) / 1000 # W to kW
            output['active_power'] = (ac * self.df['soiling_factor']) / 1000 # W to kW
            output['efficiency'] = (output['active_power'] / output['mppt_power'].replace(0, np.nan)) * 100 # Fraction to percentage

            # Compute AC phase currents (balanced assumption)
            output['a_i'] = output['active_power'] * 1000 / (400 * np.sqrt(3))
            output = output.assign(b_i=output['a_i'], c_i=output['a_i'])

            # Compute AC Voltages per phase and line (balanced assumption)
            output.loc[:, ['a_u', 'b_u', 'c_u']] = 230
            output.loc[:, ['ab_u', 'bc_u', 'ca_u']] = 400

            # Estimate inverter temperature as ambient + 35°C
            output['inv_temperature'] = self.df['temp_air'] + 35

            # Current, voltage, and power under normal (unaffected) conditions
            output['pv1_i_clean'] = dc['i_mp']
            output['pv1_u_clean'] = dc['v_mp']
            output['mppt_power_clean'] = dc['p_mp'] / 1000  # W to kW
            output['a_i_clean'] = ac / (400 * np.sqrt(3))

            # Prepare meteorological DataFrame for merging
            df_merge = self.df.copy()
            df_merge['temperature_2m'] = df_merge['temp_air']
            df_merge['diffuse_radiation'] = df_merge['dhi']
            df_merge['global_tilted_irradiance'] = df_merge['gti']
            df_merge['wind_speed_10m'] = df_merge['wind_speed']
            df_merge = df_merge[METEOROLOGICAL_COLUMNS]

            # Merge meteorological DataFrame with simulation output
            output_full = output.merge(df_merge, left_index=True, right_index=True)
            output_full = output_full.clip(lower=0).fillna(0)
            
            cols_to_export = [c for c in output_full.columns if c not in ['soiling_fraction', 'soiling_factor']]

            return output_full[cols_to_export]

    # Loop through all files in the folder
    for file in os.listdir(WEATHER_FOLDER_PRED):
        print(f"\n{condition_name.title()} | {file}:")
        
        # Collect information from file name
        local = file.removeprefix("weather_").removesuffix(".csv")
        
        # Instantiate location
        latitude, longitude = COORDINATES[local.replace('_', ' ')]
        location = Location(latitude=latitude, longitude=longitude, tz="Europe/Lisbon")
        
        # Read the CSV file
        df_input = pd.read_csv(os.path.join(WEATHER_FOLDER_PRED, file))
        
        # Convert 'date' column to datetime with UTC timezone
        df_input['date'] = pd.to_datetime(df_input['timestamp']).dt.tz_localize('UTC')
        
        # Sort by date
        df_input = df_input.sort_values('date')
        
        for year in YEARS:
            print(f"\tYear {year}")
            # Extract the year
            ts_df = df_input[df_input['date'].dt.year == year]
            
            # Restrict to April–September
            ts_df = ts_df[ts_df['date'].dt.month.isin([4, 5, 6, 7, 8, 9])]

            # Filter hours between 04:00 and 22:00
            ts_df = ts_df.set_index('date').between_time(PREDICTION_HOUR_INIT, PREDICTION_HOUR_END)
            if ts_df.empty:
                continue
            
            # --- Interpolate to 5-min timestep ---
            for col in ts_df.columns:
                ts_df[col] = pd.to_numeric(ts_df[col], errors='coerce').astype('float64')
            ts_df = ts_df.resample('5min').interpolate(method='linear').reset_index()
            
            # Instantiate plant and inverter
            plant = PVPlant(module_data, inverter_data)
            inverter = InverterTwin(inverter_data)
            initial_soiling_factor = random.uniform(soiling_min, soiling_min + 0.02)  # Start with some dirt
            print(f"\t\tInitial soiling factor: {initial_soiling_factor:.3f}")
            twin = DigitalTwinSoilingTS(plant, inverter, ts_df, condition_nr, location, initial_soiling_factor)

            # --- Run simulation ---
            results = twin.run()
            
            # --- Generate Plots with soiling ---
            if plot_samples:
                # Generate Plots
                results_plot = ts_resampling(results.between_time(TIME_INIT, TIME_END))
                condition_title = LABELS_MAP[condition_nr][0]
                output_image = f"{condition_nr}_{condition_name}_samples_{year}_{local}"
                plot_mppt_ts(results_plot, local, plot_folder, output_image, condition_title=condition_title, soiling=True)
                plot_currents_ts(results_plot, local, plot_folder, output_image, condition_title=condition_title, soiling=True)
                plot_voltage_ts(results, local, plot_folder, output_image, condition_title=condition_title)

            # --- Export CSV without clean features columns ---
            cols_to_export = [c for c in results.columns if c not in ['pv1_i_clean', 'pv1_u_clean', 'mppt_power_clean', 'a_i_clean']]
            results_to_export = results[cols_to_export]
            results_to_export.index.name = 'date'
            output_file = f"{TS_SAMPLES}/{condition_nr}_{condition_name}/{condition_nr}_{condition_name}_{year}_{local}_ts_samples.csv"
            results_to_export.to_csv(output_file)
            print(f"Exported timeseries: {output_file}")
        
            exit() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    
create_ts_samples_1_soiling(plot_samples=True)