import os
import numpy as np
import pandas as pd
import random

from classes import *
from globals import *
from utils import *
from plot_ts_samples import *

# =================================================================
# Digital Twin Simulation - Cracks Condition, Time Series Samples
# =================================================================
def create_ts_samples_3_cracks(plot_samples=False):
    """
    Runs the digital twin simulation for photovoltaic systems under cracks
    conditions using time-series meteorological data.

    This function processes multi-location weather datasets, interpolates
    measurements to a uniform 5-minute resolution, and simulates PV system
    behaviour while modeling gradually increasing cracks degradation. The
    degradation factors evolve over time, reducing photocurrent and slightly
    affecting module voltage.

    Args:
        plot_samples (bool): If True, generates time-series plots of the simulated electrical
            quantities. Default is False.

    Workflow:
        1. Initialize parameters and iterate through all available weather files.
        2. Load meteorological datasets and extract time-series for selected months.
        3. Interpolate measurements to a uniform 5-minute timestep.
        4. Instantiate PVPlant, InverterTwin, and a DigitalTwin subclass with cracks.
        5. Simulate system behaviour while applying gradually increasing cracks degradation over time.
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
    condition_nr = 3
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    plot_folder = f"{PLOT_FOLDER}/TS_samples/Plots_{condition_nr}_{condition_name}_samples"
    
    # ==============================================================
    # Subclass Time Series Digital Twin with Cracks
    # ==============================================================
    class DigitalTwinCracksTS(DigitalTwin):
        """
        Extends DigitalTwin core by including crack losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                location,
                current_degradation, 
                voltage_degradation, 
                power_degradation,
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
                location,
            )
            self.current_degradation = current_degradation 
            self.voltage_degradation = voltage_degradation
            self.power_degradation = power_degradation

        def run(self):
            # Run PVLib ModelChain Simulation
            self.mc.run_model(self.weather)
            dc = self.mc.results.dc
            ac = self.mc.results.ac

            # Create factor arrays evolving smoothly from 1.0 to the final degradation factors
            n_steps = len(self.df)
            curr_factors = np.ones(n_steps)
            volt_factors = np.ones(n_steps)
            power_factors = np.ones(n_steps)

            # Smooth exponential degradation progression (normalized)
            for i, _ in enumerate(self.df.index):
                progress = i / (n_steps - 1)
                
                # Smooth exponential curve (starts slowly, then accelerates)
                smooth_factor = 1 - np.exp(-3 * progress)
                
                # Normalize to 0–1 range
                smooth_factor /= (1 - np.exp(-3))

                # Compute degradation factors at this timestep
                curr_factors[i] = 1 - (1 - self.current_degradation) * smooth_factor
                volt_factors[i] = 1 - (1 - self.voltage_degradation) * smooth_factor
                power_factors[i] = 1 - (1 - self.power_degradation) * smooth_factor

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
            output['pv1_i'] = dc['i_mp'] * curr_factors
            output['pv1_u'] = dc['v_mp'] * volt_factors
            
            # Compute Power Quantities
            output['mppt_power'] = dc['p_mp'] * power_factors / 1000 # W to kW
            output['active_power'] = ac * power_factors / 1000 # W to kW
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

            return output_full

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
            # Randomly select a degradation scenario for this timeseries
            current_degradation, voltage_degradation, power_degradation = random.choice(
                CRACKS_DEGRADATION_SCENARIOS
            )
            print(
                f"\t\tCurrent degradation: {current_degradation:.3f}\n"
                f"\t\tVoltage degradation: {voltage_degradation:.3f}\n"
                f"\t\tPower degradation: {power_degradation:.3f}\n"
            )
            twin = DigitalTwinCracksTS(
                plant, 
                inverter, 
                ts_df, 
                condition_nr,
                location,
                current_degradation, 
                voltage_degradation, 
                power_degradation
            )

            # Run simulation
            results = twin.run()

            # Generate Plots with crack degradation evolution
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
            output_file = f"{TS_SAMPLES}/{condition_nr}_{condition_name}/{condition_nr}_{condition_name}_{year}_{local}_ts_samples.csv"
            results_to_export.to_csv(output_file)
            print(f"Exported timeseries: {output_file}\n")