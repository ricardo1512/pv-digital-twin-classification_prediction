import os
import random
from classes import *
from plot import *
from utils import *

# ==============================================================
# Digital Twin Simulation - Ground Fault Condition
# ==============================================================
def create_samples_4_ground(files_year, plot_samples=False):
    """
        Runs the digital twin simulation for a photovoltaic system under ground fault conditions.

        This function loads meteorological data for a specified year and simulates the PV system performance
        at 5-minute intervals while introducing realistic ground fault patterns by adjusting DC current
        and voltage levels according to predefined factors. It computes electrical and environmental
        parameters throughout the day. Daily-level features are extracted for machine learning analysis,
        and plots can optionally be produced.

        Args:
            files_year (tuple): Tuple containing:
                - input_file (str): Path to CSV file with meteorological data for a given year
                  (e.g., "Weather/Vila_do_Conde_weather_2023.csv").
                - output_folder (str): Directory to save output CSV and plots (e.g., "Samples_2023").
            plot_samples (bool): If True, generates daily plots of simulation results. Default is False.

        Workflow:
            1. Load and preprocess meteorological data.
            2. Define PVPlant, InverterTwin, and DigitalTwin classes.
            3. Randomly assign ground fault degradation factors for voltage and current.
            4. Simulate PV system performance with degraded conditions.
            5. Aggregate and export daily-level features for machine learning.
            6. Optionally generate diagnostic plots for visualization.
            7. Export results to CSV.

        Output:
            Saves a CSV file named "4_digital_twin_output_ground_samples.csv" containing
            daily aggregated simulation results in the specified output folder.
    """

    # ==============================================================
    # Initialization
    # ==============================================================
    input_file, output_folder = files_year

    condition_nr = 4
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    output_file = f"{output_folder}/{condition_nr}_digital_twin_output_{condition_name}_samples.csv"
    plot_folder = f"{PLOT_FOLDER}/Plots_{condition_nr}_{condition_name}_samples"

    # ==============================================================
    # Define realistic ground fault scenarios
    # Each tuple: (current, voltage)
    # ==============================================================
    curr_volt_degradation = [
        (1.02, 0.97),  # 1. Mild fault: slight current increase (MPPT effect), small voltage drop
        (0.98, 0.95),  # 2. Mild fault: small decreases in current and voltage
        (0.90, 0.93),  # 3. Moderate fault: noticeable reductions
        (0.85, 0.90),  # 4. Moderate fault: further degradation
        (0.78, 0.88),  # 5. Moderate-to-severe fault: significant current drop
        (0.65, 0.85),  # 6. Severe fault: substantial drops, safety concerns
        (0.55, 0.83),  # 7. Severe fault: heavy degradation, near trip levels
        (0.45, 0.82),  # 8. Critical fault: system compromised
        (0.35, 0.80),  # 9. Critical fault: very low current and voltage
        (0.30, 0.80),  # 10. Worst-case fault: near shutdown, urgent maintenance
    ]

    # =============================================================
    # Subclass Digital Twin with Ground Fault
    # =============================================================
    class DigitalTwinWithGround(DigitalTwin):
        """
            Extends DigitalTwin by including ground fault losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                volt_degradation,
                curr_degradation,
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
            )
            self.current_degradation = curr_degradation
            self.voltage_degradation = volt_degradation

        def run(self):
            # Run PVLib ModelChain Simulation
            self.mc.run_model(self.weather)
            dc = self.mc.results.dc

            # Apply degradation factors for ground fault condition
            dc['v_mp'] *= self.voltage_degradation
            dc['i_mp'] *= self.current_degradation
            dc['p_mp'] = dc['v_mp'] * dc['i_mp']

            # Recalculate AC power with degraded DC inputs
            ac = self.inverter.compute_ac(dc['p_mp'], dc['v_mp'])
            # Limit maximum AC power to simulate protection trip (IEC 60364-4-42 safe limit)
            ac = np.minimum(ac, 70)

            # Initialize output DataFrame for storing simulation results
            output = pd.DataFrame(index=self.df.index)

            # Set inverter_state to indicate fault condition
            output['inverter_state'] = self.condition_nr

            # Compute DC current and voltage
            output['pv1_i'] = dc['i_mp']
            output['pv1_u'] = dc['v_mp']

            # Compute Power Quantities
            output['mppt_power'] = dc['p_mp'] / 1000 # W to kW
            output['active_power'] = ac / 1000 # W to kW
            output['efficiency'] = (ac / dc['p_mp'].replace(0, np.nan)) * 100

            # Compute AC phase currents (balanced assumption)
            output['a_i'] = output['active_power'] * 1000 / (400 * np.sqrt(3))
            output = output.assign(b_i=output['a_i'], c_i=output['a_i'])

            # Compute AC Voltages per phase and line
            output.loc[:, ['a_u', 'b_u', 'c_u']] = 230
            output.loc[:, ['ab_u', 'bc_u', 'ca_u']] = 400

            # Estimate inverter temperature as ambient + 35°C
            output['inv_temperature'] = self.df['temp_air'] + 35

            return output

    # =============================================================
    # Load input meteorological CSV
    # =============================================================
    df_input = pd.read_csv(input_file)
    df_input['date'] = pd.to_datetime(df_input['date'])

    # ==============================================================
    # Main Simulation Loop (Per Day)
    # ==============================================================
    daily_features = []
    daily_groups = df_input.groupby(df_input['date'].dt.date)

    print(f"{condition_name.upper()}: Starting simulation...\n")
    rand_plots = random.sample(range(len(daily_groups)), 5)
    for i, (date, group) in enumerate(daily_groups):
        print(f"{condition_name.title():<8} | Running simulation for {date}...")

        # Prepare daily data
        group = group.copy().reset_index()
        group = group.rename(columns={"collectTime": "date"})

        # Instantiate PV system components
        plant = PVPlant(module_data, inverter_data)
        inverter = InverterTwin(inverter_data)
        voltage_degradation, current_degradation = random.choice(curr_volt_degradation)
        print(
            f"\tCurrent degradation: {voltage_degradation:.3f}\n"
            f"\tVoltage degradation: {current_degradation:.3f}\n"
        )
        twin = DigitalTwinWithGround(
            plant,
            inverter,
            group,
            condition_nr,
            voltage_degradation,
            current_degradation,
        )

        # Run daily simulation
        results = twin.run()

        # Reindex and merge with meteorological data
        group['collectTime'] = pd.to_datetime(group['date'])
        group = group.set_index('collectTime')
        results_full = results[EXPORT_COLUMNS].merge(
            group[METEOROLOGICAL_COLUMNS],
            left_index=True,
            right_index=True,
        )

        # Clean and Process Daily Data
        results_full = results_full.clip(lower=0).fillna(0)

        # Generate Daily Plots
        if i in rand_plots and plot_samples:
            output_image = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_{condition_name}_samples"
            plot_mppt(results_full, date, plot_folder, output_image)
            plot_currents(results_full, date, plot_folder, output_image)
            plot_voltages(results_full, date, plot_folder, output_image)

        # Compute and store in daily_features daily statistical features
        compute_store_daily_comprehensive_features(results_full, date, daily_features)

    # ==============================================================
    # Export Final Aggregated Results
    # ==============================================================
    df_ml_features = pd.DataFrame(daily_features)
    df_ml_features.index.name = "date"
    df_ml_features.to_csv(output_file, index=True)
    print(f"\nExported daily results to {output_file}\n")