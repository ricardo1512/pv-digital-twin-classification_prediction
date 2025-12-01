import random
from classes import *
from plot_day_samples import *
from utils import *

# ==============================================================
# Digital Twin Simulation - Crack Fault Condition
# ==============================================================
def create_samples_3_cracks(files_year, plot_samples=False):
    """
    Runs the digital twin simulation for a photovoltaic system under crack/micro-crack degradation.

    This function loads meteorological data for a specified year and simulates the PV system performance
    at 5-minute intervals while applying randomized degradation factors representing cracks in PV modules.
    It computes electrical and environmental parameters throughout the day. Daily-level features are extracted
    for machine learning analysis, and plots can optionally be produced.

    Args:
        files_year (tuple): Tuple containing:
            - input_file (str): Path to CSV file with meteorological data for a given year
                (e.g., "Weather/Vila_do_Conde_weather_2023.csv").
            - output_folder (str): Directory to save output CSV and plots (e.g., "Samples_2023").
        plot_samples (bool): If True, generates daily plots of simulation results. Default is False.

    Workflow:
        1. Load and preprocess meteorological data.
        2. Define PVPlant, InverterTwin, and DigitalTwin classes.
        3. Randomly select a crack degradation scenario for each day.
        4. Simulate PV system for each day, applying degradation to DC and AC values.
        5. Aggregate daily statistics and compute ML features.
        6. Optionally generate plots for each day.
        7. Export results to CSV.

    Output:
        Saves a CSV file named "3_crack_samples.csv" containing
        daily aggregated simulation results in the specified output folder.
    """

    # ==============================================================
    # Initialization
    # ==============================================================
    input_file, output_folder, train_test = files_year

    condition_nr = 3
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    output_file = f"{output_folder}/{condition_nr}_{train_test}_{condition_name}_samples.csv"
    plot_folder = f"{PLOT_FOLDER}/Day_samples/Plots_{condition_nr}_{condition_name}_samples"

    # ==============================================================
    # Define realistic crack degradation scenarios
    # Each tuple: (current_degradation, voltage_degradation, power_degradation)
    # ==============================================================
    degradation_scenarios = [   # for random
        (0.980, 0.990, 0.970),  # light: -2% current, -1% voltage, -3% power
        (0.960, 0.980, 0.940),  # moderate: -4% current, -2% voltage, -6% power
        (0.949, 0.956, 0.850),  # reference case: -5.1% current, -4.4% voltage, -15% power
        (0.970, 0.960, 0.930),  # -3% current, -4% voltage, -7% power
        (0.930, 0.950, 0.800),  # -7% current, -5% voltage, -20% power
        (0.990, 0.970, 0.960),  # -1% current, -3% voltage, -4% power
        (0.910, 0.940, 0.750),  # -9% current, -6% voltage, -25% power
        (0.940, 0.930, 0.810),  # -6% current, -7% voltage, -19% power
        (0.925, 0.965, 0.780),  # -7.5% current, -3.5% voltage, -22% power
        (0.970, 0.985, 0.950),  # -3% current, -1.5% voltage, -5% power
    ]

    # =============================================================
    # Subclass Digital Twin with Cracks
    # =============================================================
    class DigitalTwinWithCracks(DigitalTwin):
        """
        Extends DigitalTwin core by including crack losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                current_degradation,
                voltage_degradation,
                power_degradation,
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
            )
            self.current_degradation = current_degradation
            self.voltage_degradation = voltage_degradation
            self.power_degradation = power_degradation

        def run(self):
            # Run PVLib ModelChain Simulation
            self.mc.run_model(self.weather)
            dc = self.mc.results.dc
            ac = self.mc.results.ac

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
            dc['i_mp'] *= self.current_degradation
            dc['v_mp'] *= self.voltage_degradation
            output['pv1_i'] = dc['i_mp']
            output['pv1_u'] = dc['v_mp']

            # Compute Power Quantities
            dc['p_mp'] *= self.power_degradation
            ac = self.inverter.compute_ac(dc['p_mp'], dc['v_mp'])
            output['mppt_power'] = dc['p_mp'] / 1000 # W to kW
            output['active_power'] = ac / 1000 # W to kW
            output['efficiency'] = (ac / dc['p_mp'].replace(0, np.nan)) * 100

            # Compute AC phase currents (balanced assumption)
            output['a_i'] = output['active_power'] * 1000 / (400 * np.sqrt(3))
            output = output.assign(b_i=output['a_i'], c_i=output['a_i'])

            # Compute AC Voltages per phase and line (balanced assumption)
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
        current_degradation, voltage_degradation, power_degradation = random.choice(degradation_scenarios)
        print(
            f"\tCurrent degradation: {current_degradation:.3f}\n"
            f"\tVoltage degradation: {voltage_degradation:.3f}\n"
            f"\tPower degradation: {power_degradation:.3f}\n"
        )
        twin = DigitalTwinWithCracks(
            plant,
            inverter,
            group,
            condition_nr,
            current_degradation,
            voltage_degradation,
            power_degradation,
        )

        # Run daily simulation
        results = twin.run()

        # Reindex and merge with meteorological data
        group['collectTime'] = pd.to_datetime(group['date'])
        group = group.set_index('collectTime')
        clean_features = ['pv1_i_clean', 'pv1_u_clean', 'mppt_power_clean', 'a_i_clean']
        results_full = results[EXPORT_COLUMNS + clean_features].merge(
            group[METEOROLOGICAL_COLUMNS],
            left_index=True,
            right_index=True,
        )

        # Clean and Process Daily Data
        results_full = results_full.clip(lower=0).fillna(0)

        # Generate Daily Plots
        if i in rand_plots and plot_samples:
            condition_title = LABELS_MAP[condition_nr][0]
            output_image = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_{condition_name}_samples"
            plot_mppt(results_full, date, condition_title, plot_folder, output_image)
            plot_currents(results_full, date, condition_title, plot_folder, output_image)
            plot_voltage(results_full, date, condition_title, plot_folder, output_image)

        # Prepare end dataframe
        selected_columns = [col for col in results_full.columns if col not in clean_features]
        results_end = results_full[selected_columns]
        
        # Compute and store in daily_features daily statistical features
        compute_store_daily_comprehensive_features(results_end, date, daily_features)

    # ==============================================================
    # Export Final Aggregated Results
    # ==============================================================
    df_ml_features = pd.DataFrame(daily_features)
    df_ml_features.index.name = "date"
    df_ml_features.to_csv(output_file, index=True)
    print(f"\nExported daily results to {output_file}\n")