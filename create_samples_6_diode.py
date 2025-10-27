import os
import random
from classes import *
from plot import *
from utils import *

# =============================================================
# Digital Twin Simulation - Bypass Diode Fault Condition
# =============================================================
def create_samples_6_diode(files_year, plot_samples=False):
    """
        Runs the digital twin simulation for a photovoltaic system under bypass diode fault conditions.

        This function loads meteorological data for a specified year and simulates the PV system performance
        at 5-minute intervals while modeling the activation of bypass diodes, reducing DC voltage of affected
        modules according to the number of bypassed substrings while keeping current nearly constant.
        It computes electrical and environmental parameters throughout the day. Daily-level features are
        extracted for machine learning analysis, and plots can optionally be produced.

        Args:
            files_year (tuple): Tuple containing:
                - input_file (str): Path to CSV file with meteorological data for a given year
                    (e.g., "Weather/Vila_do_Conde_weather_2023.csv").
                - output_folder (str): Directory to save output CSV and plots (e.g., "Samples_2023").
            plot_samples (bool): If True, generates daily plots of simulation results. Default is False.

        Workflow:
            1. Load and preprocess meteorological data.
            2. Define PVPlant, InverterTwin, and DigitalTwin classes.
            3. Randomly assign diode voltage loss and number of bypassed modules.
            4. Simulate PV system performance with bypassed substrings.
            5. Aggregate daily statistics and compute ML features.
            6. Optionally generate plots for each day.
            7. Export results to CSV.

        Output:
            CSV file named "6_digital_twin_output_diode_samples.csv" containing
            daily aggregated simulation results in the specified output folder.
    """

    # ==============================================================
    # Initialization
    # ==============================================================
    input_file, output_folder = files_year

    condition_nr = 6
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    output_file = f"{output_folder}/{condition_nr}_digital_twin_output_{condition_name}_samples.csv"
    plot_folder = f"{PLOT_FOLDER}/Plots_{condition_nr}_{condition_name}_samples"

    # ==============================================================
    # Define voltage variation ranges for bypass diode fault
    # ==============================================================
    voltage_min, voltage_max = 0.05, 0.10 # for random
    bypassed_modules = 5 # for random
    substrings_per_module = 3

    # =============================================================
    # Subclass Digital Twin with Bypass Diode Fault
    # =============================================================
    class DigitalTwinWithDiode(DigitalTwin):
        """
            Extends DigitalTwin by including bypass fault losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                voltage_loss,
                nr_bypassed_modules,
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
            )
            self.voltage_loss = voltage_loss
            self.nr_bypassed_modules = nr_bypassed_modules

        def run(self):
            # Run PVLib ModelChain Simulation
            self.mc.run_model(self.weather)
            dc = self.mc.results.dc

            # Typical maximum power voltage of one module
            module_vmp = self.pvplant.module_data['V_mp_ref']

            # Voltage drop if one substring is bypassed (~1/substrings_per_module of module voltage)
            substring_drop = module_vmp / substrings_per_module

            # Total voltage drop due to bypassed substrings
            voltage_drop = self.nr_bypassed_modules * substring_drop

            # Apply reduction to the operating voltage
            pv1_u = dc['v_mp'] - voltage_drop
            dc['v_mp'] = pv1_u.clip(lower=0)  # Ensure non-negative voltage

            # Recalculate AC power with degraded DC inputs
            ac = self.inverter.compute_ac(dc['p_mp'], dc['v_mp'])

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
        diode_voltage_loss = random.uniform(voltage_min, voltage_max)
        num_bypassed_modules = min(np.random.geometric(p=0.5), bypassed_modules)
        print(
            f"\tDiode voltage loss: {diode_voltage_loss:.3f}\n"
            f"\tNumber of bypassed modules: {num_bypassed_modules}\n"
        )
        twin = DigitalTwinWithDiode(
            plant,
            inverter,
            group,
            condition_nr,
            diode_voltage_loss,
            num_bypassed_modules,
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