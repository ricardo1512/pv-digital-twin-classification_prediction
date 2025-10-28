import random
from classes import *
from plot_day_samples import *
from utils import *

# =============================================================
# Digital Twin Simulation - Soiling Fault Condition
# =============================================================
def create_samples_1_soiling(files_year, plot_samples=False):
    """
        Runs the digital twin simulation for a photovoltaic system under soiling conditions.

        This function loads meteorological data for a specified year and simulates the PV system
        performance at 5-minute intervals while modeling daily power losses caused by soiling
        accumulation and partial cleaning from rainfall events. It computes electrical and environmental
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
            3. Randomly select a soiling scenario for each day.
            4. Simulate PV system for each day, including soiling losses.
            5. Aggregate daily statistics and compute ML features.
            6. Optionally generate plots for each day.
            7. Export results to CSV.

        Output:
            CSV file named "1_digital_twin_output_soiling_samples.csv" with daily aggregated features containing
            daily aggregated simulation results in the specified output folder.
    """

    # ==============================================================
    # Initialization
    # ==============================================================
    input_file, output_folder = files_year

    condition_nr = 1
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    output_file = f"{output_folder}/{condition_nr}_digital_twin_output_{condition_name}_samples.csv"
    plot_folder = f"{PLOT_FOLDER}/Samples/Plots_{condition_nr}_{condition_name}_samples"

    # ==============================================================
    # Parameters for soiling accumulation and cleaning
    # ==============================================================
    soiling_min, soiling_max = 0.05, 0.3  # Daily soiling range, for random
    rain_threshold_mm = 1  # Minimum rainfall to partially clean PV modules
    cleaning_efficiency = 0.4  # Fraction of soiling removed per rain event

    # ==============================================================
    # Subclass Digital Twin with Soiling
    # ==============================================================
    class DigitalTwinWithSoiling(DigitalTwin):
        """
        Extends DigitalTwin core by including soiling losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                daily_soiling_loss,
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
            )
            self.daily_soiling_loss = daily_soiling_loss


        def run(self):
            # Run PVLib ModelChain Simulation
            self.mc.run_model(self.weather)
            dc = self.mc.results.dc
            ac = self.mc.results.ac

            # Compute soiling accumulation per timestep
            dt_hours = (self.df.index[1] - self.df.index[0]).total_seconds() / 3600
            daily_step = dt_hours / 24  # Fraction of a day
            days_passed = np.zeros(len(self.df))
            effective_days = 0.0

            for i in range(1, len(self.df)):
                if self.df['precipitation'].iloc[i] > rain_threshold_mm:
                    # Rain event partially cleans modules
                    effective_days *= (1 - cleaning_efficiency)
                else:
                    # Accumulate soiling based on time step
                    effective_days += daily_step

                days_passed[i] = effective_days

            # Apply exponential soiling loss
            soiling_factor = (1 - self.daily_soiling_loss) ** days_passed

            # Initialize output DataFrame for storing simulation results
            output = pd.DataFrame(index=self.df.index)

            # Set inverter_state to indicate anomaly condition
            output['inverter_state'] = self.condition_nr

            # Compute DC current and voltage
            output['pv1_i'] = dc['i_mp'] * soiling_factor
            output['pv1_u'] = dc['v_mp']

            # Compute Power Quantities
            output['mppt_power'] = (dc['p_mp'] * soiling_factor) / 1000 # W to kW
            output['active_power'] = (ac * soiling_factor) / 1000 # W to kW
            output['efficiency'] = (output['active_power'] / output['mppt_power'].replace(0, np.nan)) * 100 # Fraction to percentage

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
        daily_soling_loss = random.uniform(soiling_min, soiling_max)
        print(f"\tDaily soiling loss: {daily_soling_loss:.3f}\n")
        twin = DigitalTwinWithSoiling(
            plant,
            inverter,
            group,
            condition_nr,
            daily_soling_loss,
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