import os
import random
from classes import *
from plot_day_samples import *
from utils import *

# ==============================================================
# Digital Twin Simulation - Arc Fault Condition
# ==============================================================
def create_samples_5_arc(files_year, plot_samples=False):
    """
        Runs the digital twin simulation for a photovoltaic system under sustained series arc fault (SAF) conditions.

        This function loads meteorological data for a specified year and simulates the PV system performance at 5-minute
        intervals while introducing realistic arc fault patterns with voltage rises and current drops at a random hour,
        including a shutdown after 1 hour to mimic protection mechanisms. It computes electrical and environmental
        parameters throughout the day. Daily-level features are extracted for machine learning analysis, and plots
        can optionally be produced.

        Args:
            files_year (tuple): Tuple containing:
                - input_file (str): Path to CSV file with meteorological data for a given year
                  (e.g., "Weather/Vila_do_Conde_weather_2023.csv").
                - output_folder (str): Directory to save output CSV and plots (e.g., "Samples_2023").
            plot_samples (bool): If True, generates daily plots of simulation results. Default is False.

        Workflow:
            1. Load and preprocess meteorological data.
            2. Define PVPlant, InverterTwin, and DigitalTwin classes.
            3. Randomly select arc start time, voltage rise, and current drop.
            4. Simulate PV system performance with arc fault conditions, including shutdown after 1 hour.
            5. Aggregate and export daily-level features for machine learning.
            6. Optionally generate diagnostic plots for visualization.
            7. Export results to CSV.

        Output:
            Saves a CSV file named "5_arc_samples.csv" containing
            daily aggregated simulation results in the specified output folder.
    """

    # ==============================================================
    # Initialization
    # ==============================================================
    input_file, output_folder, train_test = files_year

    condition_nr = 5
    condition_name = LABELS_MAP[condition_nr][0].lower().replace(' ', '_')
    output_file = f"{output_folder}/{condition_nr}_{train_test}_{condition_name}_samples.csv"
    plot_folder = f"{PLOT_FOLDER}/Day_samples/Plots_{condition_nr}_{condition_name}_samples"

    # ==============================================================
    # Define voltage and current variation ranges for arc fault
    # ==============================================================
    voltage_min, voltage_max = 1, 5 # for random
    current_min, current_max = 0.3, 0.5 # for random

    # =============================================================
    # Subclass Digital Twin with Arc Fault
    # =============================================================
    class DigitalTwinWithArc(DigitalTwin):
        """
            Extends DigitalTwin by including arc fault losses in the simulation.
        """

        def __init__(
                self,
                pvplant,
                inverter,
                df,
                condition_nr,
                arc_start,
                voltage_rise,
                current_drop,
        ):
            super().__init__(
                pvplant,
                inverter,
                df,
                condition_nr,
            )
            self.arc_start = arc_start
            self.voltage_rise = voltage_rise
            self.current_drop = current_drop

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

            # Simulate Series Arc Fault
            arc_shutdown_time = self.arc_start + pd.Timedelta(hours=1)
            arc_mask = (dc.index >= arc_start) & (dc.index < arc_shutdown_time)
            shutdown_mask = dc.index >= arc_shutdown_time

            # Arc degradation during arc duration
            vmp_arc = dc.loc[arc_mask, 'v_mp']
            dc.loc[arc_mask, 'v_mp'] = np.where(vmp_arc > 0, vmp_arc + voltage_rise, vmp_arc)  # Arc voltage contribution (arc gap)
            dc.loc[arc_mask, 'i_mp'] *= current_drop  # Sustained current drop due to arc impedance
            dc.loc[arc_mask, 'p_mp'] = dc.loc[arc_mask, 'v_mp'] * dc.loc[arc_mask, 'i_mp']

            # Shutdown after 1 hour
            dc.loc[shutdown_mask, ['v_mp', 'i_mp', 'p_mp']] = 0

            # Recalculate AC power with degraded DC inputs
            ac = self.inverter.compute_ac(dc['p_mp'], dc['v_mp'])

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
        random_hour = random.randint(7, 17)
        arc_start = pd.Timestamp(f"{date} {random_hour}:00:00", tz="Europe/Lisbon")
        voltage_rise = random.uniform(voltage_min, voltage_max)
        current_drop = random.uniform(current_min, current_max)
        print(
            f"\tArc start: {arc_start.strftime('%H:%M')}\n"
            f"\tVoltage rise: {voltage_rise:.3f}\n"
            f"\tCurrent drop: {current_drop:.3f}\n"
        )
        plant = PVPlant(module_data, inverter_data)
        inverter = InverterTwin(inverter_data)
        twin = DigitalTwinWithArc(
            plant,
            inverter,
            group,
            condition_nr,
            arc_start,
            voltage_rise,
            current_drop,
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
        if i == 0 or i in rand_plots and plot_samples:
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