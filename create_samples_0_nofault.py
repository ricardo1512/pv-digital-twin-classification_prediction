from classes import *
from plot import *
from utils import *

# ==============================================================
# Digital Twin Simulation - No Fault Condition
# ==============================================================
def create_samples_0_nofault(files_year, plot_samples=False):
    """
        Runs the digital twin simulation for a photovoltaic system under normal (no-fault) conditions.

        This function loads meteorological data for a specified year and simulates the PV system performance
        at 5-minute intervals using pvlib’s ModelChain. It computes electrical and environmental parameters
        throughout the day. Daily-level features are extracted for machine learning analysis, and plots can
        optionally be produced.

        Args:
            files_year (tuple): Tuple containing:
                - input_file (str): Path to CSV file with meteorological data for a given year
                    (e.g., "Weather/Vila_do_Conde_weather_2023.csv").
                - output_folder (str): Directory to save output CSV and plots (e.g., "Samples_2023").
            plot_samples (bool): If True, generates daily plots of simulation results. Default is False.

        Workflow:
            1. Load and preprocess meteorological data.
            2. Define PVPlant, InverterTwin, and DigitalTwin classes.
            3. Simulate PV system for each day, including soiling losses.
            4. Aggregate daily statistics and compute ML features.
            5. Optionally generate plots for each day.
            6. Export results to CSV.

        Output:
            Saves a CSV file named "0_digital_twin_output_no_fault_samples.csv" containing
            daily aggregated simulation results in the specified output folder.
    """

    # ==============================================================
    # Initialization
    # ==============================================================
    input_file, output_folder = files_year

    fault_nr = 0
    fault_name = LABELS_MAP[fault_nr][0].lower().replace(' ', '_')
    output_file = f"{output_folder}/{fault_nr}_digital_twin_output_{fault_name}_samples.csv"
    plot_folder = f"{PLOT_FOLDER}/Plots_{fault_nr}_{fault_name}_samples"

    # ==============================================================
    # Load Input Weather Data
    # ==============================================================
    df_input = pd.read_csv(input_file)
    df_input['date'] = pd.to_datetime(df_input['date'])

    # ==============================================================
    # Main Simulation Loop (Per Day)
    # ==============================================================
    daily_features = []
    daily_groups = df_input.groupby(df_input['date'].dt.date)

    print(f"{fault_name.upper()}: Starting simulation...\n")
    for date, group in daily_groups:
        print(f"{fault_name.replace('_', ' ').title():<8} | Running simulation for {date}...\n")

        # Prepare daily data
        group = group.copy().reset_index()
        group = group.rename(columns={"collectTime": "date"})

        # Instantiate PV system components
        plant = PVPlant(module_data, inverter_data)
        inverter = InverterTwin(inverter_data)
        twin = DigitalTwin(plant, inverter, group, fault_nr)

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
        if plot_samples:
            output_image = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_{fault_name}_samples"
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