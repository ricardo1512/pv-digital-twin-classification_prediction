from concurrent.futures import ProcessPoolExecutor, as_completed
from create_day_samples_0_normal import *
from create_day_samples_1_soiling import *
from create_day_samples_2_shading import *
from create_day_samples_3_cracks import *
from create_day_samples_4_ground import *
from create_day_samples_5_arc import *
from create_day_samples_6_diode import *

# ==============================================================
# Dataset Generation
# ==============================================================
def create_day_samples(plot_samples=False):
    """
    Generate synthetic PV fault samples for multiple scenarios in parallel.

    This function orchestrates the creation of PV system datasets under
    different anomaly/fault and normal operating conditions. It uses parallel processing to
    accelerate sample generation across multiple anomaly/fault simulation modules, each representing
    a specific anomaly/fault type (e.g., soiling, shading, cracks, ground faults, arc faults, diode failures).

    Args:
        plot_samples (bool): If True, each simulation module will generate plots for the created samples.
            Default value: False.

    Workflow:
        1. For each dataset year (training: 2023, testing: 2024):
            - Launch all sample creation modules in parallel using a ProcessPoolExecutor.
            - Each module generates its respective fault scenario samples.
        2. Handle exceptions gracefully and log progress for each module.
    """

    print("\n" + "=" * 40)
    print("CREATING SAMPLES...")
    print("=" * 40)

    # Create samples for both training (2023) and testing (2024) datasets
    for file_year in [FILE_YEAR_TRAIN, FILE_YEAR_TEST]:

    # List of sample creation functions (each simulates a specific fault scenario)
        funcs = [
            create_samples_0_normal,
            create_samples_1_soiling,
            create_samples_2_shading,
            create_samples_3_cracks,
            create_samples_4_ground,
            create_samples_5_arc,
            create_samples_6_diode,
        ]

        # Launch all sample creation functions in parallel
        with ProcessPoolExecutor(max_workers=7) as executor:
            # Submit each function with its arguments
            futures = {
                executor.submit(func, file_year, plot_samples=plot_samples): func.__name__
                for func in funcs
            }

            # Monitor progress and handle exceptions for each function
            for future in as_completed(futures):
                func_name = futures[future]
                try:
                    future.result()
                    print(f"Finished function {func_name} for {file_year[1]}\n")
                except Exception as e:
                    print(f"Error in {func_name}: {e}")