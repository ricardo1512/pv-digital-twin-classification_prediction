from create_ts_samples_1_soiling import *
from create_ts_samples_2_shading import *
from create_ts_samples_3_cracks import *


def create_ts_samples(plot_samples=True):
    """
    Creates time-series samples for different anomaly conditions.

    Args:
        plot_samples (bool, optional): _description_. Defaults to True.
    """
    
    create_ts_samples_1_soiling(plot_samples)
    create_ts_samples_2_shading(plot_samples)
    create_ts_samples_3_cracks(plot_samples)