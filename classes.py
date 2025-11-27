import pandas as pd
import numpy as np
import pvlib
from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.irradiance import erbs
from globals import *

# ==============================================================
# Module and Inverter Data
# ==============================================================
module_data = retrieve_sam('cecmod')['Jinko_Solar_Co___Ltd_JKM410M_72HL_V']
inverter_data = retrieve_sam('cecinverter')['Huawei_Technologies_Co___Ltd___SUN2000_10KTL_USL0__240V_']
temperature_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
LOCATION = Location(latitude=LATITUDE, longitude=LONGITUDE, tz="Europe/Lisbon")

# ==============================================================
# PV Plant Class Definition
# ==============================================================
class PVPlant:
    """
    Represents the PV system layout and module/inverter characteristics.
    """
    def __init__(
            self,
            module_data,
            inverter_data,
            strings_per_inverter=1,
            modules_per_string=MODULES_PER_STRING
    ):
        self.module_data = module_data
        self.inverter_data = inverter_data
        self.strings_per_inverter = strings_per_inverter
        self.modules_per_string = modules_per_string
        self.surface_tilt = 30
        self.surface_azimuth = 180
        self.albedo = 0.2

        # Create PVSystem object for pvlib simulations
        self.system = PVSystem(
            module_parameters=self.module_data,
            temperature_model_parameters=temperature_model,
            modules_per_string=self.modules_per_string,
            strings_per_inverter=self.strings_per_inverter,
            surface_tilt=self.surface_tilt,
            surface_azimuth=self.surface_azimuth,
            albedo=self.albedo,
            inverter_parameters=self.inverter_data,
        )

# ==============================================================
# Inverter Digital Twin Class
# ==============================================================
class InverterTwin:
    """
    Represents the inverter model for converting DC to AC power.
    """
    def __init__(self, inverter_data):
        self.inverter_data = inverter_data

    def compute_ac(self, dc_power, dc_voltage):
        # Calculate AC power from DC using pvlib’s Sandia inverter model
        pac = pvlib.inverter.sandia(
            v_dc=dc_voltage,
            p_dc=dc_power,
            inverter=self.inverter_data,
        )
        return pac


# ==============================================================
# Digital Twin Core Simulation Class
# ==============================================================
class DigitalTwin:
    """
    Represents a PV system digital twin that simulates DC-to-AC conversion,
    electrical outputs, efficiency, phase currents/voltages, and inverter temperature
    using meteorological data.
    """

    def __init__(
            self,
            pvplant: PVPlant,
            inverter: InverterTwin,
            df: pd.DataFrame,
            condition_nr: int,
            location: Location = LOCATION,
    ):
        self.pvplant = pvplant
        self.inverter = inverter
        self.df = df
        self.condition_nr = condition_nr
        self.location = location

        # Initialize ModelChain for pvlib simulation
        self.mc = ModelChain(
            pvplant.system,
            self.location,
            ac_model='sandia',
            aoi_model='no_loss',
        )

        # Prepare timestamps and rename columns for pvlib
        self.df['collectTime'] = pd.to_datetime(self.df['date'])
        self.df = self.df.set_index('collectTime')
        self.df = self.df.rename(columns={
            GLOBAL_TILTED_IRRADIANCE: 'gti',
            DIFFUSE_RADIATION: 'dhi',
            TEMPERATURE_2M: 'temp_air',
            WIND_SPEED_10M: 'wind_speed',
        })

        # Solar Position and Irradiance Estimation
        solpos = LOCATION.get_solarposition(self.df.index)
        zenith = solpos['zenith']

        # Estimate GHI from GTI
        tilt_rad = np.radians(self.pvplant.surface_tilt)
        ghi_est = self.df['gti'] * np.cos(tilt_rad)
        ghi_est = ghi_est.clip(lower=0)

        # Estimate DNI and DHI using ERBS model
        erbs_res = erbs(
            ghi=ghi_est,
            zenith=zenith,
            datetime_or_doy=df.index
        )
        self.df['dni'] = erbs_res['dni']
        self.df['dhi'] = self.df['dhi']
        self.df['ghi'] = ghi_est

        # Create weather DataFrame for ModelChain
        self.weather = pd.DataFrame({
            'ghi': self.df['ghi'],
            'dhi': self.df['dhi'],
            'dni': self.df['dni'],
            'temp_air': self.df['temp_air'],
            'wind_speed': self.df['wind_speed'],
        }, index=self.df.index)

    def run(self):
        """
        Simulate the PV system operation for the given meteorological data.
        """
        
        # Run PVLib ModelChain Simulation
        self.mc.run_model(self.weather)
        dc = self.mc.results.dc
        ac = self.mc.results.ac

        # Initialize output DataFrame for storing simulation results
        output = pd.DataFrame(index=self.df.index)

        # Set inverter_state to condition_nr to indicate "normal" (0), "anomaly" (1, 2, 3) or "fault" (4, 5, 6) condition
        output['inverter_state'] = self.condition_nr

        # Compute DC current and voltage
        output['pv1_i'] = dc['i_mp']
        output['pv1_u'] = dc['v_mp']

        # Compute Power Quantities
        output['mppt_power'] = dc['p_mp'] / 1000  # W to kW
        output['active_power'] = ac / 1000  # W to kW
        output['efficiency'] = (ac / dc['p_mp'].replace(0, np.nan)) * 100  # Fraction to percentage

        # Compute AC phase currents (balanced assumption)
        output['a_i'] = output['active_power'] * 1000 / (400 * np.sqrt(3))
        output = output.assign(b_i=output['a_i'], c_i=output['a_i'])

        # Compute AC Voltages per phase and line (balanced assumption)
        output.loc[:, ['a_u', 'b_u', 'c_u']] = 230
        output.loc[:, ['ab_u', 'bc_u', 'ca_u']] = 400

        # Estimate inverter temperature as ambient + 35°C
        output['inv_temperature'] = self.df['temp_air'] + 35

        return output