#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Thu 03 Nov 22 13:54:46'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           flyTSEB.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Bryn Morgan
ORGANIZATION:   University of California, Santa Barbara
Contact:        bryn.morgan@geog.ucsb.edu
Copyright:      (c) Bryn Morgan 2022


"""


#-------------------------------------------------------------------------------
# IMPORTS
#-------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import xarray as xr




if not any('uav-et' in x for x in sys.path):
    sys.path.append('/Users/brynmorgan/dev/uav-et/code')


if not any('aeropy' in x for x in sys.path):
    sys.path.append('/Users/brynmorgan/dev/aeropy/')


# import uavet
from uavet import Flight
# import model
from model import AirLayer, radiation


# import pyTSEB.TSEB as TSEB
from pyTSEB import PyTSEB

import utils
# import pyTSEB


class FlyTSEB(PyTSEB):
    
    def __init__(self, flight : Flight, init_params : dict, res_params : dict):
        super().__init__(parameters = init_params)
        # self.p = parameters

        # Model description parameters
        # self.model_type = self.p['model']
        # self.resistance_form = self.p['resistance_form']
        # self.res_params = {}
        # self.G_form = self.p['G_form']
        # self.water_stress = self.p['water_stress']
        # self.calc_daily_ET = False
        self._set_res_params(res_params)

        # self._params = params
        self.flight = flight


        # IN PROCESS_LOCAL_IMAGE
        # in_data = dict()

    def _set_res_params(self, res_params):
        # Set the Kustas and Norman resistance parameters
        if self.resistance_form == 0:
            self.res_params = res_params
    
    def set_solar_angles(self):
        # Solar zenith angle
        theta_sun = utils.calc_zenith(
            self.flight.coords[0], self.flight.coords[1], 
            self.flight.start_time, self.flight.end_time
        )
        # Solar azimuth angle
        psi_sun = utils.calc_azimuth(
            self.flight.coords[0], self.flight.coords[1], 
            self.flight.start_time, self.flight.end_time
        )
        self.p['SZA'] = theta_sun
        self.p['SAA'] = psi_sun

    def set_img_params(self):

        image_params = {
            'T_R1' : self.flight.ortho.get_temperature(),
            'VZA' : 0.0
        }
        return image_params

    def get_met_params(self, SW_in_hat):

        air = AirLayer(**self.flight.create_air_params())

        met_params = {
            'T_A1' : air.T_a,
            'u' : air.u,
            'ea' : air.e_a * 10,
            'p' : air.p_a * 10,
            'S_dn' : SW_in_hat,
            'L_dn' : radiation.calc_LW(air.T_a, air.calc_emissivity()),
            'z_u' : air.z,
            'z_T' : air.z,
        }

        return met_params






    def _get_input_structure(self):
        return super()._get_input_structure()

    def run(self, in_data, mask=None):
        super().run(in_data=in_data, mask=mask)






class FlyTSEBPT(FlyTSEB):

    def __init__(self, params):
        super().__init__(params)

    


        input_fields = OrderedDict([
                            # General parameters
                            ("T_R1", "Land Surface Temperature"),
                            ("LAI", "Leaf Area Index"),
                            ("VZA", "View Zenith Angle for LST"),
                            ("landcover", "Landcover"),
                            ("input_mask", "Input Mask"),
                            # Vegetation parameters
                            ("f_c", "Fractional Cover"),
                            ("h_C", "Canopy Height"),
                            ("w_C", "Canopy Width Ratio"),
                            ("f_g", "Green Vegetation Fraction"),
                            ("leaf_width", "Leaf Width"),
                            ("x_LAD", "Leaf Angle Distribution"),
                            ("alpha_PT", "Initial Priestley-Taylor Alpha Value"),
                            # Spectral Properties
                            ("rho_vis_C", "Leaf PAR Reflectance"),
                            ("tau_vis_C", "Leaf PAR Transmitance"),
                            ("rho_nir_C", "Leaf NIR Reflectance"),
                            ("tau_nir_C", "Leaf NIR Transmitance"),
                            ("rho_vis_S", "Soil PAR Reflectance"),
                            ("rho_nir_S", "Soil NIR Reflectance"),
                            ("emis_C", "Leaf Emissivity"),
                            ("emis_S", "Soil Emissivity"),
                            # Illumination conditions
                            ("lat", "Latitude"),
                            ("lon", "Longitude"),
                            ("stdlon", "Standard Longitude"),
                            ("time", "Observation Time for LST"),
                            ("DOY", "Observation Day Of Year for LST"),
                            ("SZA", "Sun Zenith Angle"),
                            ("SAA", "Sun Azimuth Angle"),
                            # Meteorological parameters
                            ("T_A1", "Air temperature"),
                            ("u", "Wind Speed"),
                            ("ea", "Vapour Pressure"),
                            ("alt", "Altitude"),
                            ("p", "Pressure"),
                            ("S_dn", "Shortwave Irradiance"),
                            ("z_T", "Air Temperature Height"),
                            ("z_u", "Wind Speed Height"),
                            ("z0_soil", "Soil Roughness"),
                            ("L_dn", "Longwave Irradiance"),
                            # Resistance parameters
                            ("KN_b", "Kustas and Norman Resistance Parameter b"),
                            ("KN_c", "Kustas and Norman Resistance Parameter c"),
                            ("KN_C_dash", "Kustas and Norman Resistance Parameter c-dash"),
                            # Soil heat flux parameter
                            ("G", "Soil Heat Flux Parameter"),
                            ('S_dn_24', 'Daily shortwave irradiance')])

