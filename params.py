#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Wed 02 Nov 22 14:20:16'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           params.py
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
from multiprocessing.sharedctypes import Value
import os
import sys

import datetime
import pytz


import numpy as np
import pandas as pd
import xarray as xr


# LOCAL MODULE IMPORTS
# Temporary patch to use code from ecflux until packaged + released...
if not any('ecflux' in x for x in sys.path):
    sys.path.append('/Users/brynmorgan/dev/ecflux/ecflux/')

import fluxtower


if not any('uav-et' in x for x in sys.path):
    sys.path.append('/Users/brynmorgan/dev/uav-et/code')


if not any('aeropy' in x for x in sys.path):
    sys.path.append('/Users/brynmorgan/dev/aeropy/')


import uavet
from uavet import utils, Flight
from uavet.config import ProductionConfig

from timer import Timer


# import model
from model import AirLayer, Surface, Radiation
from model import radiation, model

import pyTSEB.TSEB as TSEB





default_params = {

}






mods_dict = {
    'TSEB-PT' : {
        'outputs' : [
            'flag', 'T_soil', 'T_veg', 'T_aero', 'LW_net_soil', 'LW_net_veg', 
            'LE_veg', 'H_veg', 'LE_soil', 'H_soil', 'G', 
            'r_soil', 'r_ex', 'r_a', 'u_star', 'L', 'n_iter'
        ],
        'model' : TSEB.TSEB_PT,
    },
    'TSEB-2T' : {
        'outputs' : [
            'flag', 'T_aero', 
            'LE_veg', 'H_veg', 'LE_soil', 'H_soil', 'G', 
            'r_soil', 'r_ex', 'r_a', 'u_star', 'L', 'n_iter'        
        ],
        'model' : TSEB.TSEB_2T,
    },
    'DTD' : {
        'outputs' : [
            'flag', 'T_soil', 'T_veg', 'T_aero', 'LW_net_soil', 'LW_net_veg', 
            'LE_veg', 'H_veg', 'LE_soil', 'H_soil', 'G', 
            'r_soil', 'r_ex', 'r_a', 'u_star', 'L', 'Ri', 'n_iter'
        ],
        'model' : TSEB.DTD,
    },
    'OSEB' : {
        'outputs' : ['flag', 'LW_net', 'LE', 'H', 'G', 'r_a', 'u_star', 'L', 'n_iter'],
        'model' : TSEB.OSEB,
    }
}


class Params:

    req_params = ['u', 'ea', 'p', 'L_dn', 'z_0M', 'd_0', 'z_u', 'z_T']

    def __init__(self, params : dict = None, flight : Flight = None):

        if params:
            self.check_params(params)
            self.params = params
        else:
            self.params = self.create_params(flight)

    def check_params(self, params):

        missing = [param for param in self.req_params if not params.has_key(param)]

        if len(missing) > 0:
            raise ValueError("Passed params missing required parameters: {missing}".format{missing=missing})

    def create_params(self, flight):
        raise NotImplementedError


class ParamsPT(Params):

    req_params = [
        'Tr_K', 'vza', 'T_A_K', 'u', 'ea', 'p', 'Sn_C', 'Sn_S', 'L_dn', 'LAI', 
        'h_C', 'emis_C', 'emis_S', 'z_0M', 'd_0', 'z_u', 'z_T'
    ]

    def __init__(self, params : dict = None, flight : Flight = None):
        super().__init__(params, flight)

    def create_params(self, flight : Flight):

        air = AirLayer(**flight.create_air_params())
        



class ParamsTSM(Params):

    req_params = [
        'T_C', 'T_S', 'T_A_K', 'u', 'ea', 'p', 'Sn_C', 'Sn_S', 'L_dn', 'LAI', 
        'h_C', 'emis_C', 'emis_S', 'z_0M', 'd_0', 'z_u', 'z_T'
    ]

    def __init__(self, params : dict = None, flight : Flight = None):
        super().__init__(params, flight)


class ParamsDTD(Params):

    req_params = [
        'Tr_K_0', 'Tr_K_1', 'T_A_K_0', 'T_A_K_1', 'u', 'ea', 'p', 'Sn_C', 'Sn_S', 'L_dn', 'LAI', 
        'h_C', 'emis_C', 'emis_S', 'z_0M', 'd_0', 'z_u', 'z_T'
    ]

    def __init__(self, params : dict = None, flight : Flight = None):
        super().__init__(params, flight)

class ParamsOSEB(Params):

    req_params = [
        'Tr_K', 'T_A_K', 'u', 'ea', 'p', 'Sn', 'L_dn', 
        'emis', 'z_0M', 'd_0', 'z_u', 'z_T'
    ]

    def __init__(self, params : dict = None, flight : Flight = None):
        super().__init__(params, flight)