#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Fri 28 Oct 22 13:14:13'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           sand_ramajal.py
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


#%% IMPORTS

import os
import sys

import datetime
import pytz

import numpy as np
import pandas as pd

from xarray import DataArray

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import rcParams
from mpl_toolkits import mplot3d

%matplotlib qt



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

# import utils

import pyTSEB.TSEB as TSEB
import pyTSEB
import tseb_models as tseb


#%% FLIGHT LOGS

# Get home directory + add Box filepath
# filepath = '/Volumes/UAV-ET/Dangermond/UAV/'
filepath = os.path.expanduser('~') + '/Box/Dangermond/UAV/'

config = ProductionConfig(init_bucket=False, root_dir=filepath)
# NOTE: This is temporary to reset the ortho dir to local until the file stuff is sorted.
config.config.set('ortho','dir','/Users/brynmorgan/Dangermond/UAV/Processing/MicaSense/Level02')
config.buckets['ortho'] = uavet.buckets.OrthoBucket(config.config['ortho'])

# config.config.set('mica','dir','/Volumes/UAV-ET/Dangermond/UAV/Level00/MicaSense/')
# config.buckets['mica'] = uavet.buckets.MicaSenseBucket(config.config['mica'])

# Get log dir
log_dir = config.config.get('log', 'dir')
# Get logs

logs = pd.DataFrame({
    'LOG_FILE': [os.path.basename(file) for file in uavet.bucket.Bucket.get_subfiles(log_dir)],
    'PATH': uavet.bucket.Bucket.get_subfiles(log_dir)
})

#%% RAMAJAL TOWER

# RAMAJAL
filepath = os.path.expanduser('~') + '/Box/Dangermond/RamajalTower/'
biomet_file = os.path.join(filepath, 'CR1000X', 'Data', 'CR1000X_SN8845_DIR_IRT_OUT.txt')
meta_file = os.path.join(filepath, 'RamajalTower_Metadata.csv')


ramajal = fluxtower.EddyProTower(filepath, meta_file, biomet_file)

# Site/tower params
h = 0.3
cover = 'homogeneous'
veg = 'GRA'

#%% FUNCTIONS


t = Timer()
t_flight = Timer()

flights = []

# ram21_i = [8, 29]
# ram18_i = [104, 105, 106, 107, 109]
# ram25_i = [110, 111, 112, 113, 114]
# ram28_i = [115, 116, 117, 118]
ramajal_i = [8, 29, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]

for n in ramajal_i:
# for n in [8,20]:
    t_flight.start()
    # Initialize flight
    flight = Flight(logs.PATH[n], config=config, get_images=False)
    if flight.ortho:
        flight.ortho.init()
    flights.append(flight)
    t_flight.stop()


#%%

gen_params = init_params | misc | spectral | structural

init_params = {
    'model' : 'TSEB_PT',
    'resistance_form' : 0,
    'G_form' : [[1], 0.35],
    'water_stress' : False
}
res_params = {
    'KN_b' : 0.012,
    'KN_c' : 0.0038,
    'KN_C_dash' : 90,
}
misc = {
    'VZA' : 0.0,
    'z0_soil' : 0.01
}
image = {
    'T_R1' : None,
    'VZA' : 0.0, 
}
spectral = {
    'emis_C' : 0.98,
    'emis_S' : 0.95,
    'rho_vis_C' : 0.07,
    'tau_vis_C' : 0.08,
    'rho_nir_C' : 0.32,
    'tau_nir_C' : 0.33, 
    'rho_vis_S' : 0.15,
    'rho_nir_S' : 0.25,
}
structural = {
    'f_c' : None,
    'f_g' : None,
    'LAI' : None,
    'h_C': 0.3,
    'w_C' : 1,
    'leaf_width' : 0.1,
    'x_LAD' : 1,
    'alpha_PT' : 1.26
}

met = {
    'T_A1' : None,
    'u' : None,
    'p': None,
    'ea' : None,
    'z_u' : None,
    'z_T' : None,
    'S_dn' : None,
    'L_dn' : None,
}

def get_img_params(flight):

    image_params = {
        'T_R1' : flight.ortho.get_temperature(),
        'VZA' : 0.0
    }
    return image_params

def get_met_params(flight):

    SW_in_hat = get_SW_IN(flight)

    air = AirLayer(**flight.create_air_params())

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

def get_solar_angles(flight):
    # Solar zenith angle
    theta_sun = utils.calc_zenith(
        flight.coords[0], flight.coords[1], 
        flight.start_time, flight.end_time
    )
    # Solar azimuth angle
    psi_sun = utils.calc_azimuth(
        flight.coords[0], flight.coords[1], 
        flight.start_time, flight.end_time
    )
    solar_params = {
        'SZA' : theta_sun,
        'SAA' : psi_sun,
    }

    return solar_params

def get_flight_params


#%%

h = 0.3
d_0 = 0.65 * h
z_0m = 0.125 * h

vza = 0.0
LAI = 1.0

emis_C = 0.98
emis_S = 0.95
alpha_c = 0.25
alpha_s = 0.2



def get_SW_IN(flight):
    # 1. Get tower data for a flight
    tower = ramajal.data.loc[flight.end_time.round('30min')]
    # 2. Get SW data
    if flight.start_time <= pytz.timezone('America/Los_Angeles').localize(datetime.datetime(2021,3,27)):
        SW_in_hat = tower.SW_IN
    else:
        SW_in_hat = flight.get_stats(altitude=50, alt_buffer=100.)['mean']['SW_IN']

    return SW_in_hat


def _get_params(flight):

    SW_in_hat = get_SW_IN(flight)

    air = AirLayer(**flight.create_air_params())

    param_dict = {
        'vza' : vza,
        'T_A_K' : air.T_a,
        'u' : air.u,
        'ea' : air.e_a * 10,
        'p' : air.p_a * 10,
        'Sn_C' : (1 - alpha_c) * SW_in_hat,
        'Sn_S' : (1 - alpha_s) * SW_in_hat,
        'L_dn' : radiation.calc_LW(air.T_a, air.calc_emissivity()),
        'LAI' : LAI,
        'h_C' : h,
        'emis_C' : emis_C,
        'emis_S' : emis_S,
        'z_0M' : z_0m,
        'd_0' : d_0,
        'z_u' : air.z,
        'z_T' : air.z,
    }

    return param_dict


def get_pt_params(flight):

    T_s = flight.ortho.get_temperature()

    params = _get_params(flight)
    params['Tr_K'] = T_s

    return params

def get_2t_params(flight):

    params = _get_params(flight)

    params['T_C'] = flight.ortho.get_temperature()
    params['T_S'] = 0.0

    return params

def get_dtd_params(flight, flight_am):

    params = _get_params(flight)
    
    params['Tr_K_0'] = flight_am.ortho.get_temperature()
    params['Tr_K_1'] = flight.ortho.get_temperature()
    params['T_A_K_0'] = AirLayer(**flight_am.create_air_params()).T_a
    params['T_A_K_1'] = params['T_A_K']
    del params['T_A_K']

    return params

def get_oseb_params(flight):

    T_s = flight.ortho.get_temperature()

    ndvi = flight.ortho.calc_ndvi()
    alpha = radiation.calc_albedo(ndvi) 
    SW_in_hat = get_SW_IN(flight)

    params = _get_params(flight)
    params['Tr_K'] = T_s
    params['emis'] = radiation.calc_surf_emissivity(ndvi)
    params['Sn'] = (1 - alpha) * SW_in_hat

    del params['Sn_C']
    del params['Sn_S']
    del params['vza']
    del params['LAI']
    del params['h_C']
    del params['emis_C']
    del params['emis_S']

    return params


mods_dict = {
    'TSEB-PT' : {
        'params' : get_pt_params,
        'object' : tseb.TSEB_PT,
        'outputs' : [
            'flag', 'T_soil', 'T_veg', 'T_aero', 'LW_net_soil', 'LW_net_veg', 
            'LE_veg', 'H_veg', 'LE_soil', 'H_soil', 'G', 
            'r_soil', 'r_ex', 'r_a', 'u_star', 'L', 'n_iter'
        ],
        'model' : TSEB.TSEB_PT,
    },
    'TSEB-2T' : {
        'params' : get_2t_params,
        'object' : tseb.TSEB_2T,
        'outputs' : [
            'flag', 'T_aero', 
            'LE_veg', 'H_veg', 'LE_soil', 'H_soil', 'G', 
            'r_soil', 'r_ex', 'r_a', 'u_star', 'L', 'n_iter'        
        ],
        'model' : TSEB.TSEB_2T,
    },
    'DTD' : {
        'params' : get_dtd_params,
        'object' : tseb.TSEB_DTD,
        'outputs' : [
            'flag', 'T_soil', 'T_veg', 'T_aero', 'LW_net_soil', 'LW_net_veg', 
            'LE_veg', 'H_veg', 'LE_soil', 'H_soil', 'G', 
            'r_soil', 'r_ex', 'r_a', 'u_star', 'L', 'Ri', 'n_iter'
        ],
        'model' : TSEB.DTD,
    },
    'OSEB' : {
        'params' : get_oseb_params,
        'object' : tseb.TSEB_OSEB,
        'outputs' : ['flag', 'LW_net', 'LE', 'H', 'G', 'r_a', 'u_star', 'L', 'n_iter'],
        'model' : TSEB.OSEB,
    }
}


def get_params(flight, mod_name, **kwargs):

    return mods_dict.get(mod_name)['params'](flight, **kwargs)


def parse_outputs(outputs, mod_name):

    keys = mods_dict.get(mod_name)['outputs']

    out_dict = dict(zip(keys,outputs))

    return out_dict


#%%

def get_footprint(flight):
    footprint = ramajal.calc_footprint(
        timestamp=flight.end_time.round('30min'), 
        domain_arr=flight.ortho.ortho_array, 
        fig=False, nx=None, dx=0.5
    )
    return footprint



def run_model(flight, model_type : str, model_params : dict, footprint : fluxtower.Footprint, return_mod=True):
    # # 1. Get tower data for a fligth
    # tower = utils.get_tower_inflight(ramajal, flight)
    # # 2. Create flight air layers
    # # TODO: ADD CONDITIONAL
    # air1 = utils.create_flight_airlayer(flight, layer='ground')       # Ground
    # air2 = utils.create_flight_airlayer(flight)                       # Cruising altitude
    # # 3. Create flight surface object
    # surf = utils.create_flight_surface(flight, h=h, cover=cover, veg=veg)
    # # 4. Create flight radiation object
    # if flight.start_time <= pytz.timezone('America/Los_Angeles').localize(datetime.datetime(2021,3,27)):
    #     SW_in_hat = tower.SW_IN
    # else:
    #     SW_in_hat = None 
    # rad = utils.create_flight_radiation(flight, air1, surf, SW_in_hat=SW_in_hat, use_hillshade=False)

    # 5. Run model
    mod_obj = mods_dict.get(model_type)['object']

    mod = mod_obj(model_params)

    mod.run()
    
    mod_stats = mod.get_summary_stats()
    # stats.insert(0, 'Model', model_type)

    # Source-weighted means
    mod_stats['SRC_WT_MEAN'] = calc_src_wt_means(mod, footprint)

    if return_mod:
        return mod, mod_stats
    else:
        return mod_stats


def calc_src_wt_means(mod, footprint):

    var_dict = mod.get_fluxes()

    src_wt_means = []

    for key,val in var_dict.items():
        if isinstance(val, DataArray):
            try:
                swm = calc_source_weighted_mean(val, footprint)
            except:
                swm = np.nan
        else:
            swm = np.nan
        src_wt_means.append(swm)
    
    return src_wt_means


def calc_source_weighted_mean(flux_arr, footprint):

    flux_resamp = flight.ortho.resample_res(flux_arr, footprint.foot_utm)

    src_wt_mean = (flux_resamp * (footprint.foot_utm * footprint.pixel_area)).sum().item()

    return src_wt_mean




# def _check_default_parameter_size(parameter, input_array):

#     parameter = np.asarray(parameter, dtype=np.float32)
#     if parameter.size == 1:
#         parameter = np.ones(input_array.shape) * parameter
#         return np.asarray(parameter, dtype=np.float32)
#     elif parameter.shape != input_array.shape:
#         raise ValueError(
#             'dimension mismatch between parameter array and input array with shapes %s and %s' %
#             (parameter.shape, input_array.shape))
#     else:
#         return np.asarray(parameter, dtype=np.float32)



#%%

params_pt = get_params(flight, 'TSEB-PT')

params_pt['Tr_K'] = params_pt['Tr_K'][3000:3300,3000:3300]

params_2t = get_params(flight, 'TSEB-2T')
# params_dtd = get_params(flight, 'DTD', flight_am)
params_oseb = get_params(flight, 'OSEB')
params_oseb['Tr_K'] = params_oseb['Tr_K'][3000:3300,3000:3300]    
params_oseb['emis'] = params_oseb['emis'][3000:3300,3000:3300]
params_oseb['Sn'] = params_oseb['Sn'][3000:3300,3000:3300]

pt_raw = TSEB.TSEB_PT(**params_pt)

oseb_raw = TSEB.OSEB(**params_oseb)


pt_out = parse_outputs(pt_raw, 'TSEB-PT')
oseb_out = parse_outputs(oseb_raw, 'OSEB')

#%%

import tseb_models

mod_pt = tseb_models.TSEB_PT(params_pt)

pt_out = mod_pt.parse_outputs(pt_raw)

mod_pt.run(return_outputs = False)

pt_arrs = mod_pt.create_out_arrays(pt_out)
#%%




#%%


mod_name = 'OSEB'


footprints = []
flights_stats = []
# flights_models = []
# flights_vars = []

for flight in flights:
    print('Running flight {name}'.format(name=flight.log_file))
    t.start()
    # 1. Get model parameters
    params = get_params(flight, mod_name)
    # # 2. Get variable values
    # flights_vars.append(get_flight_vals(flight, model_params))
    # 3. Get tower footprint
    footprint = get_footprint(flight)
    
    mod_stats = run_model(flight, model_type = mod_name, model_params = params, footprint=footprint, return_mod=False)
    mod_stats.insert(0, 'Model', mod_name)
    mod_stats.insert(0, 'Flight', flight.start_time)
    mod_stats.insert(1, 'FlightDateTime', flight.end_time.round('30min'))

    footprints.append(footprint)
    flights_stats.append(mod_stats)
    # flights_models.append(models)

    print('Finished with flight {name}'.format(name=flight.log_file))
    t.stop()


#%%

uav_flux = pd.concat(flights_stats,ignore_index=True)
uav_flux.set_index('FlightDateTime',inplace=True)


ramajal.attribute_ebr(method='all')

tower = ramajal.data.loc[[flight.end_time.round('30min') for flight in flights]].copy()

foot_fracs = [foot.footprint_fraction for foot in footprints]

out_fold = '/Users/brynmorgan/Library/Mobile Documents/com~apple~CloudDocs/Dangermond/Figures/Oct2022/'
uav_flux_file = out_fold + 'ramajal_tseb-oseb_01.csv'

uav_flux.to_csv(uav_flux_file)

#%%




#%%

uav_means = uav_flux.pivot(columns=['Variable','Model'], values='MEAN')
uav_swm = uav_flux.pivot(columns=['Variable','Model'], values='SRC_WT_MEAN')

var_list = ['H','LE']

errs = pd.DataFrame(columns=pd.MultiIndex.from_product([var_list,mods_dict.keys()], names=['Variable','Model']))
rel_errs = pd.DataFrame(columns=pd.MultiIndex.from_product([var_list,mods_dict.keys()], names=['Variable','Model']))

# errs = pd.DataFrame()
# perc = pd.DataFrame()

for var in ['H','LE']:
    for method in mods_dict.keys():
        # var_df = get_var(var, method)
        diff = uav_means[var][method] - tower[var]
        rel_err = diff / tower[var]

        errs.loc[:,(var,method)] = diff
        rel_errs.loc[:,(var,method)] = rel_err


#%%
from scipy.stats import pearsonr
def calc_r2(y_mod, y_obs):
    r,pval = pearsonr(y_mod, y_obs)
    return r**2

def calc_rmse(y_mod, y_obs):
    return np.sqrt(((y_mod - y_obs) ** 2).mean())

def calc_mbe(y_mod, y_obs):
    return (y_mod - y_obs).mean()

def calc_mape(y_mod, y_obs):
    return (abs((y_mod - y_obs) / y_obs)).mean()


#%%

# ### Quality flags
# pyTSEB might produce some *more* unreliable data that can be tracked with the quality flags:

# * 0: Al Fluxes produced with no reduction of PT parameter (i.e. positive soil evaporation)
# * 3: negative soil evaporation, forced to zero (the PT parameter is reduced in TSEB-PT and DTD)
# * 5: No positive latent fluxes found, G recomputed to close the energy balance (G=Rn-H)
# * 255: Arithmetic error. BAD data, it should be discarded

# In addition for the component temperatures TSEB (TSEB-2T):

# * 1: negative canopy latent heat flux, forced to zero
# * 2: negative canopy sensible heat flux, forced to zero
# * 4: negative soil sensible heat flux, forced to zero






    # Returns
    # -------
    # flag : int
    #     Quality flag, see Appendix for description.
    # T_AC : float
    #     Air temperature at the canopy interface (Kelvin).
    # LE_C : float
    #     Canopy latent heat flux (W m-2).
    # H_C : float
    #     Canopy sensible heat flux (W m-2).
    # LE_S : float
    #     Soil latent heat flux (W m-2).
    # H_S : float
    #     Soil sensible heat flux (W m-2).
    # G : float
    #     Soil heat flux (W m-2).
    # R_S : float
    #     Soil aerodynamic resistance to heat transport (s m-1).
    # R_x : float
    #     Bulk canopy aerodynamic resistance to heat transport (s m-1).
    # R_A : float
    #     Aerodynamic resistance to heat transport (s m-1).
    # u_friction : float
    #     Friction velocity (m s-1).
    # L : float
    #     Monin-Obuhkov length (m).
    # n_iterations : int
    #     number of iterations until convergence of L.


    # Returns
    # -------
    # flag : int
    #     Quality flag, see Appendix for description.
    # T_S : float
    #     Soil temperature  (Kelvin).
    # T_C : float
    #     Canopy temperature  (Kelvin).
    # T_AC : float
    #     Air temperature at the canopy interface (Kelvin).
    # L_nS : float
    #     Soil net longwave radiation (W m-2).
    # L_nC : float
    #     Canopy net longwave radiation (W m-2).
    # LE_C : float
    #     Canopy latent heat flux (W m-2).
    # H_C : float
    #     Canopy sensible heat flux (W m-2).
    # LE_S : float
    #     Soil latent heat flux (W m-2).
    # H_S : float
    #     Soil sensible heat flux (W m-2).
    # G : float
    #     Soil heat flux (W m-2).
    # R_S : float
    #     Soil aerodynamic resistance to heat transport (s m-1).
    # R_x : float
    #     Bulk canopy aerodynamic resistance to heat transport (s m-1).
    # R_A : float
    #     Aerodynamic resistance to heat transport (s m-1).
    # u_friction : float
    #     Friction velocity (m s-1).
    # L : float
    #     Monin-Obuhkov length (m).
    # Ri : float
    #     Richardson number.
    # n_iterations : int
    #     number of iterations until convergence of L.










#%%

vza = 0.0
h = 0.3
d_0 = h * 0.65
z_0M = 0.125 * h
z = 3.0
emis_C = 0.98
emis_S = 0.95
p = 1013.
SW_IN = 800.
LW_IN = 350.

T_rads = np.arange(0.0, 51.0, 1.0)
T_as = np.arange(0.0, 51.0, 5.0)
us = np.array([0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
e_as = np.arange(1.0, 16, 1.0)
LAIs = np.arange(0.1, 5.0, 0.1)


params_all = {
    'u' : u,
    'ea' : e_a,
    'p' : p_a,
    'L_dn' : LW_IN,
    'z_0M' : z_0m,
    'd_0' : d_0,
    'z_u' : z,
    'z_T' : z
}


for T_a in T_as:
    for u in us:
        for e_a in e_as:
            for LAI in LAIs:
                params = {
                    'Tr_K' : T_rads,
                    'vza' : 0.0,
                    'T_A_K' : T_a + 273.15,
                    'u' : u,
                    'ea' : e_a,
                    'p' : p,
                    'Sn_C' : (1-alpha_c) * SW_IN,
                    'Sn_S' : (1-alpha_s) * SW_IN,
                    'L_dn' : LW_IN,
                    'LAI' : LAI,
                    'h_C' : h,
                    'z_0M' : z_0m,
                    'd_0' : d_0,
                    'z_u' : z,
                    'z_T' : z
                }

                pt_out = TSEB.TSEB_PT(

                )


# PRIESTLEY-TAYLOR
TSEB_PT(
    Tr_K, vza, 
    T_A_K, u, ea, p, Sn_C, Sn_S, L_dn, 
    LAI, h_C, emis_C, emis_S, 
    z_0M, d_0, z_u, z_T, 
    leaf_width=0.1, z0_soil=0.01, 
    alpha_PT=1.26, x_LAD=1, f_c=1.0, f_g=1.0, w_C=1.0, 
    resistance_form=None, calcG_params=None, const_L=None, kB=0.0, massman_profile=None, verbose=True
)


# TWO-SOURCE
TSEB_2T(
    T_C, T_S, 
    T_A_K, u, ea, p, Sn_C, Sn_S, L_dn, 
    LAI, h_C, emis_C, emis_S, 
    z_0M, d_0, z_u, z_T, 
    leaf_width=0.1, z0_soil=0.01, 
    alpha_PT=1.26, x_LAD=1.0, f_c=1.0, f_g=1.0, w_C=1.0, 
    resistance_form=None, calcG_params=None, const_L=None, kB=0.0, massman_profile=None, verbose=True
)

DTD(
    Tr_K_0, Tr_K_1, vza, 
    T_A_K_0, T_A_K_1, u, ea, p, Sn_C, Sn_S, L_dn, 
    LAI, h_C, emis_C, emis_S, 
    z_0M, d_0, z_u, z_T, 
    leaf_width=0.1, z0_soil=0.01, 
    alpha_PT=1.26, x_LAD=1, f_c=1.0, f_g=1.0, w_C=1.0, 
    resistance_form=None, calcG_params=None, calc_Ri=True, kB=0.0, massman_profile=None, verbose=True
)

OSEB(
    Tr_K, T_A_K, 
    u, ea, p, Sn, L_dn, emis, z_0M, d_0, z_u, z_T, 
    calcG_params=[[1], 0.35], const_L=None, T0_K=[], kB=0.0
)











#%%

    # Parameters
    # ----------
    # Tr_K : float
    #     Radiometric composite temperature (Kelvin).
    # vza : float
    #     View Zenith Angle (degrees).
    # T_A_K : float
    #     Air temperature (Kelvin).
    # u : float
    #     Wind speed above the canopy (m s-1).
    # ea : float
    #     Water vapour pressure above the canopy (mb).
    # p : float
    #     Atmospheric pressure (mb), use 1013 mb by default.
    # Sn_C : float
    #     Canopy net shortwave radiation (W m-2).
    # Sn_S : float
    #     Soil net shortwave radiation (W m-2).
    # L_dn : float
    #     Downwelling longwave radiation (W m-2).
    # LAI : float
    #     Effective Leaf Area Index (m2 m-2).
    # h_C : float
    #     Canopy height (m).
    # emis_C : float
    #     Leaf emissivity.
    # emis_S : flaot
    #     Soil emissivity.
    # z_0M : float
    #     Aerodynamic surface roughness length for momentum transfer (m).
    # d_0 : float
    #     Zero-plane displacement height (m).
    # z_u : float
    #     Height of measurement of windspeed (m).
    # z_T : float
    #     Height of measurement of air temperature (m).
    # leaf_width : float, optional
    #     average/effective leaf width (m).
    # z0_soil : float, optional
    #     bare soil aerodynamic roughness length (m).
    # alpha_PT : float, optional
    #     Priestley Taylor coeffient for canopy potential transpiration,
    #     use 1.26 by default.
    # x_LAD : float, optional
    #     Campbell 1990 leaf inclination distribution function chi parameter.
    # f_c : float, optional
    #     Fractional cover.
    # f_g : float, optional
    #     Fraction of vegetation that is green.
    # w_C : float, optional
    #     Canopy width to height ratio.
    # resistance_form : int, optional
    #     Flag to determine which Resistances R_x, R_S model to use.

#%%

{'model': 'TSEB_PT',
 'output_file': './Output/test_image.tif',
 'resistance_form': 0,
 'water_stress': False,
 'calc_row': [0, 0],
 'G_form': [[1], 0.35],
 'KN_b': '0.012',
 'KN_c': '0.0038',
 'KN_C_dash': '90',
 'landcover': '4',
 'lat': '38.289355',
 'lon': '-121.117794',
 'alt': '97',
 'stdlon': '-105.0',
 'z_T': '5',
 'z_u': '5',
 'z0_soil': '0.01',
 'leaf_width': '0.1',
 'alpha_PT': '1.26',
 'x_LAD': '1',
 'emis_C': '0.98',
 'emis_S': '0.95',
 'rho_vis_C': '0.07',
 'tau_vis_C': '0.08',
 'rho_nir_C': '0.32',
 'tau_nir_C': '0.33',
 'rho_vis_S': '0.15',
 'rho_nir_S': '0.25',
 'T_R1': './Input/ExampleImage_Trad_pm.tif',
 'input_mask': '0',
 'T_A1': './Input/ExampleImage_Ta.tif',
 'S_dn': '861.74',
 'f_c': './Input/ExampleImage_Fc.tif',
 'VZA': '0.0',
 'LAI': './Input/ExampleImage_LAI.tif',
 'ea': '13.4',
 'f_g': '1',
 'p': '1011',
 'w_C': '1',
 'h_C': '2.4',
 'DOY': '221',
 'S_dn_24': '',
 'u': '2.15',
 'L_dn': '',
 'time': '10.9992'}
















#-------------------------------------------------------------------------------
# VARIABLES/PARAMS TO GET FROM CONFIG/USER
#-------------------------------------------------------------------------------
z0_soil = None
# Leaf spectral properties:{rho_vis_C: visible reflectance, tau_vis_C: visible transmittance, rho_nir_C: NIR reflectance, tau_nir_C: NIR transmittance}
rho_vis_C=0.07
tau_vis_C=0.08
rho_nir_C=0.32
tau_nir_C=0.33 

# Soil spectral properties:{rho_vis_S: visible reflectance, rho_nir_S: NIR reflectance}
rho_vis_S=0.15
rho_nir_S=0.25

spectra = {
    'rho_vis_C' : 0.07,
    'tau_vis_C' : 0.08,
    'rho_nir_C' : 0.32,
    'tau_nir_C' : 0.33, 
    'rho_vis_S' : 0.15,
    'rho_nir_S' : 0.25
}


LAI = None      # May move to flight or derived here
f_c = None      # May move to flight or derived here
h_c = None
w_c = None
landcover = None # 2 (grass)
x_LAD = None    # 1.0 (spherical)


#-------------------------------------------------------------------------------
# VARIABLES/PARAMS TO GET FROM FLIGHT, IMAGE, ETC.
#-------------------------------------------------------------------------------
lat = None
long = None
start_time = None
end_time = None

#-------------------------------------------------------------------------------
# MET VARIABLES/PARAMS TO GET FROM FLIGHT OR OTHER???
#-------------------------------------------------------------------------------
SW_IN = None    # W m-2
p_a = None      # kPa
LW_IN = None    # W m-2

#-------------------------------------------------------------------------------
# VARIABLES/PARAMS TO DERIVE HERE
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# PRE-PROCESSING (ALL MODELS)
#-------------------------------------------------------------------------------
# Variables derived: sza, saa, p, L_dn
# Calculate solar zenith angle and solar azimuth angles
psi_sun = utils.calc_azimuth(lat, long, start_time, end_time)
theta_sun = utils.calc_zenith(lat, long, start_time, end_time)

#-------------------------------------------------------------------------------
# PRE-PROCESSING (IMAGES)
#-------------------------------------------------------------------------------
# Calculate diffuse and direct radation
dif_vis, dif_nir, f_vis, f_nir = pyTSEB.rad.calc_difuse_ratio(SW_IN, theta_sun, p_a*10)

skyl = dif_vis * f_vis + dif_nir * f_nir

SW_IN_dir = SW_IN * (1.0 - skyl)
SW_IN_dif = SW_IN * skyl

#-------------------------------------------------------------------------------
# SOIL
#-------------------------------------------------------------------------------
# 1. MASK
# 2. Calculate roughness
# z_0m = z0_soil
# d_0 = 0.0
# 3. Calculate net SW for bare soil
# alpha_soil = calc_albedo_soil
# SW_soil = calc_SW_soil(SW_IN, alpha_soil)
def calc_albedo_soil(f_vis, f_nir, rho_vis_S, rho_nir_S):
    
    alpha_soil = f_vis * rho_vis_S + f_nir * rho_nir_S
    
    return alpha_soil

def calc_SW_soil(SW_IN, alpha_soil):

    SW_soil = (1. - alpha_soil) * SW_IN
    
    return SW_soil

# 4. Calculate fluxes for bare soil

def calc_flux_soil():

    oseb_out = TSEB.OSEB(**params_oseb_soil)

# [out_data['flag'][i],
# out_data['Ln_S1'][i],
#     out_data['LE_S1'][i],
#     out_data['H_S1'][i],
#     out_data['G1'][i],
#     out_data['R_A1'][i],
#     out_data['u_friction'][i],
#     out_data['L'][i],
#     out_data['n_iterations'][i]] = TSEB.OSEB(in_data['T_R1'][i],
#                                             in_data['T_A1'][i],
#                                             in_data['u'][i],
#                                             in_data['ea'][i],
#                                             in_data['p'][i],
#                                             out_data['Sn_S1'][i],
#                                             in_data['L_dn'][i],
#                                             in_data['emis_S'][i],
#                                             out_data['z_0M'][i],
#                                             out_data['d_0'][i],
#                                             in_data['z_u'][i],
#                                             in_data['z_T'][i],
#                                             calcG_params=[model_params["calcG_params"][0],
#                                                         model_params["calcG_params"][1][i]])

#-------------------------------------------------------------------------------
# VEGETATION
#-------------------------------------------------------------------------------
# 1. Mask
# 2. Calculate roughness
# z_0m, d_0 = calc_veg_roughness

def calc_veg_roughness(LAI, h_C, w_C, f_c, landcover=2):

    z_0m, d_0 = pyTSEB.res.calc_roughness(
        LAI = LAI, h_C = h_C, w_C = w_C, landcover = 2, f_c = f_c
    )

    return z_0m, d_0
# 3. Calculate net SW for vegetation
def calc_SW_veg():




# Net SW_veg
F = LAI / f_c
# Clumping index
omega0 = pyTSEB.CI.calc_omega0_Kustas(
    LAI = LAI,
    f_C = f_c,
    x_LAD = xLAD,
    isLAIeff = True
)

omega = pyTSEB.CI.calc_omega_Kustas(
    omega0 = omega0,
    theta = theta_sun,
    w_C = 
)
# Effective leaf area
LAI_eff = F * omega
# Net SW canopy + soil components
Sn_C, Sn_S = pyTSEB.rad.calc_Sn_Campbell(**sw_campbell_params)

sw_campbell_params = {
    'LAI' : ,
    'sza' : ,
    'S_dn_dir' : ,
    'S_dn_dif' : ,
    'fvis' : ,
    'fnir' : ,
    'rho_vis_C' : ,
    'tau_vis_C' : ,
    'rho_nir_C' : ,
    'tau_nir_C' : ,
    'rho_vis_S' : ,
    'rho_nir_S' : ,
    'x_LAD' : ,
    'LAI_eff' : ,
}

# RUN TSEB-PT (see pyTSEB.pyTSEB._call_flux_model_veg)

# BULK FLUXES
# # Calculate the bulk fluxes
# out_data['LE1'] = out_data['LE_C1'] + out_data['LE_S1']
# out_data['LE_partition'] = out_data['LE_C1'] / out_data['LE1']
# out_data['H1'] = out_data['H_C1'] + out_data['H_S1']
# out_data['R_ns1'] = out_data['Sn_C1'] + out_data['Sn_S1']
# out_data['R_nl1'] = out_data['Ln_C1'] + out_data['Ln_S1']
# out_data['R_n1'] = out_data['R_ns1'] + out_data['R_nl1']
# out_data['delta_R_n1'] = out_data['Sn_C1'] + out_data['Ln_C1']




       # Esimate diffuse and direct irradiance
        difvis, difnir, fvis, fnir = rad.calc_difuse_ratio(
            in_data['S_dn'], in_data['SZA'], press=in_data['p'])
        out_data['fvis'] = fvis
        out_data['fnir'] = fnir
        out_data['Skyl'] = difvis * fvis + difnir * fnir
        out_data['S_dn_dir'] = in_data['S_dn'] * (1.0 - out_data['Skyl'])
        out_data['S_dn_dif'] = in_data['S_dn'] * out_data['Skyl']

pyTSEB.pyTSEB.run(in_data, mask)



azimuths = [
    189.38501779464178,
    153.59120331434335,
    149.37476024152284,
    165.3328836218183,
    181.91061444142696,
    198.28903607004543,
    215.74199936549422,
    146.0165709973654,
    162.65024575494414,
    180.51866992773222,
    197.8266506568358,
    211.99675200905045,
    181.35547428207684,
    201.39944311388214,
    212.83030320572948,
    226.3066334642033
]

zeniths = [
    42.25157470231668,
    35.72088909219039,
    61.27232384321543,
    56.33738315022694,
    54.981221727377665,
    57.10042468029275,
    63.70787892852218,
    61.020525046094555,
    55.24929316121919,
    53.355848484111256,
    55.325277752505805,
    60.04090658621398,
    52.59145773947243,
    55.41829357763257,
    59.541309383879394,
    67.50985436257433
]


SW_ins = [
    851.053,
    914.442,
    512.7120896946543,
    597.6271728385573,
    641.4186536646713,
    594.7071120831929,
    466.3815198189626,
    448.1673717039957,
    662.0600614800612,
    697.7386612001504,
    658.4926169348486,
    616.9211708720932,
    730.1638214367525,
    674.145961032085,
    592.1932579516777,
    431.44092982145906
]

p_as = [
    99.96777335299952,
    99.53402870813369,
    99.77911828463688,
    99.75651492731004,
    99.6722219092807,
    99.57451043247548,
    99.58582145316889,
    99.46592296719054,
    99.46892816163572,
    99.43487545720585,
    99.37082749649731,
    99.3334735107121,
    100.26765229421298,
    100.21971332886905,
    100.15368380659936,
    100.09655319926833
]

dif_vis, dif_nir, f_vis, f_nir = pyTSEB.rad.calc_difuse_ratio(SW_ins, zeniths, [p_a*10 for p_a in p_as])

