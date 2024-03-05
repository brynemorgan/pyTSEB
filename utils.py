#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Mon 31 Oct 22 15:42:45'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           utils.py
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
import os
import sys
import pyproj

# Local module imports
# from model import AirLayer, Surface, Radiation
# from model import radiation

# Imports from other repos
# NOTE: The dependencies on fluxtower and uavet are only for setting the types 
# of the parameters for the functions, so not sure whether these should be removed?

# Temporary patch to use code from ecflux until packaged + released...
if not any('ecflux' in x for x in sys.path):
    sys.path.append('/Users/brynmorgan/dev/ecflux/ecflux/')

from fluxtower import FluxTower

if not any('uav-et' in x for x in sys.path):
    sys.path.append('/Users/brynmorgan/dev/uav-et/code')

from uavet import Flight

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

def get_tower_inflight(tower : FluxTower, flight : Flight):
    return tower.data.loc[flight.end_time.round('30min')]




import numpy as np
import datetime
import pysolar.solar as solar



#%% FUNCTIONS

def calc_azimuth(lat : float, long : float, start_time : datetime.datetime, end_time : datetime.datetime):
    """
    Calculate the average solar azimuth angle over a period of time for a point.

    Parameters
    ----------
    lat : float
        Latitude [decimal degrees]
    long : floaot
        Longitude [decimal degrees]
    start_time : datetime.datetime
        Start of time period
    end_time : datetime.datetime
        End of time period

    Returns
    -------
    psi_sun
        Solar azimuth angle [radians]
    """
    psi_sun = (
        solar.get_azimuth(lat, long, start_time) + \
        solar.get_azimuth(lat, long, end_time)
    ) / 2

    return np.radians(psi_sun)

def calc_zenith(lat : float, long : float, start_time : datetime.datetime, end_time : datetime.datetime):
    """
    Calculate the average solar zenith angle over a period of time for a point.

    Parameters
    ----------
    lat : float
        Latitude [decimal degrees]
    long : floaot
        Longitude [decimal degrees]
    start_time : datetime.datetime
        Start of time period
    end_time : datetime.datetime
        End of time period

    Returns
    -------
    theta_sun
        Solar zenith angle [radians]
    """
    # Calculate zenith angle at start time
    theta_sun_1 = 90 - solar.get_altitude(lat, long, start_time)
    # Calculate zenith angle at end time
    theta_sun_2 = 90 - solar.get_altitude(lat, long, end_time)
    # Average
    theta_sun = (theta_sun_1 + theta_sun_2) / 2

    return np.radians(theta_sun)

