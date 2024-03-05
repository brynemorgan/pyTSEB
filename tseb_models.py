#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Tue 01 Nov 22 14:31:28'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           model.py
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
import numpy as np
import pandas as pd
import xarray as xr


import pyTSEB.TSEB as TSEB


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


class TSEB_Model:

    def __init__(self, in_params : dict):

        self._params = in_params

        self._coords = self.get_coords()

        self._mod_type = None
    
    
    def _run(self):
        
        raw_out = mods_dict.get(self._mod_type)['model'](**self._params)

        return raw_out
    
    def run(self, return_outputs=False):

        self._raw_outputs = self._run()

        out_dict = self.parse_outputs(self._raw_outputs)

        self.set_out_arrays(out_dict)
        self.R_n = self.calc_Rn(self.H, self.LE, self.G)

        if return_outputs:
            outputs = self.create_out_arrays(out_dict)

            for key,val in outputs.items():
                outputs[key] = val.where(out_dict['flag'] <= 5.)

            return outputs
    
    def parse_outputs(self, outputs):

        keys = mods_dict.get(self._mod_type)['outputs']

        out_dict = dict(zip(keys,outputs))

        return out_dict

    def set_out_arrays(self, out_dict):
        raise NotImplementedError

    
    def get_coords(self):

        try:
            coords = {'x' : self._params['Tr_K'].x, 'y' : self._params['Tr_K'].y}
        except:
            coords = None
        
        return coords
    

    def get_var_stats(self, var):
        """
        Get summary statistics (min, max, mean, median, standard deviation) for 
        a variable.

        Parameters
        ----------
        var : xarray.DataArray | xarray.DataSet
            Pixel-wise variable for which to calculate summary statistics

        Returns
        -------
        stats_dict : dict
            Summary statistics for var as a dictionary.

        # NOTE: Probably will refactor.
        """
        
        stats_dict = {
            'MAX' : var.max().item(),
            'MIN' : var.min().item(),
            'MEAN' : var.mean().item(),
            'MEDIAN' : var.median().item(),
            'STD' : var.std().item()
        }

        return stats_dict

    def get_summary_stats(self, as_df=True):
        """
        Get summary statistics for all flux-related variables (zeta, u_star, 
        Psi_M, Psi_H, r_aH, L, H, LE). Note that only array-like variables will
        return summary statistics; others will return the scalar values.

        Returns
        -------
        all_dict : dict
            Dictionary of dictionaries containing flux variables (keys) and their
            summary statistics (values : dict).
        
        # TODO: Change the way non-arrays are handled (this is sloppy--all should be the same type).
        # TODO: Refactor--this is temporary and will likely want to change.
        """
        # Get flux variables as dict
        var_dict = self.get_fluxes()
        # Initialize empty dictionary for variables
        all_dict = {}
        # Iterate through flux variables and return summary stats
        for key,val in var_dict.items():
            try:
                all_dict[key] = self.get_var_stats(val)
            except:
                all_dict[key] = {'MEAN' : val}
        
        if as_df:
            return self.get_stats_df(all_dict)
        else:
            return all_dict

    def get_stats_df(self, stats_dict):

        # stats = self.get_summary_stats()

        df = pd.DataFrame(stats_dict).T.reset_index().rename(columns={'index':'Variable'})

        return df

    def get_fluxes(self):
        """
        Get flux-related variables created/output by the model.

        Returns
        -------
        dict
            Dictionary of flux variables calculated by the model run.
        """
        flux_dict =  {
            'T_soil' : self.T_soil,
            'T_veg' : self.T_veg,
            'T_aero' : self.T_aero,
            'H_veg' : self.H_veg,
            'H_soil' : self.H_soil,
            'H' : self.H,
            'LE_veg' : self.LE_veg,
            'LE_soil' : self.LE_soil,
            'LE' : self.LE,
            'G' : self.G,
            'R_n' : self.R_n,
            'r_soil' : self.r_soil,
            'r_ex' : self.r_ex,
            'r_a' : self.r_a,
            'u_star' : self.u_star,
            'L' : self.L
            # 'n_iter' : len(self.Ls)
        }

        return flux_dict

    def calc_total_flux(self, soil : xr.DataArray, veg : xr.DataArray, name : str):

        tot = soil + veg
        tot.name = name

        return tot
    
    def calc_Rn(self, H, LE, G):
        R_n = H + LE + G
        return R_n
    
    def create_out_arrays(self):
        raise NotImplementedError

    def _create_T_array(self, out_dict):

        coords = self.get_coords()
        coords.update({'band' : ['soil', 'veg', 'aero']})

        T_out = xr.DataArray(
            data = [out_dict['T_soil'], out_dict['T_veg'], out_dict['T_aero']],
            coords = coords,
            dims = ['band', 'y', 'x']
        )
        T_out.name = 'T_s'

        # TODO: add nodata/mask, add crs

        return T_out
    
    def _create_flux_array(self, out_dict, var='H'):
        
        coords = self.get_coords()
        coords.update({'band' : ['soil', 'veg', 'total']})

        data = [
            out_dict[var + '_soil'], out_dict[var + '_veg'], out_dict[var + '_soil'] + out_dict[var + '_veg']
        ]

        out_arr = xr.DataArray(
            data = data,
            coords = coords,
            dims = ['band', 'y', 'x']
        )
        out_arr.name = var

        # TODO: add nodata/mask, add crs

        return out_arr

    def _create_ra_array(self, out_dict):
        coords = self.get_coords()
        coords.update({'band' : ['soil', 'ex', 'total']})

        r_out = xr.DataArray(
            data = [out_dict['r_soil'], out_dict['r_ex'], out_dict['r_a']],
            coords = coords,
            dims = ['band', 'y', 'x']
        )
        r_out.name = 'r_aH'

        # TODO: add nodata/mask, add crs; use regex to combine all of these

        return r_out

    def _create_var_array(self, out_dict, var='G'):
        coords = self.get_coords()

        out_arr = xr.DataArray(
            data = out_dict[var],
            coords = coords,
            dims = ['y', 'x']
        )
        out_arr.name = var

        # TODO: add nodata/mask, add crs; use regex to combine all of these

        return out_arr





class TSEB_PT(TSEB_Model):

    # cls_dict = mods_dict.get('TSEB-PT')

    def __init__(self, in_params):

        super().__init__(in_params)
        
        self._mod_type = 'TSEB-PT'
    
    def set_out_arrays(self, out_dict):

        (
            self.flag, 
            self.T_soil, 
            self.T_veg, 
            self.T_aero, 
            self.LW_net_soil,
            self.LW_net_veg,
            self.LE_veg,
            self.H_veg,
            self.LE_soil,
            self.H_soil,
            self.G,
            self.r_soil,
            self.r_ex,
            self.r_a,
            self.u_star,
            self.L 
        ) = (self._create_var_array(out_dict, var) for var in list(out_dict.keys())[:-1])
    
        self.H = self.calc_total_flux(self.H_soil, self.H_veg, 'H')
        self.LE = self.calc_total_flux(self.LE_soil, self.LE_veg, 'LE')


        self.n_iter = out_dict['n_iter']
    
    def create_out_arrays(self, out_dict):

        outputs = dict(zip(
            ['H', 'LE', 'LW_net'],
            [self._create_flux_array(out_dict, var=var) for var in ['H', 'LE', 'LW_net']]
        ))

        outputs.update({'T' : self._create_T_array(out_dict)})
        outputs.update({'r_aH' : self._create_ra_array(out_dict)})
        outputs.update(
            dict(zip(
                ['G', 'u_star', 'L'], 
                [self._create_var_array(out_dict, var=var) for var in ['G', 'u_star', 'L']]
            ))
        )
        outputs.update({'n_iter' : out_dict['n_iter']})

        return outputs





class TSEB_2T(TSEB_Model):

    def __init__(self, in_params):

        super().__init__(in_params)
        
        self._mod_type = 'TSEB-2T'
    
    def set_out_arrays(self, out_dict):

        (
            self.flag,
            self.T_aero,
            self.LE_veg,
            self.H_veg,
            self.LE_soil,
            self.H_soil,
            self.G,
            self.r_soil,
            self.r_ex,
            self.r_a,
            self.u_star,
            self.L 
        ) = (self._create_var_array(out_dict, var) for var in list(out_dict.keys())[:-1])

        self.H = self.calc_total_flux(self.H_soil, self.H_veg, 'H')
        self.LE = self.calc_total_flux(self.LE_soil, self.LE_veg, 'LE')

        self.T_soil = self._create_var_array(self._params, 'T_S')
        self.T_veg = self._create_var_array(self._params, 'T_C')

        self.n_iter = out_dict['n_iter']



    def create_out_arrays(self, out_dict):

        outputs = dict(zip(
            ['H', 'LE'],
            [self._create_flux_array(out_dict, var=var) for var in ['H', 'LE']]
        ))

        outputs.update({'T' : self._create_T_array(out_dict)})
        outputs.update({'r_aH' : self._create_ra_array(out_dict)})
        outputs.update(
            dict(zip(
                ['G', 'u_star', 'L'], 
                [self._create_var_array(out_dict, var=var) for var in ['G', 'u_star', 'L']]
            ))
        )
        outputs.update({'n_iter' : out_dict['n_iter']})

        return outputs

    def _create_T_array(self, out_dict):

        coords = self.get_coords()
        coords.update({'band' : ['soil', 'veg', 'aero']})

        T_out = xr.DataArray(
            data = [self._params['T_S'], self._params['T_C'], out_dict['T_aero']],
            coords = coords,
            dims = ['band', 'y', 'x']
        )
        T_out.name = 'T_s'

        # TODO: add nodata/mask, add crs

        return T_out
    
    def get_coords(self):

        try:
            coords = {'x' : self._params['T_C'].x, 'y' : self._params['T_C'].y}
        except:
            coords = None
        
        return coords




class TSEB_OSEB(TSEB_Model):

    def __init__(self, in_params):

        super().__init__(in_params)
        
        self._mod_type = 'OSEB'
    
    def set_out_arrays(self, out_dict):

        (
            self.flag,
            self.LW_net,
            self.LE,
            self.H,
            self.G,
            self.r_a,
            self.u_star,
            self.L 
        ) = (self._create_var_array(out_dict, var) for var in list(out_dict.keys())[:-1])

        self.n_iter = out_dict['n_iter']

        self.T_s = self._params['Tr_K']

    
    def create_out_arrays(self, out_dict):

        outputs = dict(zip(
            list(out_dict.keys())[1:-1], 
            [self._create_var_array(out_dict, var=var) for var in list(out_dict.keys())[1:-1]]
        ))

        outputs.update({'n_iter' : out_dict['n_iter']})

        return outputs

    def get_fluxes(self):
        """
        Get flux-related variables created/output by the model.

        Returns
        -------
        dict
            Dictionary of flux variables calculated by the model run.
        """
        flux_dict =  {
            'T_s' : self.T_s,
            'H' : self.H,
            'LE' : self.LE,
            'G' : self.G,
            'r_a' : self.r_a,
            'u_star' : self.u_star,
            'L' : self.L
            # 'n_iter' : len(self.Ls)
        }

        return flux_dict


class TSEB_DTD(TSEB_Model):

    def __init__(self, in_params):

        super().__init__(in_params)
        
        self._mod_type = 'DTD'


    def get_coords(self):

        # img = list(self._params.values())[0]

        try:
            coords = {'x' : self._params['Tr_K_1'].x, 'y' : self._params['Tr_K_1'].y}
        except:
            coords = None
        
        return coords
    
    # def create_var_array(self, out_dict, var='G'):
    #     coords = self.get_coords()
    #     coords.update({'band' : [var]})

    #     out_arr = xr.DataArray(
    #         data = out_dict[var],
    #         coords = coords,
    #         dims = ['band', 'y', 'x']
    #     )

    #     # TODO: add nodata/mask, add crs; use regex to combine all of these

    #     return out_arr


    # def get_coords(self):

    #     try:
    #         coords = {'x' : self._params['Tr_K'].x, 'y' : self._params['Tr_K'].y}
    #     except:
    #         coords = None
        
    #     return coords



# class Footprint:

#     def __init__(self, in_params : dict, timestamp, coords : tuple = None, **kwargs):

#         self._params = in_params
#         self.timestamp = timestamp
#         self.origin = coords

#         self.ffp = self.calc_footprint(**kwargs)
#         self.set_footprint_vals()



#     def calc_footprint(self, **kwargs):

#         ffp_out = calc_ffp_clim.FFP_climatology(**self._params, **kwargs)

#         return ffp_out

#     def set_footprint_vals(self):

#         self.x_coords = self.ffp['x_2d'][0]
#         self.y_coords = self.ffp['y_2d'][:,0]
        
#         self.footprint = self.ffp['fclim_2d']

#         if self.origin:
#             self.set_footprint_utm()

#     def set_footprint_utm(self):

#         # Convert origin to UTM coordinates
#         self.origin_utm = self.get_utm_coords()
#         # Create data array with UTM coordinates
#         self.foot_utm = xr.DataArray(
#             self.footprint, 
#             coords = (self.y_coords + self.origin_utm[1], self.x_coords + self.origin_utm[0]),
#             dims = ['y', 'x']
#         )
#         # Set CRS
#         crs = utils.get_utm_crs(lat=self.origin[0], long=self.origin[1])
#         self.foot_utm.rio.set_crs(crs, inplace=True)
#         # Set resolution
#         self.resolution = self.foot_utm.rio.resolution()
#         self.pixel_area = self.resolution[0] * self.resolution[1]
#         self.footprint_fraction = (self.foot_utm * self.pixel_area).sum().item()
#         # Get contour lines in UTM coordinates
#         self.contours = self.get_contour_points(self.origin_utm)


#     def get_utm_coords(self, datum='WGS 84'):
#         coords_utm = utils.convert_to_utm(
#             self.origin[0], self.origin[1], datum=datum
#         )
#         return coords_utm