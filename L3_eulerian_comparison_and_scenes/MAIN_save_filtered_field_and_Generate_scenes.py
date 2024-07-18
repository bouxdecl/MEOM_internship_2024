#!/usr/bin/env python3

'''

Generate Drifters Scenes
########################

Inputs:
- SSH field dataset file with dims = (time, y, x) and vars = (latitude, longitude, ssh) 
- Drifters files like ragged arrays dims = (id, time) and vars = (lat, lon, u, v)

Saves:
- files for each trajectory with dims = (scene, s_obs) and vars = (time, u, v, u_filtered, v_filtered, lon, lat, lon_filtered, lat_filtered, u_geo, v_geo, u_var, v_var)

Field velocity (geostrophie and variationnal cyclostrophie) are interpolated at filtered positions and drifters time.

'''


###################################################
path_save_data = '../save_data/drifters_scenes/Scene_0point5kmfilterfield_48hfilterdrifter'
NAME = '2024-07-15_Scene_0point5kmfield_48hdrifter' #the beginning of each scene file
verbose = True

# Filter field
temporal_sigma = None   # std of the gaussian filter (hours)
spatial_sigma  = 3.5    # std of the spatial 2D gaussian filter (km)

# Filter drifters
filtering_velocities=True,
low_pass_cutoff= 1/(48*3600) #Hz
filtering_positions=False
gaussian_std=12*3600 #s 

###################################################


# --- IMPORTS

import sys
import os
import numpy as np
import xarray as xr

#own modules import
current_dir = os.path.dirname(os.path.abspath(__file__))
common_utils_dir = os.path.join(current_dir, '../common_utils')
sys.path.append(common_utils_dir)
from utils import *
import Drifters_processing
import L3Field_filtering
from compute_velocitites import add_velocities



# --- FIELD PROCESSING

#load SSHraw file obtained from the standalone file "load_field_dataset.py"
field = xr.open_dataset(os.path.join(current_dir, '../save_data/fields_data/field_L3SWOT_rawSSHonly.nc'))

if verbose:
    print('\n1. SSH filtering\n')
ssh_filtered = L3Field_filtering.ssh_filtering(field, temporal_sigma=temporal_sigma, spatial_sigma=spatial_sigma)
field['ssh'] = field.ssh.copy(data= ssh_filtered)

# compute geostrophic and cyclogeostrophic velocities
if verbose:
    print('\n2. SSC computation\n')
field_UV = add_velocities(field)
field = add_Tgrid_velocities(field_UV, replace=True) # interpolate the U/V grid velocities on the T grid

field.attrs['Field_filter'] = '{}h_{}km_filter'.format(temporal_sigma, spatial_sigma)
field.to_netcdf('../save_data/fields_data/field_L3SWOT_{}.nc'.format(field.attrs['Field_filter']))


# --- DRIFTERS PROCESSING
if verbose:
    print('\n3. Trajectories processing\n')

for id_file in range(len(FILES_SVP)):
    ds = xr.open_dataset(os.path.join(L2_DIR, FILES_SVP[id_file]))
    num_idx = ds.sizes['id']
    del ds

    for idx_id in range(num_idx):
        if verbose:
            print('Processing file {:02d}/{} id {:02d}/{}'.format(id_file, len(FILES_SVP)-1, idx_id, num_idx-1) )

        traj = open_one_traj(L2_DIR, FILES_SVP[id_file], idx_id=idx_id, L3_cleaning=True, padd_swath=0.25)

        if type(traj) == float : #open_one_traj returns np.nan if the trajectory is not in swath
            if verbose:
                print('traj not in swath\n')
        else:            
            time_span = ( traj.time.values.max() - traj.time.values.min() ) /1e9 /24 /3600
            if time_span > 5: #only long enough trajectories (5days)

                traj = traj.where(isnear_swath(traj.lon, traj.lat, dlon=0.30), drop=True)


                traj = Drifters_processing.one_trajectory_filtering_and_field_comparison(
                                                                                        traj=traj, 
                                                                                        field=field,
                                                                                        interp_func=interp_linear_velocity_field_L3, 
                                                                                        filtering_velocities=filtering_velocities,
                                                                                        low_pass_cutoff = low_pass_cutoff,
                                                                                        filtering_positions=filtering_positions,
                                                                                        gaussian_std = gaussian_std 
                                                                                        )

                traj = traj.where(isin_swath(traj.lon_filtered, traj.lat_filtered), drop=True)
                scenes = Drifters_processing.get_scenes_from_traj(traj)
                scenes.to_netcdf(os.path.join(path_save_data, '{}_file_{}_idxid_{}.nc'.format(NAME, id_file, idx_id)))

            elif verbose:
                print('traj too small')