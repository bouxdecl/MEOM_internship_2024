

'''

Input files with bash command :
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/REPROCESSING_L3_KARIN_V1.0_CALVAL/karin.tar.gz
tar -xvf karin.tar.gz



Standalone file use to produce a field dataset
############################################## 

- Load files of 1-day SWOT orbit   " SWOT_L3_LR_SSH_Expert_*_003_2023*T*_2023*T*_v0.3.nc "
- Remove unused variables
- save a single field dataset with all times


'''


# --- Imports

import sys
import os
import glob

import numpy as np
import xarray as xr
import jax.numpy as jnp

#own modules import
current_dir = os.path.dirname(os.path.abspath(__file__))
common_utils_dir = os.path.join(current_dir, 'common_utils')
sys.path.append(common_utils_dir)
import utils


# --- function

drop_var = [
 'mdt',
 'quality_flag',
 'ocean_tide',
 'ssha_noiseless',
 'ssha_unedited',
 'mss',
 'dac',
 'calibration',
 'ugos'
 'vgos'
 'longitude_nadir',
 'latitude_nadir',
 'ugosa',
 'vgosa',
 'sigma0',
 'i_num_line',
 'i_num_pixel',
 'cross_track_distance',
 'duacs_xac',
]



def get_all_times_dataset(DATAPRODUCT, files, bbox, drop_var, verbose=False):
    """
    Process multiple files to create a dataset with averaged times and domain-restrained SSH data.

    Iterates through a list of files, opens each one as an xarray Dataset after dropping specified variables,
    restrains the domain using predefined bounding box coordinates, calculates the average time for each file,
    and constructs a final xarray Dataset with SSH data, time coordinates, and geographic mesh.

    Parameters:
    DATAPRODUCT : str, attribute of the final dataset to identify it
    files : list of str
        List of file paths to be processed.
    bbox : tuple (minlon, maxlon, minlat, maxlat) to restrain the domain
    drop_var : list of str
        List of variables to be dropped from each dataset upon opening.

    Returns:
    xarray.Dataset
        Dataset containing SSH data with dimensions 'time', 'y', and 'x'. Each time slice corresponds to the
        averaged time of the data from each file processed.
    """

    dataset_results = None

    first_loop = True
    for i, file in enumerate(files): 
        if verbose:
            print('processing file {:02d} over {}'.format(i, len(files)))

        #load domain data
        ds = xr.open_dataset(file, drop_variables=drop_var)
        ds = utils.restrain_domain(ds, *bbox)
        
        #get the average time for this file
        mean_time = utils.get_mean_datetime(ds.time.values)
        time_array = np.array([mean_time], dtype='datetime64[ns]')

        #get ssh
        lat_mesh, lon_mesh, ssh = jnp.copy(ds.latitude.values), jnp.copy(ds.longitude.values), jnp.copy(ds.ssha.values)
        
        
        # Construct a Dataset for the current time slice
        results_one_time = xr.Dataset(
            data_vars=dict(
                ssh=(["time", "y", "x"], np.expand_dims(ssh, axis=0))
            ),
            coords=dict(
                time=time_array,
                latitude=(["y", "x"], lat_mesh),
                longitude=(["y", "x"], lon_mesh)
            ),
            attrs=dict(
                DATAPRODUCT=DATAPRODUCT,
                gridtype_adt='T grid : (y, x)'
            )
        )
        
        
        if first_loop:
            dataset_results = results_one_time
            first_loop = False
        else:
            dataset_results = xr.concat([dataset_results, results_one_time], dim="time")

    return dataset_results




if __name__=='__main__':

    ###
    DATAPRODUCT = 'L3SWOT_rawSSHonly'
    path_L3swot  = '/home/bouxdecl/Documents/data/FIELDS/L3_data/karin'
    path_save_data = '../save_data/fields_data/'
    ###

    files = sorted(glob.glob(os.path.join(path_L3swot, 'SWOT_L3_LR_SSH_Expert_*_003_2023*T*_2023*T*_v0.3.nc')))


    L3SWOT_rawSSHonly = get_all_times_dataset(DATAPRODUCT=DATAPRODUCT, files=files, bbox=utils.BBOX_SWATH, drop_var=drop_var)
    L3SWOT_rawSSHonly.to_netcdf(os.path.join(path_save_data, 'field_{}.nc'.format(DATAPRODUCT)))