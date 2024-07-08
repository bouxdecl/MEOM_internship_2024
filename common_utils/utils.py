
# --- Imports

import os

import numpy as np
import xarray as xr

import jax.numpy as jnp
import jaxparrow as jpw

import scipy

# --- Global variables

BBOX_MED = [-5.8, 36.5, 30, 44.5]
BBOX_DRIFTERS = [0, 13.5, 36.5, 44]
BBOX_SWATH = [0, 9, 36.5, 44]

TIME_SPAN_L3SWOT = (np.datetime64('2023-03-30'), np.datetime64('2023-07-09'))


# SVP drifters
FILES_SVP = ['L2_svp_scripps_10min_lowess_30min_v0.nc',
    'L2_svp_ogs_1h_lowess_30min_v0.nc',
    'L2_svp_ogs_10min_lowess_30min_v0.nc',
    'L2_svp_shom_10min_lowess_30min_v0.nc',
    'L2_svp_scripps_1h_lowess_30min_v0.nc',
    'L2_svp_bcg_10min_lowess_30min_v0.nc']


# unused variables of drifters dataset (only lat, lon, u, v are used)
DROP_VARS_DRIFTERS = [
 'x',
 'y',
 'platform',
 'lonc',
 'latc',
 'ax',
 'ay',
 'Axy',
 'au',
 'av',
 'Auv',
 'X',
 'U',
 'gap_mask',
 'gaps',
 'start_time',
 'start_lon',
 'start_lat',
 'end_time',
 'end_lon',
 'end_lat',
 'end_reason']




# --- open SVP drifter file and L3_cleaning : takes only data compatible with SWOT 1day swath time & loc 
def open_one_traj(dir, file, idx_id, L3_cleaning: bool, padd_swath=0.5):
    """
    Open and preprocess a single trajectory from a NetCDF file.

    Parameters:
    -----------
    dir : str
        Directory path where the NetCDF file is located.
    file : str
        Name of the NetCDF file containing trajectory data.
    idx_id : int
        Index of the trajectory ID to select.
    L3_cleaning : bool
        Flag indicating whether to perform Level 3 cleaning.
    padd_swath : float, optional
        Padding in degrees to define the swath around the trajectory (default is 0.5).

    Returns:
    --------
    xarray.Dataset or np.nan
        Processed trajectory dataset if L3 cleaning is performed and trajectory is within swath,
        otherwise np.nan if the trajectory is never within the swath.
    """
    ds = xr.open_dataset(os.path.join(dir, file), drop_variables=DROP_VARS_DRIFTERS).isel(id=idx_id).dropna(dim='time', how='any', subset=['u'])
    if not L3_cleaning:
        return ds

    ds = ds.sel(time=slice(*TIME_SPAN_L3SWOT), drop=True)
    
    if np.all(~isnear_swath(ds.lon, ds.lat, dlon=padd_swath)): #trajectory is not in the swath
        return np.nan
    
    ds = ds.where(isnear_swath(ds.lon, ds.lat, dlon=padd_swath), drop=True)

    ds.coords['dt'] = float(ds.time[1] - ds.time[0]) /1e9
    ds.attrs = {}
    ds.attrs['traj_id'] = 'file={} ; idx_id={}'.format(file, idx_id)

    return ds



# --- FIELDS functions

def _add_one_Tgrid_velocity(ds, u, axis, padding, replace=True):
    u_t = jpw.tools.operators.interpolation(jnp.copy(ds[u].values), axis=axis, padding=padding)
    if replace:
        ds[u] = ds[u].copy(data=u_t)
    else:
        ds[str(u)+'_t'] = ds[u].copy(data=u_t)
    return ds

def add_Tgrid_velocities(ds, replace=False):

    ds = _add_one_Tgrid_velocity(ds, 'u_geos', axis=1, padding='left', replace=replace)
    ds = _add_one_Tgrid_velocity(ds, 'v_geos', axis=0, padding='left', replace=replace)

    ds = _add_one_Tgrid_velocity(ds, 'u_var', axis=1, padding='left', replace=replace)
    ds = _add_one_Tgrid_velocity(ds, 'v_var', axis=0, padding='left', replace=replace)

    return ds



def restrain_domain(ds, min_lon, max_lon, min_lat, max_lat):
    '''
    spatial restrain of a dataset with coords longitude, latitude 
    '''
    extend = (min_lon <= ds.longitude) & (ds.longitude <= max_lon) & (min_lat <= ds.latitude) & (ds.latitude <= max_lat)
    in_biscay = (ds.longitude <= -0.1462) & (ds.latitude >= 43.2744)
    in_blacksea = (ds.longitude >= 27.4437) & (ds.latitude >= 40.9088)
    
    mask = extend & ~(in_biscay | in_blacksea) # exclude biscay and black sea
    return ds.where(mask, drop=True)






# --- localization

def isin_swath(lon, lat):
    ''' 
    takes two longitude, latitude arrays
    return a bolean array, True if a point is in the SWOT 1 day-swath 
    (swath is approximated to be rectilign)
    '''
    x1, x2, x3, x4, y_down, y_up, x_up = 2.85, 3.43, 3.64, 4.32, 37.14, 43.76, 4.76
    pente = (x_up - x1) / (y_up - y_down) 
    lon_down = lon - pente * (lat - y_down)

    return np.logical_or(np.logical_and(x1 <= lon_down, lon_down <= x2),  np.logical_and(x3 <= lon_down, lon_down <= x4))

def isnear_swath(lon, lat, dlon=0.25):
    ''' 
    takes two longitude, latitude arrays and dlon="nearness" parameter
    return a bolean array, True if a point is near in the SWOT 1 day-swath 
    (swath is approximated to be rectilign)
    '''    
    x1, x2, x3, x4, y_down, y_up, x_up = 2.85, 3.43, 3.62, 4.20, 37.14, 43.76, 4.76
    pente = (x_up - x1) / (y_up - y_down) 
    lon_down = lon - pente * (lat - y_down)

    return np.logical_and(x1-dlon <= lon_down, lon_down <= x4+dlon)


# --- interpolation of unregular grid

def interp_linear_velocity_field_L3(field, u: str, v: str, time_vec, lat_vec, lon_vec, dlat=0.022, dlon=0.03):
    """
    Interpolate linearly velocity field components from a dataset, with neighborhood points (dlat, dlon).
    If less than 4 points are in the neighborhood, np.nan values are return. 

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset containing the velocity field components and the 
        coordinates `time`, `latitude` and `longitude`.
    u : str
        The name of the variable representing the u-component of the velocity 
        field in the dataset.
    v : str
        The name of the variable representing the v-component of the velocity 
        field in the dataset.
        
    time_vec : array-like
        A vector of time points at which to interpolate the velocity fields.
    lat_vec : array-like
        A vector of latitude points at which to interpolate the velocity fields.
    lon_vec : array-like
        A vector of longitude points at which to interpolate the velocity fields.

    dlat, dlon : radius of points in the neighborhood taken in the interpolation

    Returns:
    --------
    np.ndarray
        The interpolated values of the u-component of the velocity field at the 
        specified times, latitudes, and longitudes. 
    np.ndarray
        The interpolated values of the v-component of the velocity field at the 
        specified times, latitudes, and longitudes.
    """
    u_interp = np.empty(shape=time_vec.shape)
    v_interp = np.empty(shape=time_vec.shape)

    for i, (lon, lat, time) in enumerate(zip(lon_vec, lat_vec, time_vec)):

        ds = field.interp(time=time).copy(deep=True)
        neighboord_points = np.logical_and(np.abs(ds.latitude - lat) < dlat, np.abs(ds.longitude - lon) < dlon) 
        ds = ds.where(neighboord_points, drop=True)
        
        # get the number of point selected
        n_points = np.sum(~np.isnan(ds[u].values.flatten()))
        
        if n_points < 4:
            u_interp[i], v_interp[i] = np.nan, np.nan
        else:
            u_interp[i] = scipy.interpolate.griddata((ds.latitude.values.flatten(), ds.longitude.values.flatten()), ds[u].values.flatten(), (lat, lon))
            v_interp[i] = scipy.interpolate.griddata((ds.latitude.values.flatten(), ds.longitude.values.flatten()), ds[v].values.flatten(), (lat, lon))
    
    return u_interp, v_interp



def interp_closest_velocity_field_L3(ds, u: str, v: str, time_vec, lat_vec, lon_vec):
    '''
    Same as interp_linear_velocity_field_L3 function but with the closest point interpolation
    '''
    #find the closest x,y coords of the interpolation points 
    x_vec = np.empty_like(lon_vec)
    y_vec = np.empty_like(lat_vec)

    for i, (lon, lat) in enumerate(zip(lon_vec, lat_vec)):
        abslat = np.abs(ds.latitude-lat)
        abslon = np.abs(ds.longitude-lon)
        c = np.maximum(abslon, abslat)

        min_idx = np.where(c == np.min(c))
        y_vec[i], x_vec[i] = min_idx[0][0], min_idx[1][0]

    #create indexers DataArrays 
    x_idx = xr.DataArray(x_vec)
    y_idx = xr.DataArray(y_vec)
    time_idx = xr.DataArray(time_vec)

    #xarray interpolate
    interped = ds.interp(x=x_idx, y=y_idx, time=time_idx)

    return interped[u].values, interped[v].values





# --- utils numpy & datetime...

def normalize_angle(angles):
    # Normalize the angles to be within -360 to 360 degrees
    angles = np.mod(angles, 360)
    
    # Adjust angles to be within -180 to 180 degrees
    angles = np.where(angles > 180, angles - 360, angles)
    angles = np.where(angles < -180, angles + 360, angles)
    
    return angles


def remove_outliers(data, density, zscore=3):
    data, idx_outliers = replace_outliers_with_nan_zscore(data, zscore)
    density[idx_outliers] = np.nan
    return data, density


def replace_outliers_with_nan_zscore(data, threshold=3):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    z_scores = (data - mean) / std
    outliers = np.abs(z_scores) > threshold
    data[outliers] = np.nan
    return data, outliers


def get_mean_datetime(datetime_array):
    # Filter out NaT values
    valid_times = datetime_array[~np.isnat(datetime_array)]

    if valid_times.size > 0:
        # Convert to timedelta in seconds relative to the reference point (e.g., 1970-01-01)
        reference_time = np.datetime64('1970-01-01')
        timedeltas = (valid_times - reference_time).astype('timedelta64[s]').astype(np.float64)

        # Compute the mean of timedeltas
        mean_timedelta = np.mean(timedeltas)

        # Convert mean timedelta back to datetime
        mean_datetime = reference_time + np.timedelta64(int(mean_timedelta), 's')
        return mean_datetime

    else:
        print("No valid datetime values found.")
        return None
    






# --- drifters file selection functions

def file_selection_mediterranean(files):
    # only mediterranean (not containing 'uwa') 

    selected_files = []
    for file in files:
         if not file.count("uwa"):
            selected_files.append(file)
    return selected_files

def file_selection_by_method(files, method: str):
    # files from one interpolation method: 'variationnal' or 'lowess'

    selected_files = []
    for file in files:
         if file.count(method):
            selected_files.append(file)
    return selected_files

def file_selection_by_sampling(files, sampling: str):
    # files from one smooth L2 sampling: '10min', '30min', '1h' at the end of the file_name

    selected_files = []
    for file in files:
         if file[-11:].count(sampling):
            selected_files.append(file)
    return selected_files
