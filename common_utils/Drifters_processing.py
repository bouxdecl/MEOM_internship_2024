

# --- Imports

import scipy
import scipy.signal

import numpy as np
import xarray as xr

import clouddrift as cd




# --- drifter filtering and field comparison

def one_trajectory_filtering_and_field_comparison(traj: xr.Dataset,
                                                  field: xr.Dataset,
                                                  interp_func, 
                                                  filtering=False, low_pass_cutoff= 1/(48*3600),
                                                  filter_positions = 'gaussian', gaussian_std=12*3600
                                                  ):
    '''
    Filters and interpolates trajectory data.

    Parameters:

    traj_raw : xr.Dataset
        Raw trajectory data with variables: time, lat, lon, u, v.

    field: xr.Dataset
        Field dataset with 
        dims= time, y, x
        coords= time, latitude, longitude
        data_vars= ssh

    DATAPRODUCT: str,
        Description of the dataset,
    interp_func : function
        Interpolation function to obtain field velocities.
    filtering : bool, optional
        Whether to apply filtering to positions and velocities (default is Fasle).
    low_pass_cutoff : float, optional
        Cutoff frequency for low-pass filtering in Hz (default is 1/(48*3600)).

    filter_positions : str, optional
        Method for filtering positions ('low_pass' or 'gaussian', default is 'gaussian').
    gaussian_width : float, optional
        Width for Gaussian filter in seconds (default is 12*3600).

    Returns:
    xr.Dataset
        Trajectory dataset with filtered positions, velocities, and interpolated field velocities.
    '''

    _traj = traj.copy(deep=True)

    time, lat, lon, u, v = _traj.time.values, _traj.lat.values, _traj.lon.values, _traj.u.values, _traj.v.values  
    dt = float(_traj.time[1] - _traj.time[0])*1e-9


    if not filtering:
        lat_filtered = lat
        lon_filtered = lon
        u_filtered   = u
        v_filtered   = v

    if filtering:
    # --- Velocities filtering
        U = u + 1j* v
        U_filtered = low_pass_complex(U, dt=dt, cutoff = low_pass_cutoff)
        
        u_filtered = U_filtered.real
        v_filtered = U_filtered.imag             
        

    # --- Position filtering
        if filter_positions == 'low_pass':
            X = lon + 1j* lat
            X_filtered = low_pass_complex(X, dt=dt, cutoff = low_pass_cutoff)

            lon_filtered = X_filtered.real
            lat_filtered = X_filtered.imag
            
        elif filter_positions == 'gaussian':
            lon_filtered = gaussian_filter(lon, dt=dt, gaussian_std=gaussian_std)
            lat_filtered = gaussian_filter(lat, dt=dt, gaussian_std=gaussian_std)
        

    # --- get field interpolated velocities
    u_geo_swot, v_geo_swot = interp_func(field, 'u_geos', 'v_geos', time, lat_filtered, lon_filtered)
    u_var_swot, v_var_swot = interp_func(field, 'u_var',  'v_var',  time, lat_filtered, lon_filtered)


    # --- Save filtered and interpolated data into the dataset
    _traj["lat_filtered"] = (["time"], lat_filtered)
    _traj["lon_filtered"] = (["time"], lon_filtered)
    
    _traj["u_filtered"]   = (["time"], u_filtered)
    _traj["v_filtered"]   = (["time"], v_filtered)
    
    _traj["u_geo_swot"]   = (["time"], u_geo_swot)
    _traj["v_geo_swot"]   = (["time"], v_geo_swot)
    
    _traj["u_var_swot"]   = (["time"], u_var_swot)
    _traj["v_var_swot"]   = (["time"], v_var_swot)
    
    _traj.attrs = traj.attrs
    _traj.attrs['filtering'] = '{}, low_pass_cutoff={}Hz, filter_positions={}, gaussian_std={}s'.format(filtering, str(low_pass_cutoff), filter_positions, gaussian_std) 

    return _traj





# --- Cut into chunks and add metrics 

def get_scenes_from_traj(traj, n_days=3, dt=np.timedelta64(30*60, "s") ):
    if not dt:
        dt = float(traj.time[1] - traj.time[0])*1e-9
    
    #find if the trajectory is in several parts, per example if it quit the SWOT swath and then go in
    row_size = cd.ragged.segment(traj.time.values.astype('datetime64[s]'), dt) 
    chunk_size = int(n_days / (dt / np.timedelta64(1, "D")))      #number of observations in a 3 day-scene

    def ragged_chunk(_arr: xr.DataArray | np.ndarray, _row_size: np.ndarray[int], _chunk_size: int) -> np.ndarray:
        return cd.ragged.apply_ragged(cd.ragged.chunk, _arr, _row_size, _chunk_size, align='middle')  # noqa

    chuncks = dict([(d, (["scene", "s_obs"], ragged_chunk(traj[d], row_size, chunk_size)))  
                    
                    for d in set(traj.data_vars).union({'time'})
                                                                                
                    ] )
    
    scenes = xr.Dataset(data_vars=chuncks)
    scenes.attrs = traj.attrs

    return scenes


def compute_scene_metric(u_drifter, v_drifter, u_field, v_field):
    U_drifter = u_drifter + 1j* v_drifter
    U_field   = u_field + 1j* v_field 

    n_points = np.nansum(~np.isnan(u_field))
    return np.nansum( np.abs(U_drifter - U_field) / np.abs(U_field) ) / n_points, n_points


def add_metrics(scenes):

    ds = scenes.copy(deep=True)

    metric_geo = np.empty(ds.sizes['scene'])
    n_points_geo = np.empty(ds.sizes['scene']) 

    metric_var = np.empty(ds.sizes['scene'])
    n_points_var = np.empty(ds.sizes['scene'])

    for i in range(ds.sizes['scene']):
        traj = ds.isel(scene=i)
        metric_geo[i], n_points_geo[i] = compute_scene_metric(traj.u, traj.v, traj.u_geo_swot, traj.v_geo_swot)
        metric_var[i], n_points_var[i] = compute_scene_metric(traj.u, traj.v, traj.u_var_swot, traj.v_var_swot)

    ds['metric_geo'] = (["scene"], metric_geo, {'metric': 'mean( L2(Udrift - Ufield) / L2(Udrift)'})
    ds['n_points_metric_geo'] = (["scene"], n_points_geo)  

    ds['metric_var'] = (["scene"], metric_var, {'metric': 'mean( L2(Udrift - Ufield) / L2(Udrift)'})
    ds['n_points_metric_var'] = (["scene"], n_points_var)

    ds['overall_metric_var'] = np.average(ds.metric_var.values, weights=ds.n_points_metric_var.values/sum(ds.n_points_metric_var.values))
    ds['overall_metric_geo'] = np.average(ds.metric_geo.values, weights=ds.n_points_metric_geo.values/sum(ds.n_points_metric_geo.values))

    return ds





# --- filters

def low_pass_complex(signal, dt, cutoff = 1/(48*3600), order=8 ):

    # filter param
    nyq = 0.5* 1/dt
    normalized_cutoff = cutoff / nyq

    b, a = scipy.signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # low pass and remove mean error
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
  
    return filtered_signal


def gaussian_filter(x, dt, gaussian_std=12*3600):
    '''
    apply gaussian filter on x, with padding. dt and gaussian_std in seconds
    '''
    # defines gaussian
    gaussian_window_width = 3*gaussian_std
    tt = np.arange(-gaussian_window_width, gaussian_window_width+dt, dt)
    
    gaussian = np.exp(-0.5 * (tt/gaussian_std)**2)
    gaussian = gaussian / np.sum(np.abs(gaussian))
    
    #padd and convolve
    padd = len(gaussian) //2 -1
    x_padded = np.pad(x, padd, mode='symmetric')

    return np.convolve(x_padded, gaussian, 'same')[padd:-padd]

