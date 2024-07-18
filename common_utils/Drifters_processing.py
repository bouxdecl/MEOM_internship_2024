
# --- Imports

import scipy
import scipy.signal

import numpy as np
import xarray as xr

import clouddrift as cd


'''

Drifters processing functions
#############################


Filtering (positions and/or velocities):
    one_trajectory_filtering()
    one_trajectory_filtering_and_field_comparison()

Cut one trajectory into scenes:
    get_scenes_from_traj()

    
Compute scene metrics for one scene or for a dataset or scenes:
    add_metrics()     
    all_add_metrics() 

Filters:
    low_pass_complex
    gaussian_filter

'''






# --- drifter filtering and field comparison


def one_trajectory_filtering(traj: xr.Dataset,                                                
                             filtering_velocities=True, low_pass_cutoff= 1/(48*3600),
                             filtering_positions = False, filter_positions='gaussian', gaussian_std=12*3600
                             ):

    _traj = traj.copy(deep=True)

    time, lat, lon, u, v = _traj.time.values, _traj.lat.values, _traj.lon.values, _traj.u.values, _traj.v.values  
    dt = float(_traj.time[1] - _traj.time[0])*1e-9


    # --- Velocities filtering
    if filtering_velocities:
        U = u + 1j* v
        U_filtered = low_pass_complex(U, dt=dt, cutoff = low_pass_cutoff)
        
        u_filtered = U_filtered.real
        v_filtered = U_filtered.imag             
    else:
        u_filtered   = u
        v_filtered   = v


    # --- Position filtering
    if filtering_positions:
        if filter_positions == 'low_pass':
            X = lon + 1j* lat
            X_filtered = low_pass_complex(X, dt=dt, cutoff = low_pass_cutoff)

            lon_filtered = X_filtered.real
            lat_filtered = X_filtered.imag
            
        elif filter_positions == 'gaussian':
            lon_filtered = gaussian_filter(lon, dt=dt, gaussian_std=gaussian_std)
            lat_filtered = gaussian_filter(lat, dt=dt, gaussian_std=gaussian_std)
    else:
        lat_filtered = lat
        lon_filtered = lon

    # --- Save filtered and interpolated data into the dataset
    _traj["lat_filtered"] = (["time"], lat_filtered)
    _traj["lon_filtered"] = (["time"], lon_filtered)
    
    _traj["u_filtered"]   = (["time"], u_filtered)
    _traj["v_filtered"]   = (["time"], v_filtered)

    _traj.attrs = traj.attrs
    _traj.attrs['filtering'] = 'filtering_velocities={}, low_pass_cutoff={}Hz, filtering_positions={}, gaussian_std={}s'.format(filtering_velocities,  str(low_pass_cutoff), filtering_positions, gaussian_std) 

    return _traj



def one_trajectory_filtering_and_field_comparison(traj: xr.Dataset,
                                                  field: xr.Dataset,
                                                  interp_func, 
                                                  filtering_velocities=True, low_pass_cutoff= 1/(48*3600),
                                                  filtering_positions=False, filter_positions = 'gaussian', gaussian_std=12*3600
                                                  ):

    _traj = one_trajectory_filtering(traj=traj,                                                
                                    filtering_velocities=filtering_velocities, low_pass_cutoff= low_pass_cutoff,
                                    filtering_positions = filtering_positions, filter_positions=filter_positions, gaussian_std=gaussian_std
                                    )

    time, lat_filtered, lon_filtered = _traj.time.values, _traj.lat_filtered.values, _traj.lon_filtered.values

    # --- get field interpolated velocities at the drifters positions/time
    u_geo_swot, v_geo_swot = interp_func(field, 'u_geos', 'v_geos', time, lat_filtered, lon_filtered)
    u_var_swot, v_var_swot = interp_func(field, 'u_var',  'v_var',  time, lat_filtered, lon_filtered)

    # --- Save filtered and interpolated data into the dataset
    _traj["u_geo"]   = (["time"], u_geo_swot)
    _traj["v_geo"]   = (["time"], v_geo_swot)
    
    _traj["u_var"]   = (["time"], u_var_swot)
    _traj["v_var"]   = (["time"], v_var_swot)
    
    return _traj







# --- Cut into 3-day scenes 

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




# --- Add field/drifters comparison metrics

def add_metrics(one_scene):

    ds = one_scene.copy(deep=True)
    ds['n_points_compa'] = np.nansum(~np.isnan(ds.u_geo_swot))

    U_drifter = ds.u_filtered + 1j* ds.v_filtered


    for U_field, name in zip([ds.u_geo_swot+1j*ds.v_geo_swot , ds.u_var_swot+1j*ds.v_var_swot], ['geo', 'var']):

        metric = np.abs(U_drifter - U_field)
        metric_norma = metric / np.abs(U_drifter)

        # clip too high values
        metric_norma = metric_norma.clip(max=5) 

        angle  = np.angle(U_field / U_drifter)
        norm_ratio = np.abs(U_field) / np.abs(U_drifter) 

        ds['angle_'+name] = ds.u_filtered.copy(data= angle)
        ds['norm_ratio_'+name] = ds.u_filtered.copy(data= norm_ratio)

        ds['metric_'+name] = ds.u_filtered.copy(data= metric)
        ds['metric_norma_'+name] = ds.u_filtered.copy(data= metric_norma)

    return ds



def all_add_metrics(all_scenes):

    DS = all_scenes.copy(deep=True)

    for name in ['geo', 'var']:
        angle = np.empty(DS.u_geo_swot.values.shape)
        norm_ratio = np.empty(DS.u_geo_swot.values.shape)
        metric = np.empty(DS.u_geo_swot.values.shape)
        metric_norma = np.empty(DS.u_geo_swot.values.shape)
        n_points_compa = np.empty(DS.sizes['scene'])

        for i in range(DS.sizes['scene']):
            ds = DS.isel(scene=i)

            n_points_compa[i] = np.nansum(~np.isnan(ds.u_geo_swot.values))

            U_drifter = ds.u_filtered.values + 1j* ds.v_filtered.values
            U_field = ds['u_'+name+'_swot'].values+1j*ds['v_'+name+'_swot'].values

            metric[i] = np.abs(U_drifter - U_field)
            
            metric_norma[i] = np.abs(U_drifter - U_field) / np.abs(U_drifter)

            angle[i]  = np.angle(U_field / U_drifter)
            norm_ratio[i] = np.abs(U_field) / np.abs(U_drifter) 

        
        DS['n_points_compa'] = xr.DataArray(data= n_points_compa, dims='scene')
        
        DS['angle_'+name] = DS.u_filtered.copy(data= angle)
        DS['norm_ratio_'+name] = DS.u_filtered.copy(data= norm_ratio)

        DS['metric_'+name] = DS.u_filtered.copy(data= metric)
        
        metric_norma[metric_norma > 5] = 5
        DS['metric_norma_'+name] = DS.u_filtered.copy(data= metric_norma)

    return DS




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

