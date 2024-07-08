
# --- Imports

import numpy as np
import xarray as xr

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


'''

Filtering functions for unregular grid fields (use on L3 SWOT Expert v0.3 files)
################################################################################


ds : xr.dataset with dims = ['time', 'y', 'x'], 
    
    y and x are the num_lines and num_pixels dimensions of SWOT_L3_LR_SSH_Expert_*_v0.3.nc files


ssh_filtering : return the dataset with the ssh filtered by gaussian with a given temporal standart deviation in HOURS and spatial standart deviation in KM 

'''



# --- main function
def ssh_filtering(ds, temporal_sigma=None, spatial_sigma=None):
    """
    Filter sea surface height (SSH) data using Gaussian filters.

    Applies temporal and/or spatial Gaussian filters to the SSH data in the given dataset.

    Parameters:
    ds : xarray.Dataset
        Dataset containing SSH data with dimensions 'time', 'latitude', 'longitude'.
    temporal_sigma : float or None, optional
        Standard deviation of the Gaussian filter for temporal filtering. If None, no temporal filtering is applied.
        units : hours
    spatial_sigma : float or None, optional
        Standard deviation of the Gaussian filter for spatial filtering. If None, no spatial filtering is applied.
        units : km
    Returns:
    xarray.Dataset
        Dataset with filtered SSH data. The 'ssh' variable in the dataset is updated with the filtered values.
    """

    filtered_data = ds.ssh.copy(deep=True)

    # Temporal filter
    if temporal_sigma:
        sanitize_data = ds.copy(deep=True)
        nan_mask = np.isnan(sanitize_data['ssh'])
        sanitize_data = sanitize_data.fillna(0)  # fill NaN with 0 to keep the same shape after Gaussian filtering

        filtered_data = xr.apply_ufunc(
            apply_time_gaussianfilter,
            sanitize_data.time,
            sanitize_data.ssh,
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['time']],
            kwargs={'sigma': temporal_sigma},
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
        )

        filtered_data = xr.where(nan_mask, np.nan, filtered_data)


    # Spatial filter
    if spatial_sigma:
        filtered_data = xr.apply_ufunc(
            apply_space_gaussianfilter,
            ds.longitude,
            ds.latitude,
            filtered_data,
            input_core_dims=[['y', 'x'], ['y', 'x'], ['y', 'x']],
            output_core_dims=[['y', 'x']],
            kwargs={'sigma': spatial_sigma},
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
        )

    return filtered_data



# --- auxiliary function

def apply_time_gaussianfilter(unregular_times, data, sigma):
    '''
    1d gaussian convolution with unregular timesteps

    Parameters:
        unregular_times ndarray, dtype = datetime64[ns]
        data (1D ndarray)
        sigma (float): standart deviation of the gaussian filter, units=hours
    '''

    # Time differences Matrix
    times = unregular_times.astype(np.float64) / 1e9 /3600   #hours : same unit as sigma
    time_diffs = np.abs(times[:, np.newaxis] - times)

    # Gaussian kernel
    gaussian_kernel = np.exp(-time_diffs**2 / (2.0 * sigma**2)) / np.sqrt(2.0 * np.pi * sigma**2)

    # Apply the Gaussian filter
    smoothed_data = np.sum(gaussian_kernel * data[:, np.newaxis], axis=0) / np.sum(gaussian_kernel, axis=0)

    return smoothed_data


def apply_space_gaussianfilter(lon, lat, ssh, sigma):
    """Apply a gaussian spatial_filter by interpolating on a regular grid

    Parameters:
        lon (2D ndarray): unregular 2D meshgrid of longitudes
        lat (2D ndarray): unregular 2D meshgrid of latitudes
        ssh (2D ndarray): unregular 2D meshgrid of ssh
        sigma (float or list): standart deviation of gaussian filter, units = km
    Returns:
        smoothed_ssh (2D ndarray): filtered ssh interpolated on th original unregular 2D meshgrid
    """    

    yx_shape = lat.shape

    latitudes = lat.flatten()
    longitudes = lon.flatten()
    ssh_values = ssh.flatten() 

    # Define the regular grid for interpolation
    n_lon, n_lat = yx_shape
    n_lon *= 3
    n_lat *= 3

    lat_min, lat_max = np.nanmin(latitudes), np.nanmax(latitudes)
    lon_min, lon_max = np.nanmin(longitudes), np.nanmax(longitudes)

    lat_regular, dlat = np.linspace(lat_min, lat_max, n_lat, retstep=True)
    lon_regular, dlon = np.linspace(lon_min, lon_max, n_lon, retstep=True)
    lon_grid, lat_grid = np.meshgrid(lon_regular, lat_regular)

    # convert the filter width in pixels for latitude, longitude
    km_per_latpx = dlat/360 *6400*2*np.pi
    km_per_lonpx = dlon/360 *6400*np.sin(np.nanmean(lat))*2*np.pi

    sigma_px = [sigma/km_per_latpx, sigma/km_per_lonpx]

    # Interpolate SSH values onto the regular grid
    ssh_regular = griddata((latitudes, longitudes), ssh_values, (lat_grid, lon_grid), method='linear')

    # Apply 2D Gaussian filter to the interpolated SSH field
    smoothed_ssh_regular = apply_gaussian_filter_preserve_shape(ssh_regular, sigma=sigma_px)

    # Interpolate smoothed SSH data back to the original irregular grid
    smoothed_ssh_irregular = griddata((lat_grid.flatten(), lon_grid.flatten()), smoothed_ssh_regular.flatten(), 
                                    (latitudes, longitudes), method='linear')

    smoothed_ssh = smoothed_ssh_irregular.reshape(yx_shape)
    return smoothed_ssh



def apply_gaussian_filter_preserve_shape(img, sigma):
    """
    Apply Gaussian filter to an image while preserving NaN values.

    Parameters:
    img : numpy.ndarray
        Input image array where NaN values represent undefined or missing data.
    sigma : float or list
        Standard deviation of the Gaussian filter, list if different along dimensions.
        Units = pixel

    Returns:
    numpy.ndarray
        Filtered image array with NaN values preserved.
    """
    nan_mask = np.isnan(img)
    filled_img = np.where(nan_mask, 0, img)
    mean_value = np.nanmean(img)
    filled_img[nan_mask] = mean_value

    filtered_img = gaussian_filter(filled_img, sigma=sigma)

    filtered_img[nan_mask] = np.nan

    return filtered_img