
# --- Imports

import numpy as np
import matplotlib.pyplot as plt


import cartopy.geodesic as geod
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic


from utils import *

# --- Plot 2D

def padd_bbox(bbox, padd):
    mlon, Mlon, mlat, Mlat = bbox
    return tuple([mlon-padd, Mlon+padd, mlat-padd, Mlat+padd])




def plot_scene(scene, bkgd_field, field_vec='both', points_hours: int=12, time_point: int=0, 
               
               padd=0.06, print_metrics=True, plot: bool=True, save_name=None, save_dir=None):

    """
    Plots a drifter trajectory scene with background field data, vectors, and specific time points.

    Parameters:
    -----------
    scene : xarray.Dataset
        The dataset containing drifter trajectory and associated data (positions, velocities, same but filtered, field vectors interpolation, metrics...)
    bkgd_field : xarray.Dataset
        The background SSH field dataset.
    field_vec : str, optional
        The type of field vectors to plot. Options are 'geo', 'cyclo', or 'both' (default is 'both').
    points_hours : int, optional
        The interval in hours for marking points on the trajectory (default is 12).
    time_point : int, optional
        The specific time point to highlight on the trajectory (default is 0).
    padd : float, optional
        The padding for the plot extent (default is 0.06).
    print_metrics : bool, optional
        Whether to print metric values in the plot title (default is True).
    plot : bool, optional
        Whether to display the plot interactively (default is True).
    save_name : str, optional
        The name of the file to save the plot (default is None, meaning the plot will not be saved).
    path_save : str, optional
        The directory path where the plot should be saved (default is None).

    Returns:
    --------
    None
    """


    ds = scene

    plt.ioff()
    if plot:
        plt.ion()
    
    # --- Figure parameters
    fig = plt.figure(tight_layout=True, figsize=(8, 8))
    crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    gl = ax.gridlines(draw_labels=True,)
    ax.add_feature(cfeature.LAND,)
    scale=3.5e-5 *4
    width=2.5e-3

    bbox = (np.nanmin(ds.lon_filtered.values), np.nanmax(ds.lon_filtered.values), np.nanmin(ds.lat_filtered.values), np.nanmax(ds.lat_filtered.values) )

    padd = 0.06
    ax.set_extent(padd_bbox(bbox, padd), crs=crs)


    # --- Drifter Trajectory
    ax.plot(ds.lon, ds.lat,transform=crs, color='grey', label='drifter trajectory', lw=0.5, zorder=0)
    ax.plot(ds.lon_filtered, ds.lat_filtered, transform=crs, color='red', label='filtered trajectory', zorder=1)

    #points to track time
    i = points_hours*2 # 30min sampling

    points_day = (ds.lon_filtered[::i], ds.lat_filtered[::i])
    ax.scatter(*points_day, transform=crs, color='black', s=7, zorder=2, label='{}h points'.format(points_hours))
    
    #plot in yellow the point when ssh is plotted  
    ax.scatter(ds.lon_filtered[int(i*time_point)], ds.lat_filtered[int(i*time_point)], transform=crs, color='blue', s=10, zorder=3, label='SSH TIME point')


    # --- Drifters Vectors and field interpolation
    ax.quiver(ds.lon_filtered, ds.lat_filtered, ds.u_filtered, ds.v_filtered, 
            angles='xy', scale_units='xy', scale=scale, width=width/2, transform=crs,
            color='black', label='filtered velocities')

    if field_vec == 'geo':
        ax.quiver(ds.lon_filtered, ds.lat_filtered, ds.u_geo_swot, ds.v_geo_swot, 
                angles='xy', scale_units='xy', scale=scale, width=width/2, transform=crs,
                color='red', label='geo interp')

    if field_vec == 'cyclo':
        ax.quiver(ds.lon_filtered, ds.lat_filtered, ds.u_var_swot, ds.v_var_swot, 
                angles='xy', scale_units='xy', scale=scale, width=width/2, transform=crs,
                color='red', label='cyclo interp')

    if field_vec == 'both':
        ax.quiver(ds.lon_filtered, ds.lat_filtered, ds.u_var_swot, ds.v_var_swot, 
                angles='xy', scale_units='xy', scale=scale, width=width/2, transform=crs,
                color='red', label='cyclo interp')


    # --- Underground field

    time_ssh = ds.time.values[int(i*time_point)]


    field = bkgd_field.interp(time=time_ssh)
    field = restrain_domain(field, *padd_bbox(bbox, padd+0.022))

    # SSH field    
    ssh_field = ax.contourf(field.longitude, field.latitude, field.ssh, transform=crs, alpha=0.35, label='SSH (swot)')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=1, axes_class=plt.Axes)
    cbar = plt.colorbar(ssh_field, cax=cax)

    # Velocity field
    if field_vec == 'geo':
        ax.quiver(field.longitude.values, field.latitude.values, field.u_var.values, field.v_var.values, label='vect_geo', angles='xy', scale_units='xy', scale=scale, width=width, transform=crs, alpha=0.2)

    elif field_vec == 'cyclo':
        ax.quiver(field.longitude.values, field.latitude.values, field.u_geos.values, field.v_geos.values, label='vect_cyclo', angles='xy', scale_units='xy', scale=scale, width=width/1.5, transform=crs, alpha=0.4)
    
    elif field_vec == 'both':
        ax.quiver(field.longitude.values, field.latitude.values, field.u_var.values, field.v_var.values, label='vect_geo', angles='xy', scale_units='xy', scale=scale, width=width, transform=crs, alpha=0.2)
        ax.quiver(field.longitude.values, field.latitude.values, field.u_geos.values, field.v_geos.values, label='vect_cyclo', angles='xy', scale_units='xy', scale=scale, width=width/1.5, transform=crs, alpha=0.4)

    title = 'One 3-day drifter scene'
    if print_metrics:
        title += '\nMetric: geo={:.2f}, cyclo={:.2f}'.format(ds.metric_geo.values, ds.metric_var.values)
    ax.set_title(title)
    ax.legend(fontsize=8)

    if save_name and save_dir:
        plt.savefig(os.path.join(save_dir, '{}.png'.format(save_name)), bbox_inches='tight', dpi = 300)