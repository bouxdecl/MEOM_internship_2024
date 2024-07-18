
# --- Imports

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


from utils import *


'''

Plots functions
###############

plot_histogram()

plot_2d_polar_hist() -> comparison between field and drifter vectors
  
welch()              -> PSD plots

plot_scenes()        -> plot one scene with all the data (SSH background, geo/cyclo fields and trajectory)

'''






# --- Histogram plots

def plot_histogram(data, color='grey', maxline=True, meanline=True):
    mean = data.mean()
    std = data.std()

    # Create the histogram
    counts, bins, patches = plt.hist(data, bins=50, alpha=0.6, color=color, edgecolor='black')

    if maxline:
        # Find the bin with the maximum count
        max_count = np.max(counts)
        max_count_index = np.argmax(counts)
        max_count_bin_center = (bins[max_count_index] + bins[max_count_index + 1]) / 2

        # Add a line for the maximum bin count
        plt.axvline(max_count_bin_center, color='r', linestyle='dashed', linewidth=1, label=f'max: {max_count_bin_center:.2f}')

    if meanline:
        # Add a line for the mean
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label=f'mean: {mean:.2f}')

    plt.legend()



def plot_2d_polar_hist(r, theta, rmax=None, vmax=None, add_title=None, plot_1sigma=False):
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})

    if rmax is None:
        rmax = r.max()
        
    # Define bin edges for r and theta
    res = 50
    r_bins = np.linspace(0, rmax, res)
    theta_bins = np.linspace(theta.min(), theta.max(), res)

    # Create 2D histogram using numpy's histogram2d function
    counts, theta_edges, r_edges = np.histogram2d(theta, r, bins=(theta_bins, r_bins))

    # Plot using pcolormesh
    if vmax is None:
        vmax = counts.max()
    h = ax.pcolormesh(theta_bins, r_bins, counts.T, cmap='plasma', zorder=1, vmax=vmax)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=-1, axes_class=plt.Axes)
    cbar = plt.colorbar(h, cax=cax)

    # Add the mean point
    ax.scatter(theta.mean(), r.mean(), marker='o', s=3, c='red', label='Weighted Mean', zorder=2)

    if plot_1sigma:
        # Add 1-sigma limit
        theta_grid, r_grid = np.meshgrid(theta_bins, r_bins)

        gaussian = np.exp(-((theta_grid - theta.mean())**2 / (2 * theta.std()**2) +
                        (r_grid - r.mean())**2 / (2 * r.std()**2)))

        ax.contour(theta_grid, r_grid, gaussian, levels=[0.6065], colors='green', linestyles='--', linewidths=1, zorder=3, label='1-sigma Contour')


    # Customize plot
    title = '2D Histogram of Field velocity normalized by drifter velocity'
    if add_title:
        title += '\n' + add_title 

    ax.set_title(title, fontsize=10)
    ax.set_theta_zero_location('N')

    ax.set_rmax(rmax)
    ax.set_rticks(np.arange(0, rmax+1, 1), )  
    ax.set_rlabel_position(285)
    ax.grid(True)

    ax.legend(bbox_to_anchor=(1,1), fontsize=5)



# --- PSD welch plot

def welch(sig_list, name_list, dt, freq_list=[], nfenetre = None, save=None, save_path=None):

    if nfenetre is None:
        nfenetre = len(sig_list[0]) //3

    fig, ax = plt.subplots()

    results = []
    for sig, lb in zip(sig_list, name_list):
        frequencies, psd = scipy.signal.welch(sig, fs=1/dt, nperseg = nfenetre, noverlap=nfenetre//4)
        ax.semilogy(frequencies, psd, label=lb, zorder=4)
        results.append((frequencies, psd))

    for hour, label, color in freq_list:
        ax.axvline(-1/(hour*3600), c=color, ls='--', lw=1.1, label=label)
        
    ax.axvline(-1/(19*3600), c='black', ls='--', lw=1.5, label='19h (inertial freq)', zorder=1)

    xlim = 10**(-4)
    ax.set_xlim(-0.5e-4, 0.4e-4)
    ax.set_ylim(0.7, 5e4)

    ax.set_title('PSD of the complex velocity (Welch\'s Method)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V^2/Hz)')
    ax.grid()
    ax.legend()

    # Set the x-axis to display in powers of ten
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    if save and save_path:
        save_name = 'PSD'+ name_list[1] + '.png'
        plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight', dpi = 300)

    return results






# --- Spatial plots : plot_scenes


def padd_bbox(bbox, padd):
    mlon, Mlon, mlat, Mlat = bbox
    return tuple([mlon-padd, Mlon+padd, mlat-padd, Mlat+padd])


def plot_scene(scene, bkgd_field, field_vec='both', points_hours: int=12, time_point: int=0, 
               
               padd=0.06, print_metrics=False, add_title=None, plot: bool=True, save_name=None, save_dir=None):

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
        title += '\nMetric: geo={:.2f}, cyclo={:.2f}'.format(ds.overall_metric_geo.values, ds.overall_metric_var.values)
    if add_title:
        title += add_title
    ax.set_title(title)
    ax.legend(fontsize=8)

    if save_name and save_dir:
        plt.savefig(os.path.join(save_dir, '{}.png'.format(save_name)), bbox_inches='tight', dpi = 300)