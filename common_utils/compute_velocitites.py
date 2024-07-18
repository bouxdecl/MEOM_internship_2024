
import numpy as np
import xarray as xr
import jax.numpy as jnp
import optax
import jaxparrow as jpw

'''
Add geos and cyclo velocity fields to a SSH field with dims = (time, y, x) and coords = (lat, lon) by using jaxparrow package
'''


def add_velocities(SSH_field):

    def compute_geos_and_var(ssh, lat_mesh, lon_mesh):
        mask = jnp.isnan(ssh)
        
        '''
        # Calculate geostrophic velocities
        u_geos, v_geos = jpw.geostrophy(ssh, lat_mesh, lon_mesh, mask=mask, return_grids=False)
        '''

        # Define optimizer
        lr_scheduler = optax.exponential_decay(1e-2, 200, .5)  # decrease the learning rate
        optim = optax.sgd(learning_rate=lr_scheduler)  # basic SGD works nicely
        optim = optax.chain(optax.clip(1), optim)  # prevent updates from exploding
        
        # Calculate velocities
        u_var, v_var, u_geos, v_geos = jpw.cyclogeostrophy(ssh, lat_mesh, lon_mesh, mask, optim=optim, return_geos=True, return_grids=False, return_losses=False)
        
        return u_geos, v_geos, u_var, v_var


    u_geos, v_geos, u_var, v_var = xr.apply_ufunc(
    compute_geos_and_var,
    SSH_field.ssh,
    SSH_field.latitude,
    SSH_field.longitude,
    input_core_dims=[['y', 'x'], ['y', 'x'], ['y', 'x']],
    output_core_dims=[['y', 'x'], ['y', 'x'], ['y', 'x'], ['y', 'x']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[SSH_field.ssh.dtype, SSH_field.ssh.dtype, SSH_field.ssh.dtype, SSH_field.ssh.dtype]
    )

    Field_dataset = SSH_field.copy(deep=True)
    Field_dataset['u_geos'] = u_geos
    Field_dataset['v_geos'] = v_geos
    Field_dataset['u_var'] = u_var
    Field_dataset['v_var'] = v_var

    return Field_dataset