import jax
import jax.numpy as jnp
import jaxopt
from jaxopt import prox

@jax.jit
def chi_square(mod_flux, obs_flux, obs_err, train_mask=None):
    residuals = (obs_flux - mod_flux) / obs_err
    return jnp.sum((residuals**2) * train_mask)


@jax.jit
def max_entropy(rel_map, grid, hp):
    areas = grid[2][None, :]
    m = hp['def_bright']
    b = jnp.clip(rel_map, 1e-6, hp['max_bright'])
    entropy = jnp.sum(areas * (b * jnp.log(b / m) - b + m))
    return entropy

@jax.jit
def l2_norm(b_map, grid, phases):
import jax
import jax.numpy as jnp

@jax.jit
def calc_l2_penalty(b_map, grid, phases):
    if b_map.ndim == 1:
        b_map = b_map.reshape(1, -1)
        
    n_t = b_map.shape[0]
    n_total_grids = b_map.shape[1]

    n_lat_grid = int((n_total_grids / 2) ** 0.5)
    n_lon_grid = int(n_lat_grid * 2)
    
    b_map_3d = b_map.reshape(n_t, n_lat_grid, n_lon_grid)
    
    diff_lat = jnp.diff(b_map_3d, axis=1)
    diff_lon = jnp.diff(b_map_3d, axis=2)
    spatial_l2 = jnp.sum(diff_lat**2) + jnp.sum(diff_lon**2)

    if n_t == 1:
        return spatial_l2
    else:
        areas = grid[2]
        time_l2 = jnp.sum(areas[None, :] * (jnp.diff(b_map, axis=0)**2) / (jnp.diff(phases)[:, None] + 1e-8))
        return spatial_l2 + time_l2


def l1_norm(b_map, hyperparam_prox, scale=1.0):
    shift = 1.0  
    areas = grid_cv[2][None, :]

    shifted_prox = prox.prox_lasso(b_map - shift, hyperparam_prox * scale * areas)

    return jnp.clip(shift + shifted_prox, 1e-6, 1.0)
