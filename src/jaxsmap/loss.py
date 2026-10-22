import jax
import jax.numpy as jnp
from jaxopt import prox


def _extract_grid_geom(grid, n_total_grids):
    if hasattr(grid, "areas"):
        areas = grid.areas
        lat_flat = grid.lat_flat
        n_lat = grid.n_lat
        n_lon = grid.n_lon
    else:
        lat_flat, _, areas = grid[0], grid[1], grid[2]
        n_lat = int((n_total_grids / 2) ** 0.5)
        n_lon = int(n_lat * 2)

    lats = lat_flat.reshape(n_lat, n_lon)[:, 0]
    dlam = 2.0 * jnp.pi / n_lon
    lat_edges = jnp.arcsin(jnp.linspace(-1.0, 1.0, n_lat + 1))
    dphi = jnp.diff(lat_edges)
    phi_half = 0.5 * (lats[:-1] + lats[1:])
    cos_phi_half = jnp.cos(phi_half)

    return areas, n_lat, n_lon, dphi, dlam, cos_phi_half

@jax.jit
def chi_square(mod_flux, obs_flux, obs_err, mask=None):
    res = (obs_flux - mod_flux) / obs_err
    if mask is None:
        return 0.5 * jnp.sum(res ** 2)
    return 0.5 * jnp.sum((res ** 2) * mask)


@jax.jit
def max_entropy(f_map, grid, def_val=1.0, max_val=1., eps=1e-6):

    if f_map.ndim == 1:
        f_map = f_map.reshape(1, -1)
    areas, _, _, _, _, _ = _extract_grid_geom(grid, f_map.shape[1])
    b = jnp.clip(f_map, eps, max_val - eps)
    m = jnp.clip(def_val, eps, max_val - eps)
    neg_s = b * jnp.log(b / m) + (max_val - b) * jnp.log((max_val - b) / (max_val - m))

    return jnp.sum(areas[None, :] * neg_s)


@jax.jit
def tv_space_norm(f_map, grid, eps=1e-12):

    if f_map.ndim == 1:
        f_map = f_map.reshape(1, -1)
    n_t, n_total_grids = f_map.shape
    _, n_lat, n_lon, dphi, dlam, cos_phi_half = _extract_grid_geom(grid, n_total_grids)
    f_3d = f_map.reshape(n_t, n_lat, n_lon)

    diff_lat = jnp.sqrt((f_3d[:, 1:, :] - f_3d[:, :-1, :]) ** 2 + eps)
    weight_lat = (dlam * cos_phi_half)[None, :, None]

    diff_lon = jnp.sqrt((jnp.roll(f_3d, -1, axis=2) - f_3d) ** 2 + eps)
    weight_lon = dphi[None, :, None]

    return jnp.sum(weight_lat * diff_lat) + jnp.sum(weight_lon * diff_lon)


@jax.jit
def tsv_space_norm(f_map, grid):

    if f_map.ndim == 1:
        f_map = f_map.reshape(1, -1)
    n_t, n_total_grids = f_map.shape
    _, n_lat, n_lon, dphi, dlam, cos_phi_half = _extract_grid_geom(grid, n_total_grids)
    f_3d = f_map.reshape(n_t, n_lat, n_lon)

    diff_lat_sq = (f_3d[:, 1:, :] - f_3d[:, :-1, :]) ** 2
    weight_lat = (dlam * cos_phi_half)[None, :, None]

    diff_lon_sq = (jnp.roll(f_3d, -1, axis=2) - f_3d) ** 2
    weight_lon = dphi[None, :, None]

    return jnp.sum(weight_lat * diff_lat_sq) + jnp.sum(weight_lon * diff_lon_sq)


@jax.jit
def tsv_time_norm(f_map, grid, times):

    if f_map.ndim == 1 or f_map.shape[0] <= 1:
        return 0.0
    n_t, n_total_grids = f_map.shape
    areas, _, _, _, _, _ = _extract_grid_geom(grid, n_total_grids)

    dt = jnp.diff(times)[:, None]
    df_sq = jnp.diff(f_map, axis=0) ** 2
    time_weight = 1.0 / (dt + 1e-8)

    return jnp.sum(areas[None, :] * time_weight * df_sq)


@jax.jit
def l1_norm(f_map, grid, shift=1.0, eps=1e-12):

    if f_map.ndim == 1:
        f_map = f_map.reshape(1, -1)
    areas, _, _, _, _, _ = _extract_grid_geom(grid, f_map.shape[1])
    abs_dev = jnp.sqrt((f_map - shift) ** 2 + eps)
    return jnp.sum(areas[None, :] * abs_dev)


@jax.jit
def group_l1_time_norm(f_map, grid, shift=1.0, eps=1e-12):

    if f_map.ndim == 1:
        f_map = f_map.reshape(1, -1)
    areas, _, _, _, _, _ = _extract_grid_geom(grid, f_map.shape[1])
    temporal_l2 = jnp.sqrt(jnp.sum((f_map - shift) ** 2, axis=0) + eps)
    return jnp.sum(areas * temporal_l2)


@jax.jit
def l2_norm(f_map, grid, shift=1.0):

    if f_map.ndim == 1:
        f_map = f_map.reshape(1, -1)
    areas, _, _, _, _, _ = _extract_grid_geom(grid, f_map.shape[1])
    return jnp.sum(areas[None, :] * ((f_map - shift) ** 2))



@jax.jit
def prox_l1(f_map, grid, lam_lr, shift=1.0, min_val=0.0, max_val=1.0):

    is_1d = (f_map.ndim == 1)
    if is_1d:
        f_map = f_map.reshape(1, -1)
    areas, _, _, _, _, _ = _extract_grid_geom(grid, f_map.shape[1])
    res = shift + prox.prox_lasso(f_map - shift, lam_lr * areas[None, :])
    res = jnp.clip(res, min_val, max_val)
    return jnp.squeeze(res) if is_1d else res


@jax.jit
def prox_group_l1_time(f_map, grid, lam_lr, shift=1.0, min_val=0.0, max_val=1.0):

    is_1d = (f_map.ndim == 1)
    if is_1d:
        f_map = f_map.reshape(1, -1)
    areas, _, _, _, _, _ = _extract_grid_geom(grid, f_map.shape[1])
    u = f_map - shift
    prox_fn = jax.vmap(prox.prox_group_lasso, in_axes=(1, 0), out_axes=1)
    res = shift + prox_fn(u, lam_lr * areas)
    res = jnp.clip(res, min_val, max_val)
    return jnp.squeeze(res) if is_1d else res
    shifted_prox = prox.prox_lasso(b_map - shift, hyperparam_prox * scale * areas)
    return jnp.clip(shift + shifted_prox, 1e-6, 1.0)
