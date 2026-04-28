import jax
import jax.numpy as jnp
from jax import jit

@jit
def x_cart(phi, lam):
  return jnp.cos(phi)[:, None] * jnp.sin(lam)[None, :]

@jit
def y_cart(phi, lam, incl):
  return jnp.sin(incl) * jnp.sin(phi)[:, None] - jnp.cos(incl) * jnp.cos(phi)[:, None] * jnp.cos(lam)[None, :]

@jit
def z_cart(phi, lam, incl):
  return jnp.cos(incl) * jnp.sin(phi)[:, None] + jnp.sin(incl) * jnp.cos(phi)[:, None] * jnp.cos(lam)[None, :]

@jit
def cart(phi, lam, incl):
    return x_cart(phi, lam), y_cart(phi, lam, incl), z_cart(phi, lam, incl)

@jit
def spherical_grid(n_lat, n_lon):
    lat_edges = jnp.linspace(-1.0, 1.0, n_lat + 1)
    lats = jnp.arcsin((lat_edges[:-1] + lat_edges[1:]) / 2.0)
    dlat = jnp.diff(jnp.arcsin(lat_edges))
    dlam = jnp.pi * 2. / n_lon
    lons = jnp.linspace(-jnp.pi + dlam/2.0, jnp.pi - dlam/2.0, n_lon)
    lat_grid, lon_grid = jnp.meshgrid(lats, lons, indexing='ij')
    areas = jnp.full_like(lat_flat, 2.0/n_lat* dlam)
    return lat_grid.ravel(), lon_grid.ravel(), areas

@jit
def vlos(phi, lam, vsini):
    return vsini * jnp.cos(phi) * jnp.sin(lam)

@jit
def limb_darkening(mu, ld):
    return 1. - ld[0] * (1. - mu**0.5) - ld[1] * (1. - mu) - ld[2] * (1. - mu**1.5) - ld[3] * (1. - mu**2)
