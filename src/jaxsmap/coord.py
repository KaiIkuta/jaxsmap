import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial


@jit
def x_cart(phi, lam):
  return jnp.cos(phi)[:, None] * jnp.sin(lam)[None, :]

@jit
def y_cart(phi, lam, incl):
  return jnp.sin(incl) * jnp.sin(phi)[:, None] - jnp.cos(incl) * jnp.cos(phi)[:, None] * jnp.cos(lam)[None, :]

@jit
def z_cart(phi, lam, incl):
  return jnp.cos(incl) * jnp.sin(phi)[:, None] + jnp.sin(incl) * jnp.cos(phi)[:, None] * jnp.cos(lam)[None, :]


@jit # Kipping 2013
def q_to_u(q):
    u1, u2 = 2.*(q[0]**0.5)*q[1], (q[0]**0.5)*(1.-2.*q[1])
    return jnp.array([u1,u2])
@jit # Ikuta et al. 2020
def u_to_c(u):
    return jnp.array([0.,u[0]+2*u[1],0.,-u[1]])

@jit
def mu(lat_flat, lon_flat, t, period, incl_rad):
    t = jnp.atleast_1d(t)
    phase = t / period
    lon_shifted = lon_flat[None, :] + 2.0 * jnp.pi * phase[:, None]
    mu = jnp.sin(lat_flat)[None, :] * jnp.cos(incl_rad) + \
         jnp.cos(lat_flat)[None, :] * jnp.sin(incl_rad) * jnp.cos(lon_shifted)
    return jnp.squeeze(mu)

@jax.jit
def vlos(lat_flat, lon_flat, t, period, vsini):
    t = jnp.atleast_1d(t)
    phase = t / period
    lon_shifted = lon_flat[None, :] + 2.0 * jnp.pi * phase[:, None]
    vlos = vsini * jnp.cos(lat_flat)[None, :] * jnp.sin(lon_shifted)
    return jnp.squeeze(vlos)

@jax.jit
def limb_darkening(mu, c):
    mu_clip = jnp.maximum(mu, 0.0)
    return 1.0 - c[0]*(1.0 - mu_clip**0.5) \
               - c[1]*(1.0 - mu_clip) \
               - c[2]*(1.0 - mu_clip**1.5) \
               - c[3]*(1.0 - mu_clip**2)

@register_pytree_node_class
class StarGrid:
    def __init__(self, n_lat=36, n_lon=72):
        self.n_lat = n_lat
        self.n_lon = n_lon
        
        lat_edges = jnp.linspace(-1.0, 1.0, n_lat + 1)
        lats = jnp.arcsin((lat_edges[:-1] + lat_edges[1:]) / 2.0)
        
        dlam = jnp.pi * 2. / n_lon
        lons = jnp.linspace(-jnp.pi + dlam/2.0, jnp.pi - dlam/2.0, n_lon)
        
        lat_grid, lon_grid = jnp.meshgrid(lats, lons, indexing='ij')
        self.lat_flat = lat_grid.ravel()
        self.lon_flat = lon_grid.ravel()

        self.areas = jnp.full_like(self.lat_flat, 2.0 / n_lat * dlam)

        self.num_grids = self.lat_flat.size

    def tree_flatten(self):
        children = (self.lat_flat, self.lon_flat, self.areas)
        aux_data = (self.n_lat, self.n_lon, self.num_grids)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.lat_flat, obj.lon_flat, obj.areas = children
        obj.n_lat, obj.n_lon, obj.num_grids = aux_data
        return obj
