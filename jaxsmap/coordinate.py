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
def vlos(phi, lam, vsini):
    return vsini * jnp.cos(phi) * jnp.sin(lam)

@jit
def limb_darkening(mu, ld):
    return 1. - ld[0] * (1. - mu**0.5) - ld[1] * (1. - mu) - ld[2] * (1. - mu**1.5) - ld[3] * (1. - mu**2)

