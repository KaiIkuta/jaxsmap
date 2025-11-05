import jax
import jax.numpy as jnp
from jax import jit

incl = 60.

@jit
def x_cart(phi, lam):
  return jnp.cos(phi)[:, None] * jnp.sin(lam)[None, :]

@jit
def y_cart(phi, lam):
  return jnp.sin(incl) * jnp.sin(phi)[:, None] - jnp.cos(incl) * jnp.cos(phi)[:, None] * jnp.cos(lam)[None, :]

@jit
def z_cart(phi, lam):
  return jnp.cos(incl) * jnp.sin(phi)[:, None] + jnp.sin(incl) * jnp.cos(phi)[:, None] * jnp.cos(lam)[None, :]
