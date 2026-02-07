import jax
import jax.numpy as jnp
from jax import custom_jvp, vmap, jit
from scipy import special

@custom_jvp
def wofz(z):
    return jax.pure_callback(special.wofz(z), z, z, vmap_method="expand_dims")

@jit
def voigt_profile(nu, sigma, gamma):
    s = jnp.maximum(sigma, 1e-9)
    z = (nu + 1j * gamma) / (s * jnp.sqrt(2.0))
    return jnp.real(wofz(z)) / (s * jnp.sqrt(2 * jnp.pi))
