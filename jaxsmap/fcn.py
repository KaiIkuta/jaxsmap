import jax
import jax.numpy as jnp
from jax import custom_jvp, vmap, jit
from scipy import special

@custom_jvp
def wofz(z):
    result_shape = jax.ShapeDtypeStruct(z.shape, jnp.complex64)
    return jax.pure_callback(special.wofz, result_shape, z, vmap_method="expand_dims")

@wofz.defjvp
def wofz_jvp(primals, tangents):
    (z,) = primals
    (z_dot,) = tangents
    w = wofz(z)
    dw_dz = -2.0 * z * w + 2.0j / jnp.sqrt(jnp.pi)
    return w, dw_dz * z_dot

@jit
def voigt_profile(nu, sigma, gamma):
    s = jnp.maximum(sigma, 1e-9)
    z = (nu + 1j * gamma) / (s * jnp.sqrt(2.0))
    return jnp.real(wofz(z)) / (s * jnp.sqrt(2 * jnp.pi))
