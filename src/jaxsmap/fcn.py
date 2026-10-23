import jax
import jax.numpy as jnp

_N_WEIDEMAN = 24
_L_WEIDEMAN = jnp.sqrt(_N_WEIDEMAN / jnp.sqrt(2.0))
_M_IDX = jnp.arange(1, 2 * _N_WEIDEMAN)
_THETA = _M_IDX * jnp.pi / _N_WEIDEMAN
_T_GRID = _L_WEIDEMAN * jnp.tan(0.5 * _THETA)
_F_VALS = jnp.exp(-_T_GRID**2) * (_L_WEIDEMAN**2 + _T_GRID**2)
_F_PADDED = jnp.concatenate([jnp.array([0.0]), _F_VALS])
_POLY_COEFFS = jnp.fft.fft(jnp.fft.fftshift(_F_PADDED)).real / (2 * _N_WEIDEMAN)
_POLY_COEFFS_REV = _POLY_COEFFS[1:_N_WEIDEMAN + 1][::-1]

@jax.jit
def wofz_jax(z):
    z_in = _L_WEIDEMAN + 1j * z
    z_out = _L_WEIDEMAN - 1j * z
    q = z_in / z_out
    poly = jnp.polyval(_POLY_COEFFS_REV, q)
    w = 2.0 * poly / (z_out**2) + (1.0 / jnp.sqrt(jnp.pi)) / z_out
    return w

@jax.jit
def voigt_profile(dl, sigma, gamma):
    z = (dl + 1j * gamma) / (sigma * jnp.sqrt(2.0))
    return jnp.real(wofz_jax(z)) / (sigma * jnp.sqrt(2.0 * jnp.pi))
