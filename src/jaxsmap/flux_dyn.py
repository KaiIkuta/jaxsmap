import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from functools import partial
from coord import StarGrid, flux_lim, limb_darkening, mu

@register_pytree_node_class
class SpottedFluxModel:
    
    def __init__(self, grid: StarGrid):
        self.grid = grid

    def tree_flatten(self):
        return ((self.grid,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.grid = children[0]
        return obj


    @jax.jit
    def cos_beta(self, t, period, incl_rad):
        phi_flat = self.grid.lat_flat
        lam_flat = self.grid.lon_flat
        
        z1 = jnp.cos(incl_rad) * jnp.sin(phi_flat)
        z1 = jnp.broadcast_to(z1[None, :], (t.size, self.grid.num_grids))
        
        phase = 2. * jnp.pi * t[:, None] / period
        
        z2 = jnp.sin(incl_rad) * jnp.cos(phi_flat) * jnp.cos(lam_flat)
        z2 = z2[None, :] * jnp.cos(phase)
        
        z3 = jnp.sin(incl_rad) * jnp.cos(phi_flat) * jnp.sin(lam_flat)
        z3 = z3[None, :] * jnp.sin(phase)
        
        return z1 + z2 - z3

　　@jax.jit
    def cos_phi_beta(self, phi_flat, lam_flat, t, period, incl_rad):
        cosb = self.cos_beta(t, period, incl_rad)
        phi_flat = self.grid.lat_flat
        return cosb * jnp.cos(phi_flat)

    @jax.jit
    def _spotted_flux_single(self, params, t):
        period = params["period"]
        incl_rad = params["incl"] * jnp.pi / 180.
        ld_star = params["ld_star"]
        ld_spot = params["ld_spot"]
        f_spot = params["f_spot"]

        mu = mu(self.grid.lat_flat, self.grid.lon_flat, t, period, incl_rad)
        mu_clip = jnp.where(mu < 0, 0.0, mu)

        I_star = limb_darkening(mu_clip, ld_star)
        I_spot = limb_darkening(mu_clip, ld_spot)
        
        integrand = self.grid.areas * mu_clip * (I_star - f_spot * I_spot)
        return integrand / jnp.pi

    @jax.jit
    def _relative_flux_single(self, params, t):
        ld_star = params["ld_star"]
        spot_contrib = jnp.sum(self._spotted_flux_single(params, t), axis=1)
        f = flux_lim(ld_star) - spot_contrib
        f_ave = jnp.mean(f) + 1e-12
        return f / f_ave - 1.

    @partial(jax.jit, static_argnums=(0,))
    def relative_flux(self, params, t):
        ld_star = params.get("ld_star")
        ld_spot = params.get("ld_spot")
        f_spot = params.get("f_spot")

        num_color = ld_star.shape[0] if (ld_star is not None and jnp.ndim(ld_star) == 2) else 1
        t_size = t.size
        n_grids = self.grid.num_grids
        
        f_spot_axis = None
        
        if f_spot is not None:
            f_size = f_spot.size
            if f_size == n_grids:
                f_spot = f_spot.reshape(1, n_grids)
            elif f_size == t_size * n_grids and f_spot.shape[0] == t_size:
                f_spot = f_spot.reshape(t_size, n_grids)
            elif f_size == num_color * n_grids and f_spot.shape[0] == num_color:
                f_spot = f_spot.reshape(num_color, 1, n_grids)
                f_spot_axis = 0
            elif f_size == num_color * t_size * n_grids and f_spot.shape[0] == num_color:
                f_spot = f_spot.reshape(num_color, t_size, n_grids)
                f_spot_axis = 0
            else:
                raise ValueError(f"Error the shape of the surface intensity: {f_f_spot.shape}")
            params = {**params, "f_spot": f_spot}

        ld_star_axis = 0 if (ld_star is not None and jnp.ndim(ld_star) == 2) else None
        ld_spot_axis = 0 if (ld_spot is not None and jnp.ndim(ld_spot) == 2) else None

        if ld_star_axis is not None or ld_spot_axis is not None or f_spot_axis is not None:
            axes_dict = {k: None for k in params.keys()}
            if "ld_star" in axes_dict: axes_dict["ld_star"] = ld_star_axis
            if "ld_spot" in axes_dict: axes_dict["ld_spot"] = ld_spot_axis
            if "f_spot"  in axes_dict: axes_dict["f_spot"]  = f_spot_axis

            mapped_func = jax.vmap(
                self.__class__._relative_flux_single, 
                in_axes=(None, axes_dict, None)
            )
            return mapped_func(self, params, t)
        else:
            return self._relative_flux_single(params, t)
