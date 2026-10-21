import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial
from .coord import StarGrid, limb_darkening, mu, vlos
from .fcn import voigt_profile

C_LIGHT = 299792.458  # km/s

@register_pytree_node_class
class DopplerImagingModel:
    def __init__(self, grid: StarGrid, velocity_grid, wl0=550.0, inst_res=None):
        self.grid = grid
        self.vel_grid = jnp.asarray(velocity_grid)
        self.wl0 = wl0
        self.n_grids = grid.num_grids

        if inst_res is not None and inst_res > 0:
            fwhm_v = C_LIGHT / inst_res
            dv = self.vel_grid[1] - self.vel_grid[0]
            num_pts = 2 * int(3.0 * fwhm_v / dv) + 1
            v_g = jnp.linspace(-3.0 * fwhm_v, 3.0 * fwhm_v, num_pts)
            prof = (0.939437 / fwhm_v) * jnp.exp(-2.772589 * (v_g / fwhm_v)**2)
            self.inst_kernel = jnp.array(prof / jnp.sum(prof))
        else:
            self.inst_kernel = jnp.array([1.0])

    def tree_flatten(self):
        children = (self.grid, self.vel_grid, self.inst_kernel)
        aux_data = (self.n_grids, self.wl0)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.grid, obj.vel_grid, obj.inst_kernel = children
        obj.n_grids, obj.wl0 = aux_data
        return obj

    @jax.jit
    def get_wavelength_grid(self, vel_r):
        return (self.vel_grid - vel_r) / C_LIGHT * self.wl0 + self.wl0

    @jax.jit
    def _compute_profile_single(self, params, t_val, vel_r):
        period = params["period"]
        incl_rad = params["incl"] * jnp.pi / 180.0
        vsini = params["vsini"]
        domega = params.get("domega", 0.0)
        u_spec = params["u_spec"]
        spot_contrast = params["spot_contrast"]
        f_spot_flat = params["f_spot"]


        sigma_wl = params["sigma_wl"]  
        gamma_wl = params["gamma_wl"]  
        line_str = params["line_str"]


        wl_syn = self.get_wavelength_grid(vel_r)

        mu_val = mu(self.grid.lat_flat, self.grid.lon_flat, t_val, period, incl_rad, domega)
        vloss = vlos(self.grid.lat_flat, self.grid.lon_flat, t_val, period, vsini, domega)

        is_visible = (mu_val > 0.0).astype(jnp.float64)
        mu_clip = jnp.maximum(mu_val, 0.0)
        I_star = limb_darkening(mu_clip, u_spec)

        base_weight = self.grid.areas * mu_clip * I_star * is_visible
        pixel_intensity = 1.0 - f_spot_flat * (1.0 - spot_contrast)


        def voigt_and_weight(vlos_i, int_i, w_i):

            wl_center = self.wl0 * (1.0 + vlos_i / C_LIGHT)

            local_depth = line_str * voigt_profile(wl_syn - wl_center, sigma_wl, gamma_wl)
            spot_weight = w_i * int_i
            return spot_weight * local_depth, spot_weight

        contribs, spot_weights = jax.vmap(voigt_and_weight)(vloss, pixel_intensity, base_weight)

        total_weight = jnp.maximum(jnp.sum(spot_weights), 1e-10)
        normalized_flux = 1.0 - (jnp.sum(contribs, axis=0) / total_weight)


        pad_width = len(self.inst_kernel) // 2
        return jnp.convolve(jnp.pad(normalized_flux, pad_width, mode='edge'), self.inst_kernel, mode='valid')

    @partial(jax.jit, static_argnums=(0,))
    def compute_profiles(self, params, t, vel_rs=None):
        f_spot = params.get("f_spot")
        t_size = t.size
        if vel_rs is None:
            vel_rs = jnp.zeros_like(t)

        if f_spot is not None:
            f_size = f_spot.size
            axes_dict = {k: None for k in params.keys()}
            if f_size == self.n_grids:
                f_spot = f_spot.reshape(1, self.n_grids)
                axes_dict["f_spot"] = None
            elif f_size == t_size * self.n_grids:
                f_spot = f_spot.reshape(t_size, self.n_grids)
                axes_dict["f_spot"] = 0
            else:
                raise ValueError("f_spot shape mismatch for Doppler Imaging.")

            params_safe = {**params, "f_spot": f_spot}
            mapped_fn = jax.vmap(
                self.__class__._compute_profile_single,
                in_axes=(None, axes_dict, 0, 0)
            )
            return mapped_fn(self, params_safe, t, vel_rs)
        else:
            raise ValueError("f_spot is missing from params.")

    @staticmethod
    @jax.jit
    def calc_mean_ew(profiles, wl_syn_set):
        def ew_single(prof, wl):
            depth = 1.0 - prof
            dwl = wl[1:] - wl[:-1]
            return jnp.sum(depth[:-1] * dwl) + depth[-1] * dwl[-1]

        ews = jax.vmap(ew_single)(profiles, wl_syn_set)
        return jnp.mean(ews)

    def fit_line_strength(self, obs_profiles, params, t, vel_rs=None):

        if vel_rs is None:
            vel_rs = jnp.zeros_like(t)

        wl_syn_set = jax.vmap(self.get_wavelength_grid)(vel_rs)

        mean_ew_obs = self.calc_mean_ew(obs_profiles, wl_syn_set)

        line_str_0 = params.get("line_str", 1.0)
        syn_profiles_0 = self.compute_profiles({**params, "line_str": line_str_0}, t, vel_rs)
        mean_ew_syn = self.calc_mean_ew(syn_profiles_0, wl_syn_set)

        best_line_str = line_str_0 * (mean_ew_obs / jnp.maximum(mean_ew_syn, 1e-12))
        return {**params, "line_str": best_line_str}
