import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial
from coord import StarGrid, limb_darkening, mu, vlos
from fcn import voigt_profile


@register_pytree_node_class
class DopplerImagingModel:
    def __init__(self, grid: StarGrid, velocity_grid, inst_res=None):
        self.grid = grid
        self.vel_grid = jnp.asarray(velocity_grid)
        self.n_grids = grid.num_grids
        
        if inst_res is not None and inst_res > 0:
            C_LIGHT = 299792.458
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
        aux_data = (self.n_grids,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.grid, obj.vel_grid, obj.inst_kernel = children
        obj.n_grids = aux_data[0]
        return obj

    @jax.jit
    # 修正: 引数から f_spot_flat を消し、paramsから直接読み込む形に統一
    def _compute_profile_single(self, params, t_val, local_flux, local_vels):
        period = params["period"]
        incl_rad = params["incl"] * jnp.pi / 180.
        vsini = params["vsini"]
        u_spec = params["u_spec"]
        spot_contrast = params["spot_contrast"]
        
        # 追加: ここで f_spot を取得
        f_spot_flat = params["f_spot"]

        mu_val = mu(self.grid.lat_flat, self.grid.lon_flat, t_val, period, incl_rad)
        vloss = vlos(self.grid.lat_flat, self.grid.lon_flat, t_val, period, vsini)
        
        is_visible = (mu_val > 0.0).astype(jnp.float32)
        mu_clip = jnp.maximum(mu_val, 0.0)

        I_star = limb_darkening(mu_clip, u_spec)
        base_weight = self.grid.areas * mu_clip * I_star * is_visible
        
        pixel_intensity = 1.0 - f_spot_flat * (1.0 - spot_contrast)

        def shift_and_weight(vlos_i, int_i, w_i):
            shifted = jnp.interp(self.vel_grid, local_vels + vlos_i, 
                                 1.0 - int_i * (1.0 - local_flux), 
                                 left=1.0, right=1.0)
            return w_i * (1.0 - shifted), w_i

        contribs, weights = jax.vmap(shift_and_weight)(vloss, pixel_intensity, base_weight)
        
        total_weight = jnp.maximum(jnp.sum(weights), 1e-10)
        normalized_flux = 1.0 - (jnp.sum(contribs, axis=0) / total_weight)

        pad_width = len(self.inst_kernel) // 2
        return jnp.convolve(jnp.pad(normalized_flux, pad_width, mode='edge'), self.inst_kernel, mode='valid')

    @partial(jax.jit, static_argnums=(0,))
    def compute_profiles(self, params, t, local_flux, local_vels):
        f_spot = params.get("f_spot")
        t_size = t.size
        
        if f_spot is not None:
            f_size = f_spot.size
            
            # 追加: params の動的な axes マッピングを作成
            axes_dict = {k: None for k in params.keys()}
            
            if f_size == self.n_grids:
                f_spot = f_spot.reshape(1, self.n_grids)
                axes_dict["f_spot"] = None # 時間方向にはブロードキャスト
            elif f_size == t_size * self.n_grids:
                f_spot = f_spot.reshape(t_size, self.n_grids)
                axes_dict["f_spot"] = 0    # 時間方向(t)と同時にマップする
            else:
                raise ValueError("f_spot shape mismatch for Doppler Imaging.")
            
            params_safe = {**params, "f_spot": f_spot}
            
            # 修正: _compute_profile_single に渡す引数は (self, params, t, local_flux, local_vels) の5つ
            mapped_fn = jax.vmap(self.__class__._compute_profile_single, in_axes=(None, axes_dict, 0, None, None))
            return mapped_fn(self, params_safe, t, local_flux, local_vels)
        else:
            raise ValueError("f_spot is missing from params.")
