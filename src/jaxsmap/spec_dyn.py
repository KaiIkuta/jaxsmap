import jax
import jax.numpy as jnp
from jax import jit
from jaxsmap.coord import spherical_grid
from jaxsmap.fcn import voigt_profile

C_LIGHT = 299792.458  # (km/s)
period = 2.766 #(day)
t_nor = 59665.276 #BJD-2400000


@jit
def create_inst_kernel_velocity(vel_grid, inst_res):
    fwhm_v = C_LIGHT / inst_res
    dv = vel_grid[1] - vel_grid[0]
    num_pts = 2 * int(3.0 * fwhm_v / dv) + 1
    v_g = jnp.linspace(-3.0 * fwhm_v, 3.0 * fwhm_v, num_pts)
    prof = (0.939437 / fwhm_v) * jnp.exp(-2.772589 * (v_g / fwhm_v)**2)
    return jnp.where(inst_res > 0, jnp.array(prof / jnp.sum(prof)), jnp.array([1.0]))

@jit
def mu(lat, lon,incl,phase):
    lon_shifted = lon + 2.*jnp.pi*phase
    return (jnp.sin(lat) * jnp.cos(incl) +
          jnp.cos(lat) * jnp.sin(incl) * jnp.cos(lon_shifted))
@jit
def vlos(lat, lon, vsini, phase):
    lon_shifted = lon + 2.0 * jnp.pi * phase
    return vsini * jnp.cos(lat) * jnp.sin(lon_shifted)

@jit
def limb_darkening(mu, u1=0.26853869, u2=0.70448628,u3=-0.30565246,u4=0.15820857): #Claret (2023): V-band, nonlinear
    mu = jnp.maximum(mu, 0.0)
    return 1.0 - u1 * (1.0 - mu**0.5) - u2 * (1.0 - mu) - u3 * (1.0 - mu**1.5) - u4 * (1.0 - mu**2)

def shift_and_weight_pixel(pixel_vlos, pixel_intensity, pixel_mu, pixel_area,
                           velocity_grid, local_profile_flux, local_profile_vels):

    is_visible = (pixel_mu > 0.0).astype(jnp.float32)
    weight =  pixel_area * pixel_mu * limb_darkening(pixel_mu) * is_visible

    shifted_profile = jnp.interp(
        velocity_grid,
        local_profile_vels + pixel_vlos,
        1.-pixel_intensity *(1.-local_profile_flux),
        left=1.0, right=1.0
    )

    return weight * (1.- shifted_profile), weight



@jit
def compute_flux(surface_map, lats, lons, areas, incl_rad,phase, vsini,
                 velocity_grid, local_profile_flux, local_profile_vels,inst_kernel):
    mus = mu(lats, lons, incl_rad, phase)
    vloss = vlos(lats, lons, vsini, phase)

    pixel_contribs, pixel_weights = vmap(
        shift_and_weight_pixel,
        in_axes=(0, 0, 0, 0, None, None, None)
    )(vloss, surface_map, mus, areas, velocity_grid, local_profile_flux, local_profile_vels)


    total_weight = jnp.maximum(jnp.sum(pixel_weights, axis=0),1e-10)

    normalized_flux = 1.0 - (jnp.sum(pixel_contribs, axis=0) / total_weight)

    pad_width = len(inst_kernel) // 2
    return jnp.convolve(jnp.pad(normalized_flux, pad_width, mode='edge'), inst_kernel, mode='valid')


#Parameters for Voigt profile (v_shift, sigma, gamma, depth) should be determined from the immaculate sphere
if __name__ == "__main__":
    grid = spherical_grid(n_lat=36, n_lon=72)
    local_vels = jnp.linspace(-20, 20, 500)
    v_shift = 0.3629
    v_prof = voigt_profile(local_vels - v_shift, sigma=2.5732, gamma=2.4631)
    v_prof_norm = v_prof / jnp.max(v_prof)
    depth = 0.1515
    local_f = 1.0 - (v_prof_norm * depth)
    obs_vel_grid = jnp.linspace(-32, 32, 200)
    inst_kernel = create_inst_kernel_velocity(obs_vel_grid, inst_res=65000)

  
    incl_rad = jnp.deg2rad(60.)
    vsini=17.6
    f_spot = 0.27
    phase = jnp.array([0.,0.25,0.5,0.75,1.0])
  
    spotted_map = jnp.ones((len(phase),len(grid[0])))
    for i in range(len(spotted_map[0])):
        lon_shifted = grid[1] + 2.0 * jnp.pi * phase
        dist = jnp.arccos(jnp.cos(grid[0])*jnp.cos(jnp.deg2rad(spot_lat))*jnp.cos(lon_shifted - jnp.deg2rad(spot_lon)) + jnp.sin(grid[0])*jnp.sin(jnp.deg2rad(spot_lat)))
        spotted_map[i, dist < jnp.deg2rad(spot_rad)] = f_spot

    mod_flux = compute_flux(
        spotted_map, grid[0], grid[1], grid[2], incl_rad,
        0.0, vsini, obs_vel_grid, local_f, local_vels, inst_kernel
    )
