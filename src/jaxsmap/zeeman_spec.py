


@jit
def calc_blos(lat, lon, incl, phase, br, blat, blon):
    lon_shifted = lon + 2.0 * jnp.pi * phase
    pr = jnp.sin(lat) * jnp.cos(incl) + jnp.cos(lat) * jnp.sin(incl) * jnp.cos(lon_shifted)
    plat = jnp.cos(lat) * jnp.cos(incl) - jnp.sin(lat) * jnp.sin(incl) * jnp.cos(lon_shifted)
    plon = -jnp.sin(incl) * jnp.sin(lon_shifted)
    return br * pr + blat * plat + blon * plon

def shift_and_weight_pixel(pixel_vlos, pixel_intensity, pixel_blos, pixel_mu, pixel_area,
                           velocity_grid, local_profile_flux, local_profile_dI_dv, local_vels, zeeman_factor):

    is_visible = (pixel_mu > 0.0).astype(jnp.float32)
    weight =  pixel_area * pixel_mu * limb_darkening(pixel_mu) * is_visible

    shifted_profile_I = jnp.interp(
        velocity_grid,
        local_vels + pixel_vlos,
        1.0 - pixel_intensity * (1.0 - local_profile_flux),
        left=1.0, right=1.0
    )
    
    shifted_dI_dv = jnp.interp(
        velocity_grid,
        local_vels + pixel_vlos,
        pixel_intensity * local_profile_dI_dv,
        left=0.0, right=0.0
    )
    shifted_profile_V = -zeeman_factor * pixel_blos * shifted_dI_dv

    return weight * (1.0 - shifted_profile_I), weight * shifted_profile_V, weight


@jit
def compute_stokes_IV(intensity_map, br_map, blat_map, blon_map, lats, lons, areas, incl_rad, phase, vsini,
                      velocity_grid, local_profile_flux, local_vels, inst_kernel, zeeman_factor):
    mus = mu(lats, lons, incl_rad, phase)
    vloss = vlos(lats, lons, vsini, phase)
    bloss = calc_blos(lats, lons, incl_rad, phase, br_map, blat_map, blon_map)


    dv = local_vels[1] - local_vels[0]
    local_profile_dI_dv = jnp.gradient(local_profile_flux, dv)

    pixel_contribs_I, pixel_contribs_V, pixel_weights = vmap(
        shift_and_weight_pixel,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None)
    )(vloss, intensity_map, bloss, mus, areas, velocity_grid, local_profile_flux, local_profile_dI_dv, local_vels, zeeman_factor)

    total_weight = jnp.maximum(jnp.sum(pixel_weights, axis=0), 1e-10)


    normalized_flux_I = 1.0 - (jnp.sum(pixel_contribs_I, axis=0) / total_weight)
    normalized_flux_V = jnp.sum(pixel_contribs_V, axis=0) / total_weight

    pad_width = len(inst_kernel) // 2
    stokes_I_conv = jnp.convolve(jnp.pad(normalized_flux_I, pad_width, mode='edge'), inst_kernel, mode='valid')
    stokes_V_conv = jnp.convolve(jnp.pad(normalized_flux_V, pad_width, mode='constant', constant_values=0.0), inst_kernel, mode='valid')

    return stokes_I_conv, stokes_V_conv



if __name__ == "__main__":
    grid = spherical_grid(n_lat=36, n_lon=72)
    lats, lons, areas = grid[0], grid[1], grid[2]
    
    local_vels = jnp.linspace(-20, 20, 500)
    v_shift = 0.3629
    v_prof = voigt_profile(local_vels - v_shift, sigma=2.5732, gamma=2.4631)
    v_prof_norm = v_prof / jnp.max(v_prof)
    depth = 0.1515
    local_f = 1.0 - (v_prof_norm * depth)
    
    obs_vel_grid = jnp.linspace(-32, 32, 200)
    inst_kernel = create_inst_kernel_velocity(obs_vel_grid, inst_res=65000) #Velocity bin for GOES-RV

    incl_rad = jnp.deg2rad(60.)
    vsini = 17.6
    f_spot = 0.27
    phases = jnp.array([0., 0.25, 0.5, 0.75, 1.0])
    
    # Spotパラメータ（元のコードで未定義だった部分を補足）
    spot_lat = 30.0
    spot_lon = 0.0
    spot_rad = 20.0
    
    # 1. Intensity Map (輝度マップ)
    dist = jnp.arccos(jnp.cos(lats)*jnp.cos(jnp.deg2rad(spot_lat))*jnp.cos(lons - jnp.deg2rad(spot_lon)) + jnp.sin(lats)*jnp.sin(jnp.deg2rad(spot_lat)))
    intensity_map = jnp.where(dist < jnp.deg2rad(spot_rad), f_spot, 1.0)
    
    # 2. Magnetic Map (磁場マップ: テスト用に同位置に動径方向磁場1000Gの磁気スポットを設定)
    b_max = 1000.0
    br_map = jnp.where(dist < jnp.deg2rad(spot_rad), b_max, 0.0)
    blat_map = jnp.zeros_like(lats)
    blon_map = jnp.zeros_like(lats)
    
    # ゼーマン効果によるスケーリング係数 (4.67e-13 * lambda0(A) * geff * c(km/s))
    lambda_0 = 6000.0 # オングストローム
    geff = 1.2
    zeeman_factor = 4.67e-13 * lambda_0 * geff * c_light
    
    # 複数位相(Phases)に対する計算をjax.vmapでベクトル化処理し高速化
    compute_phases = vmap(
        compute_stokes_IV,
        in_axes=(None, None, None, None, None, None, None, None, 0, None, None, None, None, None, None)
    )
    
    mod_flux_I, mod_flux_V = compute_phases(
        intensity_map, br_map, blat_map, blon_map, lats, lons, areas, incl_rad,
        phases, vsini, obs_vel_grid, local_f, local_vels, inst_kernel, zeeman_factor
    )
    
    print("Stokes I shape:", mod_flux_I.shape)
    print("Stokes V shape:", mod_flux_V.shape)


