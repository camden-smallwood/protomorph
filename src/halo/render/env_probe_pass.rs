//! Atmosphere helpers — `GpuAtmosphereData` (the runtime cbuffer
//! mirroring Halo's `c_atmosphere_fog_interface::set_constant`
//! packing), `GpuSkyParams`, and `precompute_betas` (spectral
//! integration of Rayleigh + Mie scattering coefficients).
//!
//! Originally lived alongside the `EnvProbePass` cubemap renderer
//! (a protomorph-only fabrication, no engine analog). The pass was
//! removed in the 2026-05-09 audit; these helpers are kept because
//! patchy fog and the per-frame sky uniform pipeline depend on
//! them. File still named `env_probe_pass.rs` for import-path
//! stability — rename deferred.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// GPU atmosphere cbuffer — mirrors Halo's
/// `c_atmosphere_fog_interface::set_constant` 7-vec4 packing
/// (cbuffer slots 0x29 VS / 0x3F PS) with extras for our screen-space
/// sky shader (sun disc, horizon fade, etc.).
///
/// All `beta_*` fields are the RUNTIME-DERIVED scattering coefficients.
/// In Halo, `c_atmosphere_fog_interface::precompute_scattering_coefficients
/// @ 0x1803aea00` does spectral integration over 300-800nm using
/// `k_spectrum` (solar irradiance) + `k_n2_1Amplitudes` (Mie aerosol)
/// at the channel reference wavelengths (R=650nm, G=570nm, B=475nm)
/// then multiplies by `multiplier × 1000`. We approximate this with
/// 1/λ⁴ for Rayleigh and broadband for Mie — visual approximation,
/// not bit-exact.
/// Each slot is a packed [f32; 4] so vec3+scalar combinations don't
/// hit Rust vs WGSL alignment mismatches. WGSL side declares as
/// `vec4<f32>` and unpacks into vec3+scalar accessors. 11 slots × 16 = 176B.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuAtmosphereData {
    // Halo cbuffer slot 0: (sun_dir.xyz, distance_bias)
    // Per atmosphere_fx.hlsl line 13/20: SUN_DIR + DIST_BIAS share slot 0.
    pub slot0_sun_dir_dist_bias: [f32; 4],
    // Halo cbuffer slot 1: (sun_intensity_normalized.xyz, max_fog_thickness)
    // slot1.w doubles as the ATMOSPHERE_ENABLE flag (negative = disabled,
    // line 21-22 of the HLSL). Caller stores -1 to disable.
    //   sun_intensity_normalized = sun_intensity / (β_m + β_p) per channel
    //   — Halo's SUN_INTENSITY_OVER_TR_PLUS_TM normalization.
    pub slot1_sun_int_norm_thickness: [f32; 4],
    // Halo cbuffer slot 2: (β_m × log2(e) per channel, mie_g + 1)
    pub slot2_beta_m_log2e_g1: [f32; 4],
    // Halo cbuffer slot 3: (β_p × log2(e) per channel, reference_height)
    pub slot3_beta_p_log2e_refh: [f32; 4],
    // Halo cbuffer slot 4: (β_m_angular per channel, mie_height_scale)
    pub slot4_beta_m_angular_mieh: [f32; 4],
    // Halo cbuffer slot 5: ((1-g²) × β_p_angular per channel, rayleigh_height_scale)
    pub slot5_beta_p_angular_rayh: [f32; 4],
    // Halo cbuffer slot 6: (2g, padding × 3)
    pub slot6_g2: [f32; 4],
    // Sky-pass extras (sun disc + horizon — for the screen sky shader).
    pub slot7_sun_disc: [f32; 4],     // (luminance, angular_radius, edge_softness, disc_intensity)
    pub slot8_sun_glow: [f32; 4],     // (inner_glow, air_mass_scale, zenith_factor, pad)
    pub slot9_sun_tint_horizon: [f32; 4], // (tint.xyz, horizon_fade_start)
    pub slot10_horizon_pad: [f32; 4],     // (horizon_fade_end, pad, pad, pad)
}

/// Faithful port of dllcache `c_atmosphere_fog_interface::precompute_scattering_coefficients`
/// @ 0x1803ae9c0. Computes Rayleigh+Mie extinction and angular prefix
/// coefficients per RGB channel by sampling spectral curves at the
/// channel reference wavelengths (650/570/475 nm).
///
/// The spectral sources (`K_SPECTRUM`, `K_N2_1AMPLITUDES`) are
/// extracted from `halo3_dllcache_play.dll` — see
/// [`crate::halo::render::spectral_constants`]. Returns
/// `((β_m, β_p), (β_m_angular_prefix, β_p_angular_prefix))` — the
/// four 3-vector coefficient sets Halo stores on the in-memory
/// `c_atmosphere_setting` and uploads to the engine cbuffer.
///
/// **No approximation.** The dllcache function builds 5nm-step curves
/// from 300nm to 800nm, then samples them at exact 5nm boundaries
/// (650/570/475 are all on the grid). We evaluate the formulas
/// directly at those three wavelengths — mathematically identical
/// because linear interpolation at exact knot points is identity.
pub(crate) fn precompute_betas(rayleigh_multiplier: f32, mie_multiplier: f32, desaturation: f32)
    -> (([f32; 3], [f32; 3]), ([f32; 3], [f32; 3]))
{
    use crate::halo::render::spectral_constants::{K_N2_1AMPLITUDES, K_SPECTRUM};

    /// Linearly interpolate a sampled curve. `step_nm` is the wavelength
    /// step between adjacent samples; the curve covers 300..800nm.
    fn lerp_curve(table: &[f32], step_nm: f32, lambda_nm: f32) -> f32 {
        let idx = (lambda_nm - 300.0) / step_nm;
        let i0 = idx.floor() as usize;
        let i1 = (i0 + 1).min(table.len() - 1);
        let t = idx - (i0 as f32);
        table[i0] * (1.0 - t) + table[i1] * t
    }

    /// Evaluate the four β-curve formulas at one wavelength. Mirrors
    /// the loop body of `precompute_scattering_coefficients` exactly.
    fn beta_at(lambda_nm: f32) -> [f32; 4] {
        let lambda_m = lambda_nm * 1.0e-9;
        // K_SPECTRUM is sampled in 1nm steps (501 points / 500nm range).
        // K_N2_1AMPLITUDES is sampled in 10nm steps (51 points).
        let n2 = lerp_curve(&K_N2_1AMPLITUDES, 10.0, lambda_nm);
        let solar = lerp_curve(&K_SPECTRUM, 1.0, lambda_nm);

        // Rayleigh denominator: 3 × N × λ⁴ where N = 2.545e25 (number
        // density of air molecules at STP, per dllcache constant).
        let lambda4 = lambda_m * lambda_m * lambda_m * lambda_m;
        let inv_ray = 1.0 / (3.0 * 2.545e25_f32 * lambda4);

        // Mie wavenumber-squared: (2π/λ)².
        let two_pi = 2.0 * std::f32::consts::PI;
        let k = two_pi / lambda_m;
        let k_squared = k * k;

        // Constants from the dllcache decompile:
        //   248.05023, 1.0599999 — Rayleigh extinction prefactor
        //   8.9694953e-17        — Mie extinction prefactor (solar-weighted)
        //   1.4275395e-17        — Mie angular prefactor
        //   19.73921, 1.0599999, 0.7629  — Rayleigh angular prefactor
        //   0.01                  — common Mie scale
        let beta_m       = (n2 * 248.05023 * 1.0599999) * inv_ray;
        let beta_p       = (k_squared * 8.9694953e-17 * solar) * 0.01;
        let beta_p_ang   = (k_squared * 1.4275395e-17) * 0.01;
        let beta_m_ang   = (n2 * 19.73921 * 1.0599999 * 0.7629) * inv_ray;

        [beta_m, beta_p, beta_m_ang, beta_p_ang]
    }

    let r = beta_at(650.0);
    let g = beta_at(570.0);
    let b = beta_at(475.0);

    // Multiply by user multipliers × 1000 (per dllcache: each output is
    // `curve_value * multiplier * 1000.0`).
    let mut beta_m         = [r[0]*rayleigh_multiplier*1000.0, g[0]*rayleigh_multiplier*1000.0, b[0]*rayleigh_multiplier*1000.0];
    let mut beta_p         = [r[1]*mie_multiplier*1000.0,      g[1]*mie_multiplier*1000.0,      b[1]*mie_multiplier*1000.0];
    let mut beta_m_angular = [r[2]*rayleigh_multiplier*1000.0, g[2]*rayleigh_multiplier*1000.0, b[2]*rayleigh_multiplier*1000.0];
    let mut beta_p_angular = [r[3]*mie_multiplier*1000.0,      g[3]*mie_multiplier*1000.0,      b[3]*mie_multiplier*1000.0];

    // Desaturation pass (dllcache lines 124+): blend each channel
    // toward the mean by `desaturation`. v25 = mean × desat;
    // ch_new = (1 - desat) × ch + v25.
    if desaturation > 0.0 {
        let blend = |arr: &mut [f32; 3]| {
            let mean = (arr[0] + arr[1] + arr[2]) * (1.0 / 3.0);
            let v25 = mean * desaturation;
            for v in arr.iter_mut() {
                *v = (1.0 - desaturation) * *v + v25;
            }
        };
        blend(&mut beta_m);
        blend(&mut beta_p);
        blend(&mut beta_m_angular);
        blend(&mut beta_p_angular);
    }

    ((beta_m, beta_p), (beta_m_angular, beta_p_angular))
}

/// `c_color_xyY` — CIE xyY chromaticity + luminance. Mirrors engine
/// struct used by `convert_RGB_to_xyY @ 0x1803B0780` and friends.
#[allow(non_snake_case)]
#[derive(Copy, Clone, Default)]
struct ColorXyY {
    x: f32,
    y: f32,
    Y: f32,
}

/// `convert_RGB_to_xyY @ 0x1803B0780` — verbatim port. sRGB-D65 RGB →
/// CIE XYZ → xyY (x, y from chromaticity; Y = the XYZ Y component).
#[allow(non_snake_case)]
fn convert_rgb_to_xyy(rgb: blam_tags::math::RealRgbColor) -> ColorXyY {
    let G = rgb.green;
    let B = rgb.blue;
    let v4 = (rgb.red * 0.412424)     + (G * 0.35757899)   + (B * 0.180464);
    let v5 = (rgb.red * 0.21265601)   + (G * 0.71515799)   + (B * 0.072185598);
    let v6 = (v5 + v4)
        + ((rgb.red * 0.0193324) + (G * 0.119193) + (B * 0.95044398));
    if v6 == 0.0 {
        ColorXyY { x: 0.0, y: 0.0, Y: 0.0 }
    } else {
        let v7 = 1.0 / v6;
        ColorXyY { x: v7 * v4, y: v7 * v5, Y: v5 }
    }
}

/// Scenario-sky sun pulled from `get_sun_constants_from_sky @ 0x1803ADCB0`
/// (sky model's `lightgen_lights[last]`). Threaded into the non-override
/// branch of [`get_sun_parameters`]. `intensity` is linear RGB already
/// scaled by `solid_angle × 0.2 × g_render_light_intensity` in the loader;
/// `direction` is the z-up unit vector read verbatim from the lightgen light.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SkySun {
    pub intensity: [f32; 3],
    pub direction: [f32; 3],
}

/// `c_atmosphere_fog_interface::get_sun_parameters @ 0x1803AF990` —
/// verbatim port of the `flags & 2` ("Override Real Sun Values") branch.
/// Writes `sun_intensity` (linear RGB) and `sun_direction` (z-up unit
/// vector). The non-override branch is the engine's
/// `get_sun_constants_from_sky` path (sky model's `lightgen_lights[last]`
/// computed offline) — protomorph doesn't have that data yet, so the
/// caller passes `sky_sun` as a fallback for that branch.
#[allow(non_snake_case)]
pub(crate) fn get_sun_parameters(
    parameters: &blam_tags::sky_atmosphere::AtmosphereSettings,
    sky_sun: Option<SkySun>,
) -> ([f32; 3], [f32; 3]) {
    use blam_tags::sky_atmosphere::AtmosphereFlags;
    if parameters.flags.contains(AtmosphereFlags::OverrideRealSunValues) {
        // override_color = parameters->m_dominant_light_color;
        let override_color = parameters.color;
        // convert_RGB_to_xyY(&override_color, &temp);
        let temp = convert_rgb_to_xyy(override_color);
        let mut m_dominant_light_intensity = parameters.intensity;
        let mut v10 = 0.0_f32;
        let v11;
        if temp.y == 0.0 {
            m_dominant_light_intensity = 0.0;
            v11 = 0.0;
        } else {
            v10 = (m_dominant_light_intensity / temp.y) * temp.x;
            v11 = ((1.0 - temp.x) - temp.y) * (m_dominant_light_intensity / temp.y);
        }
        // sun_intensity (linear RGB) — sRGB-D65 inverse matrix applied to (v10, intensity, v11).
        let sun_intensity = [
            (v10 *  3.240479)   - (m_dominant_light_intensity * 1.53715)   - (v11 * 0.49853501),
            (m_dominant_light_intensity *  1.875991) - (v10 * 0.96925598)  + (v11 * 0.041556001),
            (v10 *  0.055647999) - (m_dominant_light_intensity * 0.204043) + (v11 * 1.0573111),
        ];
        // sun_direction from (m_dominant_light_phi, m_dominant_light_theta).
        // Tag schema names them "Heading [0..360]" (φ) and "Pitch [0..90]" (θ).
        // 0.0055555557 ≈ 1/180; the full factor is (deg × π / 180).
        let phi_rad   = (0.0055555557_f32 * parameters.sun_heading) * std::f32::consts::PI;
        let theta_rad = (0.0055555557_f32 * parameters.sun_pitch)   * std::f32::consts::PI;
        let v12 = phi_rad.cos();
        let v13 = phi_rad.sin();
        let sun_direction = [
            v12 * theta_rad.sin(),
            v13 * theta_rad.sin(),
            theta_rad.cos(),
        ];
        (sun_intensity, sun_direction)
    } else {
        // Engine `get_sun_constants_from_sky @ 0x1803ADCB0` (non-override
        // branch): both intensity and direction come from the sky model's
        // `lightgen_lights[last]` — intensity already folded as
        // `×solid_angle ×0.2 ×g_render_light_intensity` in the loader,
        // direction read verbatim. When the sky has no lightgen lights the
        // engine defaults to a white sun pointing straight down
        // (`sun_intensity=(1,1,1)`, `*sun_direction=global_down3d=(0,0,-1)`).
        match sky_sun {
            Some(s) => (s.intensity, s.direction),
            None => {
                use std::sync::atomic::{AtomicBool, Ordering};
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[atmosphere] sky has no lightgen lights — using engine default \
                         white sun pointing down (warned once)",
                    );
                }
                ([1.0, 1.0, 1.0], [0.0, 0.0, -1.0])
            }
        }
    }
}

impl GpuAtmosphereData {
    /// Build from a `blam_tags::sky_atmosphere::SkyAtmosphere` walked
    /// from the scenario's `atmospheric` tag-ref. Picks the primary
    /// (first-enabled) atmosphere setting per
    /// `SkyAtmosphere::primary_setting`. Mirrors Halo's
    /// `populate_atmosphere_parameters` + `set_constant` chain:
    /// derives β_* via `precompute_betas` (spectral approximation),
    /// then packs the 7-vec4 cbuffer Halo expects (with the
    /// `sun_intensity / (β_m+β_p)` normalization).
    pub fn from_sky_atmosphere(atm: &blam_tags::sky_atmosphere::SkyAtmosphere) -> Self {
        Self::from_sky_atmosphere_with_exposure(atm, 0.67, None)
    }

    pub fn from_sky_atmosphere_with_exposure(
        atm: &blam_tags::sky_atmosphere::SkyAtmosphere,
        view_exposure: f32,
        sky_sun: Option<SkySun>,
    ) -> Self {
        match atm.primary_setting() {
            None => Self::neutral(),
            Some(s) => Self::from_atmosphere_setting(s, view_exposure, sky_sun),
        }
    }

}

/// Find the BSP cluster containing `eye`, by AABB containment test.
/// Falls back to nearest-cluster-by-center when no AABB contains the
/// point. Mirrors the protomorph decorator path's `find_cluster`. Engine
/// uses portal-aware BFS for the full multi-cluster blend; for the MP
/// fast-path the starting cluster is sufficient.
pub(crate) fn find_cluster_for_eye(
    clusters: &[blam_tags::structure_bsp::BspCluster],
    eye: glam::Vec3,
) -> Option<usize> {
    if clusters.is_empty() {
        return None;
    }
    for (i, c) in clusters.iter().enumerate() {
        if c.bounds_x.contains(eye.x)
            && c.bounds_y.contains(eye.y)
            && c.bounds_z.contains(eye.z)
        {
            return Some(i);
        }
    }
    let mut best = (f32::INFINITY, 0usize);
    for (i, c) in clusters.iter().enumerate() {
        let cx = (c.bounds_x.lower + c.bounds_x.upper) * 0.5;
        let cy = (c.bounds_y.lower + c.bounds_y.upper) * 0.5;
        let cz = (c.bounds_z.lower + c.bounds_z.upper) * 0.5;
        let d2 = (eye - glam::Vec3::new(cx, cy, cz)).length_squared();
        if d2 < best.0 {
            best = (d2, i);
        }
    }
    Some(best.1)
}

impl GpuAtmosphereData {

    /// Build atmosphere cbuffer data from a single `AtmosphereSettings`
    /// element. Used both for primary-setting selection and per-setting
    /// diagnostic dumps.
    ///
    /// `sky_sun` is the engine's `get_sun_constants_from_sky @
    /// 0x1803adcb0` result for the active scenario sky — i.e.,
    /// `lightgen_lights[last].intensity × solid_angle × 0.2 ×
    /// g_render_light_intensity`. The engine consumes this WHEN the
    /// atmosphere setting's flag bit 1 ("Override Real Sun Values") is
    /// UNSET. When bit 1 IS set, the engine uses the atmosphere's own
    /// `dominant_light_color × dominant_light_intensity` (with an
    /// xyY→XYZ→linear-RGB chromaticity-preserving conversion that we
    /// approximate as direct multiply for v1).
    pub fn from_atmosphere_setting(
        s: &blam_tags::sky_atmosphere::AtmosphereSettings,
        view_exposure: f32,
        sky_sun: Option<SkySun>,
    ) -> Self {
        let ((beta_m, beta_p), (beta_m_angular, beta_p_angular)) =
            precompute_betas(s.rayleigh_multiplier, s.mie_multiplier, s.desaturation);

        let log2e = 1.442695_f32;
        // Engine `c_atmosphere_fog_interface::get_sun_parameters @
        // 0x1803AF990` produces both `sun_intensity` and `sun_direction`
        // from the setting (override-real-sun branch reads the
        // atmosphere's own `m_dominant_light_*` fields).
        let (sun_int, sun_dir) = get_sun_parameters(s, sky_sun);
        let denom = [
            (beta_m[0] + beta_p[0]).max(1e-6),
            (beta_m[1] + beta_p[1]).max(1e-6),
            (beta_m[2] + beta_p[2]).max(1e-6),
        ];
        let g = s.sun_phase_function;
        let one_minus_g2 = 1.0 - g * g;

        // Halo's slot1.w doubles as the disable flag — engine
        // `set_constant @ 0x1803AE530` writes `max(max_fog_thickness,
        // 0.1)` when atmosphere is enabled (the 0.1 floor prevents
        // authored-zero `max_fog_thickness` from being mis-read as
        // disable). When NOT enabled, engine writes -1 to disable.
        let thickness_or_disable = if s.is_enabled() {
            s.max_fog_thickness.max(0.1)
        } else {
            -1.0
        };
        Self {
            slot0_sun_dir_dist_bias: [sun_dir[0], sun_dir[1], sun_dir[2], s.distance_bias],
            // sun_int_norm = sun_intensity / (β_m + β_p) — Halo's
            // `SUN_INTENSITY_OVER_TR_PLUS_TM`.
            slot1_sun_int_norm_thickness: [sun_int[0]/denom[0], sun_int[1]/denom[1], sun_int[2]/denom[2], thickness_or_disable],
            slot2_beta_m_log2e_g1: [beta_m[0]*log2e, beta_m[1]*log2e, beta_m[2]*log2e, 1.0 + g],
            slot3_beta_p_log2e_refh: [beta_p[0]*log2e, beta_p[1]*log2e, beta_p[2]*log2e, s.sea_level],
            slot4_beta_m_angular_mieh: [beta_m_angular[0], beta_m_angular[1], beta_m_angular[2], s.mie_height_scale],
            slot5_beta_p_angular_rayh: [beta_p_angular[0]*one_minus_g2, beta_p_angular[1]*one_minus_g2, beta_p_angular[2]*one_minus_g2, s.rayleigh_height_scale],
            slot6_g2: [2.0*g, 0.0, 0.0, 0.0],
            // slot7.x = view_exposure (Halo's `g_exposure.r`).
            // slot7.y/z/w = sun disc params (angular_radius, edge_softness, disc_intensity).
            slot7_sun_disc: [view_exposure, 0.0093, 0.001, s.intensity],
            slot8_sun_glow: [0.5, 1.0, 1.0, 0.0],
            slot9_sun_tint_horizon: [s.color.red, s.color.green, s.color.blue, -0.1],
            slot10_horizon_pad: [0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Phase 5 — build cbuffer from the engine-faithful accumulated
    /// [`WeightedAtmosphereParameters`] instead of a single picked
    /// `AtmosphereSettings`. Slots 0-6 are populated per
    /// `c_atmosphere_fog_interface::set_constant @ 0x1803AE530`:
    /// ```text
    /// slot 0 = (normalize3d_with_default(sun_dir, global_down3d), distance_bias)
    /// slot 1 = (sun_intensity / (β_m + β_p) per channel, max(thickness, 0.1))
    /// slot 2 = (β_m × log2(e), heyey_greenstein + 1)
    /// slot 3 = (β_p × log2(e), reference_datum_plane)
    /// slot 4 = (β_m_angular, mie_height)
    /// slot 5 = (β_p_angular × (1 - g²), rayleigh_height)
    /// slot 6 = (2g, 0, 0, 0)        // engine broadcasts; we pack .x
    /// ```
    /// `params.atmosphere_enabled == 0` ⇒ disable path: only slot1.w =
    /// −1 set (engine's `disable_atmosphere`). Rest neutral.
    ///
    /// Slots 7-10 (sun disc, glow, tint, horizon — protomorph sky-pass
    /// extras, not engine `set_constant` output) source from
    /// `sky_extras_source` when provided. Phase 5.3 will move these onto
    /// a dedicated sky-pass cbuffer per engine separation.
    pub fn from_weighted_parameters(
        params: &super::atmosphere_fog_interface::WeightedAtmosphereParameters,
        sky_extras_source: Option<&blam_tags::sky_atmosphere::AtmosphereSettings>,
        view_exposure: f32,
    ) -> Self {
        if params.atmosphere_enabled == 0 {
            return Self::neutral();
        }

        // Engine `normalize3d_with_default(sun_direction, global_down3d)`.
        let len_sq = params.sun_direction.i * params.sun_direction.i
            + params.sun_direction.j * params.sun_direction.j
            + params.sun_direction.k * params.sun_direction.k;
        let sun = if len_sq > 1e-12 {
            let inv_len = len_sq.sqrt().recip();
            [
                params.sun_direction.i * inv_len,
                params.sun_direction.j * inv_len,
                params.sun_direction.k * inv_len,
            ]
        } else {
            // `global_down3d` per engine.
            [0.0, 0.0, -1.0]
        };

        let log2e = 1.442695_f32;
        let g = params.heyey_greenstein;
        let one_minus_g2 = 1.0 - g * g;

        // Engine guards: `if (β_m.x + β_p.x <= 0.0001 || β_m.y + β_p.y
        // <= 0.0001 || β_m.z + β_p.z <= 0.0001) sun_int_norm = 0; else
        // sun_int_norm = sun_intensity / (β_m + β_p)`.
        let sum_x = params.beta_m.i + params.beta_p.i;
        let sum_y = params.beta_m.j + params.beta_p.j;
        let sum_z = params.beta_m.k + params.beta_p.k;
        let sun_int_norm = if sum_x <= 1.0e-4 || sum_y <= 1.0e-4 || sum_z <= 1.0e-4 {
            [0.0_f32, 0.0, 0.0]
        } else {
            [
                params.sun_intensity.red / sum_x,
                params.sun_intensity.green / sum_y,
                params.sun_intensity.blue / sum_z,
            ]
        };

        // Engine `slot1.w = max(max_fog_thickness, 0.1)`.
        let thickness = params.max_fog_thickness.max(0.1);

        // Sky extras — caller picks the dominant setting and passes it
        // through. Neutral fallback when no setting is available.
        let (sun_disc_intensity, sun_color) = match sky_extras_source {
            Some(s) => (s.intensity, [s.color.red, s.color.green, s.color.blue]),
            None => (1.0, [1.0, 1.0, 1.0]),
        };

        Self {
            slot0_sun_dir_dist_bias: [sun[0], sun[1], sun[2], params.distance_bias],
            slot1_sun_int_norm_thickness: [sun_int_norm[0], sun_int_norm[1], sun_int_norm[2], thickness],
            slot2_beta_m_log2e_g1: [
                params.beta_m.i * log2e,
                params.beta_m.j * log2e,
                params.beta_m.k * log2e,
                1.0 + g,
            ],
            slot3_beta_p_log2e_refh: [
                params.beta_p.i * log2e,
                params.beta_p.j * log2e,
                params.beta_p.k * log2e,
                params.reference_datum_plane,
            ],
            slot4_beta_m_angular_mieh: [
                params.beta_m_angular.i,
                params.beta_m_angular.j,
                params.beta_m_angular.k,
                params.mie_height,
            ],
            slot5_beta_p_angular_rayh: [
                params.beta_p_angular.i * one_minus_g2,
                params.beta_p_angular.j * one_minus_g2,
                params.beta_p_angular.k * one_minus_g2,
                params.rayleigh_height,
            ],
            slot6_g2: [2.0 * g, 0.0, 0.0, 0.0],
            slot7_sun_disc: [view_exposure, 0.0093, 0.001, sun_disc_intensity],
            slot8_sun_glow: [0.5, 1.0, 1.0, 0.0],
            slot9_sun_tint_horizon: [sun_color[0], sun_color[1], sun_color[2], -0.1],
            slot10_horizon_pad: [0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn neutral() -> Self {
        Self {
            // Disable flag in slot1.w (-1) — atmosphere off.
            slot0_sun_dir_dist_bias: [0.0, 0.0, 1.0, 0.0],
            slot1_sun_int_norm_thickness: [0.0, 0.0, 0.0, -1.0],
            slot2_beta_m_log2e_g1: [0.0, 0.0, 0.0, 1.0],
            slot3_beta_p_log2e_refh: [0.0, 0.0, 0.0, 0.0],
            slot4_beta_m_angular_mieh: [0.0, 0.0, 0.0, 1.0],
            slot5_beta_p_angular_rayh: [0.0, 0.0, 0.0, 1.0],
            slot6_g2: [0.0, 0.0, 0.0, 0.0],
            slot7_sun_disc: [1.0, 0.0, 0.0, 0.0],
            slot8_sun_glow: [0.0, 0.0, 1.0, 0.0],
            slot9_sun_tint_horizon: [1.0, 1.0, 1.0, 0.0],
            slot10_horizon_pad: [0.0, 0.0, 0.0, 0.0],
        }
    }

}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSkyParams {
    pub inverse_view_projection: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub _pad: f32,
}
