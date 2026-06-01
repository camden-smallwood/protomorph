// `atmosphere_fx.hlsl` port — faithful, structure-and-name-matching.
//
// HLSL signature (verbatim):
//   void compute_scattering(
//       in float3 view_point,
//       in float3 world_scene_point,
//       out float3 extinction,
//       out float3 inscatter)
//
// WGSL uses `ptr<function, vec3<f32>>` for HLSL `out` params.
//
// HLSL macros (UPPERCASE) expand to atmosphere cbuffer fields:
//   ATMOSPHERE_ENABLE                     = atmosphere_enabled
//   SUN_DIR                               = sun_direction
//   DIST_BIAS                             = distance_bias
//   MAX_FOG_THICKNESS                     = max_fog_thickness
//   REFERENCE_DATUM_PLANE                 = reference_height
//   REFERENCE_MIE_HEIGHT_SCALE            = mie_height_scale
//   REFERENCE_RAY_HEIGHT_SCALE            = rayleigh_height_scale
//   TOTAL_RAYLEIGH_LOG2E                  = β_m × log2(e)
//   TOTAL_MIE_LOG2E                       = β_p × log2(e)
//   RAYLEIGH_THETA_PREFIX                 = β_m_angular_prefix
//   HEYEY_GREENSTEIN_CONSTANT_PLUS_ONE    = 1 + g
//   HEYEY_GREENSTEIN_CONSTANT_TIMES_TWO   = 2g
//   MIE_THETA_PREFIX_HGC                  = (1 - g²) × β_p_angular_prefix
//   SUN_INTENSITY_OVER_TR_PLUS_TM         = sun_intensity / (β_m + β_p)
//   k_log2_e                              = 1.4426950
//
// All sourced from `engine_atmosphere` (slot map mirrors Halo's
// `c_atmosphere_fog_interface::set_constant @ 0x1803ae600`).

const k_log2_e: f32 = 1.4426950;

// DIAGNOSTIC: when true, force every opaque entry-point's atmosphere
// term to a no-op (extinction=1, inscatter=0). Used to bisect the
// "everything washed-out white" symptom — if bypass restores
// saturated lit color across distance, the problem is in the
// atmosphere term (cbuffer values or formula); if it doesn't, the
// problem is upstream (exposure / sqrt / lit).
const ATMOSPHERE_BYPASS: bool = false;

fn compute_scattering(
    view_point: vec3<f32>,
    world_scene_point: vec3<f32>,
    extinction: ptr<function, vec3<f32>>,
    inscatter: ptr<function, vec3<f32>>,
) {
    if (ATMOSPHERE_BYPASS) {
        *extinction = vec3<f32>(1.0);
        *inscatter = vec3<f32>(0.0);
        return;
    }

    let atm = make_engine_atmosphere();

    // Halo uses slot1.w (max_fog_thickness) as the disable sentinel —
    // negative when atmosphere is off (atmosphere_fx.hlsl line 21+37).
    if (atm.max_fog_thickness < 0.0) {
        *extinction = vec3<f32>(1.0, 1.0, 1.0);
        *inscatter = vec3<f32>(0.0);
        return;
    }

    var view_vector = view_point - world_scene_point;
    var dist = length(view_vector);
    if (dist < 1e-6) {
        *extinction = vec3<f32>(1.0);
        *inscatter = vec3<f32>(0.0);
        return;
    }
    view_vector = view_vector / dist;
    let c_theta = -dot(view_vector, atm.sun_direction);

    // HLSL line 49-50: dist = max(dist + DIST_BIAS, 0); dist = min(dist, MAX_FOG_THICKNESS).
    // Riverworld's haze_level setting has Distance Bias = -15, meaning
    // close-range fog is suppressed for the first 15 world units.
    dist = max(dist + atm.distance_bias, 0.0);
    dist = min(dist, atm.max_fog_thickness);

    var view_height = max(view_point.z - atm.reference_height, 0.0);
    var scene_height = max(world_scene_point.z - atm.reference_height, 0.0);
    let diff = view_height - scene_height;

    view_height = view_height * k_log2_e;
    scene_height = scene_height * k_log2_e;

    let mie_h = max(atm.mie_height_scale, 1e-4);
    let ray_h = max(atm.rayleigh_height_scale, 1e-4);

    if (diff * diff > 0.001) {
        let dp = -(exp2(-view_height / mie_h) - exp2(-scene_height / mie_h)) * dist * mie_h / diff;
        let dm = -(exp2(-view_height / ray_h) - exp2(-scene_height / ray_h)) * dist * ray_h / diff;
        *extinction = exp2(-(atm.beta_m_log2e * dm + atm.beta_p_log2e * dp));
    } else {
        let dp = exp2(-view_height / mie_h) * dist;
        let dm = exp2(-view_height / ray_h) * dist;
        *extinction = exp2(-(atm.beta_m_log2e * dm + atm.beta_p_log2e * dp));
    }

    // Rayleigh phase: RAYLEIGH_THETA_PREFIX × (1 + cos²θ)
    let beta_m_theta = atm.beta_m_angular * (1.0 + c_theta * c_theta);

    // Mie HG phase: MIE_THETA_PREFIX_HGC × (1+g²-2g·cosθ)^-1.5
    // (β_p_angular is already pre-multiplied by (1-g²) in set_constant.)
    let heyey_term = atm.mie_g_plus_one - atm.mie_g_times_two * c_theta;
    let heyey_term_one_pt_five = pow(max(heyey_term, 1e-4), -1.5);
    let beta_p_theta = atm.beta_p_angular * heyey_term_one_pt_five;

    // SUN_INTENSITY_OVER_TR_PLUS_TM × (β_m_θ + β_p_θ) × (1 - extinction)
    *inscatter = atm.sun_intensity_normalized
              * (beta_m_theta + beta_p_theta)
              * (vec3<f32>(1.0) - *extinction);
}

// `BLEND_FOG_INSCATTER_SCALE` from `blend_fx.hlsl` — picks whether
// inscatter is added to the PS output. Substituted per-variant during
// WGSL assembly (mirrors HLSL's offline-compiler `#define` per blend_mode):
//   opaque                : 1.0
//   alpha_blend           : 1.0
//   pre_multiplied_alpha  : 1.0
//   additive              : 0.0   ← clouds_hazy_wisps + every additive cloud
//   multiply / double_multiply : take BLEND_MULTIPLICATIVE branch instead
//                                (no scattering applied — not yet ported)
const BLEND_FOG_INSCATTER_SCALE: f32 = __BLEND_FOG_INSCATTER_SCALE__;

// Backwards-compat alias (entry points still use FOG_INSCATTER_SCALE
// at call site; one short transitional turn before the entry rename).
const FOG_INSCATTER_SCALE: f32 = BLEND_FOG_INSCATTER_SCALE;

// `g_exposure.r` — Halo's `c_camera_fx_values::get_render_exposure
// @ 0x18068e3e0`. Sourced from `scenario.camera_fx_settings.exposure
// .exposure` (stops); for riverworld → ~0.67.
//
// Engine layout (B1): full vec4 = `(view_exposure, pow(2, HDR_target_stops),
// 1.0, 1.0)`. The shader corpus uses `.r/.rrr` exclusively (`g_exposure.y/z/w`
// not read in the extracted HLSL); helper returns the .x scalar. Use
// `g_exposure_vec4()` for the full vector.
fn g_exposure() -> f32 {
    return engine_exposure.g_exposure.x;
}

fn g_exposure_vec4() -> vec4<f32> {
    return engine_exposure.g_exposure;
}

// `ILLUM_SCALE` aka `g_alt_exposure.r` — engine constant the HLSL
// umbrellas multiply self-illumination by (entry_points_fx.hlsl
// line 390 in the linear-with-dominant umbrella, plus quadratic /
// per_vertex variants).
//
// Engine layout (B1): full vec4 = `(illum_scale, illum_scale × view_exposure,
// 0, 0)`. `.r` (ILLUM_SCALE) and `.g` (ILLUM_EXPOSURE — used by the alt-
// exposure umbrella variant) are the live components; .b/.a are 0. Sourced
// from `c_camera_fx_values::self_illum_scale` (camera_fx_settings.rs:344).
fn g_alt_exposure() -> f32 {
    return engine_exposure.g_alt_exposure.x;
}

fn g_alt_exposure_vec4() -> vec4<f32> {
    return engine_exposure.g_alt_exposure;
}

// `ILLUM_EXPOSURE` (#define in HLSL) = `g_alt_exposure.g` —
// `illum_scale × view_exposure`. Used by the alt-exposure umbrella variant.
fn illum_exposure() -> f32 {
    return engine_exposure.g_alt_exposure.y;
}
