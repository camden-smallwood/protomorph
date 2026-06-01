// Engine bindings shared across every assembled Halo shader variant.
// (Camera, engine lighting, per-model uniforms, bone matrices.)
// Doesn't change per variant; emitted verbatim before the per-variant
// material bindings + cbuffer struct.

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// dllcache `LightingPS` cbuffer — slot 0x30/0x2E. Ravi-packed SH3
// only (10 vec4). Written per-cluster by
// `calculate_and_set_ravi_constants_internal`. The Rust-side packer
// `pack_ravi_constants` (see `halo::lighting`) shuffles the 9-float
// SH3 per channel into this layout including sign flips and the √3
// L2-m=0 multiplier.
struct EngineLightingPs {
    ravi: array<vec4<f32>, 10>,
}
@group(0) @binding(1) var<uniform> engine_lighting_ps: EngineLightingPs;

// dllcache dominant-light cbuffer — slot 0x24. Direction is the
// world-space unit vector pointing FROM fragment TO the dominant
// light (engine convention); intensity is RGB radiance. Written
// per-cluster alongside `EngineLightingPs` by `c_lighting_interface`.
// Bound separately at binding 13 to mirror the engine's two-slot
// layout (instead of fusing into one struct). Per-draw dynamic
// offset shares the value used for binding 1.
struct EngineDominantLight {
    direction: vec4<f32>,
    intensity: vec4<f32>,
}
@group(0) @binding(13) var<uniform> dominant_light: EngineDominantLight;

// Atmospheric scattering parameters — sourced from
// `scenario.atmospheric` → `.sky_atm_parameters` tag's primary
// atmosphere setting (haze_level, etc.). Layout matches
// `GpuAtmosphereData` in env_probe_pass.rs (already used elsewhere
// in the renderer for sky/env probe rendering); reusing here so all
// halo shaders share the same source-of-truth.
// Mirrors Halo's `c_atmosphere_fog_interface::set_constant` 7-vec4
// packing (cbuffer slots 0x29 VS / 0x3F PS) plus screen-sky extras.
// See `GpuAtmosphereData` in env_probe_pass.rs for the wire layout.
// Each slot is a vec4 to dodge Rust↔WGSL vec3 alignment confusion.
struct EngineAtmosphereRaw {
    // Halo HLSL slot map:
    //   slot0.w = DIST_BIAS (atmosphere_fx.hlsl line 20)
    //   slot1.w = ATMOSPHERE_ENABLE / MAX_FOG_THICKNESS (double-purposed,
    //             line 21-22 — negative when disabled, positive when
    //             enabled where the value is the actual fog clamp).
    slot0_sun_dir_dist_bias: vec4<f32>,        // (sun_dir.xyz, distance_bias)
    slot1_sun_int_norm_thickness: vec4<f32>,   // (sun_int / (β_m+β_p), max_fog_thickness OR -1 to disable)
    slot2_beta_m_log2e_g1: vec4<f32>,          // (β_m × log2(e), g+1)
    slot3_beta_p_log2e_refh: vec4<f32>,        // (β_p × log2(e), reference_height)
    slot4_beta_m_angular_mieh: vec4<f32>,      // (β_m_angular, mie_height)
    slot5_beta_p_angular_rayh: vec4<f32>,      // ((1-g²)×β_p_angular, rayleigh_height)
    slot6_g2: vec4<f32>,                       // (2g, _, _, _)
    slot7_sun_disc: vec4<f32>,                 // (luminance, ang_radius, edge_soft, disc_intensity)
    slot8_sun_glow: vec4<f32>,                 // (inner_glow, air_mass_scale, zenith_factor, _)
    slot9_sun_tint_horizon: vec4<f32>,         // (tint.xyz, horizon_fade_start)
    slot10_horizon_pad: vec4<f32>,             // (horizon_fade_end, _, _, _)
}
@group(0) @binding(2) var<uniform> engine_atmosphere_raw: EngineAtmosphereRaw;

// Friendly accessor — keeps the rest of the WGSL referencing
// `engine_atmosphere.<field>` rather than the slot tuples.
struct EngineAtmosphere {
    sun_direction: vec3<f32>,
    distance_bias: f32,
    sun_intensity_normalized: vec3<f32>,
    /// Positive: fog distance clamp. Negative: atmosphere disabled
    /// (mirrors Halo's `ATMOSPHERE_ENABLE < 0.0` check).
    max_fog_thickness: f32,
    beta_m_log2e: vec3<f32>,
    mie_g_plus_one: f32,
    beta_p_log2e: vec3<f32>,
    reference_height: f32,
    beta_m_angular: vec3<f32>,
    mie_height_scale: f32,
    beta_p_angular: vec3<f32>,
    rayleigh_height_scale: f32,
    mie_g_times_two: f32,
    view_exposure: f32,
    sun_luminance: f32,
    sun_angular_radius: f32,
    sun_edge_softness: f32,
    sun_disc_intensity: f32,
    sun_inner_glow_intensity: f32,
    sun_air_mass_scale: f32,
    sun_tint: vec3<f32>,
    zenith_air_mass_factor: f32,
    horizon_fade_start: f32,
    horizon_fade_end: f32,
}
fn make_engine_atmosphere() -> EngineAtmosphere {
    let r = engine_atmosphere_raw;
    var out: EngineAtmosphere;
    out.sun_direction = r.slot0_sun_dir_dist_bias.xyz;
    out.distance_bias = r.slot0_sun_dir_dist_bias.w;
    out.sun_intensity_normalized = r.slot1_sun_int_norm_thickness.xyz;
    out.max_fog_thickness = r.slot1_sun_int_norm_thickness.w;
    out.beta_m_log2e = r.slot2_beta_m_log2e_g1.xyz;
    out.mie_g_plus_one = r.slot2_beta_m_log2e_g1.w;
    out.beta_p_log2e = r.slot3_beta_p_log2e_refh.xyz;
    out.reference_height = r.slot3_beta_p_log2e_refh.w;
    out.beta_m_angular = r.slot4_beta_m_angular_mieh.xyz;
    out.mie_height_scale = r.slot4_beta_m_angular_mieh.w;
    out.beta_p_angular = r.slot5_beta_p_angular_rayh.xyz;
    out.rayleigh_height_scale = r.slot5_beta_p_angular_rayh.w;
    out.mie_g_times_two = r.slot6_g2.x;
    // slot7.x repurposed for view_exposure (g_exposure.r). Sun
    // luminance is now the same as sun_disc_intensity (.w).
    out.view_exposure = r.slot7_sun_disc.x;
    out.sun_luminance = r.slot7_sun_disc.w;
    out.sun_angular_radius = r.slot7_sun_disc.y;
    out.sun_edge_softness = r.slot7_sun_disc.z;
    out.sun_disc_intensity = r.slot7_sun_disc.w;
    out.sun_inner_glow_intensity = r.slot8_sun_glow.x;
    out.sun_air_mass_scale = r.slot8_sun_glow.y;
    out.zenith_air_mass_factor = r.slot8_sun_glow.z;
    out.sun_tint = r.slot9_sun_tint_horizon.xyz;
    out.horizon_fade_start = r.slot9_sun_tint_horizon.w;
    out.horizon_fade_end = r.slot10_horizon_pad.x;
    return out;
}

struct ModelUniforms { model: mat4x4<f32>, }
@group(1) @binding(0) var<uniform> model_u: ModelUniforms;

@group(3) @binding(0) var<storage, read> node_matrices: array<mat4x4<f32>>;

// Lightmap atlas + dominant-light intensity atlas + SH compression
// scalars. Halo `LightmapCompressPS` cbuffer (DX11 ps register cb19;
// see `hlsl_constant_persist.h:118-122`).
//
// Atlas formats per riverworld inspection:
//   lightprobe_atlas:           1024x1024 BC3 array, 8 slices
//                                (4 SH coefs × 2 L*UVW pairs)
//   lightprobe_intensity_atlas: 1024x1024 BC3 array, 2 slices
//                                (1 vec3 × L*UVW pair)
//
// `lightmap_compress` carries the per-coefficient L-range scalars from
// `LightmapBspData.compression_vectors[2k].x` for k ∈ [0..4]:
//   .x.xyz = (sh_coef_0, sh_coef_1, sh_coef_2) ranges
//   .y.x   = sh_coef_3 range  (.y/.z unused at order-2)
//   .y.w   = dominant_light_intensity range
struct LightmapCompress {
    constant_0: vec4<f32>,
    constant_1: vec4<f32>,
    constant_2: vec4<f32>,
}
@group(0) @binding(3) var<uniform> lightmap_compress: LightmapCompress;
@group(0) @binding(4) var lightprobe_atlas: texture_2d_array<f32>;
@group(0) @binding(5) var lightprobe_intensity_atlas: texture_2d_array<f32>;
@group(0) @binding(6) var lightmap_atlas_sampler: sampler;

// `EngineUnderwater` — global underwater-fog state for transparent
// shaders. Engine source: `g_is_underwater` + `g_underwater_murkiness`
// + `g_underwater_fog_color` (uploaded by `c_water_renderer::render_*`).
// Phase B6: every transparent PS samples this to apply the canonical
// `apply_underwater_fog(color, view_dist)` lerp toward fog_color when
// `k_is_camera_underwater`. Mirrors the in-shader underwater branch
// from `water_shading_fx.hlsl:1042-1046` and the fullscreen
// `underwater_ps` in `water_ripple_hlsl.hlsl:919-920`.
struct EngineUnderwater {
    fog_color: vec4<f32>,
    /// .x = `k_ps_underwater_murkiness`. .y/.z/.w unused.
    murkiness: vec4<f32>,
    /// .x = `g_is_underwater` (0 or 1, `u32` reinterpreted as f32 to
    /// keep camera_bgl uniformly std140-aligned). PS treats any
    /// non-zero value as underwater.
    is_underwater_flags: vec4<f32>,
}
@group(0) @binding(7) var<uniform> engine_underwater: EngineUnderwater;

// `EngineExposure` (B1) — `g_exposure` + `g_alt_exposure` PS reg slots
// written by engine `setup_render_target_globals_with_exposure
// @ 0x180670AD0`. Both are vec4 in HLSL; protomorph plumbs them faithfully
// here. Bound at `@group(0) @binding(9)`.
//
// Layout:
//   g_exposure     = (view_exposure, pow(2, HDR_target_stops), 1.0, 1.0)
//   g_alt_exposure = (illum_scale,   illum_scale × view_exposure, 0, 0)
//
// HLSL fields used: `g_exposure.r` (`.rrr` everywhere), `g_alt_exposure.r`
// (#define ILLUM_SCALE), `g_alt_exposure.g` (#define ILLUM_EXPOSURE in the
// alt-exposure umbrella variant).
struct EngineExposure {
    g_exposure: vec4<f32>,
    g_alt_exposure: vec4<f32>,
}
@group(0) @binding(9) var<uniform> engine_exposure: EngineExposure;

// `MiscPS` cbuffer — engine `hlsl_constant_persist_fx.hlsl:101`. Holds
// per-frame globals consumed across material shaders: render-target
// dims, dynamic env-probe blend (formerly per-material baked at 0.5),
// debug pin, and packed flags (pc_specular_enabled / pc_albedo_lighting
// / LDR_gamma2 / HDR_gamma2). Layout mirrors `GpuEngineMiscPs` in
// `src/halo/render/shared.rs`.
struct EngineMiscPs {
    /// `.xy` = render-target dims (pixels). `.zw` pad.
    texture_size: vec4<f32>,
    /// 3-channel blend between two dynamic env probes (current /
    /// previous). Default `(0.5,0.5,0.5,0)` matches the legacy
    /// per-material baseline.
    dynamic_environment_blend: vec4<f32>,
    /// `p_render_debug_mode` — engine debug pin (unused at runtime).
    render_debug_mode: vec4<f32>,
    /// `.x` = pc_specular_enabled, `.y` = pc_albedo_lighting,
    /// `.z` = LDR_gamma2, `.w` = HDR_gamma2.
    flags: vec4<f32>,
    /// `.x` = `ps_total_time` (seconds since engine boot).
    ps_total_time: vec4<f32>,
    /// `.x` = `actually_calc_albedo` (1.0 = sample material textures).
    actually_calc_albedo: vec4<f32>,
}
@group(0) @binding(12) var<uniform> engine_misc_ps: EngineMiscPs;

// G-buffer textures populated by `_entry_point_albedo` (first pass).
// Static-lighting PSs sample these via `textureLoad(t, frag_pos, 0)`
// to skip recomputing albedo + bump_normal from material textures.
// Engine equivalent: `albedo_texture` + `normal_texture` externs in
// entry_points_fx.hlsl::get_albedo_and_normal (line 25-46 of the
// extracted HLSL).
//
// `_surface_albedo` — RGB albedo, A = specular_mask (per
// `convert_to_albedo_target`'s output).
@group(0) @binding(10) var albedo_texture: texture_2d<f32>;
// `_surface_post_HDR` MRT[1] — encoded world normal (`n*0.5+0.5`).
// W = albedo.w copy.
@group(0) @binding(11) var normal_texture: texture_2d<f32>;

// Engine `Kernel5PS` cbuffer (pool slot 0x33 in the engine cbuffer
// inventory). Carries PS bool flags written via
// `c_rasterizer::set_shader_constant_bool(slot_id | 0x8000000, …)` —
// the wrapper ORs bit 27 and routes to the slot ID's high-byte cbuffer.
//
// We declare a 6-vec4 array for entries 0..=5 because the entry we
// care about (`actually_calc_albedo` at byte 88 = entry 5, sub_byte 8)
// is the highest known offset. Each entry is a `vec4<u32>` because
// `set_shader_constant_bool` packs ints, and a bool occupies the int's
// low byte. To read `actually_calc_albedo`:
//
//   let calc = misc_bool_ps.entries[5].z;   // entry 5, sub_byte 8 → .z lane
//
// `calc == 0u` means "use G-buffer" (engine semantics: actually_calc_albedo
// is the LOGICAL NEGATION of `using_albedo_sampler`).
struct MiscBoolPs {
    entries: array<vec4<u32>, 6>,
}
@group(0) @binding(14) var<uniform> misc_bool_ps: MiscBoolPs;

// `apply_underwater_fog` — engine port of the underwater-fog mix used
// by every transparent PS when the camera is below water. Mirrors
// `water_ripple_hlsl.hlsl:919-920` (fullscreen pass) and
// `water_shading_fx.hlsl:1042-1046` (water-surface pass):
//   transparence = 0.5 * saturate(1 - compute_fog_factor(murkiness, dist))
//   output       = lerp(fog_color, color, transparence)
// `view_dist` is the per-pixel camera-to-fragment distance (already
// computed by most PSs for atmospheric scattering).
fn apply_underwater_fog(color: vec3<f32>, view_dist: f32) -> vec3<f32> {
    if (engine_underwater.is_underwater_flags.x == 0.0) {
        return color;
    }
    let murk = engine_underwater.murkiness.x;
    let fog_factor = 1.0 - clamp(1.0 / exp(murk * view_dist), 0.0, 1.0);
    let transparence = 0.5 * clamp(1.0 - fog_factor, 0.0, 1.0);
    return mix(engine_underwater.fog_color.rgb, color, transparence);
}
