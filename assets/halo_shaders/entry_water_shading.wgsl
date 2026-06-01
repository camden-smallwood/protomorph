// Halo `water_shading_non_tessellation_*` entry point — minimal port
// of `water_shading()` (water_shading_fx.hlsl:658-1062). PC riverworld
// path skips the tessellated branch — water_fx.hlsl:51-57 redirects
// `water_shading_non_tessellation_vs/ps` onto the engine's
// `static_per_vertex_vs/ps` slot, with the rmw rmt2 binding the water
// pixl bytecode for that slot.
//
// Phase A4d scope: refraction sample from scene_ldr_snapshot + Schlick
// fresnel + sky-tint reflection. Single-material — no option-driven
// branching yet (riverworld has one rmw material; multi-material
// levels need Phase A4b/c). Foam, dynamic lights, env cubemap,
// bank alpha, atmospheric scattering, VS displacement, ripple all
// land in Phases A6-A9.

// ---- Camera + lightmap atlas (engine camera_bgl, @group(0)) ----
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// `lightmap_compress` carries the L-range scalars used to dequantize
// the BC3 atlas samples back to floating-point SH coefficients. Layout
// per `engine_bindings.wgsl`:
//   constant_0.x = sh[0] L-range  (the DC / constant term — ALL water needs)
//   constant_0.yz, constant_1.x = sh[1..3] L-ranges (unused by water)
//   constant_1.y = dominant_light L-range (unused by water)
struct LightmapCompress {
    constant_0: vec4<f32>,
    constant_1: vec4<f32>,
    constant_2: vec4<f32>,
}
@group(0) @binding(3) var<uniform> lightmap_compress: LightmapCompress;
@group(0) @binding(4) var lightprobe_atlas: texture_2d_array<f32>;
@group(0) @binding(6) var lightmap_atlas_sampler: sampler;

// ---- Engine atmosphere cbuffer (Phase A6d) ----
// Same `EngineAtmosphereRaw` shape as `engine_bindings.wgsl` — slot
// map mirrors Halo's `c_atmosphere_fog_interface::set_constant @
// 0x1803ae600`. Visible to BOTH stages on camera_bgl per
// `shared.rs::create_camera_bgl` slot 2 (VERTEX_FRAGMENT). VS computes
// `compute_scattering` per-vertex; PS applies the resulting extinction
// + inscatter to the composed water color.
struct EngineAtmosphereRaw {
    slot0_sun_dir_dist_bias: vec4<f32>,
    slot1_sun_int_norm_thickness: vec4<f32>,
    slot2_beta_m_log2e_g1: vec4<f32>,
    slot3_beta_p_log2e_refh: vec4<f32>,
    slot4_beta_m_angular_mieh: vec4<f32>,
    slot5_beta_p_angular_rayh: vec4<f32>,
    slot6_g2: vec4<f32>,
    slot7_sun_disc: vec4<f32>,
    slot8_sun_glow: vec4<f32>,
    slot9_sun_tint_horizon: vec4<f32>,
    slot10_horizon_pad: vec4<f32>,
}
@group(0) @binding(2) var<uniform> engine_atmosphere_raw: EngineAtmosphereRaw;

// EngineExposure cbuffer @ binding(9). Engine
// `setup_render_target_globals_with_exposure @ 0x180683580` writes
// `g_exposure = (view_exposure, 2^hdr_target_stops, 1, 1)` for the
// rasterizer-globals PS register that `water_shading_fx.hlsl:1034`
// reads as `g_exposure.rrr` in the subtract-fog-add-back dance.
// Was previously sourced from `engine_atmosphere_raw.slot7_sun_disc.x`
// (sun_disc_intensity) — wrong cbuffer, wrong value, dimmed all
// reflection/diffuse contributions below visibility on dim scenes.
struct EngineExposure {
    g_exposure: vec4<f32>,
    g_alt_exposure: vec4<f32>,
}
@group(0) @binding(9) var<uniform> engine_exposure: EngineExposure;

const ATMOSPHERE_K_LOG2_E: f32 = 1.4426950;

// `compute_scattering` (port of `atmosphere_fx.hlsl`). Returns
// `(extinction, inscatter)` for the segment from `view_point` to
// `world_scene_point`. Output range is HDR — apply as
// `output = output * extinction + inscatter * (1 - bed_contrib)`
// per `water_shading_fx.hlsl:1040`.
fn compute_scattering_water(
    view_point: vec3<f32>,
    world_scene_point: vec3<f32>,
    extinction: ptr<function, vec3<f32>>,
    inscatter: ptr<function, vec3<f32>>,
) {
    let r = engine_atmosphere_raw;
    let max_fog = r.slot1_sun_int_norm_thickness.w;
    if (max_fog < 0.0) {
        *extinction = vec3<f32>(1.0);
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
    let sun_direction = r.slot0_sun_dir_dist_bias.xyz;
    let c_theta = -dot(view_vector, sun_direction);

    let dist_bias = r.slot0_sun_dir_dist_bias.w;
    dist = max(dist + dist_bias, 0.0);
    dist = min(dist, max_fog);

    let reference_height = r.slot3_beta_p_log2e_refh.w;
    var view_height = max(view_point.z - reference_height, 0.0);
    var scene_height = max(world_scene_point.z - reference_height, 0.0);
    let diff = view_height - scene_height;
    view_height = view_height * ATMOSPHERE_K_LOG2_E;
    scene_height = scene_height * ATMOSPHERE_K_LOG2_E;

    let mie_h = max(r.slot4_beta_m_angular_mieh.w, 1e-4);
    let ray_h = max(r.slot5_beta_p_angular_rayh.w, 1e-4);
    let beta_m_log2e = r.slot2_beta_m_log2e_g1.xyz;
    let beta_p_log2e = r.slot3_beta_p_log2e_refh.xyz;
    if (diff * diff > 0.001) {
        let dp = -(exp2(-view_height / mie_h) - exp2(-scene_height / mie_h)) * dist * mie_h / diff;
        let dm = -(exp2(-view_height / ray_h) - exp2(-scene_height / ray_h)) * dist * ray_h / diff;
        *extinction = exp2(-(beta_m_log2e * dm + beta_p_log2e * dp));
    } else {
        let dp = exp2(-view_height / mie_h) * dist;
        let dm = exp2(-view_height / ray_h) * dist;
        *extinction = exp2(-(beta_m_log2e * dm + beta_p_log2e * dp));
    }

    let beta_m_angular = r.slot4_beta_m_angular_mieh.xyz;
    let beta_p_angular = r.slot5_beta_p_angular_rayh.xyz;
    let mie_g_plus_one = r.slot2_beta_m_log2e_g1.w;
    let mie_g_times_two = r.slot6_g2.x;
    let beta_m_theta = beta_m_angular * (1.0 + c_theta * c_theta);
    let heyey_term = mie_g_plus_one - mie_g_times_two * c_theta;
    let heyey_term_one_pt_five = pow(max(heyey_term, 1e-4), -1.5);
    let beta_p_theta = beta_p_angular * heyey_term_one_pt_five;
    let sun_intensity_norm = r.slot1_sun_int_norm_thickness.xyz;
    *inscatter = sun_intensity_norm
              * (beta_m_theta + beta_p_theta)
              * (vec3<f32>(1.0) - *extinction);
}

// `decode_bpp16_luvw` (`lightmap_sampling_fx.hlsl:15-23` — bit-for-bit
// port). Reconstructs one SH coefficient from a pair of BC3 atlas
// samples that encode (L, direction) halves — alpha = L scale, RGB =
// direction. Water only uses the constant term (sh[0]) but the helper
// is the same. Cannot `#include` lightmap_sampling_fx.wgsl because
// our `LightmapCompress` declaration lives here; importing the full
// `sample_lightprobe_texture` would double-declare.
fn decode_bpp16_luvw(s0: vec4<f32>, s1: vec4<f32>, l_range: f32) -> vec3<f32> {
    let l = s0.a * s1.a * l_range;
    let uvw = s0.xyz + s1.xyz;
    return (uvw * 2.0 - 2.0) * l;
}

// ---- Water engine cbuffer trio (Phase A2 / A4a, @group(1)) ----
struct WaterPS {
    k_water_view_depth_constant: vec4<f32>,
    k_is_lightmap_exist: u32,
    k_is_water_interaction: u32,
    k_is_water_tessellated: u32,
    k_is_camera_underwater: u32,
    // Non-engine: host sets from `PROTOMORPH_DEBUG_WATER_VIZ`. 0 = normal.
    // 1 = lightmap_intensity × 5. 2 = water_color × 5. 3 = lm_tex.xy gradient.
    k_debug_viz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
struct WaterSharedPS {
    k_water_view_xform_inverse: mat4x4<f32>,
    k_water_player_view_constant: vec4<f32>,
    k_ps_camera_position: vec4<f32>,
    k_ps_underwater_murkiness: vec4<f32>,
    k_ps_underwater_fog_color: vec4<f32>,
}
struct WaterSharedVS {
    k_ripple_buffer_radius: vec4<f32>,
    k_ripple_buffer_center: vec4<f32>,
}
@group(1) @binding(0) var<uniform> water_ps: WaterPS;
@group(1) @binding(1) var<uniform> water_shared_ps: WaterSharedPS;
@group(1) @binding(2) var<uniform> water_shared_vs: WaterSharedVS;

// ---- Water extern resources (Phase A5, @group(2)) ----
// Mirrors the dllcache `c_player_view::render_water @ 0x18068c120`
// extern dispatch:
//   binding 0: extern 45 (`_render_method_extern_scene_ldr_texture`)
//              → `_surface_screenshot_simple_resolve` (LDR copy of
//              `lighting_base` taken before transparency).
//   binding 1: clamp + bilinear sampler — engine
//              `_sampler_filter_bilinear` + `_sampler_address_mode_clamp`
//              per `c_water_renderer::render_underwater_fog`.
//   binding 2: extern 3 (`_render_method_extern_texture_global_target_z`)
//              → `_surface_depth_stencil` read-only, depth aspect.
//   binding 3: clamp + point sampler for the depth read.
//   binding 4: cbuf 0x2F (`LDRTextureSize`) — `(w, h, 1/w, 1/h)`.
//   binding 5: cbuf 0x3C (`DepthConstants`) — engine formula
//              `view_z = depth_constants.x / (depth - depth_constants.y)`.
//
// Phase A6 starts reading scene_depth + DepthConstants for soft-edge
// fade and atmospheric fog distance; A5 only sources LDR (refraction).
@group(2) @binding(0) var scene_ldr_snapshot: texture_2d<f32>;
@group(2) @binding(1) var scene_ldr_sampler: sampler;
@group(2) @binding(2) var scene_depth: texture_depth_2d;
@group(2) @binding(3) var scene_depth_sampler: sampler;
struct LDRTextureSize {
    size: vec4<f32>,
}
struct DepthConstants {
    depth_constants: vec4<f32>,
}
@group(2) @binding(4) var<uniform> ldr_texture_size: LDRTextureSize;
@group(2) @binding(5) var<uniform> depth_constants: DepthConstants;

// ---- Per-rmw-material resources (Phase A6a, @group(3)) ----
// Single-material slot for now — riverworld has 1 rmw shader_water.
// Phase A4b promotes this to a per-material cache when multi-material
// levels arrive.
//
// `environment_map` is the rmt2 user-bound texture cube the shader
// samples for static reflection (water_shading_fx.hlsl:937-957). The
// engine binds via the rmt2 sampler routing (PS slot 16 on DX11) —
// here we live in @group(3) since rmw doesn't share rmsh's
// `material_bgl` shape. `WaterMaterialPS` carries the scalar rmw
// parameters the shading PS needs.
@group(3) @binding(0) var environment_map: texture_cube<f32>;
@group(3) @binding(1) var environment_map_sampler: sampler;

// `WaterMaterialPS` mirrors the rmw rmt2 user-cbuffer layout exactly
// — one `vec4<f32>` per slot in rmt2-declared source order, matching
// the bytes uploaded verbatim from `mat.cbuffer.bytes` (= the engine's
// `rmsh.postprocess.real_constants` blob). Real-typed parameters
// broadcast their scalar across all 4 components per the cache-builder
// packing (`reference_canonical_cbuffer_merge.md`); xform / color
// parameters store a vec4 directly.
//
// Order observed for `riverworld_water_rough.shader_water` (66 slots).
// `_xform` suffix denotes texture xform vec4s (scale.xy, translation.xy).
// All other slots carry a Real broadcast or a Color rgba.
struct WaterMaterialPS {
    slope_range_x: vec4<f32>,
    slope_range_y: vec4<f32>,
    wave_displacement_array_xform: vec4<f32>,
    time_warp: vec4<f32>,
    wave_slope_array_xform: vec4<f32>,
    time_warp_aux: vec4<f32>,
    detail_slope_scale_x: vec4<f32>,
    detail_slope_scale_y: vec4<f32>,
    detail_slope_scale_z: vec4<f32>,
    detail_slope_steepness: vec4<f32>,
    watercolor_coefficient: vec4<f32>,
    reflection_coefficient: vec4<f32>,
    sunspot_cut: vec4<f32>,
    shadow_intensity_mark: vec4<f32>,
    refraction_texcoord_shift: vec4<f32>,
    water_murkiness: vec4<f32>,
    refraction_extinct_distance: vec4<f32>,
    minimal_wave_disturbance: vec4<f32>,
    refraction_depth_dominant_ratio: vec4<f32>,
    bankalpha_infuence_depth: vec4<f32>,
    fresnel_coefficient: vec4<f32>,
    water_diffuse: vec4<f32>,
    foam_texture_xform: vec4<f32>,
    foam_texture_detail_xform: vec4<f32>,
    foam_height: vec4<f32>,
    foam_pow: vec4<f32>,
    displacement_range_x: vec4<f32>,
    displacement_range_y: vec4<f32>,
    displacement_range_z: vec4<f32>,
    wave_height: vec4<f32>,
    wave_height_aux: vec4<f32>,
    choppiness_forward: vec4<f32>,
    choppiness_backward: vec4<f32>,
    choppiness_side: vec4<f32>,
    wave_visual_damping_distance: vec4<f32>,
    waveshape: vec4<f32>,
    global_shape: vec4<f32>,
    base_map_xform: vec4<f32>,
    detail_map_xform: vec4<f32>,
    change_color_map_xform: vec4<f32>,
    bump_map_xform: vec4<f32>,
    bump_detail_map_xform: vec4<f32>,
    self_illum_map_xform: vec4<f32>,
    albedo_color: vec4<f32>,
    bump_detail_coefficient: vec4<f32>,
    diffuse_coefficient: vec4<f32>,
    specular_coefficient: vec4<f32>,
    normal_specular_power: vec4<f32>,
    glancing_specular_power: vec4<f32>,
    fresnel_curve_steepness: vec4<f32>,
    albedo_specular_tint_blend: vec4<f32>,
    analytical_specular_contribution: vec4<f32>,
    area_specular_contribution: vec4<f32>,
    environment_map_specular_contribution: vec4<f32>,
    analytical_anti_shadow_control: vec4<f32>,
    self_illum_intensity: vec4<f32>,
    normal_specular_tint: vec4<f32>,
    glancing_specular_tint: vec4<f32>,
    env_tint_color: vec4<f32>,
    self_illum_color: vec4<f32>,
    primary_change_color: vec4<f32>,
    secondary_change_color: vec4<f32>,
    watercolor_texture_xform: vec4<f32>,
    environment_map_xform: vec4<f32>,
    no_dynamic_lights: vec4<f32>,
    global_shape_texture_xform: vec4<f32>,
    // Phase W1 — non-riverworld variants. Order MUST match
    // `WATER_WGSL_FIELDS` tail in `render_water.rs`. The repack writes
    // each field's vec4 at offset `i * 16` matching its index in that
    // array.
    water_color_pure: vec4<f32>,           // watercolor=pure
    globalshape_infuence_depth: vec4<f32>, // global_shape=depth
}
@group(3) @binding(2) var<uniform> water_material: WaterMaterialPS;
// Phase A7a — wave_displacement_array. Engine: VS-side rmt2 texture
// sampled by `transform_vertex` (`water_shading_fx.hlsl:368-485`).
// Halo's `type=array` bitm with depth=N slices renders as a 2D array
// here. The HLSL DX11 path treats the "z" coordinate as a slice index
// (`convert_3d_texture_coord_to_array_texture` lines 399-409); for our
// minimal MVP `time_warp = 0` so we always read slice 0.
@group(3) @binding(3) var wave_displacement_array: texture_2d_array<f32>;
@group(3) @binding(4) var wave_displacement_array_sampler: sampler;
// Phase A6f — `watercolor_texture` (PS 2D). Engine line 788:
//   `sampleBiasGlobal2D(watercolor_texture, base_tex.xy)`
// Sampled at `tv.base_tex` (per-vertex base texcoord interpolated to
// per-pixel) and modulated by `watercolor_coefficient` × lightmap_intensity
// to produce the depth-tint color.
@group(3) @binding(5) var watercolor_texture: texture_2d<f32>;
@group(3) @binding(6) var watercolor_texture_sampler: sampler;
// Phase A6e — foam path. Engine line 873-921.
@group(3) @binding(7) var global_shape_texture: texture_2d<f32>;
@group(3) @binding(8) var foam_texture: texture_2d<f32>;
@group(3) @binding(9) var foam_texture_detail: texture_2d<f32>;
// Phase A6f — wave_slope_array. Engine PS texture sampled by
// `compose_slope` (`water_shading_fx.hlsl:549-626`). Same-shape
// 2D-array bitmap as wave_displacement_array but storing per-pixel
// slope (xy) keyframes. The per-pixel normal is built by rotating
// `(slope_shading.xy, 1)` by the per-vertex tangent frame; without
// this the cubemap reflection appears polygon-flat across each
// triangle (the missing per-pixel normal is the visible "blocky
// cubemap" artifact).
@group(3) @binding(10) var wave_slope_array: texture_2d_array<f32>;

// `transform_texcoord` (`texture_xform.fx`) — identical to rmsh's
// helper. xform.xy = scale, xform.zw = translation.
fn transform_texcoord(uv: vec2<f32>, xform: vec4<f32>) -> vec2<f32> {
    return uv * xform.xy + xform.zw;
}

// `restore_displacement` (`water_shading_fx.hlsl:305-316`). Maps the
// signed-normalized sample back to geometry-space displacement using
// the per-axis range/min/height authored in the rmsh.
fn restore_displacement(
    displacement: vec3<f32>,
    range: vec3<f32>,
    min_v: vec3<f32>,
    height: f32,
) -> vec3<f32> {
    return (displacement * range + min_v) * height;
}

// `apply_choppiness` (`water_shading_fx.hlsl:318-327`). Direction-
// dependent scaling: y is multiplied by `chop_side`, x by either
// `chop_forward` (when x < 0) or `chop_backward` (when x ≥ 0).
// Engine comments label `chop_side` as "backward choppiness" — the
// label is misleading but the math is direct.
fn apply_choppiness(
    displacement: vec3<f32>,
    chop_forward: f32,
    chop_backward: f32,
    chop_side: f32,
) -> vec3<f32> {
    var d = displacement;
    d.y = d.y * chop_side;
    d.x = d.x * select(chop_backward, chop_forward, d.x < 0.0);
    return d;
}

// `compute_fog_factor` (`water_shading_fx.hlsl:111-114`):
//   1 - saturate(1 / exp(murkiness * depth))
// Goes from 0 at depth=0 to 1 at large depth. Used by the refraction
// tint to fade scene_ldr toward water_color in deep areas.
fn compute_fog_factor(murkiness: f32, depth: f32) -> f32 {
    return 1.0 - clamp(1.0 / exp(murkiness * depth), 0.0, 1.0);
}

// Sample the wave_displacement_array at a given (uv, slice_position,
// mip_level). slice_position is the fractional slice index (cyclic
// over the array's layer count). Mirrors the HLSL DX11 path's
// `convert_3d_texture_coord_to_array_texture` + 2-slice lerp
// (`water_shading_fx.hlsl:399-414`). `mip_level` is the LOD passed
// to `sample3Dlod` per line 396.
fn sample_wave_displacement(uv: vec2<f32>, slice_position: f32, mip_level: f32) -> vec3<f32> {
    let n = i32(textureNumLayers(wave_displacement_array));
    let z_scaled = slice_position * f32(n);
    let z_floor = floor(z_scaled);
    let slice_lo = (i32(z_floor) % n + n) % n;
    let slice_hi = (slice_lo + 1) % n;
    let t_blend = z_scaled - z_floor;
    let s_lo = textureSampleLevel(
        wave_displacement_array, wave_displacement_array_sampler,
        uv, slice_lo, mip_level,
    ).xyz;
    let s_hi = textureSampleLevel(
        wave_displacement_array, wave_displacement_array_sampler,
        uv, slice_hi, mip_level,
    ).xyz;
    return mix(s_lo, s_hi, t_blend);
}

// `restore_slope` (`water_shading_fx.hlsl:535-547`):
//   slope = sample.xy * range + min
// where range = (slope_range_x, slope_range_y) and min = -range/2,
// so the [0,1] sample maps to [-range/2, +range/2].
fn restore_slope(slope01: vec2<f32>, range: vec2<f32>) -> vec2<f32> {
    let mn = -range * 0.5;
    return slope01 * range + mn;
}

// Sample wave_slope_array at (uv, slice_position, mip_level), lerping
// between adjacent slices like `sample_wave_displacement` does for the
// displacement array. Mirrors the HLSL DX11 path
// (`water_shading_fx.hlsl:585-607`) which does
// `convert_3d_texture_coord_to_array_texture` + 2-slice lerp.
fn sample_wave_slope(uv: vec2<f32>, slice_position: f32, mip_level: f32) -> vec2<f32> {
    let n = i32(textureNumLayers(wave_slope_array));
    let z_scaled = slice_position * f32(n);
    let z_floor = floor(z_scaled);
    let slice_lo = (i32(z_floor) % n + n) % n;
    let slice_hi = (slice_lo + 1) % n;
    let t_blend = z_scaled - z_floor;
    // wave_slope_array uses the same sampler as the displacement
    // array (engine pairs both with `_sampler_filter_anisotropic` +
    // `_sampler_address_mode_wrap`).
    let s_lo = textureSampleLevel(
        wave_slope_array, wave_displacement_array_sampler,
        uv, slice_lo, mip_level,
    ).xy;
    let s_hi = textureSampleLevel(
        wave_slope_array, wave_displacement_array_sampler,
        uv, slice_hi, mip_level,
    ).xy;
    return mix(s_lo, s_hi, t_blend);
}

// `compose_slope` (`water_shading_fx.hlsl:549-626`). Outputs two
// slope vectors:
//   slope_shading   — drives the per-pixel surface normal (TBN-space).
//   slope_refraction — drives the screen-space refraction perturbation
//                       (Phase A6g; clamped by `minimal_wave_disturbance`
//                       to keep wide-angle waves rippling the bed).
// Three samples from `wave_slope_array`:
//   slope        : main wave xform     × `time_warp`
//   slope_aux    : aux wave xform      × `time_warp_aux`
//   slope_detail : detail xform (= main × detail_slope_scale_xy) at
//                  mip+1, time = `time_warp * detail_slope_scale_z`
// All three are restored from [0,1] to [-range/2, +range/2], then
// the aux + detail contributions are added to the main per the
// engine's `slope_aux = slope_aux*hAuxScale + slope_detail*hDetail`,
// `slope_shading = slope*hScale + slope_aux`, and
// `slope_refraction = slope*max(hScale, minimal_wave_disturbance) + slope_aux`.
struct ComposedSlope {
    shading: vec2<f32>,
    refraction: vec2<f32>,
}

fn compose_slope(
    base_uv: vec2<f32>,
    height_scale: f32,
    height_aux_scale: f32,
    height_detail_scale: f32,
    mip_level: f32,
) -> ComposedSlope {
    let slope_range = vec2<f32>(
        water_material.slope_range_x.x,
        water_material.slope_range_y.x,
    );

    // Engine line 560:
    //   wave_detail_xform = wave_displacement_array_xform
    //                     * (detail_slope_scale_x, detail_slope_scale_y, 1, 1)
    // i.e. element-wise; only the scale half (.xy) takes the detail
    // factors.
    let xform_main   = water_material.wave_displacement_array_xform;
    let xform_aux    = water_material.wave_slope_array_xform;
    let xform_detail = vec4<f32>(
        xform_main.x * water_material.detail_slope_scale_x.x,
        xform_main.y * water_material.detail_slope_scale_y.x,
        xform_main.z,
        xform_main.w,
    );

    let uv_main   = transform_texcoord(base_uv, xform_main);
    let uv_aux    = transform_texcoord(base_uv, xform_aux);
    let uv_detail = transform_texcoord(base_uv, xform_detail);

    let time_main   = water_material.time_warp.x;
    let time_aux    = water_material.time_warp_aux.x;
    let time_detail = time_main * water_material.detail_slope_scale_z.x;

    let s_main_01   = sample_wave_slope(uv_main,   time_main,   mip_level);
    let s_aux_01    = sample_wave_slope(uv_aux,    time_aux,    mip_level);
    let s_detail_01 = sample_wave_slope(uv_detail, time_detail, mip_level + 1.0);

    let s_main   = restore_slope(s_main_01,   slope_range);
    let s_aux_r  = restore_slope(s_aux_01,    slope_range);
    let s_detail = restore_slope(s_detail_01, slope_range);

    // Engine lines 620-625.
    let slope_aux_combined = s_aux_r * height_aux_scale
                           + s_detail * height_detail_scale;
    let shading = s_main * height_scale + slope_aux_combined;
    let min_disturb = water_material.minimal_wave_disturbance.x;
    let refraction = s_main * max(height_scale, min_disturb)
                   + slope_aux_combined;
    return ComposedSlope(shading, refraction);
}

// ---- Vertex inputs: 3 streams in `_vertex_type_water` order
// (per Ares/source/rasterizer/rasterizer_resource_definitions.cpp:41).
// Stream 0 (per-instance, 156 bytes) = control points.
// Stream 1 (per-vertex, 12 bytes)    = barycentric.
// Stream 2 (per-instance, 72 bytes)  = lighting attrs (NORMAL0..4 —
// per-control-point local_info + base_texcoord). ----
struct VsIn {
    @location(0) pos1_tc1x: vec4<f32>,
    @location(1) tc1y_tan1: vec4<f32>,
    @location(2) bin1_lm1x: vec4<f32>,
    @location(3) lm1y_pos2: vec4<f32>,
    @location(4) tc2_tan2xy: vec4<f32>,
    @location(5) tan2z_bin2: vec4<f32>,
    @location(6) lm2_pos3xy: vec4<f32>,
    @location(7) pos3z_tc3_tan3x: vec4<f32>,
    @location(8) tan3yz_bin3xy: vec4<f32>,
    @location(9) bin3z_lm3: vec3<f32>,
    @location(10) bary: vec3<f32>,
    // Stream 2 — `s_raw_water_append` data per control point. li1/li2/li3
    // are `local_info`; bt1/bt2/bt3 are `base_texcoord`. li.y carries
    // the per-vertex pre-baked water depth (`s_water_render_vertex.water_depth`
    // per `water_fx.hlsl`), used by bankalpha=depth and waveshape damping.
    @location(11) li1_bt1x: vec4<f32>,    // NORMAL0
    @location(12) bt1yz_li2xy: vec4<f32>, // NORMAL1
    @location(13) li2z_bt2: vec4<f32>,    // NORMAL2
    @location(14) li3_bt3x: vec4<f32>,    // NORMAL3
    @location(15) bt3yz: vec2<f32>,       // NORMAL4
}

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_to_camera: vec3<f32>,
    /// `position_ss` from water_shading_fx.hlsl:248 — projected vertex
    /// position passed to PS for screen-space refraction lookup
    /// (`texcoord_ss = position_ss.xy / position_ss.w * 0.5 + 0.5`,
    /// y-flipped per D3D / wgpu coordinate convention).
    @location(3) position_ss: vec4<f32>,
    /// `OUT.position_ws.w` per `water_shading_fx.hlsl:519` (PC path) —
    /// the per-vertex baked water depth (`misc_info.w = IN.local_info.y`).
    /// Used by bankalpha=depth (`water_shading_fx.hlsl:1014`) and the
    /// VS waveshape damping path.
    @location(4) water_depth: f32,
    /// `OUT.lm_tex.xy` per `water_shading_fx.hlsl:524`. Per-vertex
    /// lightmap atlas UV interpolated from the 3 control points'
    /// `lm1/lm2/lm3`. Used by the lightmap_intensity SH lookup
    /// (lines 760-786).
    @location(5) lm_tex: vec2<f32>,
    /// `OUT.base_tex` per `water_shading_fx.hlsl:522-523`:
    ///   .xy = per-control-point `base_texcoord`
    ///   .z  = `water_height_relative * 10` (current displacement.z)
    ///   .w  = `max_height_relative * 10` (peak displacement scale)
    /// Phase A6f reads .xy for watercolor_texture sampling. Phase A6e
    /// reads .zw for the auto-foam threshold:
    ///   foam_factor_auto = pow(saturate((base_tex.z - foam_height)
    ///                                / saturate(base_tex.w - foam_height)),
    ///                          foam_pow);
    @location(6) base_tex: vec4<f32>,
    /// `OUT.texcoord.xy` per `water_shading_fx.hlsl:497` — the
    /// wave UV (used by `foam_texture` / `foam_texture_detail`
    /// sampling at xform'd texcoord). Distinct from `base_tex.xy`,
    /// which is the watercolor / global_shape UV.
    @location(7) wave_texcoord: vec2<f32>,
    /// `OUT.fog_extinction.xyz` per `water_shading_fx.hlsl:489` —
    /// per-vertex `compute_scattering` extinction (HDR). Multiplies
    /// the composed surface color in PS to add atmospheric haze.
    @location(8) fog_extinction: vec3<f32>,
    /// `OUT.fog_inscatter.xyz` per `water_shading_fx.hlsl:490` —
    /// per-vertex inscattered light. Added in PS scaled by
    /// `(1 - color_refraction_bed_contribution)` so deep refraction
    /// areas don't get the inscatter applied twice (the LDR sample
    /// already contains the underwater scene's inscatter).
    @location(9) fog_inscatter: vec3<f32>,
    /// `OUT.tangent` per `water_shading_fx.hlsl:501,517`. .xyz =
    /// world-space tangent. .w = `sqrt(height_scale_global * wave_height)`
    /// (HLSL line 510, packed into `tangent.w` on PC for `compose_slope`
    /// per line 720).
    @location(10) tangent_w: vec4<f32>,
    /// `OUT.binormal` per `water_shading_fx.hlsl:502,518`. .xyz =
    /// world-space binormal. .w = `sqrt(height_scale_global * wave_height_aux)`
    /// (HLSL line 511, packed into `binormal.w` on PC for `compose_slope`
    /// per line 721).
    @location(11) binormal_w: vec4<f32>,
    /// `OUT.texcoord.w` per `water_shading_fx.hlsl:497` — the per-vertex
    /// `mipmap_level = max(incident_ws.w / wave_visual_damping_distance, 1.0)`.
    /// Used by `compose_slope` (line 565) and the `normal_hack_ratio`
    /// damping (line 740).
    @location(12) mipmap_level: f32,
}

// `get_tessellated_vertex` (`water_shading_fx.hlsl:51-105`) — interpolates
// 3 control points by barycentric weights into a per-vertex
// `s_water_render_vertex`. Inlined here as a struct + helper.
struct TessellatedVertex {
    position: vec3<f32>,
    texcoord: vec2<f32>,
    tangent: vec3<f32>,
    binormal: vec3<f32>,
    normal: vec3<f32>,
    local_info: vec3<f32>,
    base_tex: vec2<f32>,
    lm_tex: vec2<f32>,
}

fn unpack_and_blend(in: VsIn) -> TessellatedVertex {
    // ---- Control-point positions ----
    let pos1 = in.pos1_tc1x.xyz;
    let pos2 = vec3<f32>(in.lm1y_pos2.y, in.lm1y_pos2.z, in.lm1y_pos2.w);
    let pos3 = vec3<f32>(in.lm2_pos3xy.z, in.lm2_pos3xy.w, in.pos3z_tc3_tan3x.x);

    // ---- Per-CP texcoord ----
    let tc1 = vec2<f32>(in.pos1_tc1x.w, in.tc1y_tan1.x);
    let tc2 = vec2<f32>(in.tc2_tan2xy.x, in.tc2_tan2xy.y);
    let tc3 = vec2<f32>(in.pos3z_tc3_tan3x.y, in.pos3z_tc3_tan3x.z);

    // ---- Per-CP tangent ----
    let tan1 = vec3<f32>(in.tc1y_tan1.y, in.tc1y_tan1.z, in.tc1y_tan1.w);
    let tan2 = vec3<f32>(in.tc2_tan2xy.z, in.tc2_tan2xy.w, in.tan2z_bin2.x);
    let tan3 = vec3<f32>(in.pos3z_tc3_tan3x.w, in.tan3yz_bin3xy.x, in.tan3yz_bin3xy.y);

    // ---- Per-CP binormal ----
    let bin1 = in.bin1_lm1x.xyz;
    let bin2 = vec3<f32>(in.tan2z_bin2.y, in.tan2z_bin2.z, in.tan2z_bin2.w);
    let bin3 = vec3<f32>(in.tan3yz_bin3xy.z, in.tan3yz_bin3xy.w, in.bin3z_lm3.x);

    // ---- Per-CP local_info (li.x = displacement_scale, li.y = water_depth,
    //      li.z = unused; from `water_shading_fx.hlsl` IN.local_info) ----
    let li1 = in.li1_bt1x.xyz;
    let li2 = vec3<f32>(in.bt1yz_li2xy.z, in.bt1yz_li2xy.w, in.li2z_bt2.x);
    let li3 = in.li3_bt3x.xyz;

    // ---- Per-CP base_texcoord (bt1/bt2/bt3 — for foam / global_shape /
    //      watercolor PS sampling) ----
    let bt1 = vec2<f32>(in.li1_bt1x.w, in.bt1yz_li2xy.x);
    let bt2 = vec2<f32>(in.li2z_bt2.y, in.li2z_bt2.z);
    let bt3 = vec2<f32>(in.li3_bt3x.w, in.bt3yz.x);

    // ---- Per-CP lightmap UV ----
    let lm1 = vec2<f32>(in.bin1_lm1x.w, in.lm1y_pos2.x);
    let lm2 = vec2<f32>(in.lm2_pos3xy.x, in.lm2_pos3xy.y);
    let lm3 = vec2<f32>(in.bin3z_lm3.y, in.bin3z_lm3.z);

    // ---- Barycentric blend ----
    let w = in.bary;
    var out: TessellatedVertex;
    out.position   = w.x * pos1 + w.y * pos2 + w.z * pos3;
    out.texcoord   = w.x * tc1  + w.y * tc2  + w.z * tc3;
    out.tangent    = w.x * tan1 + w.y * tan2 + w.z * tan3;
    // Engine `water_fx.hlsl:154/182/224/296` flips the post-interpolation
    // binormal to right-handed: `OUT.binormal = -barycentric_interpolate(...)`
    // with explicit comment `###xwan inversion binormal to right hand`.
    // The source vertex stream stores left-handed binormals; downstream
    // TBN reconstruction (`normal = cross(tangent, binormal)`) only
    // produces an upward-facing normal once the binormal is flipped.
    out.binormal   = -(w.x * bin1 + w.y * bin2 + w.z * bin3);
    out.local_info = w.x * li1  + w.y * li2  + w.z * li3;
    out.base_tex   = w.x * bt1  + w.y * bt2  + w.z * bt3;
    out.lm_tex     = w.x * lm1  + w.y * lm2  + w.z * lm3;
    // Per `water_fx.hlsl:165/234` (and `water_shading_fx.hlsl:747` PC
    // path): normal = cross(tangent, binormal). With binormal already
    // flipped above, this matches engine; we use the equivalent
    // `-cross(B, T)` form here.
    out.normal = -normalize(cross(out.binormal, out.tangent));
    return out;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // ---- Inline get_tessellated_vertex ----
    let tv = unpack_and_blend(in);
    var position = tv.position;

    // ---- Wave displacement (water_shading_fx.hlsl:368-485) ----
    // PC-faithful path. Engine wraps in `if (k_is_water_tessellated)`
    // and `if (TEST_CATEGORY_OPTION(waveshape, default))`. Riverworld
    // uses waveshape=default; we set k_is_water_tessellated=1 from the
    // host.
    //
    // `water_height_relative` and `max_height_relative` are forwarded
    // through `OUT.base_tex.zw` (×10 per HLSL line 523) for the foam
    // path's auto-factor calculation (Phase A6e). Non-tessellated
    // path leaves them at the engine defaults (0, 1) per HLSL line
    // 370-371.
    var water_height_relative: f32 = 0.0;
    var max_height_relative: f32 = 1.0;

    // ---- Distance-fade LOD for wave sampling (water_shading_fx.hlsl:338) ----
    //   incident_ws.w = length(camera - position);
    //   mipmap_level = max(incident_ws.w / wave_visual_damping_distance, 1.0);
    // The base VS position is pre-displacement here; engine computes
    // incident_ws BEFORE the displacement loop too, so we match. The
    // resulting mipmap_level drives `sample3Dlod(wave_displacement_array,
    // texcoord.xyz, mipmap_level)` — distant water hits higher mips
    // and waves smooth out, near water hits mip 0 for sharp detail.
    let camera_world_vs = water_shared_ps.k_ps_camera_position.xyz;
    let incident_dist_vs = length(camera_world_vs - position);
    let damping = max(water_material.wave_visual_damping_distance.x, 1e-3);
    let mipmap_level = max(incident_dist_vs / damping, 1.0);

    // ---- Global shape control (water_shading_fx.hlsl:341-352) ----
    // Engine has three branches:
    //   global_shape=none  → height_scale_global=1, choppy_scale_global=1
    //   global_shape=paint → sample global_shape_texture.xy
    //   global_shape=depth → height_scale_global = saturate(local_info.y / globalshape_infuence_depth)
    //
    // Phase W2 — per-rmt2 substitution. `__GLOBAL_SHAPE_PAINT__` /
    // `__GLOBAL_SHAPE_DEPTH__` get replaced with `true`/`false` literals
    // at WGSL compile-time per the material's rmdf choice; naga DCE
    // drops the dead branches.
    var height_scale_global: f32 = 1.0;
    var choppy_scale_global: f32 = 1.0;
    if (__GLOBAL_SHAPE_PAINT__) {
        // Engine relies on the global_shape bitmap rim being zero, so
        // ClampToEdge returns 0 outside the painted silhouette and waves
        // / foam stop at the river edge. The riverworld_water_multipurpose
        // bitmap has TOP-row G=255 (silhouette extends to the bitmap's top
        // edge), so ClampToEdge streaks `choppy_scale_global=1` across
        // every mesh-8 vertex with `v < 0`. Explicit bounds-check zeroes
        // OOB samples, matching engine intent for "outside the painted
        // silhouette = no waves".
        let _gs_uv = tv.base_tex.xy;
        let _gs_in_range =
            f32(_gs_uv.x >= 0.0 && _gs_uv.x <= 1.0 && _gs_uv.y >= 0.0 && _gs_uv.y <= 1.0);
        let shape_control = textureSampleLevel(
            global_shape_texture, watercolor_texture_sampler,
            _gs_uv, mipmap_level,
        ) * _gs_in_range;
        height_scale_global = shape_control.x;
        choppy_scale_global = shape_control.y;
    } else if (__GLOBAL_SHAPE_DEPTH__) {
        // Engine `water_shading_fx.hlsl:351`:
        //   height_scale_for_shallow_water = saturate(local_info.y / globalshape_infuence_depth)
        // `local_info.y` is per-vertex baked water depth; the divisor
        // lives in `globalshape_depth` rmop (default 1.0).
        let depth = tv.local_info.y;
        let influence = max(water_material.globalshape_infuence_depth.x, 1e-6);
        height_scale_global = clamp(depth / influence, 0.0, 1.0);
        // choppy_scale_global stays 1.0 in the depth branch (engine does
        // not touch it).
    }

    if (water_ps.k_is_water_tessellated != 0u) {
        let displacement_range = vec3<f32>(
            water_material.displacement_range_x.x,
            water_material.displacement_range_y.x,
            water_material.displacement_range_z.x,
        );
        let displacement_min = -displacement_range * 0.5;

        // Engine: two displacement layers from water_shading_fx.hlsl:380-438.
        //   layer 1: wave_displacement_array_xform + time_warp + wave_height
        //   layer 2: wave_slope_array_xform + time_warp_aux + wave_height_aux
        // Both layers sample the SAME `wave_displacement_array` texture
        // (the `wave_slope_array_xform` here is just a different UV
        // tiling, not a different texture — the engine reuses the
        // displacement array for the secondary layer too per HLSL line
        // 396-397). After restore_displacement they're scaled by
        // `wave_scale = sqrt(xform.x * xform.y)`, summed, then run
        // through `apply_choppiness` for directional scaling, then
        // multiplied by `local_info.x` (per-vertex tapering).
        //
        // time_warp / time_warp_aux are both cyclic ∈ [0, 1) over
        // their `time period` (0.8s in riverworld) — see TagFunction
        // time-aware overlay. The engine HLSL DX11 path
        // (`water_shading_fx.hlsl:399-414`) multiplies the cyclic
        // value by the array slice count to drive a smooth two-slice
        // lerp through the keyframes; `sample_wave_displacement`
        // ports that.
        let xform_main = water_material.wave_displacement_array_xform;
        let xform_aux  = water_material.wave_slope_array_xform;
        let uv_main = transform_texcoord(tv.texcoord, xform_main);
        let uv_aux  = transform_texcoord(tv.texcoord, xform_aux);

        let s_main = sample_wave_displacement(uv_main, water_material.time_warp.x, mipmap_level);
        let s_aux  = sample_wave_displacement(uv_aux,  water_material.time_warp_aux.x, mipmap_level);

        let d_main = restore_displacement(
            s_main, displacement_range, displacement_min,
            water_material.wave_height.x,
        );
        let d_aux = restore_displacement(
            s_aux,  displacement_range, displacement_min,
            water_material.wave_height_aux.x,
        );

        // Engine line 431-433: wave_scale = sqrt(xform.x * xform.y).
        // The engine notes (line 432-433) that wave_scale_aux SHOULD
        // be sqrt(wave_slope_array_xform.x * wave_slope_array_xform.y)
        // but uses `wave_scale` for both — the comment marks it as a
        // shipped quirk. We mirror that.
        let wave_scale = sqrt(xform_main.x * xform_main.y);
        var displacement = (d_main + d_aux) / wave_scale;

        // apply_choppiness — directional x/y scaling per HLSL line 440.
        // choppiness inputs are pre-scaled by `choppy_scale_global` from
        // the global_shape branch above (engine HLSL lines 441-444).
        displacement = apply_choppiness(
            displacement,
            water_material.choppiness_forward.x  * choppy_scale_global,
            water_material.choppiness_backward.x * choppy_scale_global,
            water_material.choppiness_side.x     * choppy_scale_global,
        );

        // Engine line 447: displacement *= IN.local_info.x (per-vertex
        // displacement scale — authored to 0 at shore and 1 in deep
        // water so waves taper at edges).
        displacement = displacement * tv.local_info.x;

        // Engine HLSL line 451: displacement.z *= height_scale_global.
        // Modulates the vertical wave amplitude per-pixel via the painted
        // global_shape_texture (or per-vertex water_depth for `depth` mode).
        displacement.z = displacement.z * height_scale_global;

        // Track relative heights for foam factor (Phase A6e).
        // Engine line 457: water_height_relative = displacement.z.
        water_height_relative = displacement.z;
        // Engine line 448: max_height_relative = 0.5 * local_info.x *
        //   displacement_range_z * (wave_height + wave_height_aux) /
        //   wave_scale. This is the peak height the surface could reach
        //   given the rmw's authored ranges; foam_factor_auto compares
        //   the current height to this max.
        max_height_relative = 0.5
            * tv.local_info.x
            * water_material.displacement_range_z.x
            * (water_material.wave_height.x + water_material.wave_height_aux.x)
            / wave_scale;

        // Engine line 481: position += tangent*disp.x + binormal*disp.y
        //                          + normal*disp.z;
        position = position
            + tv.tangent * displacement.x
            + tv.binormal * displacement.y
            + tv.normal * displacement.z;
    }

    // ---- Project ----
    let view_to_cam = camera_world_vs - position;
    let clip = camera.projection * camera.view * vec4<f32>(position, 1.0);

    // ---- Atmospheric scattering (water_shading_fx.hlsl:488-491) ----
    // Engine: compute_scattering(Camera_Position, position.xyz, ...);
    //         OUT.fog_extinction = float4(extinction, 0);
    //         OUT.fog_inscatter  = float4(inscatter, 0);
    var fog_ext: vec3<f32>;
    var fog_in:  vec3<f32>;
    compute_scattering_water(camera_world_vs, position, &fog_ext, &fog_in);

    var out: VsOut;
    out.clip_position = clip;
    out.world_position = position;
    out.world_normal = tv.normal;
    out.view_to_camera = view_to_cam;
    out.position_ss = clip;
    out.water_depth = tv.local_info.y;
    out.lm_tex = tv.lm_tex;
    // Engine line 523: OUT.base_tex = float4(IN.base_tex.xy,
    //   water_height_relative*10, max_height_relative*10).
    out.base_tex = vec4<f32>(
        tv.base_tex,
        water_height_relative * 10.0,
        max_height_relative * 10.0,
    );
    // Engine line 497: OUT.texcoord = float4(IN.texcoord.xyz, mipmap_level).
    // Foam uses .xy (with the foam xforms applied in PS).
    out.wave_texcoord = tv.texcoord;
    out.fog_extinction = fog_ext;
    out.fog_inscatter = fog_in;
    // Engine lines 517-518 (PC): pack height-scale factors into
    // tangent.w / binormal.w so `compose_slope` can read them
    // (line 720-721).
    //   tangent.w   = sqrt(height_scale_global * wave_height)
    //   binormal.w  = sqrt(height_scale_global * wave_height_aux)
    // height_scale_global comes from the global_shape branch above
    // (paint = global_shape_texture.x; none = 1.0; depth = depth-clamped).
    let wave_h_scale     = sqrt(max(height_scale_global * water_material.wave_height.x,     0.0));
    let wave_h_aux_scale = sqrt(max(height_scale_global * water_material.wave_height_aux.x, 0.0));
    out.tangent_w  = vec4<f32>(tv.tangent,  wave_h_scale);
    out.binormal_w = vec4<f32>(tv.binormal, wave_h_aux_scale);
    out.mipmap_level = mipmap_level;
    return out;
}

// Engine `accum_pixel` (render_target_fx.hlsl:8-18). `water_shading_fx.hlsl
// :1061` uses `convert_to_render_target(...)` → writes BOTH RT0
// (`_surface_screenshot_composite_depth` = our `lighting_base`) AND RT1
// (`_surface_screenshot_composite_cubemap` = our `hdr_dark`, the bloom-
// extract feed source). On MCC PC `DARK_COLOR_MULTIPLIER = g_exposure.g = 1.0`
// so RT0 content == RT1 content. Without RT1 writes, auto-exposure's bloom
// -feed sample misses bright water and over-corrects upward — visible as a
// blown-out scene whenever water is bright (post-`g_exposure` fix).
struct AccumPixel {
    @location(0) color: vec4<f32>,
    @location(1) dark_color: vec4<f32>,
}

@fragment
fn fs_main(in: VsOut) -> AccumPixel {
    // ---- Early-out depth test (water_shading_fx.hlsl:660-677) ----
    // Engine: `depth_water = depth_buffer.Load(int3(pixel, 0)).r;
    //          if (depth_water > position.z) return scene_ldr`.
    // POINT-fetched depth (no filter) compared against the rasterizer
    // depth value — bail out and return scene if scene is closer than
    // water (water occluded). Engine uses `.Load(int3)` because a
    // linear sample of depth blurs between adjacent pixels, producing
    // half-pixel-wide false positives along water/shore edges that
    // render as a bright scene_ldr fringe.
    //
    // `@builtin(position).xy` in WGSL is the framebuffer pixel
    // coordinate (already snapped); `.z` is the reverse-Z depth.
    // `textureLoad` does a point fetch matching engine's `Load`.
    let _eo_px = vec2<i32>(in.clip_position.xy);
    let _eo_depth_water = textureLoad(scene_depth, _eo_px, 0);
    if (_eo_depth_water > in.clip_position.z) {
        let _eo_scene = textureLoad(scene_ldr_snapshot, _eo_px, 0).rgb;
        let accum = vec4<f32>(_eo_scene, 1.0);
        return AccumPixel(accum, accum);
    }

    // ---- Per-pixel normal via compose_slope (water_shading_fx.hlsl:710-753) ----
    // Build the TBN matrix from the per-vertex world-space tangent /
    // binormal and the geometry normal reconstructed from their cross
    // product (engine PC path, line 747). Then sample wave_slope_array
    // 3× and combine into `slope_shading`. Apply the per-pixel
    // `normal_hack_ratio = max(mipmap_level, 1)` damp (line 740) so
    // distant water blends to the smooth geometry normal. Finally
    // rotate `(slope_shading.xy, 1)` into world space.
    let world_t = in.tangent_w.xyz;
    let world_b = in.binormal_w.xyz;
    let world_n_geo = -normalize(cross(world_b, world_t));

    // Engine `water_shading_fx.hlsl:691-696` samples
    // `tex_ripple_buffer_slope_height` — a runtime texture populated
    // when objects splash into water (Halo's ripple-interaction
    // system). Without active splash events the buffer is all-zero
    // and `ripple_slope` evaluates to 0. Static water in the engine
    // produces the same `vec2(0.0)` here. Phase A9 will plumb the
    // splash event → ripple compute path; until then this is
    // engine-correct for the static case.
    let ripple_slope = vec2<f32>(0.0);
    let ripple_slope_length_pre = ripple_slope.x + ripple_slope.y + 1.0;
    let ripple_slope_length = clamp(ripple_slope_length_pre, 0.3, 2.1);
    let ripple_slope_weak = 1.0 / ripple_slope_length;

    // ---- Waveshape branches (water_shading_fx.hlsl:710-737) ----
    // Engine HLSL gates the slope computation on the waveshape category:
    //   waveshape=default (HLSL 712-726) — compose_slope from wave_slope_array
    //   waveshape=bump    (HLSL 727-737) — sample bump_map + bump_detail_map
    //   waveshape=none    (implicit)     — both slopes stay 0
    //
    // Phase W4.1 — `__WAVESHAPE_DEFAULT__/BUMP__` substituted per rmdf.
    // The `default` branch is currently wired; `bump` is scaffolded but
    // falls through to zero slopes pending bump_map / bump_detail_map
    // binding plumbing (no current scenario uses waveshape=bump).
    var slope_shading_raw: vec2<f32> = vec2<f32>(0.0);
    var slope_refraction_raw: vec2<f32> = vec2<f32>(0.0);
    if (__WAVESHAPE_DEFAULT__) {
        // HLSL 712-726: compose_slope(wave_slope_array, ...).
        let composed = compose_slope(
            in.wave_texcoord,
            in.tangent_w.w  * ripple_slope_weak,
            in.binormal_w.w * ripple_slope_weak,
            water_material.detail_slope_steepness.x * ripple_slope_weak,
            in.mipmap_level,
        );
        slope_shading_raw = composed.shading;
        slope_refraction_raw = composed.refraction;
    } else if (__WAVESHAPE_BUMP__) {
        // FIXME(water): port HLSL 727-737:
        //   bump        = sample_bumpmap(bump_map, transform_texcoord(uv, bump_map_xform));
        //   bump_detail = sample_bumpmap(bump_detail_map, transform_texcoord(uv, bump_detail_map_xform));
        //   bump.xy += bump_detail.xy;
        //   slope_shading    = bump.xy / max(bump.z, 0.01);
        //   slope_refraction = slope_shading;
        // Needs bump_map / bump_detail_map texture bindings + the
        // _xform cbuffer slots threaded into WaterMaterialPS. Deferred
        // until a scenario using waveshape=bump (`_2_0_1_1_1_0_0_0`
        // rmt2 variant) is added to the test set. For now this branch
        // leaves slopes at zero (engine-equivalent to waveshape=none).
    }
    // else: waveshape=none — slopes stay 0.

    // Engine lines 740-744:
    //   normal_hack_ratio = max(texcoord.w, 1.0)
    //   slope_shading    /= normal_hack_ratio
    //   slope_shading    += ripple_slope
    //   slope_refraction += ripple_slope
    let normal_hack_ratio = max(in.mipmap_level, 1.0);
    let slope_shading_p = slope_shading_raw / normal_hack_ratio + ripple_slope;
    let slope_refraction_p = slope_refraction_raw + ripple_slope;

    // Engine lines 746-753: build TBN matrix, transform tangent-space
    // `(slope_shading, 1.0)` to world. WGSL `mat3x3` columns map to
    // HLSL row-major rows when used with `mul(v, M)`, so we build the
    // matrix with tangent / binormal / normal as ROWS by constructing
    // it column-major and multiplying `M * v`.
    let tangent_frame = mat3x3<f32>(
        vec3<f32>(world_t.x, world_b.x, world_n_geo.x),
        vec3<f32>(world_t.y, world_b.y, world_n_geo.y),
        vec3<f32>(world_t.z, world_b.z, world_n_geo.z),
    );
    let n_ts = normalize(vec3<f32>(slope_shading_p, 1.0));
    let n = normalize(tangent_frame * n_ts);
    let view_to_cam = in.view_to_camera;
    let v = normalize(view_to_cam);
    let n_dot_v = max(dot(n, v), 0.0);

    // ---- Screen-space refraction texcoord (water_shading_fx.hlsl:660-665) ----
    // INTERPOLATORS.position_ss /= position_ss.w; → NDC.
    // texcoord_ss = ndc/2 + 0.5; y-flip; viewport scale/offset.
    let pos_ss_w = max(in.position_ss.w, 1e-4);
    let ndc = in.position_ss.xy / pos_ss_w;
    var texcoord_ss = ndc * 0.5 + vec2<f32>(0.5, 0.5);
    texcoord_ss.y = 1.0 - texcoord_ss.y;
    // k_water_player_view_constant.xy = viewport offset; .zw = scale.
    // For fullscreen single-viewport this is (0,0,1,1) — the engine
    // applies `texcoord_ss = .xy + texcoord_ss * .zw` for split-screen
    // adjustment (water_shading_fx.hlsl:665).
    texcoord_ss = water_shared_ps.k_water_player_view_constant.xy
                + texcoord_ss * water_shared_ps.k_water_player_view_constant.zw;

    // ---- Slope-driven refraction perturbation (water_shading_fx.hlsl:820-828) ----
    // Engine flow:
    //   1. Sample scene_depth at UNPERTURBED texcoord_ss → bed_depth_raw0
    //   2. Back-project to world → depth_water = length(bed - position_ws)
    //   3. bump = slope_refraction.xy * incident_ws.yx * refraction_texcoord_shift
    //           * saturate(3 * depth_water)
    //          *= min(max(2 / incident_ws.w, 0), 1)
    //          *= k_water_player_view_constant.zw
    //          *= ripple_slope_length
    //   4. texcoord_refraction = clamp(texcoord_ss + bump, view+δ, view+size-δ)
    //   5. Sample scene_ldr + scene_depth at texcoord_refraction.
    //
    // Without this the cubemap reflection (peeking through fresnel) shows
    // a perfectly straight underwater plane — the bump introduces the
    // per-pixel ripple distortion that gives water its characteristic
    // shimmer.
    let pvc0 = water_shared_ps.k_water_player_view_constant;
    let texcoord_ss_unperturbed = clamp(texcoord_ss, vec2<f32>(0.0), vec2<f32>(1.0));
    let bed_depth_raw0 = textureSampleLevel(
        scene_depth, scene_depth_sampler, texcoord_ss_unperturbed, 0,
    );
    let raw_uv0 = vec2<f32>(
        (texcoord_ss_unperturbed.x - pvc0.x) / max(pvc0.z, 1e-6),
        1.0 - (texcoord_ss_unperturbed.y - pvc0.y) / max(pvc0.w, 1e-6),
    );
    let bed_ndc0 = (raw_uv0 - vec2<f32>(0.5)) * 2.0;
    let bed_clip0 = vec4<f32>(bed_ndc0, bed_depth_raw0, 1.0);
    let bed_world_h0 = water_shared_ps.k_water_view_xform_inverse * bed_clip0;
    let bed_world0 = bed_world_h0.xyz / max(bed_world_h0.w, 1e-6);
    let depth_water_for_bump = length(bed_world0 - in.world_position);

    // incident_ws follows engine: world-space dir from surface to camera
    // (`Camera_Position - position_ws`, normalized; .w = pre-norm length).
    let incident_ws_dir = water_shared_ps.k_ps_camera_position.xyz
                        - in.world_position;
    let incident_ws_w = length(incident_ws_dir);

    var bump = slope_refraction_p
             * vec2<f32>(incident_ws_dir.y, incident_ws_dir.x)
             * water_material.refraction_texcoord_shift.x
             * clamp(3.0 * depth_water_for_bump, 0.0, 1.0);
    bump = bump * clamp(2.0 / max(incident_ws_w, 1e-3), 0.0, 1.0);
    bump = bump * pvc0.zw;
    bump = bump * ripple_slope_length;

    let delta_clamp = vec2<f32>(0.001);
    let texcoord_refraction = clamp(
        texcoord_ss + bump,
        pvc0.xy + delta_clamp,
        pvc0.xy + pvc0.zw - delta_clamp,
    );
    let refraction_uv = texcoord_refraction;
    let scene_ldr = textureSample(scene_ldr_snapshot, scene_ldr_sampler, refraction_uv).rgb;

    // ---- Schlick fresnel (water_shading_fx.hlsl:629-636 + 988-993) ----
    // Engine `compute_fresnel`:
    //   return r0 + (1 - r0) * pow(1 - n_dot_v, 2.5);
    // Exponent is HARDCODED to 2.5 in water (the cbuffer
    // `fresnel_curve_steepness` field exists but is consumed by
    // OTHER shader subclasses — glass/terrain/two_lobe_phong — not
    // water). Previously we used the cbuffer value (5.0 for
    // riverworld) which double-amplified glancing fresnel.
    //
    // `fresnel_normal` flips when underwater (HLSL 988-991):
    //   fresnel_normal = normal * 2 * (0.5 - k_is_camera_underwater)
    // = -normal underwater, +normal above. We then recompute
    // n_dot_v against the flipped normal for the fresnel dot.
    let r0 = water_material.fresnel_coefficient.x;
    let fresnel_normal = n * (1.0 - 2.0 * f32(water_ps.k_is_camera_underwater));
    let fresnel_n_dot_v = clamp(dot(v, fresnel_normal), 0.0, 1.0);
    let fresnel = r0 + (1.0 - r0) * pow(1.0 - fresnel_n_dot_v, 2.5);

    // ---- Lightmap intensity SH lookup (water_shading_fx.hlsl:756-786) ----
    // Engine path:
    //   if (k_is_lightmap_exist) {
    //     sh_dxt_vector_0 = TFETCH_3D(lightprobe_texture_array, slice 0);
    //     sh_dxt_vector_1 = TFETCH_3D(lightprobe_texture_array, slice 1);
    //     sh_coefficients_0 = decode_bpp16_luvw(v0, v1, p_lightmap_compress_constant_0.x);
    //     lightmap_intensity = saturate(sh_coefficients_0);
    //   } else {
    //     lightmap_intensity = (1, 1, 1);  // implicit from declaration line 756
    //   }
    var lightmap_intensity: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    if (water_ps.k_is_lightmap_exist != 0u) {
        let s0 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, in.lm_tex, 0);
        let s1 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, in.lm_tex, 1);
        let sh_coefficients_0 = decode_bpp16_luvw(s0, s1, lightmap_compress.constant_0.x);
        lightmap_intensity = clamp(sh_coefficients_0, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // ---- water_color (water_shading_fx.hlsl:788-800) ----
    // Engine always samples watercolor_texture (the .a channel feeds the
    // `bankalpha=paint` branch even when watercolor=pure). Then:
    //   watercolor=pure    → water_color = water_color_pure
    //   watercolor=texture → water_color = sample.rgb * watercolor_coefficient
    // and both multiply by lightmap_intensity at line 800.
    //
    // Phase W2 — `__WATERCOLOR_PURE__` substituted per rmdf choice.
    let water_color_from_texture_rgba = textureSample(
        watercolor_texture, watercolor_texture_sampler, in.base_tex.xy,
    );
    let water_color_from_texture =
        water_color_from_texture_rgba.rgb * water_material.watercolor_coefficient.x;
    var water_color: vec3<f32>;
    if (__WATERCOLOR_PURE__) {
        water_color = water_material.water_color_pure.rgb;
    } else {
        // texture (or any other variant — pure is the only non-texture
        // branch in the rmdf)
        water_color = water_color_from_texture;
    }
    water_color = water_color * lightmap_intensity;

    // ---- Refraction tint (water_shading_fx.hlsl:802-870) ----
    // Engine has two branches:
    //   refraction=none    (HLSL 807-811) — no scene sample, water-only
    //   refraction=dynamic (HLSL 812-870) — back-project scene_depth,
    //                                       sample scene_ldr at perturbed UV
    //
    // Phase W4.4 — `__REFRACTION_NONE__` / `__REFRACTION_DYNAMIC__`
    // substituted per rmdf choice.
    let camera_world = water_shared_ps.k_ps_camera_position.xyz;
    let incident_dist = length(camera_world - in.world_position);

    var color_refraction: vec3<f32>;
    var color_refraction_bed: vec3<f32>;
    var color_refraction_bed_contribution: f32;
    if (__REFRACTION_NONE__) {
        // Engine HLSL 807-811:
        //   color_refraction = water_color;
        //   color_refraction_bed = water_color;
        //   (color_refraction_bed_contribution stays 0 — declared 0 at
        //    line 804 and never set in this branch)
        color_refraction = water_color;
        color_refraction_bed = water_color;
        color_refraction_bed_contribution = 0.0;
    } else {
        // refraction=dynamic — runtime depth back-projection
        //
        // 1. Sample scene_depth at refraction_uv → raw depth ∈ [0, 1]
        // 2. Convert refraction_uv back to NDC, build (ndc_x, ndc_y, raw_depth, 1)
        // 3. Multiply by k_water_view_xform_inverse → world position of
        //    the scene bed visible through the water at this pixel
        // 4. depth_refraction = lerp(distance, abs(z-diff), refraction_depth_dominant_ratio)
        //    (HLSL line 862-864 — the dominant ratio mostly weights the
        //    z-difference for riverworld at 0.95, with a small distance
        //    contribution to capture oblique view geometry)
        // 5. transparence = saturate(1 - compute_fog_factor(water_murkiness, depth_refraction))
        //                 * saturate(1 - incident_distance / refraction_extinct_distance)
        // 6. color_refraction = lerp(water_color, scene_ldr, transparence)
        //
        // Differs from the engine in that we skip the slope-driven UV
        // bump (`compose_slope` from wave_slope_array isn't ported yet)
        // and the depth-rejection re-sample. Both are refinements; the
        // shape (deep = water_color, shallow = scene_ldr) is correct.

        // Sample scene_depth at the refraction UV. Returns raw [0, 1]
        // device depth (reverse-Z; 1.0 at near, 0.0 at infinity).
        let bed_depth_raw = textureSampleLevel(
            scene_depth, scene_depth_sampler, refraction_uv, 0,
        );

        // Convert refraction_uv back to NDC, mirroring the
        // ndc → texcoord_ss flow inverted:
        //   texcoord_ss' = (refraction_uv - player_view.xy) / player_view.zw
        //   texcoord_ss'.y = 1 - texcoord_ss'.y     (un-y-flip)
        //   ndc = (texcoord_ss' - 0.5) * 2
        // For full-screen single-viewport, k_water_player_view_constant =
        // (0, 0, 1, 1) and the xform is identity — but we still apply
        // the formula so split-screen variations carry through.
        let pvc = water_shared_ps.k_water_player_view_constant;
        var raw_uv = (refraction_uv - pvc.xy) / max(pvc.zw, vec2<f32>(1e-6));
        raw_uv.y = 1.0 - raw_uv.y;
        let bed_ndc = (raw_uv - vec2<f32>(0.5)) * 2.0;

        // Project NDC point at scene_depth back to world space.
        // k_water_view_xform_inverse = inverse(view * projection) — column-
        // major in WGSL, so `M * v` is the standard column-major NDC→world
        // transform. (Engine HLSL: `mul(v, M)` row-major produces same
        // logical operation with opposite operand order.)
        let bed_clip = vec4<f32>(bed_ndc, bed_depth_raw, 1.0);
        let bed_world_h = water_shared_ps.k_water_view_xform_inverse * bed_clip;
        let bed_world = bed_world_h.xyz / max(bed_world_h.w, 1e-6);

        // depth_refraction (HLSL line 862-864):
        //   depth_by_z        = abs(point_refraction.z - position_ws.z);
        //   depth_by_distance = length(point_refraction.xyz - position_ws.xyz);
        //   depth_refraction  = lerp(by_distance, by_z, refraction_depth_dominant_ratio);
        let depth_by_z = abs(bed_world.z - in.world_position.z);
        let depth_by_distance = length(bed_world - in.world_position);
        let dom = water_material.refraction_depth_dominant_ratio.x;
        let depth_refraction = mix(depth_by_distance, depth_by_z, dom);

        // Engine HLSL 867:
        //   transparence = saturate(1 - compute_fog_factor(murk, depth) * ripple_slope_length)
        // The ripple_slope_length factor amplifies depth-fog when ripples
        // disturb the surface. Inert at rest (ripple_slope_length = 1).
        let fog = compute_fog_factor(water_material.water_murkiness.x, depth_refraction);
        let transparence_depth = clamp(1.0 - fog * ripple_slope_length, 0.0, 1.0);
        let transparence_distance = clamp(
            1.0 - incident_dist / max(water_material.refraction_extinct_distance.x, 1e-3),
            0.0, 1.0,
        );
        let transparence = transparence_depth * transparence_distance;
        // `color_refraction_bed` = bare scene_ldr sample (the "pure
        // underwater color" per HLSL line 847). Tracked separately so the
        // engine's subtract-then-add-back dance at output time can avoid
        // double-applying fog + exposure to the refracted scene (which
        // already went through both in the rmsh pass).
        color_refraction_bed = scene_ldr;
        color_refraction = mix(water_color, color_refraction_bed, transparence);
        // Engine line 870: `color_refraction_bed_contribution = transparence`
        color_refraction_bed_contribution = transparence;
    }

    // ---- Reflection (water_shading_fx.hlsl:931-962) ----
    // Engine HLSL has 3 branches:
    //   reflection=none    (HLSL 933-936) — color_reflection = 0
    //   reflection=static  (HLSL 937-957) — cubemap reflection w/ HDR alpha
    //   reflection=dynamic (HLSL 958-962) — scene_ldr at (x, y-0.2)
    //
    // `reflection=static_ssr` (Reach+ corpus) has no separate H3 HLSL
    // branch; it falls back to `static`. Substituted by the host as
    // `__REFLECTION_STATIC__ || static_ssr → true`.
    //
    // Phase A6a uses lightmap_intensity from the SH lookup above.
    // `sampleBiasGlobalCUBE` applies a global mip bias (~-1.0) on
    // sampling. wgpu's `textureSample` computes mip from screen-space
    // derivatives of reflect_dir — but reflect_dir varies wildly per
    // pixel on heavily-displaced waterfall water, producing huge
    // gradients → wgpu picks high mips → blocky face content shows
    // through. Force mip 0 explicitly to mirror the engine's sharp-bias
    // behavior. (At 64×64 the cubemap is mip 0 always for riverworld;
    // per-LOD reflection roughness is a different feature not used by
    // this rmw.)
    var color_reflection: vec3<f32>;
    if (__REFLECTION_STATIC__) {
        // HLSL 937-957: cubemap reflection.
        //   reflect_dir = reflect(-incident_ws, normal);
        //   reflect_dir.z = abs(reflect_dir.z);  // flip down-pointing rays
        //   env = sampleBiasGlobalCUBE(environment_map, reflect_dir);
        //   parts.x = saturate(env.a - sunspot_cut);
        //   parts.y = min(env.a, sunspot_cut);
        //   sun_light_rate = saturate(lightmap_intensity - shadow_intensity_mark);
        //   sun_scale = dot(sun_light_rate, sun_light_rate);
        //   alpha = parts.x * sun_scale + parts.y;
        //   color_reflection = env.rgb * alpha * reflection_coefficient;
        let reflect_dir_raw = reflect(-v, n);
        let reflect_dir = vec3<f32>(reflect_dir_raw.x, reflect_dir_raw.y, abs(reflect_dir_raw.z));
        let env_sample = textureSampleLevel(environment_map, environment_map_sampler, reflect_dir, 0.0);
        let sun_light_rate = clamp(
            lightmap_intensity - vec3<f32>(water_material.shadow_intensity_mark.x),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let sun_scale = dot(sun_light_rate, sun_light_rate);
        let parts_x = clamp(env_sample.a - water_material.sunspot_cut.x, 0.0, 1.0);
        let parts_y = min(env_sample.a, water_material.sunspot_cut.x);
        let env_alpha = parts_x * sun_scale + parts_y;
        color_reflection =
            env_sample.rgb * env_alpha * water_material.reflection_coefficient.x;
    } else if (__REFLECTION_DYNAMIC__) {
        // HLSL 958-962: sample scene_ldr at (x, y-0.2), modulated by
        // reflection_coefficient. No HDR-alpha knee — the engine relies
        // on the LDR scene already being tone-mapped.
        let dyn_uv = vec2<f32>(texcoord_ss.x, texcoord_ss.y - 0.2);
        let dyn_uv_clamped = clamp(dyn_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let dyn_refl = textureSample(scene_ldr_snapshot, scene_ldr_sampler, dyn_uv_clamped).rgb;
        color_reflection = dyn_refl * water_material.reflection_coefficient.x;
    } else {
        // reflection=none (HLSL 933-936) — engine sets float3(0,0,0).
        color_reflection = vec3<f32>(0.0);
    }

    // ---- Diffuse n.l term (water_shading_fx.hlsl:924-929) ----
    // Engine HARDCODES sun direction to (0, 0, 1) and explicitly comments
    // out the normalize(sun_dir_ws):
    //
    //   float3 water_kd= water_diffuse;
    //   float3 sun_dir_ws= float3(0.0, 0.0, 1.0);   //  sun direction
    //   //sun_dir_ws= normalize(sun_dir_ws);
    //   float n_dot_l= saturate(dot(sun_dir_ws, normal));
    //   float3 color_diffuse= water_kd * n_dot_l;
    //
    // Quirk — water diffuse ignores the actual atmosphere sun direction.
    // The intent is a top-down zenith fill that gives flat horizontal
    // water consistent diffuse independent of time-of-day sun rotation.
    // `color_diffuse *= lightmap_intensity` lands at line 965. Phase W5.1.
    let sun_dir_ws = vec3<f32>(0.0, 0.0, 1.0);
    let water_n_dot_l = max(dot(sun_dir_ws, n), 0.0);
    let water_kd = water_material.water_diffuse.rgb;
    let color_diffuse = water_kd * water_n_dot_l * lightmap_intensity;

    // ---- Foam (water_shading_fx.hlsl:873-921) ----
    // Engine HLSL has 4 branches gated on the foam category:
    //   foam=none:  foam_factor = 0  (skipped block; foam_color stays 0)
    //   foam=auto:  foam_factor_auto computed from height ramp
    //   foam=paint: foam_factor_paint sampled from global_shape_texture.b
    //   foam=both:  max(foam_factor_auto, foam_factor_paint)
    // base_tex.zw carries water_height_relative*10 and max_height_relative*10
    // (×10 per HLSL line 523), so we divide back when comparing to foam_height.
    //
    // Phase W4.2 — `__FOAM_NONE__/AUTO/PAINT/BOTH__` substituted per
    // rmdf choice. Naga DCE drops unused factors / texture samples.
    var foam_factor_auto: f32 = 0.0;
    if (__FOAM_AUTO__ || __FOAM_BOTH__) {
        // HLSL 880-884
        let water_h_rel = in.base_tex.z * 0.1;
        let max_h_rel = in.base_tex.w * 0.1;
        let foam_height = water_material.foam_height.x;
        let foam_pow_v = water_material.foam_pow.x;
        let denom = clamp(max_h_rel - foam_height, 0.0, 1.0);
        let auto_t = clamp((water_h_rel - foam_height) / max(denom, 1e-6), 0.0, 1.0);
        foam_factor_auto = pow(auto_t, foam_pow_v);
    }
    var foam_factor_paint: f32 = 0.0;
    if (__FOAM_PAINT__ || __FOAM_BOTH__) {
        // HLSL 886-891: sample global_shape_texture.b at base_tex.xy.
        // Same OOB guard as the VS shape_control: zero foam_factor_paint
        // when `base_tex.xy` exits the painted silhouette so the bitmap's
        // rim doesn't streak foam across OOB texels.
        let _foam_in_range = f32(
            in.base_tex.x >= 0.0 && in.base_tex.x <= 1.0
            && in.base_tex.y >= 0.0 && in.base_tex.y <= 1.0,
        );
        foam_factor_paint = textureSample(
            global_shape_texture, watercolor_texture_sampler, in.base_tex.xy,
        ).b * _foam_in_range;
    }
    // HLSL 893-905: output factor per branch.
    var foam_factor: f32;
    if (__FOAM_AUTO__) {
        foam_factor = foam_factor_auto;
    } else if (__FOAM_PAINT__) {
        foam_factor = foam_factor_paint;
    } else if (__FOAM_BOTH__) {
        foam_factor = max(foam_factor_auto, foam_factor_paint);
    } else {
        // foam=none — engine leaves foam_factor = 0 at declaration (875).
        foam_factor = 0.0;
    }
    // HLSL 908: add ripple foam (ripple_foam_factor = 0 until Phase W7
    // ripple interaction lands).
    foam_factor = max(0.0, foam_factor);

    // Sample foam_texture × foam_texture_detail when factor is non-trivial
    // (engine guards with > 0.002 — same threshold, HLSL 913). Each texture
    // has its own xform applied to wave_texcoord (line 916-917).
    var foam_color = vec4<f32>(0.0);
    if (foam_factor > 0.002) {
        let foam_uv = transform_texcoord(in.wave_texcoord, water_material.foam_texture_xform);
        let foam_detail_uv = transform_texcoord(in.wave_texcoord, water_material.foam_texture_detail_xform);
        // Engine `riverworld_water_rough.shader_water`
        // `parameters[32]/bitmap address mode = 0` (Wrap) for both
        // foam_texture and foam_texture_detail. They're tiling wave-
        // detail patterns, not painted silhouettes — sampled through
        // the Clamp `watercolor_texture_sampler`, OOB UVs would streak
        // the bitmap edge across the lake. Reuse the Repeat
        // `wave_displacement_array_sampler` for foam.
        let f_main = textureSample(
            foam_texture, wave_displacement_array_sampler, foam_uv,
        );
        let f_detail = textureSample(
            foam_texture_detail, wave_displacement_array_sampler, foam_detail_uv,
        );
        foam_color = vec4<f32>(f_main.rgb * f_detail.rgb, f_main.a * f_detail.a);
        // Engine line 920: foam_factor *= foam_color.w (alpha gates how
        // much of the foam pattern actually applies).
        foam_factor = foam_color.w * foam_factor;
    }

    // ---- Engine-faithful output composition ----
    // Tracks `bed_contrib` (= color_refraction_bed_contribution in
    // HLSL) through every step that scales the bed's effective
    // contribution, so the final subtract-fog-exposure-add dance
    // removes EXACTLY the bed's share of output_color — not more,
    // not less.
    //
    // HLSL line 870: bed_contrib starts at `transparence`
    var bed_contrib = color_refraction_bed_contribution;

    // HLSL line 995: lerp(refraction, reflection, fresnel) — bed
    // share is scaled by (1 - fresnel) since reflection contributes
    // pure non-bed.
    var output_color = mix(color_refraction, color_reflection, fresnel);
    // HLSL line 996: bed_contrib *= 1 - fresnel
    bed_contrib = bed_contrib * (1.0 - fresnel);
    // HLSL line 999: + color_diffuse (pure non-bed, no contrib change).
    output_color = output_color + color_diffuse;

    // ---- bankalpha (water_shading_fx.hlsl:1002-1022) ----
    // Engine:
    //   bankalpha=none  → skip the mix entirely (output_color unchanged)
    //   bankalpha=depth → alpha = saturate(position_ws.w / bankalpha_infuence_depth)
    //   bankalpha=paint → alpha = saturate(water_color_from_texture.w)
    // Then `output = lerp(bed, output, alpha)` and
    // `bed_contrib = (1-alpha) + bed_contrib*alpha`.
    //
    // Phase W2 — `__BANKALPHA_NONE__` / `__BANKALPHA_PAINT__` /
    // `__BANKALPHA_DEPTH__` substituted per rmdf choice.
    if (!__BANKALPHA_NONE__) {
        var bank_alpha: f32 = 1.0;
        if (__BANKALPHA_PAINT__) {
            bank_alpha = clamp(water_color_from_texture_rgba.w, 0.0, 1.0);
        } else if (__BANKALPHA_DEPTH__) {
            // HLSL `position_ws.w` (PC path) is the per-vertex baked
            // water_depth from `local_info.y` — we forward that as
            // `in.water_depth`.
            let divisor = max(water_material.bankalpha_infuence_depth.x, 1e-6);
            bank_alpha = clamp(in.water_depth / divisor, 0.0, 1.0);
        }
        output_color = mix(color_refraction_bed, output_color, bank_alpha);
        bed_contrib = (1.0 - bank_alpha) + bed_contrib * bank_alpha;
    }

    // HLSL line 1023: foam mix.
    let foam_lit = foam_color.rgb * lightmap_intensity;
    output_color = mix(output_color, foam_lit, foam_factor);
    // HLSL line 1026: bed_contrib *= 1 - foam_factor
    bed_contrib = bed_contrib * (1.0 - foam_factor);

    // ---- Subtract-fog-exposure-add dance (HLSL line 1031-1037) ----
    // Avoids double-applying fog + exposure to the refracted scene_ldr
    // sample, which already went through atmosphere + g_exposure in
    // its rmsh-pass output to lighting_base.
    output_color = output_color - color_refraction_bed * bed_contrib;
    output_color = output_color * in.fog_extinction
                 + in.fog_inscatter * (1.0 - bed_contrib);
    let g_exposure = engine_exposure.g_exposure.x;
    output_color = output_color * g_exposure;
    output_color = output_color + color_refraction_bed * bed_contrib;

    // ---- Underwater fog (water_shading_fx.hlsl:1042-1046) ----
    // When the camera is below a water surface, mix the surface color
    // toward `k_ps_underwater_fog_color` by `transparence`. Engine:
    //   transparence = 0.5 * saturate(1 - compute_fog_factor(
    //                                   k_ps_underwater_murkiness,
    //                                   incident_ws.w));
    //   output      = lerp(k_ps_underwater_fog_color, output, transparence);
    // The `0.5*` factor halves the murkiness contribution so even
    // close-up water has some underwater haze; far water saturates
    // toward fog color.
    if (water_ps.k_is_camera_underwater != 0u) {
        let uw_t_raw = 1.0 - compute_fog_factor(
            water_shared_ps.k_ps_underwater_murkiness.x,
            incident_ws_w,
        );
        let uw_transparence = 0.5 * clamp(uw_t_raw, 0.0, 1.0);
        output_color = mix(
            water_shared_ps.k_ps_underwater_fog_color.rgb,
            output_color,
            uw_transparence,
        );
    }

    // Engine emits opaque alpha=1.0 (`water_shading_fx.hlsl:1057/1061`).
    // Shore softness is in the COLOR mix at line 1019, not blending.

    // Engine-investigation diagnostic (PROTOMORPH_DEBUG_WATER_VIZ in host).
    // viz=1: lightmap_intensity * 5  — verifies whether atlas-empty regions
    //         are the cause of missing blue. If river surface renders BLACK
    //         under this viz, the lightprobe atlas IS zero at water UVs
    //         (engine-faithful zero — open investigation
    //         project_waterfall_atlas_empty_fallback_2026_05_15). If river
    //         renders bright colored, atlas has data and the bug is elsewhere.
    // viz=2: water_color (texture × coefficient × lightmap_intensity) — the
    //         actual blue tint that gets lerp'd against scene_ldr.
    // viz=3: lm_tex.xy as RG — verifies the lightmap UVs water vertices land at.
    let viz = water_ps.k_debug_viz;
    if (viz == 1u) {
        let accum = vec4<f32>(lightmap_intensity * 5.0, 1.0);
        return AccumPixel(accum, accum);
    } else if (viz == 2u) {
        let accum = vec4<f32>(water_color * 5.0, 1.0);
        return AccumPixel(accum, accum);
    } else if (viz == 3u) {
        let accum = vec4<f32>(fract(in.lm_tex.x), fract(in.lm_tex.y), 0.0, 1.0);
        return AccumPixel(accum, accum);
    }

    let accum = vec4<f32>(output_color, 1.0);
    return AccumPixel(accum, accum);
}
