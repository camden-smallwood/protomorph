// `decorators.hlsl` Static-variant port — no #defines (no
// DECORATOR_WIND, no DECORATOR_DYNAMIC_LIGHTS, no DECORATOR_DOMINANT_LIGHT,
// no DECORATOR_SHADED_LIGHT, no DECORATOR_WAVY). The simplest of the 6
// engine variants. Ships as the v1 fallback for every decorator_set
// while per-variant ports land.
//
// Engine path (decorators.hlsl @ default_vs/default_ps with no macros):
//   VS: world = quaternion_transform(vert) * scale + instance_pos
//       compute_scattering(camera, world) -> extinction, inscatter
//       ambient_light.rgb = instance_color.rgb * 2^(a*63.75-31.75)
//       ambient_light.rgb *= extinction
//   PS: light = ambient_light
//       color = sampleBiasGlobal2D(diffuse, uv) * light + inscatter
//       color.rgb *= g_exposure.rrr
//       clip(color.a - k_decorator_alpha_test_threshold)
//
// Notes:
//   - We don't have the runtime-baked HDR `instance_color.a` exponent
//     (cache-builder fills it from cluster light samples; MCC strips
//     that data — see reference_mcc_strips_decorator_data.md). Use the
//     painted `tint_color.rgb` directly as ambient.
//   - Quaternion / instance_position / scale are pre-composed Rust-side
//     into a 4×4 model matrix (see `build_instance` in render_decorators.rs).
//   - Single MRT (lighting_base) — engine writes color + normal+depth
//     to MRT0/1 via `convert_to_albedo_target`. v1 skips the normal MRT
//     write so SSAO can't AO foliage; deferred until per-variant lands.

// ---------------------------------------------------------------------------
// Engine bindings (mirror engine_bindings.wgsl shape — atmosphere + camera)
// ---------------------------------------------------------------------------

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Engine default-lighting cbuffer (dominant light + SH probe). Populated
// by `c_lighting_interface::setup_default_lighting @ 0x1806AA190` at
// scenario load. We use the SH probe as the "no rays hit" fallback for
// per-placement ambient — Reach's `light_placement` falls back to
// `c_lighting_interface::get_default_sky_lightprobe` when its 10-sample
// BSP3D raycast finds no hits, then evaluates `ravi_order_3` at
// world_up. Same path here, just applied to all placements (cluster-uniform
// approximation; full Reach bake is filed as a follow-up).
struct EngineDefaultLighting {
    dominant_light_direction: vec4<f32>,
    dominant_light_intensity: vec4<f32>,
    ravi: array<vec4<f32>, 10>,
};
@group(0) @binding(1) var<uniform> default_lighting: EngineDefaultLighting;

// Lightmap-compress cbuffer — 3 vec4 of L-range values per
// `LightmapBspData.compression_vectors`, used to scale the BC3-packed
// SH coefficients back to physical units. Mirror of
// `lightmap_sampling_fx.wgsl::LightmapCompress` (binding 3 of camera_bgl).
struct LightmapCompress {
    constant_0: vec4<f32>,
    constant_1: vec4<f32>,
    constant_2: vec4<f32>,
};
@group(0) @binding(3) var<uniform> lightmap_compress: LightmapCompress;

// Lightprobe atlas — 8 BC3 array slices encoding the 4 SH coefs (DC + 3
// L1 bands) as (L, UVW) half-pairs. Sampled at the per-placement UV
// computed via `lightmap_raycast::Raycaster::cast_vertical` at scenario
// load.
@group(0) @binding(4) var lightprobe_atlas: texture_2d_array<f32>;
// Lightprobe intensity atlas — currently unused by decorators but kept
// in the binding set so the BGL stays in sync with consumers.
@group(0) @binding(5) var lightprobe_intensity_atlas: texture_2d_array<f32>;
@group(0) @binding(6) var lightmap_atlas_sampler: sampler;

// Simple-lights cbuffer — up to 8 dynamic point/spot lights from
// LightsView. Populated each frame by `Renderer::render`. Layout
// matches `simple_lights_fx.hlsl`. Engine binding slot: 0x310000+0x310001.
struct SimpleLight {
    position_size: vec4<f32>,
    inv_direction_sphere: vec4<f32>,
    color_smooth: vec4<f32>,
    falloff_scale_offset: vec4<f32>,
    bounding_radius2: vec4<f32>,
};
struct SimpleLights {
    count: vec4<f32>,
    lights: array<SimpleLight, 8>,
};
@group(0) @binding(8) var<uniform> simple_lights_cb: SimpleLights;

// `calc_simple_lights_analytical_diffuse_translucent` — verbatim port
// of `simple_lights_fx.hlsl:136`. Diffuse-only accumulation across
// active simple lights. Engine line 169 commented out the cosine
// modulation (pure radiance accumulation); we mirror that.
fn simple_lights_diffuse(world_pos: vec3<f32>) -> vec3<f32> {
    var diffuse = vec3<f32>(0.0);
    let n = u32(simple_lights_cb.count.x);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let l = simple_lights_cb.lights[i];
        let to_light_test = l.position_size.xyz - world_pos;
        let dist2 = dot(to_light_test, to_light_test);
        if (dist2 >= l.bounding_radius2.x) {
            continue;
        }
        let inv_dist = inverseSqrt(max(dist2, 1e-12));
        let to_light = to_light_test * inv_dist;
        let light_size = l.position_size.w;
        let inv_dir = l.inv_direction_sphere.xyz;
        let sphere = l.inv_direction_sphere.w;
        let smooth_exp = l.color_smooth.w;
        let falloff_scale = l.falloff_scale_offset.xy;
        let falloff_offset = l.falloff_scale_offset.zw;

        var falloff = vec2<f32>(
            1.0 / (light_size + dist2),
            dot(to_light, inv_dir),
        );
        falloff = max(vec2<f32>(0.0001), falloff * falloff_scale + falloff_offset);
        falloff.y = pow(falloff.y, smooth_exp) + sphere;
        let combined = clamp(falloff.x, 0.0, 1.0) * clamp(falloff.y, 0.0, 1.0);
        diffuse = diffuse + l.color_smooth.xyz * combined;
    }
    return diffuse;
}

// Engine atmosphere — populated from `scenario.atmospheric` →
// haze_level setting. Used for `compute_scattering` + `g_exposure`.
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
};
@group(0) @binding(2) var<uniform> engine_atmosphere_raw: EngineAtmosphereRaw;

// `EngineExposure` (B1) — engine-canonical render-target globals.
// Mirrors `setup_render_target_globals_with_exposure` and the layout in
// `assets/halo_shaders/engine_bindings.wgsl`. Uploaded once per frame
// from `GpuEngineExposure::from_camera_fx`. The decorator PS reads
// `g_exposure.x` only (`color.rgb *= g_exposure.rrr`); the rest of the
// vec4 (.y = pow(2, HDR_target_stops), .z = 1.0, .w = 1.0) is reserved
// for the `convert_to_render_target` LDR/HDR alpha-adjust paths.
struct EngineExposure {
    g_exposure: vec4<f32>,
    g_alt_exposure: vec4<f32>,
};
@group(0) @binding(9) var<uniform> engine_exposure: EngineExposure;

const k_log2_e: f32 = 1.4426950;

// Engine `g_exposure.r` — view exposure scalar from
// `c_camera_fx_values::get_render_exposure` × `g_HDR_target_stops`.
// Engine binds via dedicated render-target-globals cbuffer (NOT the
// atmosphere cbuffer's slot7.x — that aliasing was a v1 stop-gap that
// happened to land the right value but wouldn't survive a future
// exposure-system port).
fn g_exposure() -> f32 {
    return engine_exposure.g_exposure.x;
}

// `ravi_order_3` — full SH3 evaluation at a normal direction. Faithful
// port of `spherical_harmonics_fx.hlsl::ravi_order_3` (mirrored in
// assets/halo_shaders/spherical_harmonics_fx.wgsl). Inlined here so this
// v1 shader stays self-contained (the assembled shader pipeline path
// is the cleaner long-term home — moves with Phase 5's variant ports).
const RAVI_C1: f32 = 0.429043;
const RAVI_C2: f32 = 0.511664;
const RAVI_C4: f32 = 0.886227;
const RAVI_INV_PI: f32 = 0.31830989;

// Engine `ravi_order_3` (`spherical_harmonics_fx.hlsl:169`):
//   lightprobe_color = (c4·DC + (-2c2)·x1 + (-2c1)·x2 - c1·x3) / π
// Decorator carries its own copy because the decorator shader pipeline
// is built outside the render_methods/mod.rs assembler that #-includes
// `spherical_harmonics_fx.wgsl`. Keep this body in lock-step with the
// shared function to avoid drift; an earlier divergence dropped the L2
// normalization (`+ x2 + x3` instead of `(-2c1)·x2 - c1·x3`) and added
// an unwarranted max(0) clamp — both fixed 2026-05-15 in the SH audit.
fn ravi_order_3(normal: vec3<f32>, c: array<vec4<f32>, 10>) -> vec3<f32> {
    var x1: vec3<f32>;
    x1.r = dot(normal, c[1].rgb);
    x1.g = dot(normal, c[2].rgb);
    x1.b = dot(normal, c[3].rgb);

    let a = normal.xyz * normal.yzx;
    var x2: vec3<f32>;
    x2.r = dot(a, c[4].rgb);
    x2.g = dot(a, c[5].rgb);
    x2.b = dot(a, c[6].rgb);

    let b = vec4<f32>(normal.x * normal.x, normal.y * normal.y, normal.z * normal.z, 1.0 / 3.0);
    var x3: vec3<f32>;
    x3.r = dot(b, c[7]);
    x3.g = dot(b, c[8]);
    x3.b = dot(b, c[9]);

    let dc = c[0].rgb;
    return (RAVI_C4 * dc + (-2.0 * RAVI_C2) * x1 + (-2.0 * RAVI_C1) * x2 - RAVI_C1 * x3) * RAVI_INV_PI;
}

// `decode_bpp16_luvw` — port of `lightmap_sampling_fx.hlsl`:15. Two
// BC3 atlas samples encode (L, U), (L, V) as a "half pair"; this
// reconstructs one SH coefficient (vec3) from them.
fn decode_bpp16_luvw(s0: vec4<f32>, s1: vec4<f32>, l_range: f32) -> vec3<f32> {
    let l = s0.a * s1.a * l_range;
    let uvw = s0.xyz + s1.xyz;
    return (uvw * 2.0 - 2.0) * l;
}

// Sample the per-cluster SH atlas at `uv`, evaluate at world_up=(0,0,1),
// return an RGB brightness. Engine path: `sample_lightprobe_texture`
// (4 SH coefs from 8 BC3 slices) → `ravi_order_2(world_up, sh)`.
//
// Halo's ravi packing applied at world_up = (0, 0, 1):
//   c[1] = (x_axis, y_axis, -z_axis, 0)  (per pack_ravi_constants)
//   dot(world_up, c[1].rgb) = -z_axis = -sh[2]   (sh[2] = L1 m=0)
//
// So `eval_at_world_up = (RAVI_C4 * sh[0] + 2 * RAVI_C2 * sh[2]) * RAVI_INV_PI`
//                     = (0.886227*sh[0] + 1.023328*sh[2]) * 0.31830989.
// Only sh[0] (DC) and sh[2] (L1 z-axis) contribute when normal = +Z;
// the other bands' Y_l_m at +Z are zero. Saves 4 atlas samples vs the
// full ravi_order_2.
fn lightmap_eval_at_world_up(uv: vec2<f32>) -> vec3<f32> {
    // Two BC3 slice pairs: slices 0/1 = SH DC, slices 4/5 = SH L1 m=0.
    // `textureSampleLevel` (lod 0) — VS shaders can't use derivatives.
    let s0 = textureSampleLevel(lightprobe_atlas, lightmap_atlas_sampler, uv, 0, 0.0);
    let s1 = textureSampleLevel(lightprobe_atlas, lightmap_atlas_sampler, uv, 1, 0.0);
    let s4 = textureSampleLevel(lightprobe_atlas, lightmap_atlas_sampler, uv, 4, 0.0);
    let s5 = textureSampleLevel(lightprobe_atlas, lightmap_atlas_sampler, uv, 5, 0.0);

    // Compression scalars: sh[0] uses constant_0.x, sh[2] uses constant_0.z
    // (lightmap_sampling_fx.wgsl:62-64).
    let sh0 = decode_bpp16_luvw(s0, s1, lightmap_compress.constant_0.x);
    let sh2 = decode_bpp16_luvw(s4, s5, lightmap_compress.constant_0.z);
    return max((RAVI_C4 * sh0 + 2.0 * RAVI_C2 * sh2) * RAVI_INV_PI, vec3<f32>(0.0));
}

// ---------------------------------------------------------------------------
// Material (per-set diffuse texture)
// ---------------------------------------------------------------------------

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

// Per-frame wind globals — engine-faithful layout matching
// `g_wind_values.{wind_data, wind_data2}` produced by
// `s_wind_definition::update @ 0x180907F80`. See
// `src/halo/render/render_decorators.rs::WindGlobals` for the full
// encoding. `sample_wind(world_xy)` consumes wind_data + wind_data2;
// WAVY variants additionally read wind_data2.w (global time). The
// `wave_flow` knobs are per-set — see `DecoratorSetUniforms.wave_flow`.
struct WindGlobals {
    wind_data: vec4<f32>,   // (ring_pos.x, ring_pos.y, 1/gust_size, 0)
    wind_data2: vec4<f32>,  // (bend × sin_dir, bend × cos_dir, oscillation, time)
};
@group(1) @binding(2) var<uniform> wind_globals: WindGlobals;

// Wind noise bitmap — `random4_warp.bitmap` (BC5/DXN signed
// 2-channel 128×128 noise) by default. Engine binds at vertex
// sampler 0 via `setup_wind_constants @ 0x180908290` with wrap+linear.
@group(1) @binding(4) var t_wind_noise: texture_2d<f32>;
@group(1) @binding(5) var s_wind_noise: sampler;

// Procedural stand-in for the engine's `sample2Dlod(wind_texture, ...)`
// call. The real bitmap (`gust noise bitmap`, default
// `shaders/default_bitmaps/random4_warp.bitmap`) is a 4-channel noise
// pattern, bilinearly-filtered with wrap addressing. The bilinear
// filter band-limits the spatial frequency of the noise — neighbouring
// world points produce smoothly-varying samples (gradient changes over
// many world units, not per-fragment).
//
// Engine VS body (`wind_fx.hlsl::sample_wind`, lines 10-25):
//   float2 texc = position.xy * wind_data.z + wind_data.xy;
//   float4 wind_vector = sample2Dlod(wind_texture, texc, 0);
//   wind_vector.xy = wind_vector.xy * wind_data2.z + wind_data2.xy;
//   return wind_vector.xy;
//
// Verbatim port of `wind_fx.hlsl::sample_wind` (lines 10-25):
//   float2 texc = position.xy * wind_data.z + wind_data.xy;
//   float4 wind_vector = sample2Dlod(wind_texture, texc, 0);
//   wind_vector.xy = wind_vector.xy * wind_data2.z + wind_data2.xy;
//   return wind_vector.xy;
//
// Halo's noise bitmap is BC5/DXN signed (2-channel, [-1, 1]). wgpu's
// `Bc5RgSnorm` decodes the BC blocks hardware-side and returns
// already-signed `.xy` from the sample. Wrap addressing on the sampler
// makes the ring-buffer wrap invisible — `texc=0.0` and `texc=1.0`
// sample the same texel. `textureSampleLevel(..., 0.0)` is the WGSL
// equivalent of `sample2Dlod(..., 0)`.
fn sample_wind(world_xy: vec2<f32>) -> vec2<f32> {
    let texc = world_xy * wind_globals.wind_data.z + wind_globals.wind_data.xy;
    let n = textureSampleLevel(t_wind_noise, s_wind_noise, texc, 0.0);
    return n.xy * wind_globals.wind_data2.z + wind_globals.wind_data2.xy;
}

// Per-set uniforms — variant index + tag-driven knobs.
//   knobs.x = variant_idx (0..5; see `render_shader_index`)
//   knobs.y = translucency
//   knobs.z = contrast.x  (ShadedDynamicLights cosine offset)
//   knobs.w = contrast.y  (ShadedDynamicLights cosine slope)
//   fade.x  = start_fade_distance (world units)
//   fade.y  = end_fade_distance   (world units)
//   wave_flow = (wavelength_x, wavelength_y, wave_speed, wave_frequency)
//     — engine slot 0x160005, consumed only by the WAVY variant.
struct DecoratorSetUniforms {
    knobs: vec4<f32>,
    fade: vec4<f32>,
    wave_flow: vec4<f32>,
};
@group(1) @binding(3) var<uniform> set_uniforms: DecoratorSetUniforms;

// `s_decorator_set::e_render_shader` (matches `render_shader_index` in
// render_decorators.rs; matches `RenderShader` enum on the tag side).
const VARIANT_WIND_DYNAMIC: f32     = 0.0;
const VARIANT_DYNAMIC_LIGHTS: f32   = 1.0;
const VARIANT_STATIC: f32           = 2.0;
const VARIANT_DOMINANT_LIGHT: f32   = 3.0;
const VARIANT_WAVY_DYNAMIC: f32     = 4.0;
const VARIANT_SHADED_DYNAMIC: f32   = 5.0;

// `k_decorator_alpha_test_threshold` from `decorators.h:5` — engine
// value is 0.3, NOT the 0.5 used by other alpha-tested paths. Lower
// threshold preserves the silhouette of thin foliage cards instead of
// clipping the alpha gradient mid-leaf. The same constant gates the
// per-vertex LOD-fade collapse (`saturate(distance × LOD.x + LOD.y)`)
// — engine VS discards a vertex when fade alpha falls below this.
const k_decorator_alpha_test_threshold: f32 = 0.3;

// ---------------------------------------------------------------------------
// Vertex / interpolators
// ---------------------------------------------------------------------------

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
    // Per-instance attributes (stream 1) — model matrix + tint + lightmap UV.
    @location(7) inst_m0: vec4<f32>,
    @location(8) inst_m1: vec4<f32>,
    @location(9) inst_m2: vec4<f32>,
    @location(10) inst_m3: vec4<f32>,
    @location(11) inst_tint: vec4<f32>,
    /// `[u, v, raycast_valid, _pad]` — `raycast_valid = 1.0` when the
    /// per-placement vertical raycast hit a BSP triangle (use atlas
    /// SH); `0.0` when the raycast missed (fall back to global SH).
    @location(12) inst_lightmap_uv: vec4<f32>,
};

// Mirrors decorators.hlsl `default_vs` outputs:
//   TEXCOORD0 = texcoord (xy + ShadedDynamicLights cosine .z/.w pair)
//   TEXCOORD1 = ambient_light  (rgb = per-instance ambient × extinction; a unused)
//   TEXCOORD2 = inscatter      (rgb = atmospheric inscatter; a unused)
//   TEXCOORD3 = world_normal   (xyz; PS encodes into normal MRT for SSAO)
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tex_coords: vec4<f32>,
    @location(1) ambient_light: vec3<f32>,
    @location(2) inscatter: vec3<f32>,
    @location(3) world_normal: vec3<f32>,
    /// LOD fade alpha — `saturate((end_fade − distance) / (end_fade −
    /// start_fade))`. PS multiplies albedo.a so alpha-test culls
    /// past-fade fragments cleanly.
    @location(4) fade_alpha: f32,
    /// Host-surface normal (the BSP surface the decorator is planted on
    /// = the instance's up/local-Z axis), written to the normal MRT so
    /// the screenspace shadow + SSAO treat the decorator like its host
    /// surface — orientation-independent (ground/wall/ceiling). The
    /// per-leaf `world_normal` above stays the LIGHTING normal; using it
    /// for the shadow cosine zeroed shadows on grass (billboard normals
    /// are ~perpendicular to the up-ish light).
    @location(5) surface_normal: vec3<f32>,
};

// PS output — two MRTs matching engine `albedo_pixel`
// (`albedo_pass_fx.hlsl:4-11`). Engine `default_ps` does sample +
// light + alpha-test + writes the LIT color to slot 0
// (`_surface_accum_HDR`); slot 1 carries the encoded normal +
// albedo.w (mirror of `convert_to_albedo_target`). Decorators
// render ONCE in the albedo pass — there is no SL entry. The lit
// color flows through to the SL target via the albedo→SL copy-blit.
struct PsOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

// Decorator vertex normal is packed as oct-snorm8 in
// `ModelVertex::normal_oct` (locations 1 of stream 0). Decode mirrors
// `geometry::oct_decode_snorm8` Rust-side.
fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
}

// `compute_scattering` — atmosphere_fx.hlsl port. Reused verbatim from
// assets/halo_shaders/atmosphere_fx.wgsl. Inlined here so this v1
// shader doesn't depend on the assembled-shader include path.
//
// DIAGNOSTIC: bypass mirrors `atmosphere_fx.wgsl::ATMOSPHERE_BYPASS`.
// Toggle both in lockstep when bisecting the "washed-out white" bug.
const ATMOSPHERE_BYPASS_DECORATOR: bool = false;

fn compute_scattering(
    view_point: vec3<f32>,
    world_scene_point: vec3<f32>,
    extinction: ptr<function, vec3<f32>>,
    inscatter: ptr<function, vec3<f32>>,
) {
    if (ATMOSPHERE_BYPASS_DECORATOR) {
        *extinction = vec3<f32>(1.0);
        *inscatter = vec3<f32>(0.0);
        return;
    }
    let r = engine_atmosphere_raw;
    let sun_direction = r.slot0_sun_dir_dist_bias.xyz;
    let distance_bias = r.slot0_sun_dir_dist_bias.w;
    let sun_int_norm = r.slot1_sun_int_norm_thickness.xyz;
    let max_fog_thickness = r.slot1_sun_int_norm_thickness.w;
    let beta_m_log2e = r.slot2_beta_m_log2e_g1.xyz;
    let mie_g_plus_one = r.slot2_beta_m_log2e_g1.w;
    let beta_p_log2e = r.slot3_beta_p_log2e_refh.xyz;
    let reference_height = r.slot3_beta_p_log2e_refh.w;
    let beta_m_angular = r.slot4_beta_m_angular_mieh.xyz;
    let mie_height_scale = r.slot4_beta_m_angular_mieh.w;
    let beta_p_angular = r.slot5_beta_p_angular_rayh.xyz;
    let rayleigh_height_scale = r.slot5_beta_p_angular_rayh.w;
    let mie_g_times_two = r.slot6_g2.x;

    if (max_fog_thickness < 0.0) {
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
    let c_theta = -dot(view_vector, sun_direction);
    dist = max(dist + distance_bias, 0.0);
    dist = min(dist, max_fog_thickness);

    var view_height = max(view_point.z - reference_height, 0.0);
    var scene_height = max(world_scene_point.z - reference_height, 0.0);
    let diff = view_height - scene_height;
    view_height = view_height * k_log2_e;
    scene_height = scene_height * k_log2_e;
    let mie_h = max(mie_height_scale, 1e-4);
    let ray_h = max(rayleigh_height_scale, 1e-4);

    if (diff * diff > 0.001) {
        let dp = -(exp2(-view_height / mie_h) - exp2(-scene_height / mie_h)) * dist * mie_h / diff;
        let dm = -(exp2(-view_height / ray_h) - exp2(-scene_height / ray_h)) * dist * ray_h / diff;
        *extinction = exp2(-(beta_m_log2e * dm + beta_p_log2e * dp));
    } else {
        let dp = exp2(-view_height / mie_h) * dist;
        let dm = exp2(-view_height / ray_h) * dist;
        *extinction = exp2(-(beta_m_log2e * dm + beta_p_log2e * dp));
    }

    let beta_m_theta = beta_m_angular * (1.0 + c_theta * c_theta);
    let heyey = mie_g_plus_one - mie_g_times_two * c_theta;
    let heyey_n1p5 = pow(max(heyey, 1e-4), -1.5);
    let beta_p_theta = beta_p_angular * heyey_n1p5;
    *inscatter = sun_int_norm * (beta_m_theta + beta_p_theta) * (vec3<f32>(1.0) - *extinction);
}

// HLSL camera position recovered from the inverse view matrix's
// translation column (row-major: view⁻¹ third column when stored as
// transposed). For our column-major `camera.view`, the camera world
// position is `-(view^T * view.w_col).xyz`. Cheap closed form using the
// rotation submatrix: `-view[3].xyz * view_rot⁻¹`. For a view matrix
// without scale this is just `-(R^T) * t`.
fn camera_position() -> vec3<f32> {
    let rt = mat3x3<f32>(
        vec3<f32>(camera.view[0].x, camera.view[1].x, camera.view[2].x),
        vec3<f32>(camera.view[0].y, camera.view[1].y, camera.view[2].y),
        vec3<f32>(camera.view[0].z, camera.view[1].z, camera.view[2].z),
    );
    let t = vec3<f32>(camera.view[3].x, camera.view[3].y, camera.view[3].z);
    return rt * (-t);
}

@vertex
fn default_vs(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.inst_m0, in.inst_m1, in.inst_m2, in.inst_m3);
    let variant = set_uniforms.knobs.x;
    let motion_scale = in.inst_lightmap_uv.w;

    // Instance world XY — model matrix's translation column. Used by
    // wind/wavy phase calcs.
    let inst_xy = vec2<f32>(in.inst_m3.x, in.inst_m3.y);

    // Local position + height-scale bend. Wind variants compute a
    // height_scale < 1 that shortens the leaf as it bends.
    var local_pos = in.position;
    var height_scale: f32 = 1.0;
    var wind_offset_xy: vec2<f32> = vec2<f32>(0.0);

    if (variant == VARIANT_WIND_DYNAMIC) {
        // `decorators.hlsl:223-234` — sample the wind grid at the
        // instance's world XY, scale by `motion_scale × sat(vertex.z)`
        // (top-of-leaf bends most), and store the offset for the
        // post-quat-transform world-position add.
        let wind_vec = sample_wind(inst_xy);
        let mscale = motion_scale * clamp(local_pos.z, 0.0, 1.0);
        let scaled = wind_vec * mscale;
        wind_offset_xy = scaled;

        // height_scale = sqrt(h² / (h² + |wind|²)) — `decorators_hlsl.hlsl:231-233`:
        //   instance_scale = dot(instance_quaternion, instance_quaternion)  // = scale²
        //   height_squared = instance_scale * vertex_position.z + 0.01       // LINEAR z
        // The z term is LINEAR (was wrongly squared here). instance_scale
        // (the per-instance quaternion scale) isn't plumbed through the
        // model-matrix path yet, so it's treated as 1 — exact for the
        // common unit-scale decorator instances.
        let wind_sq = dot(scaled, scaled);
        let height_sq = local_pos.z + 0.01;
        height_scale = sqrt(height_sq / (height_sq + wind_sq));
    } else if (variant == VARIANT_WAVY_DYNAMIC) {
        // `decorators.hlsl:236-240` — sin-driven sway:
        //   phase = vertex.z × wave_flow.w
        //         + wind_data2.w × wave_flow.z
        //         + dot(inst.xy, wave_flow.xy)
        //   wave = motion_scale × sat(|z|) × sin(phase)
        //   vertex.x += wave
        let wave_flow = set_uniforms.wave_flow;
        let phase = local_pos.z * wave_flow.w
                  + wind_globals.wind_data2.w * wave_flow.z
                  + dot(inst_xy, wave_flow.xy);
        let wave = motion_scale * saturate(abs(local_pos.z)) * sin(phase);
        local_pos.x = local_pos.x + wave;
    }
    local_pos.z = local_pos.z * height_scale;

    let world_pos4 = model * vec4<f32>(local_pos, 1.0);
    // HLSL decorators_hlsl.hlsl:248 — `world_position.xy += wind_vector.xy * height_scale`
    // (the post-transform wind offset is scaled by the bend factor; height_scale
    // is 1.0 for non-wind variants so wind_offset_xy is 0 there anyway).
    let world_pos = vec3<f32>(world_pos4.xy + wind_offset_xy * height_scale, world_pos4.z);

    // World normal — model matrix's rotation submatrix × object normal.
    // ShadedDynamicLights uses the rotated_position vector instead, so
    // sun-facing vertices get bright modulation in PS.
    let nrm_mat = mat3x3<f32>(in.inst_m0.xyz, in.inst_m1.xyz, in.inst_m2.xyz);
    let rotated_position = nrm_mat * local_pos;
    var world_normal: vec3<f32>;
    if (variant == VARIANT_SHADED_DYNAMIC) {
        world_normal = normalize(rotated_position);
    } else {
        world_normal = normalize(nrm_mat * oct_decode(in.normal_oct));
    }

    var extinction: vec3<f32>;
    var inscatter: vec3<f32>;
    compute_scattering(camera_position(), world_pos, &extinction, &inscatter);

    // Static-variant ambient is `instance_color.rgb * 2^(a*63.75-31.75)`
    // in the engine; the cache-builder bakes per-placement HDR light
    // there from a 10-sample BSP3D raycast against the lightmap atlas.
    // MCC strips the bake.
    //
    // Per-placement approximation:
    //   1. At scenario load, vertical-raycast each placement against
    //      BSP geometry → barycentric → interpolated atlas UV
    //      (one ray of Reach `light_placement`'s 10-sample bake).
    //   2. Here in VS: sample the lightprobe atlas at that UV, decode
    //      SH, evaluate at world_up = (0, 0, 1) per channel.
    //   3. Multiply the painted tint by the SH brightness.
    // Placements whose raycast missed (off-BSP / vertical wall under
    // them) fall back to the scenario's global SH probe (Reach's
    // no-rays-hit `get_default_sky_lightprobe` path).
    // Per-placement HDR ambient — engine `instance_color`. Phase 1c
    // bake (`bake_placement` Reach 10-sample loop) produces this and
    // packs it into `inst_tint` as `(R, G, B, exp_byte/255)`. Decode
    // mirrors `decorators.hlsl:282`:
    //   ambient = inst_tint.rgb × exp2(inst_tint.a × 63.75 − 31.75)
    // For unbaked placements (no atlas) the bake encodes the painted
    // tint at exp=127 → exp2(0) = 1.0, so this branch recovers the
    // pre-bake fallback gracefully.
    let baked_ambient = in.inst_tint.rgb * exp2(in.inst_tint.a * 63.75 - 31.75);

    // Optional analytical sun contribution for DominantLightOnly.
    // Engine: `motion_scale * sun_color * calc_diffuse_lobe(world_normal,
    // sun_direction, translucency)`. We skip translucency for v1.
    var dynamic_light = vec3<f32>(0.0);
    if (variant == VARIANT_DOMINANT_LIGHT) {
        // Decorator VS — pull sun from engine_atmosphere instead of
        // `dominant_light` (FRAGMENT-only at binding 13 to dodge the
        // Metal MSL slot conflict). atmosphere's `slot0.xyz` IS the
        // engine analytical sun direction; `slot1.xyz` carries the
        // normalized sun intensity divided by (β_m+β_p) — multiply
        // back to recover RGB radiance.
        let atm = engine_atmosphere_raw;
        let sun_dir = atm.slot0_sun_dir_dist_bias.xyz;
        let sun_color = atm.slot1_sun_int_norm_thickness.xyz
                      * (atm.slot2_beta_m_log2e_g1.xyz + atm.slot3_beta_p_log2e_refh.xyz);
        let cos_term = max(dot(world_normal, sun_dir), 0.0);
        dynamic_light = motion_scale * sun_color * cos_term;
    }

    // DECORATOR_DYNAMIC_LIGHTS — accumulate simple-lights diffuse
    // (decorators.hlsl:263-273). Engine path uses
    // `calc_simple_lights_analytical_diffuse_translucent` which
    // skips the cosine modulation by design (engine line 169).
    // Variants 0/1/4/5 set the macro; 2/3 don't.
    if (variant == VARIANT_WIND_DYNAMIC
        || variant == VARIANT_DYNAMIC_LIGHTS
        || variant == VARIANT_WAVY_DYNAMIC
        || variant == VARIANT_SHADED_DYNAMIC) {
        dynamic_light = dynamic_light + simple_lights_diffuse(world_pos);
    }

    let ambient = (baked_ambient + dynamic_light) * extinction;

    // ShadedDynamicLights: PS modulates `light` by a per-pixel cosine
    // computed from `dot(rotated_position, sun_direction) /
    // length(rotated_position)`. Pass these as texcoord.zw to PS.
    var sun_cos_z: f32 = 1.0;
    var sun_cos_w: f32 = 1.0;
    if (variant == VARIANT_SHADED_DYNAMIC) {
        // VS-side sun_dir — same atmosphere fallback as above.
        let sun_dir = engine_atmosphere_raw.slot0_sun_dir_dist_bias.xyz;
        sun_cos_z = dot(rotated_position, sun_dir);
        sun_cos_w = max(length(rotated_position), 1e-4);
    }

    var out: VsOut;
    let clip_pos = camera.projection
        * camera.view
        * vec4<f32>(world_pos, 1.0);
    out.clip = clip_pos;
    out.tex_coords = vec4<f32>(in.tex_coords, sun_cos_z, sun_cos_w);
    out.ambient_light = ambient;
    out.inscatter = inscatter;
    out.world_normal = world_normal;
    // Host-surface normal = the instance's up axis (local +Z in world =
    // 3rd column of the instance rotation). The decorator is planted with
    // local-Z along the surface normal, so this is the ground/wall/ceiling
    // normal it grows from — correct for any orientation.
    out.surface_normal = normalize(in.inst_m2.xyz);

    // LOD fade — engine: `saturate(distance × LOD_constants.x + LOD_constants.y)`
    // per `decorators.hlsl:165`. Substituting our fade.xy:
    //   alpha = saturate((end − distance) / (end − start))
    let cam_to_world = world_pos - camera_position();
    let dist = length(cam_to_world);
    let start_fade = set_uniforms.fade.x;
    let end_fade = set_uniforms.fade.y;
    let span = max(end_fade - start_fade, 0.001);
    out.fade_alpha = clamp((end_fade - dist) / span, 0.0, 1.0);
    return out;
}

// Faithful port of `decorators.hlsl::default_ps` (lines 321-363).
// Engine flow: sample diffuse → light with ambient + inscatter →
// alpha-test clip → multiply by g_exposure → return
// `convert_to_albedo_target(color, normal, position_w)` which writes
// the LIT color to `_surface_accum_HDR` (RT0) and encoded normal +
// albedo.w to `_surface_post_HDR` (RT1).
@fragment
fn default_ps(in: VsOut) -> PsOut {
    let albedo = textureSample(t_diffuse, s_diffuse, in.tex_coords.xy);
    // Multiply texture alpha by per-placement LOD fade so foliage
    // dithers / cuts out cleanly past `end_fade_distance`.
    let alpha_with_fade = albedo.a * in.fade_alpha;
    if (alpha_with_fade < k_decorator_alpha_test_threshold) {
        discard;
    }

    // ShadedDynamicLights — modulate ambient by per-pixel cosine.
    // Engine: `light.rgb *= saturate(texcoord.z) * contrast.y + contrast.x`
    // (decorators.hlsl:336).
    var light = in.ambient_light;
    if (set_uniforms.knobs.x == VARIANT_SHADED_DYNAMIC) {
        let cos_per_pixel = clamp(in.tex_coords.z / in.tex_coords.w, 0.0, 1.0);
        let contrast_x = set_uniforms.knobs.z;
        let contrast_y = set_uniforms.knobs.w;
        light = light * (cos_per_pixel * contrast_y + contrast_x);
    }

    // decorators.hlsl:347 — `float4 color = texcoord * light + inscatter;`
    var color = vec4<f32>(albedo.rgb * light + in.inscatter, albedo.a);
    // decorators.hlsl:353 — `color.rgb *= g_exposure.rrr;`
    color = vec4<f32>(color.rgb * vec3<f32>(g_exposure()), color.a);

    // decorators.hlsl:359 — `convert_to_albedo_target(color, normal, position_w)`.
    // Mirrors `albedo_pass_fx.hlsl:23-40`:
    //   result.albedo_specmask = albedo;          (slot 0 = lit color)
    //   result.normal.xyz = normal * 0.5 + 0.5;
    //   result.normal.w = albedo.w;               (slot 1 .w = original albedo.a)
    // Write the HOST-SURFACE normal (not the per-leaf billboard normal)
    // to the normal MRT, so the screenspace shadow's cosine term + SSAO
    // treat the decorator like the surface it grows from. Using the leaf
    // normal here zeroed object shadows on grass (leaf normals are ~90°
    // to the up-ish light → cosine≈0). The lit `color` already used the
    // leaf `world_normal`, so this doesn't change how the grass is shaded.
    let n = normalize(in.surface_normal);
    var out: PsOut;
    out.color = color;
    out.normal = vec4<f32>(n * 0.5 + vec3<f32>(0.5), color.w);
    return out;
}
