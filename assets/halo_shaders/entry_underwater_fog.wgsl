// Engine reference: `c_water_renderer::render_underwater_fog @ 0x180694080`.
// HLSL body: `water_ripple_hlsl.hlsl::underwater_vs / underwater_ps`
// (defined-out as `shadow_apply_vs / shadow_apply_ps` aliases).
//
// Fullscreen quad pass that runs inside `c_player_view::render_water`
// BEFORE `render_shading` when `g_is_underwater`. Reads the snapshot
// of `lighting_base` taken before water (extern 45,
// `_surface_screenshot_simple_resolve`) and the scene depth (extern
// 3); writes a fogged copy back to `lighting_base`. Uses the same
// engine cbuffer state that `render_shading` does
// (`k_water_view_xform_inverse`, `k_water_player_view_constant`,
// `k_ps_camera_position`, `k_ps_underwater_murkiness`,
// `k_ps_underwater_fog_color`).

// ---- @group(0): camera (unused but bound for layout consistency).
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// ---- @group(1): WaterSharedPS cbuf (slot 0x560000-0x560007 in engine).
// Identical layout to `entry_water_shading.wgsl`'s WaterSharedPS — same
// host-uploaded buffer is bound at @group(1) @binding(1) of both.
struct WaterSharedPS {
    k_water_view_xform_inverse: mat4x4<f32>,
    k_water_player_view_constant: vec4<f32>,
    k_ps_camera_position: vec4<f32>,
    k_ps_underwater_murkiness: vec4<f32>,
    k_ps_underwater_fog_color: vec4<f32>,
}
@group(1) @binding(1) var<uniform> water_shared_ps: WaterSharedPS;

// ---- @group(2): water_extern_bgl (scene LDR + scene depth).
// Reuses the EXACT layout of the water_shading pass so we share one
// bind group. Bindings 4 and 5 (LDRTextureSize / DepthConstants) are
// bound but unused in this shader (we read raw device-space depth and
// re-project via `k_water_view_xform_inverse`, matching the engine
// underwater_ps's screen→world reconstruction).
@group(2) @binding(0) var scene_ldr_snapshot: texture_2d<f32>;
@group(2) @binding(1) var scene_ldr_sampler: sampler;
@group(2) @binding(2) var scene_depth: texture_depth_2d;
@group(2) @binding(3) var scene_depth_sampler: sampler;
struct LDRTextureSize { dims: vec4<f32>, }
struct DepthConstants { depth_constants: vec4<f32>, }
@group(2) @binding(4) var<uniform> ldr_texture_size: LDRTextureSize;
@group(2) @binding(5) var<uniform> depth_constants: DepthConstants;

// `compute_fog_factor` per `water_ripple_hlsl.hlsl:874-879`:
//   1 - saturate(1 / exp(murkiness * depth))
fn compute_fog_factor(murkiness: f32, depth: f32) -> f32 {
    return 1.0 - clamp(1.0 / exp(murkiness * depth), 0.0, 1.0);
}

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    /// `position_ss` per HLSL `s_underwater_interpolators::position_ss`
    /// — verbatim copy of `OUT.position` for the PS to recover NDC
    /// after the rasterizer's homogeneous divide.
    @location(0) position_ss: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Engine reads from a `k_screen_corners` cbuffer array, but for a
    // fullscreen quad the corners are always NDC ±1. Emit 4 verts as
    // a triangle strip:
    //   vid 0 → (-1, -1)    vid 1 → ( 1, -1)
    //   vid 2 → (-1,  1)    vid 3 → ( 1,  1)
    let x = select(-1.0, 1.0, (vid & 1u) != 0u);
    let y = select(-1.0, 1.0, (vid & 2u) != 0u);
    let p = vec4<f32>(x, y, 0.0, 1.0);
    var o: VsOut;
    o.clip_position = p;
    o.position_ss = p;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Mirrors HLSL `underwater_ps` (water_ripple_hlsl.hlsl:882-923).
    let pos_ss_w = max(in.position_ss.w, 1e-4);
    let ndc = in.position_ss.xy / pos_ss_w;
    var texcoord_ss = ndc * 0.5 + vec2<f32>(0.5, 0.5);
    texcoord_ss.y = 1.0 - texcoord_ss.y;
    let pvc = water_shared_ps.k_water_player_view_constant;
    texcoord_ss = pvc.xy + texcoord_ss * pvc.zw;

    // Sample scene depth (raw device-space, reverse-Z).
    let pixel_depth = textureSampleLevel(
        scene_depth, scene_depth_sampler, texcoord_ss, 0,
    );

    // Re-project to world space:
    //   pixel_world_h = inv(view*proj) * (ndc.xy, depth, 1)
    //   pixel_world   = pixel_world_h.xyz / pixel_world_h.w
    let pixel_clip = vec4<f32>(ndc, pixel_depth, 1.0);
    let pixel_world_h = water_shared_ps.k_water_view_xform_inverse * pixel_clip;
    let pixel_world = pixel_world_h.xyz / max(pixel_world_h.w, 1e-6);

    let cam = water_shared_ps.k_ps_camera_position.xyz;
    let distance = length(cam - pixel_world);

    // Sample LDR snapshot (= scene before water/transparency).
    let pixel_color = textureSample(
        scene_ldr_snapshot, scene_ldr_sampler, texcoord_ss,
    ).rgb;

    // Engine fog formula: closer pixels = less fog (more pixel),
    // farther pixels = saturate to fog_color. The 0.5 cap keeps
    // even close-up pixels at ≤ 50% pixel mix — diving feels heavy.
    let fog_factor = compute_fog_factor(
        water_shared_ps.k_ps_underwater_murkiness.x,
        distance,
    );
    let transparence = 0.5 * clamp(1.0 - fog_factor, 0.0, 1.0);
    let output_color = mix(
        water_shared_ps.k_ps_underwater_fog_color.rgb,
        pixel_color,
        transparence,
    );
    return vec4<f32>(output_color, 1.0);
}
