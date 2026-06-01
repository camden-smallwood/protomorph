// Halo `foliage / static_prt_ambient` entry point.
//
// Faithful port of `foliage_fx.hlsl::static_prt_ambient_vs`
// (line 432-479). Selected for foliage placements whose BSP cluster
// carries PRT-ambient transfer data (single-float per-vertex AO).
//
// Engine VS (foliage_fx.hlsl:432, PC branch):
//   in float prt_c0_c3 : BLENDWEIGHT1   // per-vertex AO scalar
//   vsout = static_common_vs(vertex, ...);   // ravi_order_3 SH eval at normal
//   vsout.lighting *= prt_c0;                // modulate by AO
//
// Engine PS: `static_per_vertex_ps` returns `static_common_ps` — same
// alpha-test + composition as static_sh.
//
// Protomorph PRT-Ambient secondary VB: `PrtAmbientVertex` (single f32,
// `geometry/mod.rs`). Same stream consumed by the rmsh equivalent at
// `entry_static_prt_ambient.wgsl`.
//
// Per the blueprint: foliage is PURE DIFFUSE — no specular, no
// envmap real contribution, no transmission/subsurface. The HLSL
// `calc_material_foliage_ps` always sets `specular_radiance = 0`.
// All lighting comes from `ravi_order_3` evaluated AT THE VERTEX
// (foliage_fx.hlsl:256: `vsout.lighting = ravi_order_3(normal, sh)`)
// and interpolated to the pixel.
//
// PS does alpha-test on `base_map.a` (foliage_fx.hlsl:276:
// `calc_alpha_test_ps(texcoord, output_alpha)`) — discards leaf-card
// transparent regions. NOT alpha-blend.
//
// v1 simplifications vs full HLSL:
//   - Atmospheric scattering implemented per-vertex via
//     `compute_scattering` (foliage_fx.hlsl:258).
//
// Wind animation: per-vertex `vibration` + `animation_offset` ported
// from foliage_fx.hlsl:46-87 below. `g_tree_animation_coeff` driven
// from `engine_misc_ps.ps_total_time.x` (engine MiscPS cbuffer).
//
// Render-state overrides (set by pipeline_cache for rmfl group_tag):
//   - cull_mode = None (leaf cards 2-sided)
//   - depth_write = true (alpha-test, opaque pass)

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
    // Per-vertex PRT transfer scalar (`PrtAmbientVertex` secondary VB,
    // single f32). Engine equivalent: foliage `static_prt_ambient_vs`
    // `prt_c0_c3` BLENDWEIGHT1 input.
    @location(11) prt_transfer: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) sh_lighting: vec3<f32>,
    @location(4) extinction: vec3<f32>,
    @location(5) inscatter: vec3<f32>,
}

fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
}

// Port of `vibration` + `animation_offset` (foliage_fx.hlsl:46-87).
// See entry_albedo_foliage.wgsl for the line-by-line correspondence;
// duplicated here so the static_sh entry doesn't need to share a
// helper module. Reads `engine_exposure.g_alt_exposure.w` (camera_bgl
// binding 9, VS+FS-visible) for the global wind phase — the engine
// `g_tree_animation_coeff` slot. MiscPS would be more semantic but
// can't be VS-visible without colliding with the water pipeline's
// Metal slot layout.
fn foliage_vibration(offset: f32) -> f32 {
    let coeff = engine_exposure.g_alt_exposure.w * 0.5;
    let vibration_base = abs(fract(offset + coeff) - 0.5) * 2.0;
    let x = (0.5 - vibration_base) * 3.14159265;
    return sin(x);
}

fn foliage_animation_offset(texcoord: vec2<f32>) -> vec3<f32> {
    let distance = fract(texcoord.x);
    var id = texcoord.x - distance + 3.0;
    let vib_h = foliage_vibration(id / 0.53);
    id = id + floor(texcoord.y) * 7.0;
    let vib_v = foliage_vibration(id / 1.1173);
    let dirx = fract(id / 0.727) - 0.5;
    let diry = fract(id / 0.371) - 0.5;
    return vec3<f32>(
        dirx * vib_h,
        diry * vib_h,
        vib_v * 0.3,
    ) * distance * material.animation_amplitude_horizontal.x;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Skinning (BSP instances: identity 1-bone; characters: real bones)
    var skin = mat4x4<f32>(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    skin += node_matrices[in.node_indices[0]] * in.node_weights[0];
    skin += node_matrices[in.node_indices[1]] * in.node_weights[1];
    skin += node_matrices[in.node_indices[2]] * in.node_weights[2];
    skin += node_matrices[in.node_indices[3]] * in.node_weights[3];
    let weight_sum = in.node_weights[0] + in.node_weights[1] + in.node_weights[2] + in.node_weights[3];
    if (weight_sum < 0.0001) {
        skin = mat4x4(vec4(1.,0.,0.,0.), vec4(0.,1.,0.,0.), vec4(0.,0.,1.,0.), vec4(0.,0.,0.,1.));
    }
    let skinned_model = model_u.model * skin;

    // Wind animation — port of `tree_animation_special_local_to_view`
    // (foliage_fx.hlsl:95). Per-vertex displacement applied to local
    // position before world transform.
    let local_offset = foliage_animation_offset(in.tex_coords);
    let local_pos = vec4<f32>(in.position + local_offset, 1.0);
    let world_pos = skinned_model * local_pos;
    let nrm_mat = mat3x3<f32>(skinned_model[0].xyz, skinned_model[1].xyz, skinned_model[2].xyz);
    let obj_normal = oct_decode(in.normal_oct);
    let world_normal = normalize(nrm_mat * obj_normal);

    // foliage_fx.hlsl:256 — SH evaluation AT VERTEX, interpolated to pixel.
    // sh_lighting_coefficients comes from `v_lighting_constant_*` (cb15) in
    // dllcache; we use the default-lighting cbuf via `build_default_sh_array`.
    // foliage_fx.hlsl:476 — `vsout.lighting *= prt_c0;` modulates the SH3
    // result by the per-vertex AO scalar. Clamp the scalar to [0,1] to
    // protect against sign-flipped or out-of-range packings.
    let sh_lighting = ravi_order_3(world_normal, build_default_sh_array())
                    * clamp(in.prt_transfer, 0.0, 1.0);

    // Atmospheric scattering — extinction + inscatter computed per-vertex
    // (foliage_fx.hlsl:258 `compute_scattering`).
    let inv_view = mat3x3<f32>(
        vec3<f32>(camera.view[0].x, camera.view[1].x, camera.view[2].x),
        vec3<f32>(camera.view[0].y, camera.view[1].y, camera.view[2].y),
        vec3<f32>(camera.view[0].z, camera.view[1].z, camera.view[2].z),
    );
    let camera_world = -(inv_view * camera.view[3].xyz);
    var extinction: vec3<f32>;
    var inscatter: vec3<f32>;
    compute_scattering(camera_world, world_pos.xyz, &extinction, &inscatter);

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    out.world_position = world_pos.xyz;
    out.world_normal = world_normal;
    out.sh_lighting = sh_lighting;
    out.extinction = extinction;
    out.inscatter = inscatter;
    return out;
}

// Alpha-test threshold per HLSL `calc_alpha_test_ps` default (0.5).
// `NO_ALPHA_TO_COVERAGE` is the path most shaders take; explicit
// discard below this value.
const FOLIAGE_ALPHA_THRESHOLD: f32 = 0.5;

// Single MRT — `_surface_post_HDR` slot 0 only. Normal G-buffer was
// written by `_entry_point_albedo` (foliage albedo entry) and is
// sampled (when needed) via camera_bgl @ binding 11.

// Engine `accum_pixel` (render_target_fx.hlsl:8-18). See entry_static_sh.wgsl.
struct AccumPixel {
    @location(0) color: vec4<f32>,
    @location(1) dark_color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> AccumPixel {
    // Engine `foliage_fx.hlsl::static_common_ps` (line 265):
    //   calc_alpha_test_ps(vsout.texcoord, output_alpha);  // samples alpha_test_map
    //   float4 albedo = get_albedo(vsout.position);        // Load G-buffer
    //   out_color.w = output_alpha;
    // Engine relies on alpha-to-coverage at draw time. wgpu doesn't
    // expose ATOC cleanly so we hard-discard. Required here because
    // the albedo pass's discard merely skips the G-buffer write —
    // SKY then OVERWRITES the G-buffer + depth at those pixels in its
    // own albedo-pass dispatch, leaving the SL depth-test passing
    // (leaf depth > sky depth in reverse-Z) and the SL Load returning
    // sky color → ghost leaf-card silhouettes lit by leaf SH.
    let alpha = textureSample(alpha_test_map, alpha_test_map_sampler, in.tex_coords).a;
    if (alpha < FOLIAGE_ALPHA_THRESHOLD) {
        discard;
    }
    let fp = vec2<i32>(in.clip_position.xy);
    let base = textureLoad(albedo_texture, fp, 0);

    // Diffuse composition — verbatim port of `foliage_fx.hlsl::static_common_ps:280`:
    //   out_color.rgb = (vsout.lighting * albedo.rgb * extinction
    //                   + inscatter * fog_scale) * g_exposure.r
    // foliage_material_fx.hlsl: specular_radiance = 0 (always).
    // Engine does NOT add simple_lights at PS — `vsout.lighting` is
    // the only lighting input (per-vertex SH3 baked in VS). Adding
    // PS-side simple_lights would double-count any cluster light that
    // gets baked into the VS-side SH.
    let lit = in.sh_lighting * base.rgb;
    let composed = (lit * in.extinction + in.inscatter * FOG_INSCATTER_SCALE) * g_exposure();

    // Engine `convert_to_render_target` clamps RGB ≥ 0 before RT write
    // (`render_target_fx.hlsl:29`).
    let accum = vec4<f32>(max(composed, vec3<f32>(0.0)), 1.0);
    return AccumPixel(accum, accum);
}
