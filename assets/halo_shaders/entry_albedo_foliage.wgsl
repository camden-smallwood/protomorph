// Halo `foliage_albedo` entry point — first-pass G-buffer fill for
// foliage geometry (rmfl). Outputs albedo + normal to G-buffer; no
// lighting, no atmosphere, no exposure. Sibling of
// entry_foliage_static_sh.wgsl which produces the FINAL lit color.
//
// VS keeps the same vertex inputs as the SL variant but skips the
// per-vertex SH evaluation + atmospheric scattering — those are
// static_lighting concerns. PS does the same alpha-test + base_map
// sample and writes the G-buffer.
//
// Render-state overrides (set by pipeline_cache for rmfl group_tag):
//   - cull_mode = None (leaf cards 2-sided)
//   - depth_write = true (alpha-test, opaque-pass G-buffer fill)

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
}

fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
}

// Port of `vibration` (foliage_fx.hlsl:50). Repeating triangle wave
// through a sin spring approximation:
//   base = abs(frac(offset + g_tree_animation_coeff) - 0.5) × 2
//   x    = (0.5 - base) × π
//   return sin(x)
// Engine `g_tree_animation_coeff` is a per-frame phase. Protomorph
// stuffs `total_time` into `g_alt_exposure.w` (VS+FS-visible) at
// per-frame upload time — see `upload_view_exposure` in render/mod.rs.
// MiscPS isn't VS-visible (Metal slot conflict with water pipeline).
fn foliage_vibration(offset: f32) -> f32 {
    let coeff = engine_exposure.g_alt_exposure.w * 0.5;
    let vibration_base = abs(fract(offset + coeff) - 0.5) * 2.0;
    let x = (0.5 - vibration_base) * 3.14159265;
    return sin(x);
}

// Port of `animation_offset` (foliage_fx.hlsl:70). Derives a per-vertex
// 3D displacement from leaf texcoord — `distance` scales the magnitude
// (leaf-tip vertices sway more than the stem), `id` decorrelates the
// vertical and horizontal phases per leaf card.
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
    // (foliage_fx.hlsl:95). Engine applies the per-vertex displacement
    // to the LOCAL position before world transform; we mirror that.
    let local_offset = foliage_animation_offset(in.tex_coords);
    let local_pos = vec4<f32>(in.position + local_offset, 1.0);
    let world_pos = skinned_model * local_pos;
    let nrm_mat = mat3x3<f32>(skinned_model[0].xyz, skinned_model[1].xyz, skinned_model[2].xyz);
    let obj_normal = oct_decode(in.normal_oct);

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    out.world_normal = normalize(nrm_mat * obj_normal);
    return out;
}

const FOLIAGE_ALPHA_THRESHOLD: f32 = 0.5;

struct FsOut {
    @location(0) albedo_specmask: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    // Engine `foliage_fx.hlsl::albedo_ps` (line 189):
    //   calc_alpha_test_ps(vsout.texcoord, output_alpha);  // samples alpha_test_map
    //   calc_albedo_ps(vsout.texcoord, albedo, ...);       // rmt2-chosen variant
    //   albedo.w = output_alpha;
    // Engine relies on alpha-to-coverage at draw time (no hard clip
    // here). wgpu doesn't expose ATOC cleanly, so we clip on the
    // alpha_test_map.a value at threshold 0.5 to match the DX11
    // codepath in `alpha_test_fx.hlsl:36-40`.
    // HLSL alpha_test_fx.hlsl:20 — alpha sampled through transform_texcoord.
    let alpha = textureSample(alpha_test_map, alpha_test_map_sampler, transform_texcoord(in.tex_coords, material.alpha_test_map_xform)).a;
    if (alpha < FOLIAGE_ALPHA_THRESHOLD) {
        discard;
    }
    // `calc_albedo` is the rmt2-chosen `calc_albedo_*_ps` variant
    // prepended by render_methods/mod.rs:479 rmfl branch. For default
    // foliage that's `calc_albedo_default_ps @ albedo_fx.hlsl:76` =
    // `base.rgb * (detail.rgb * DETAIL_MULTIPLIER) * albedo_color.rgb`.
    var albedo: vec4<f32>;
    calc_albedo(in.tex_coords, &albedo, in.world_normal, vec4<f32>(0.0));
    var out: FsOut;
    // Foliage has no specular_mask param in foliage_material_fx; A=1.0.
    out.albedo_specmask = vec4<f32>(albedo.rgb, 1.0);
    // RT1.w = output_alpha (engine `albedo.w = output_alpha`) — passed
    // to the SL pass via the G-buffer Load path.
    out.normal = vec4<f32>(in.world_normal * 0.5 + vec3<f32>(0.5), alpha);
    return out;
}
