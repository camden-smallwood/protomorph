// Halo `terrain / albedo` entry point — first-pass G-buffer fill
// for terrain (rmtr). Sibling of entry_terrain_static_sh.wgsl which
// produces the FINAL lit color.
//
// The 4-layer blend logic (sample blend_map → mask inactive → normalize
// → per-layer albedo + tangent-space bump accumulation → world bump)
// is shared with the SL variant; the differentiator is what we output:
// (albedo, world_normal) here vs (lit_color, normal) in SL.
//
// Substitution tokens (per-variant, by render_methods::assemble):
//   __MATERIAL_N_ACTIVE__       — 1.0 for active layers, 0.0 for "off"
//   __DETAIL_BUMP_ENABLED_N__   — 1.0 when detail_bump applies for layer N

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
    @location(1) world_position: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) world_binormal: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
}

fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
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

    let local_pos = vec4<f32>(in.position, 1.0);
    let world_pos = skinned_model * local_pos;

    let nrm_mat = mat3x3<f32>(skinned_model[0].xyz, skinned_model[1].xyz, skinned_model[2].xyz);
    let obj_normal = oct_decode(in.normal_oct);
    let obj_tangent = normalize(in.tangent_sign.xyz);
    let obj_binormal = cross(obj_normal, obj_tangent) * sign(in.tangent_sign.w);

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    out.world_position = world_pos.xyz;
    out.world_tangent = normalize(nrm_mat * obj_tangent);
    out.world_binormal = normalize(nrm_mat * obj_binormal);
    out.world_normal = normalize(nrm_mat * obj_normal);
    return out;
}

// 4-layer terrain helpers (same as entry_terrain_static_sh.wgsl).
// `transform_texcoord` is provided by the prepended utilities.wgsl.

fn sample_terrain_layer_albedo(
    base_tex: texture_2d<f32>,
    base_samp: sampler,
    detail_tex: texture_2d<f32>,
    detail_samp: sampler,
    base_xform: vec4<f32>,
    detail_xform: vec4<f32>,
    uv: vec2<f32>,
) -> vec4<f32> {
    let base_uv = transform_texcoord(uv, base_xform);
    let detail_uv = transform_texcoord(uv, detail_xform);
    let base = textureSample(base_tex, base_samp, base_uv);
    let detail = textureSample(detail_tex, detail_samp, detail_uv);
    return vec4<f32>(base.rgb * detail.rgb, base.a * detail.a);
}

fn sample_terrain_layer_bump_ts(
    bump_tex: texture_2d<f32>,
    bump_samp: sampler,
    detail_bump_tex: texture_2d<f32>,
    detail_bump_samp: sampler,
    bump_xform: vec4<f32>,
    detail_bump_xform: vec4<f32>,
    uv: vec2<f32>,
    detail_bump_enabled: f32,
) -> vec3<f32> {
    let bump_uv = transform_texcoord(uv, bump_xform);
    let bump_ts = sample_bumpmap(bump_tex, bump_samp, bump_uv);
    let detail_uv = transform_texcoord(uv, detail_bump_xform);
    let detail_ts = sample_bumpmap(detail_bump_tex, detail_bump_samp, detail_uv) * detail_bump_enabled;
    var sum = vec3<f32>(bump_ts.x + detail_ts.x, bump_ts.y + detail_ts.y, 0.0);
    sum.z = sqrt(max(1.0 - sum.x * sum.x - sum.y * sum.y, 0.0));
    return sum;
}

// Engine writes albedo (with spec_mask in RT0.a) and normal to two RTs.
// Now that RT0 is Rgba16Float (was Rg11b10Ufloat — no alpha channel),
// engine-faithful spec_mask routing into RT0.a is possible.
// `terrain_new_fx.hlsl:381-387` accumulates albedo.w as the per-layer
// spec_mask (= sum_N(blend.N * base_N.a * detail_N.a)).
struct FsOut {
    @location(0) albedo_specmask: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    let tangent = normalize(in.world_tangent);
    let binormal = normalize(in.world_binormal);
    let normal = normalize(in.world_normal);

    let blend_uv = transform_texcoord(in.tex_coords, material.blend_map_xform);
    var blend = textureSample(blend_map, blend_map_sampler, blend_uv);
    let layer_active = vec4<f32>(
        __MATERIAL_0_ACTIVE__,
        __MATERIAL_1_ACTIVE__,
        __MATERIAL_2_ACTIVE__,
        __MATERIAL_3_ACTIVE__,
    );
    blend = blend * layer_active;
    let blend_sum = max(blend.x + blend.y + blend.z + blend.w, 0.0001);
    blend = blend / blend_sum;

    var albedo_4 = vec4<f32>(0.0);
    var bump_ts_accum = vec3<f32>(0.0);

    let dbe_0: f32 = __DETAIL_BUMP_ENABLED_0__;
    let dbe_1: f32 = __DETAIL_BUMP_ENABLED_1__;
    let dbe_2: f32 = __DETAIL_BUMP_ENABLED_2__;
    let dbe_3: f32 = __DETAIL_BUMP_ENABLED_3__;

    if (blend.x > 0.0) {
        let layer = sample_terrain_layer_albedo(
            base_map_m_0, base_map_m_0_sampler, detail_map_m_0, detail_map_m_0_sampler,
            material.base_map_m_0_xform, material.detail_map_m_0_xform,
            in.tex_coords,
        );
        albedo_4 += layer * blend.x;
        let bump_ts = sample_terrain_layer_bump_ts(
            bump_map_m_0, bump_map_m_0_sampler, detail_bump_m_0, detail_bump_m_0_sampler,
            material.bump_map_m_0_xform, material.detail_bump_m_0_xform,
            in.tex_coords, dbe_0,
        );
        bump_ts_accum += bump_ts * blend.x;
    }
    if (blend.y > 0.0) {
        let layer = sample_terrain_layer_albedo(
            base_map_m_1, base_map_m_1_sampler, detail_map_m_1, detail_map_m_1_sampler,
            material.base_map_m_1_xform, material.detail_map_m_1_xform,
            in.tex_coords,
        );
        albedo_4 += layer * blend.y;
        let bump_ts = sample_terrain_layer_bump_ts(
            bump_map_m_1, bump_map_m_1_sampler, detail_bump_m_1, detail_bump_m_1_sampler,
            material.bump_map_m_1_xform, material.detail_bump_m_1_xform,
            in.tex_coords, dbe_1,
        );
        bump_ts_accum += bump_ts * blend.y;
    }
    if (blend.z > 0.0) {
        let layer = sample_terrain_layer_albedo(
            base_map_m_2, base_map_m_2_sampler, detail_map_m_2, detail_map_m_2_sampler,
            material.base_map_m_2_xform, material.detail_map_m_2_xform,
            in.tex_coords,
        );
        albedo_4 += layer * blend.z;
        let bump_ts = sample_terrain_layer_bump_ts(
            bump_map_m_2, bump_map_m_2_sampler, detail_bump_m_2, detail_bump_m_2_sampler,
            material.bump_map_m_2_xform, material.detail_bump_m_2_xform,
            in.tex_coords, dbe_2,
        );
        bump_ts_accum += bump_ts * blend.z;
    }
    if (blend.w > 0.0) {
        let layer = sample_terrain_layer_albedo(
            base_map_m_3, base_map_m_3_sampler, detail_map_m_3, detail_map_m_3_sampler,
            material.base_map_m_3_xform, material.detail_map_m_3_xform,
            in.tex_coords,
        );
        albedo_4 += layer * blend.w;
        let bump_ts = sample_terrain_layer_bump_ts(
            bump_map_m_3, bump_map_m_3_sampler, detail_bump_m_3, detail_bump_m_3_sampler,
            material.bump_map_m_3_xform, material.detail_bump_m_3_xform,
            in.tex_coords, dbe_3,
        );
        bump_ts_accum += bump_ts * blend.w;
    }

    // Engine `terrain_new_fx.hlsl:91` declares
    //   PARAM(float, global_albedo_tint);
    // — a SCALAR. Engine line 387: `albedo.xyz *= global_albedo_tint *
    // DETAIL_MULTIPLIER` — scalar broadcast on all 3 channels.
    //
    // We read `.x` (and broadcast via scalar multiply). The cbuffer
    // slot is filled as `[scalar, 0, 0, 0]` per blam-tags' scalar
    // packing. Reading `.xyz` would zero G+B — wrong; `.x` is right.
    let albedo_rgb = albedo_4.rgb * material.global_albedo_tint.x * 4.59479;
    let albedo_w = albedo_4.a;

    let bump_ts_norm = normalize(bump_ts_accum + vec3<f32>(0.0, 0.0, 1e-6));
    let bump_normal = normalize(
        tangent * bump_ts_norm.x + binormal * bump_ts_norm.y + normal * bump_ts_norm.z
            + 1e-6 * normal,
    );

    var fs_out: FsOut;
    fs_out.albedo_specmask = vec4<f32>(albedo_rgb, albedo_w);
    // Normal RT alpha unused — engine `convert_to_albedo_target` writes
    // clip-distance here, which protomorph doesn't consume downstream.
    fs_out.normal = vec4<f32>(bump_normal * 0.5 + vec3<f32>(0.5), 0.0);
    return fs_out;
}
