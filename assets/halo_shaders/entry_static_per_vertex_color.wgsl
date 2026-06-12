// Halo `_entry_point_vertex_color_lighting` — faithful port of
// `static_per_vertex_color_vs/_ps` (entry_points_fx.hlsl:799/836).
//
// The engine routes sky `.render_model` mesh parts whose mesh has the
// `_mesh_has_vertex_color_bit` (`mesh->flags & 1`) through this entry
// point (`render_mesh_part_default @ 0x18069EBC0:64-82`). The diffuse
// LIGHTING term is the per-vertex baked `vert_color` (interpolated from
// a SECONDARY VERTEX BUFFER) instead of the SH probe that
// `entry_static_sh.wgsl` evaluates — everything else (albedo sample,
// self-illumination, aerial-perspective fog, exposure, per-blend-mode
// render-target convert) is identical to `static_sh`.
//
// HLSL PS body (the line that matters):
//   out_color.xyz = ((vert_color + simple_light_diffuse_light) * albedo
//                    + self_illum_radiance);
//   out_color.xyz = (out_color.xyz * extinction
//                    + inscatter * BLEND_FOG_INSCATTER_SCALE) * g_exposure.rrr;
// (`#ifdef BLEND_MULTIPLICATIVE`: (vert_color*albedo + self_illum)*FACTOR.)
//
// Diffuse-ONLY: no bump map (uses the interpolated vertex normal), no
// specular, no envmap, no PRT. `simple_light_diffuse_light` is zero for
// sky (the sky pass binds the empty simple-lights slot).
//
// VS attribute: slot 0-6 = the usual `ModelVertex`; slot 1 (location 12)
// = `SkyVertColorVertex` (vec4 RGB + 0 pad). See `geometry/mod.rs`.

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
    // Per-vertex baked sky color — engine `_vertex_buffer_usage_vert_color`,
    // in tag format read from `raw_vertex.vertex color`.
    @location(12) vert_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) world_binormal: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
    @location(5) fragment_to_camera_world: vec3<f32>,
    @location(6) extinction: vec3<f32>,
    @location(7) inscatter: vec3<f32>,
    @location(8) lightmap_texcoord: vec2<f32>,
    @location(9) vert_color: vec3<f32>,
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

    let inv_view = mat3x3<f32>(
        vec3<f32>(camera.view[0].x, camera.view[1].x, camera.view[2].x),
        vec3<f32>(camera.view[0].y, camera.view[1].y, camera.view[2].y),
        vec3<f32>(camera.view[0].z, camera.view[1].z, camera.view[2].z),
    );
    let camera_world = -(inv_view * camera.view[3].xyz);
    let fragment_to_camera_world = camera_world - world_pos.xyz;

    // HLSL: compute_scattering(Camera_Position, vertex.position, ...).
    var extinction: vec3<f32>;
    var inscatter: vec3<f32>;
    compute_scattering(camera_world, world_pos.xyz, &extinction, &inscatter);

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    out.world_position = world_pos.xyz;
    out.world_tangent = normalize(nrm_mat * obj_tangent);
    out.world_binormal = normalize(nrm_mat * obj_binormal);
    out.world_normal = normalize(nrm_mat * obj_normal);
    out.fragment_to_camera_world = fragment_to_camera_world;
    out.extinction = extinction;
    out.inscatter = inscatter;
    out.lightmap_texcoord = in.lightmap_texcoord;
    // HLSL: vsout.vert_color = vert_color;
    out.vert_color = in.vert_color.rgb;
    return out;
}

struct AccumPixel {
    @location(0) color: vec4<f32>,
    @location(1) dark_color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> AccumPixel {
    let tangent = normalize(in.world_tangent);
    let binormal = normalize(in.world_binormal);
    // Diffuse-only: use the interpolated VERTEX normal (no bump map).
    let normal = normalize(in.world_normal);

    let view_dir = normalize(in.fragment_to_camera_world);
    let view_dir_in_tangent_space = vec3<f32>(
        dot(tangent, view_dir),
        dot(binormal, view_dir),
        dot(normal, view_dir),
    );

    var texcoord: vec2<f32>;
    calc_parallax(in.tex_coords, view_dir_in_tangent_space, &texcoord);

    var output_alpha: f32;
    calc_alpha_test(texcoord, &output_alpha);
    if (output_alpha < 0.5) {
        discard;
    }

    // Sky meshes ALWAYS compute albedo inline (the constant_color combine
    // → `albedo_color`, etc.). The opaque G-buffer Load path
    // (`SL_USE_GBUFFER`) is intentionally bypassed here: the sky albedo
    // pass's G-buffer write isn't a reliable source for these materials
    // (e.g. constant_color), and computing inline guarantees the diffuse
    // term `vert_color × albedo` gets the real albedo, not a stale/zero
    // G-buffer texel.
    var albedo: vec4<f32>;
    // misc.xyz carries world-space view_dir (fragment→camera) for albedo
    // variants that need it (chameleon's N·V); .w reserved.
    let misc = vec4<f32>(view_dir, 0.0);
    calc_albedo(texcoord, &albedo, normal, misc);

    // HLSL line 836:
    //   self_illum_radiance = calc_self_illumination_ps(texcoord, albedo.xyz, 0) * ILLUM_SCALE
    // (albedo passed BY VALUE — the diffuse term below uses the original
    // albedo, unlike static_sh's inout `from_albedo` zeroing.)
    var albedo_for_illum = albedo.xyz;
    let self_illum_radiance = calc_self_illumination(texcoord, &albedo_for_illum, view_dir_in_tangent_space)
        * g_alt_exposure();

    // Simple lights are zero for the sky pass (empty simple-lights slot).
    let simple_light_diffuse_light = vec3<f32>(0.0);
    // HLSL diffuse lighting term = per-vertex baked color.
    let diffuse_radiance = in.vert_color + simple_light_diffuse_light;

    const BLEND_MULTIPLICATIVE_ENABLED: f32 = __BLEND_MULTIPLICATIVE_ENABLED__;
    const BLEND_MULTIPLICATIVE_FACTOR:  f32 = __BLEND_MULTIPLICATIVE_FACTOR__;

    var out_rgb: vec3<f32>;
    if (BLEND_MULTIPLICATIVE_ENABLED > 0.5) {
        // HLSL `#ifdef BLEND_MULTIPLICATIVE`: no fog, no exposure.
        out_rgb = (diffuse_radiance * albedo.xyz + self_illum_radiance) * BLEND_MULTIPLICATIVE_FACTOR;
    } else {
        // HLSL default branch.
        out_rgb = diffuse_radiance * albedo.xyz + self_illum_radiance;
        out_rgb = (out_rgb * in.extinction + in.inscatter * BLEND_FOG_INSCATTER_SCALE) * g_exposure();
    }

    let alpha_out: f32 = __ALPHA_CHANNEL_OUTPUT__;
    out_rgb = out_rgb * __ALPHA_PREMULTIPLY__;

    let accum = vec4<f32>(max(out_rgb, vec3<f32>(0.0)), alpha_out);
    return AccumPixel(accum, accum);
}
