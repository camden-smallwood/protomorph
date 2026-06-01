// Halo `albedo` entry point — VS + PS faithful ports of
// `albedo_vs` (entry_points_fx.hlsl:71) and `static_default_ps`'s
// G-buffer-fill body, output via `convert_to_albedo_target`
// (albedo_pass_fx.hlsl:23-40).
//
// First geometry pass per `c_player_view::render_albedo`. Writes the
// pre-lighting G-buffer:
//   MRT[0] = `_surface_albedo`        — RGB albedo + spec_mask in A
//   MRT[1] = `_surface_post_HDR`      — encoded normal + albedo.w in A
//
// NO lighting, NO atmosphere, NO exposure, NO simple_lights, NO
// envmap. Those run later in `_entry_point_static_per_pixel` /
// `_static_sh` etc which sample from this G-buffer.

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
    @location(5) fragment_to_camera_world: vec3<f32>,
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

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    out.world_position = world_pos.xyz;
    out.world_tangent = normalize(nrm_mat * obj_tangent);
    out.world_binormal = normalize(nrm_mat * obj_binormal);
    out.world_normal = normalize(nrm_mat * obj_normal);
    out.fragment_to_camera_world = camera_world - world_pos.xyz;
    return out;
}

// `albedo_pixel` from albedo_pass_fx.hlsl:4-11.
struct FsOut {
    @location(0) albedo_specmask: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    let tangent = normalize(in.world_tangent);
    let binormal = normalize(in.world_binormal);
    let normal = normalize(in.world_normal);

    let view_dir = normalize(in.fragment_to_camera_world);
    let view_dir_in_tangent_space = vec3<f32>(
        dot(tangent, view_dir),
        dot(binormal, view_dir),
        dot(normal, view_dir),
    );
    let _unused_view_dir_ts = view_dir_in_tangent_space;

    // calc_parallax_ps — usually a no-op stub for albedo pass; the
    // engine may also call it here to keep the parallax-shifted UVs
    // consistent across albedo + static_lighting walks of the same
    // geometry. Mirrors `albedo_vs/ps` umbrella in entry_points_fx.
    var texcoord: vec2<f32>;
    calc_parallax(in.tex_coords, view_dir_in_tangent_space, &texcoord);

    var output_alpha: f32;
    calc_alpha_test(texcoord, &output_alpha);
    if (output_alpha < 0.5) {
        discard;
    }

    let misc = vec4<f32>(0.0);
    var bump_normal_unnorm: vec3<f32>;
    calc_bumpmap(texcoord, in.fragment_to_camera_world, tangent, binormal, normal, &bump_normal_unnorm);
    var albedo: vec4<f32>;
    calc_albedo(texcoord, &albedo, bump_normal_unnorm, misc);

    let bump_normal = normalize(bump_normal_unnorm + 1e-6 * normal);

    var specular_mask: f32;
    calc_specular_mask(texcoord, albedo.w, &specular_mask);

    // `convert_to_albedo_target_no_srgb` (albedo_pass_fx.hlsl:42).
    // RT0 is Rgba16Float now (matches engine `_surface_accum_HDR`),
    // so spec_mask goes in RT0.w per engine — the SL pass reads it
    // back via `Load(...).w` on the albedo texture without needing
    // a second calc_specular_mask call.
    var fs_out: FsOut;
    fs_out.albedo_specmask = vec4<f32>(albedo.xyz, specular_mask);
    fs_out.normal = vec4<f32>(bump_normal * 0.5 + vec3<f32>(0.5), 0.0);
    return fs_out;
}
