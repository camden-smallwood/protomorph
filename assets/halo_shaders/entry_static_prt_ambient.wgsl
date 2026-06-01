// Halo `_entry_point_static_lighting_prt_ambient` — `static_sh`
// lighting math, but each vertex carries a UNORM scalar transfer
// coefficient (per-vertex ambient visibility) that attenuates the SH
// probe's indirect-light contribution at the fragment.
//
// Engine path (per `project_research_per_mesh_prt_2026_05_11.md`):
//   * Picker (`select_instance_entry_point @ 0x180691340`) routes
//     here when `instance.lightmapping_policy == 2 (single-probe)`
//     AND `mesh.vertex_buffer_indices[3] != 0xFFFF`.
//   * Stream is baked offline by `create_prt_vertex_buffer @
//     0x82E080F0` (Reach) from `per_mesh_prt_data[i].mesh pca data`:
//     3 floats RGB per vertex → 1 grayscale value per vertex. MCC PC
//     declaration is `R32_FLOAT BLENDWEIGHT1` slot 2 (Ares
//     `rasterizer_resource_definitions.cpp:46`); Reach quantizes to 1
//     byte UNORM.
//   * Per-vertex `transfer` is the fraction of incident ambient light
//     reaching this vertex (1.0 = fully exposed, 0.0 = fully occluded).
//     Multiplies the umbrella-computed `diffuse_radiance_initial`
//     before the material model sees it — captures static AO that
//     was baked at lightmap time but stored per-vertex (not per-pixel
//     via the atlas).
//
// Differs from `entry_static_sh.wgsl` ONLY in:
//   (1) VertexInput adds `@location(11) prt_transfer: f32` (slot 1 VB,
//       single f32 per vertex).
//   (2) VertexOutput passes through interpolated `prt_transfer` to PS.
//   (3) PS scales `diffuse_radiance_initial` by `prt_transfer` before
//       passing to `calc_material`. `prt_ravi_diff` keeps the same
//       `(1, 0, 1, dot(N, L))` shape — only the indirect ambient term
//       is occluded.

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
    // PRT Ambient transfer scalar — slot 1 VB, R32_FLOAT.
    @location(11) prt_transfer: f32,
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
    // Engine `entry_points_fx.hlsl:977-980` packs the 4-channel PRT
    // ratio in the VS so each component is per-vertex-interpolated.
    //   .x = prt_ravi_ratio (= prt_mono/ravi_mono) — diffuse AO %
    //   .y = prt_mono                              — engine-marked "unused"
    //   .z = ambient_occlusion * π/0.886227        — specular AO %
    //   .w = min(dot(N, analytical_L), prt_mono)   — kills back-spec
    @location(13) prt_ravi_diff: vec4<f32>,
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

    var extinction: vec3<f32>;
    var inscatter: vec3<f32>;
    compute_scattering(camera_world, world_pos.xyz, &extinction, &inscatter);

    // Engine `entry_points_fx.hlsl:969-980` — PRT ratio computation
    // in VS so the four channels per-vertex interpolate at PS.
    //   ambient_occlusion = prt_c0;                                   // [0,1] AO
    //   lighting_c0       = dot(L0.xyz, vec3(1/3));                   // mono SH DC
    //   ravi_mono         = (0.886227 * lighting_c0) / π;             // monoramp
    //   prt_mono          = ambient_occlusion * lighting_c0;
    //   prt_mono          = max(prt_mono, 0.01);                      // clamp
    //   ravi_mono         = max(ravi_mono, 0.01);                     // clamp
    //   prt_ravi_ratio    = prt_mono / ravi_mono;
    //   dom_dir           = -normalize(L1_R + L1_G + L1_B);           // get_constant_analytical_light_dir_vs
    let ambient_occlusion = clamp(in.prt_transfer, 0.0, 1.0);
    let lighting_c0 = dot(engine_lighting_ps.ravi[0].xyz, vec3<f32>(1.0 / 3.0));
    var prt_mono = max(ambient_occlusion * lighting_c0, 0.01);
    let ravi_mono = max((0.886227 * lighting_c0) / 3.1415926535, 0.01);
    let prt_ravi_ratio = prt_mono / ravi_mono;
    let constant_analytical_light_dir = -normalize(
        engine_lighting_ps.ravi[1].xyz
            + engine_lighting_ps.ravi[2].xyz
            + engine_lighting_ps.ravi[3].xyz,
    );
    let world_normal_out = normalize(nrm_mat * obj_normal);
    let prt_ravi_diff = vec4<f32>(
        prt_ravi_ratio,
        prt_mono,
        (ambient_occlusion * 3.1415926535) / 0.886227,
        min(dot(world_normal_out, constant_analytical_light_dir), prt_mono),
    );

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    out.world_position = world_pos.xyz;
    out.world_tangent = normalize(nrm_mat * obj_tangent);
    out.world_binormal = normalize(nrm_mat * obj_binormal);
    out.world_normal = world_normal_out;
    out.fragment_to_camera_world = fragment_to_camera_world;
    out.extinction = extinction;
    out.inscatter = inscatter;
    out.lightmap_texcoord = in.lightmap_texcoord;
    out.prt_ravi_diff = prt_ravi_diff;
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
    let normal = normalize(in.world_normal);

    let view_dir = normalize(in.fragment_to_camera_world);
    let view_dir_in_tangent_space = vec3<f32>(
        dot(tangent, view_dir),
        dot(binormal, view_dir),
        dot(normal, view_dir),
    );

    // Parallax + alpha test still run in SL — texcoord feeds
    // calc_self_illumination + calc_material (env_map, area_specular).
    var texcoord: vec2<f32>;
    calc_parallax(in.tex_coords, view_dir_in_tangent_space, &texcoord);

    var output_alpha: f32;
    calc_alpha_test(texcoord, &output_alpha);
    if (output_alpha < 0.5) {
        discard;
    }

    // Engine `entry_points_fx.hlsl::get_albedo_and_normal` —
    // opaque variants Load; transparents recompute (no albedo pass).
    let SL_USE_GBUFFER: bool = __SL_USE_GBUFFER__;
    var albedo: vec4<f32>;
    var bump_normal: vec3<f32>;
    var specular_mask: f32;
    if (SL_USE_GBUFFER) {
        let fp = vec2<i32>(in.clip_position.xy);
        let albedo_full = textureLoad(albedo_texture, fp, 0);
        let normal_packed = textureLoad(normal_texture, fp, 0);
        bump_normal = normalize(normal_packed.xyz * 2.0 - 1.0 + 1e-6 * normal);
        specular_mask = albedo_full.w;
        albedo = vec4<f32>(albedo_full.rgb, specular_mask);
    } else {
        let misc = vec4<f32>(0.0);
        var bump_normal_unnorm: vec3<f32>;
        calc_bumpmap(texcoord, in.fragment_to_camera_world, tangent, binormal, normal, &bump_normal_unnorm);
        calc_albedo(texcoord, &albedo, bump_normal_unnorm, misc);
        bump_normal = normalize(bump_normal_unnorm + 1e-6 * normal);
        calc_specular_mask(texcoord, albedo.w, &specular_mask);
    }

    let view_dot_normal = dot(view_dir, bump_normal);
    let view_reflect_dir = (view_dot_normal * bump_normal - view_dir) * 2.0 + view_dir;

    // Same single-probe SH source as `entry_static_sh.wgsl`. Engine
    // PRT-Ambient routes the per-vertex AO through `prt_ravi_diff.x`
    // inside the material model (which multiplies diffuse_radiance by
    // it). Don't pre-scale `diffuse_radiance_initial` here — the
    // material model handles the AO multiply once, in line with how
    // every other entry point passes prt_ravi_diff downstream.
    let sh = build_default_sh_array();
    let diffuse_radiance_initial = ravi_order_3(bump_normal, sh);

    let dominant_light_dir = normalize(dominant_light.direction.xyz);
    let dominant_light_intensity = dominant_light.intensity.xyz;
    let prt_ravi_diff = in.prt_ravi_diff;

    let mat = calc_material(
        view_dir,
        in.fragment_to_camera_world,
        bump_normal,
        view_reflect_dir,
        sh,
        dominant_light_dir,
        dominant_light_intensity,
        albedo.xyz,
        specular_mask,
        texcoord,
        prt_ravi_diff,
        diffuse_radiance_initial,
        in.world_position,
    );

    let envmap_area = max(mat.envmap_area_specular_only, vec3<f32>(0.001));
    let envmap_radiance = calc_environment_map(
        view_dir,
        bump_normal,
        view_reflect_dir,
        mat.envmap_specular_reflectance_and_roughness,
        envmap_area,
    );

    var albedo_for_illum = albedo.xyz;
    let self_illum_radiance = calc_self_illumination(texcoord, &albedo_for_illum, view_dir_in_tangent_space)
        * g_alt_exposure();

    const BLEND_MULTIPLICATIVE_ENABLED: f32 = __BLEND_MULTIPLICATIVE_ENABLED__;
    const BLEND_MULTIPLICATIVE_FACTOR:  f32 = __BLEND_MULTIPLICATIVE_FACTOR__;

    var out_rgb: vec3<f32>;
    if (BLEND_MULTIPLICATIVE_ENABLED > 0.5) {
        out_rgb = (albedo_for_illum + self_illum_radiance) * BLEND_MULTIPLICATIVE_FACTOR;
    } else {
        out_rgb = mat.diffuse_radiance * albedo_for_illum
            + mat.specular_color.xyz
            + self_illum_radiance
            + envmap_radiance;
        out_rgb = (out_rgb * in.extinction + in.inscatter * BLEND_FOG_INSCATTER_SCALE) * g_exposure();
    }
    // Engine applies underwater fog via the separate `render_underwater_fog`
    // fullscreen post-pass (engine `c_water_renderer::render_underwater_fog
    // @ 0x180694080`). Per-fragment `apply_underwater_fog` here doubles
    // the fog when underwater. Match the engine-faithful `entry_static_sh`
    // / `entry_static_per_pixel` paths — leave fog to the fullscreen pass.

    let alpha_out: f32 = __ALPHA_CHANNEL_OUTPUT__;
    out_rgb = out_rgb * __ALPHA_PREMULTIPLY__;

    // Engine `convert_to_render_target` clamps RGB ≥ 0 before RT write
    // (`render_target_fx.hlsl:29`).
    let accum = vec4<f32>(max(out_rgb, vec3<f32>(0.0)), alpha_out);
    return AccumPixel(accum, accum);
}
