// Halo `static_sh` entry point — VS + PS faithful ports of
// `static_sh_vs` (entry_points_fx.hlsl:544) and `static_sh_ps` (582).
// PS body mirrors `calc_output_color_with_explicit_light_quadratic`
// (entry_points_fx.hlsl:161) line for line.
//
// Pipeline:
//   VS:  build tangent_frame, fragment_to_camera_world (= cam - vert),
//        per-vertex extinction + inscatter via compute_scattering.
//   PS:  build sh_lighting_coefficients[10] from default_lighting,
//        prt_ravi_diff = (1, 0, 1, dot(N, dominant_light_dir)),
//        call umbrella → composed (lit*ext + inscat*scale)*g_exposure.
//
// Stubs (to be lit up later):
//   - calc_parallax_ps: identity (no parallax map plumbed).
//   - calc_alpha_test_ps: no discard (G-buffer fill handles alpha).
//   - normal_lengthsq blended-normal-attenuate: no MSAA G-buffer yet.
//   - APPLY_OVERLAYS: empty by default.
//   - misc: zero (misc_attr_animation only used by select shaders).

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
    @location(6) extinction: vec3<f32>,
    @location(7) inscatter: vec3<f32>,
    @location(8) lightmap_texcoord: vec2<f32>,
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

    // Camera world from inverse view (rotation transpose · -translation).
    let inv_view = mat3x3<f32>(
        vec3<f32>(camera.view[0].x, camera.view[1].x, camera.view[2].x),
        vec3<f32>(camera.view[0].y, camera.view[1].y, camera.view[2].y),
        vec3<f32>(camera.view[0].z, camera.view[1].z, camera.view[2].z),
    );
    let camera_world = -(inv_view * camera.view[3].xyz);

    // HLSL: vsout.fragment_to_camera_world.rgb = Camera_Position - vertex.position
    // (un-normalized world-space direction to camera).
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
    return out;
}

// Faithful port of `calc_output_color_with_explicit_light_quadratic`
// (entry_points_fx.hlsl:161). Inlined into fs_main since WGSL doesn't
// take struct-of-array params naturally; the body matches the HLSL
// statement-by-statement.
//
// Single MRT — `_surface_post_HDR` slot 0 only. Normal G-buffer is
// already populated by `_entry_point_albedo`; SL samples via binding 11.

// Engine `accum_pixel` (render_target_fx.hlsl:8-18). RT0 = LDR
// (engine `_surface_screenshot_composite_depth`); RT1 = DARK
// (`_surface_screenshot_composite_cubemap`, bloom-feed source).
// On MCC PC `DARK_COLOR_MULTIPLIER = g_exposure.g = 1.0` so RT0 == RT1.
struct AccumPixel {
    @location(0) color: vec4<f32>,
    @location(1) dark_color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> AccumPixel {
    // Normalize interpolated tangent frame (HLSL `static_sh_ps` does
    // this when ALPHA_OPTIMIZATION isn't defined — same here).
    let tangent = normalize(in.world_tangent);
    let binormal = normalize(in.world_binormal);
    let normal = normalize(in.world_normal);

    // HLSL umbrella:
    //   float3 view_dir = normalize(fragment_to_camera_world);
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
    // Opaque variants read the runtime `actually_calc_albedo` cbuffer
    // bit; transparent variants pin `false` at WGSL-assembly time.
    // `let` (vs `const`) because the opaque substitution is a runtime
    // expression (cbuffer fetch).
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

    // HLSL line 207-211:
    //   float view_dot_normal = dot(view_dir, bump_normal);
    //   float3 view_reflect_dir = (view_dot_normal * bump_normal - view_dir) * 2 + view_dir;
    let view_dot_normal = dot(view_dir, bump_normal);
    let view_reflect_dir = (view_dot_normal * bump_normal - view_dir) * 2.0 + view_dir;

    // HLSL line 216:
    //   float3 diffuse_radiance = ravi_order_3(bump_normal, sh_lighting_coefficients);
    // sh_lighting_coefficients[10] is built from p_lighting_constant_0..9
    // in the HLSL static_sh_ps body (line 600-613). We use the
    // build_default_sh_array() adapter — populated from default_lighting
    // cbuffer (slot 0x24+0x30 in dllcache).
    //
    // NOTE: this `static_sh` entry uses the single-probe cbuffer. The
    // per-pixel atlas path is a SEPARATE entry point — `static_per_pixel`
    // — and uses ORDER-2 SH via pack_constants_texture_array_linear, NOT
    // ravi_order_3 with zero-padded L2 coefs (which produces black-wedge
    // ringing). Don't fold the atlas into this entry.
    let sh = build_default_sh_array();
    let diffuse_radiance_initial = ravi_order_3(bump_normal, sh);

    // HLSL line 615:
    //   prt_ravi_diff = (1, 0, 1, dot(N, k_ps_dominant_light_direction))
    let dominant_light_dir = normalize(dominant_light.direction.xyz);
    let dominant_light_intensity = dominant_light.intensity.xyz;
    let prt_ravi_diff = vec4<f32>(1.0, 0.0, 1.0, dot(normal, dominant_light_dir));

    // HLSL line 220-241: CALC_MATERIAL(material_type)(...)
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

    // HLSL line 244-245:
    //   envmap_area_specular_only = max(envmap_area_specular_only, 0.001);
    //   envmap_radiance = CALC_ENVMAP(envmap_type)(view_dir, bump_normal,
    //                       view_reflect_dir, env_spec_refl_and_rough,
    //                       envmap_area_specular_only);
    let envmap_area = max(mat.envmap_area_specular_only, vec3<f32>(0.001));
    let envmap_radiance = calc_environment_map(
        view_dir,
        bump_normal,
        view_reflect_dir,
        mat.envmap_specular_reflectance_and_roughness,
        envmap_area,
    );

    // HLSL line 248: self_illum_radiance = calc_self_illumination_ps(...) * ILLUM_SCALE;
    // HLSL `inout albedo` — `from_albedo` variant zeroes albedo so the
    // `diffuse * albedo` term drops out (pure emissive).
    // ILLUM_SCALE = g_alt_exposure.r; placeholder helper returns 1.0
    // until alt exposure plumbing lands.
    var albedo_for_illum = albedo.xyz;
    let self_illum_radiance = calc_self_illumination(texcoord, &albedo_for_illum, view_dir_in_tangent_space)
        * g_alt_exposure();

    // HLSL has TWO color-output paths in `calc_output_color_with_explicit_light_quadratic`:
    //   * default          — `out_color = (lit_color * extinction + inscatter * SCALE) * exposure`
    //   * BLEND_MULTIPLICATIVE — `out_color = (albedo + self_illum) * BLEND_MULTIPLICATIVE`
    // (entry_points_fx.hlsl, the `#ifdef BLEND_MULTIPLICATIVE` block.)
    //
    // BLEND_MULTIPLICATIVE_ENABLED + BLEND_MULTIPLICATIVE_FACTOR + ALPHA_CHANNEL_OUTPUT +
    // ALPHA_PREMULTIPLY are substituted per-variant by `render_methods/mod.rs` at WGSL
    // assembly. See `blend_fx.hlsl` for the full table. The fixed-const branch optimizes
    // out cleanly — equivalent to HLSL preprocessor's `#ifdef`.
    const BLEND_MULTIPLICATIVE_ENABLED: f32 = __BLEND_MULTIPLICATIVE_ENABLED__;
    const BLEND_MULTIPLICATIVE_FACTOR:  f32 = __BLEND_MULTIPLICATIVE_FACTOR__;

    // Simple lights are evaluated INSIDE the material model (engine
    // pattern). `mat.diffuse_radiance` and `mat.specular_color` already
    // include the per-light contribution with the authored spec_power.

    // Cascade shadow gating reserved for dynamic objects (see
    // entry_static_per_pixel.wgsl for the rationale).
    var out_rgb: vec3<f32>;
    if (BLEND_MULTIPLICATIVE_ENABLED > 0.5) {
        // HLSL `#ifdef BLEND_MULTIPLICATIVE` — no lighting, no fog, no exposure.
        // APPLY_OVERLAYS not yet ported (no decal-overlay materials shipped).
        out_rgb = (albedo_for_illum + self_illum_radiance) * BLEND_MULTIPLICATIVE_FACTOR;
    } else {
        // HLSL default branch.
        out_rgb = mat.diffuse_radiance * albedo_for_illum
            + mat.specular_color.xyz
            + self_illum_radiance
            + envmap_radiance;
        out_rgb = (out_rgb * in.extinction + in.inscatter * BLEND_FOG_INSCATTER_SCALE) * g_exposure();
    }
    // Underwater fog is applied by the separate `render_underwater_fog`
    // fullscreen post-pass (engine `c_water_renderer::render_underwater_fog
    // @ 0x180694080`). Don't apply here — would double-fog when underwater.

    // `convert_to_render_target_premultiplied_alpha` premultiplies rgb by alpha
    // before render-target convert (render_target_fx.hlsl:1-4). For other modes
    // ALPHA_PREMULTIPLY substitutes to 1.0 (no-op).
    let alpha_out: f32 = __ALPHA_CHANNEL_OUTPUT__;
    out_rgb = out_rgb * __ALPHA_PREMULTIPLY__;

    let accum = vec4<f32>(max(out_rgb, vec3<f32>(0.0)), alpha_out);
    return AccumPixel(accum, accum);
}
