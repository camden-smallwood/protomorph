// Halo `static_per_pixel` entry point — VS + PS faithful ports of
// `static_per_pixel_vs` (entry_points_fx.hlsl:437) and
// `static_per_pixel_ps` (472). Used by BSP cluster + structure-instanced
// rmsh surfaces — anything with a per-pixel lightmap atlas chart.
//
// Differs from `static_sh` in the diffuse pipeline:
//   - SH source: lightprobe atlas via `sample_lightprobe_texture` (order-2,
//     4 sh_coefs) — NOT the default-lighting cbuffer's 10-vec4 ravi.
//   - Diffuse: `ravi_order_2_with_dominant_light` (subtracts dominant
//     from L0+L1, evaluates order-2, re-adds dominant as analytical
//     lambertian). NOT ravi_order_3 — that ringing on zero-padded L2
//     coefs is what produced the black-wedge artifacts.
//   - Dominant light: extracted from the SH L1 coefs via the atlas's
//     intensity slice, NOT dominant_light.*.
//   - prt_ravi_diff: (1, 1, 1, dot(N, dom_dir)) — second component
//     differs from static_sh's (1, 0, 1, ...).
// All material/envmap/self_illum/scattering/exposure plumbing matches
// static_sh — only the SH source and ravi evaluator change.

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

// Faithful port of `calc_output_color_with_explicit_light_linear_with_dominant_light`
// (entry_points_fx.hlsl:275). Inlined into fs_main matching HLSL line by line.
//
// Single MRT — `_surface_post_HDR` slot 0 only. The normal G-buffer
// at slot 1 was written by `_entry_point_albedo` already; SL pass
// samples it via camera_bgl @ binding 11 instead of re-emitting it.

// Engine `accum_pixel` (render_target_fx.hlsl:8-18). See entry_static_sh.wgsl.
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
    // calc_self_illumination + calc_material (env_map, area_specular
    // lookup); alpha test gates the discard for transparent rmsh
    // variants. Engine keeps both in `static_per_pixel_ps` even when
    // get_albedo_and_normal takes the Load() branch.
    var texcoord: vec2<f32>;
    calc_parallax(in.tex_coords, view_dir_in_tangent_space, &texcoord);

    var output_alpha: f32;
    calc_alpha_test(texcoord, &output_alpha);
    if (output_alpha < 0.5) {
        discard;
    }

    // Engine `entry_points_fx.hlsl::get_albedo_and_normal` —
    //   #ifdef maybe_calc_albedo
    //     if (actually_calc_albedo) recompute  // transparents
    //     else                      Load       // opaques
    // SL_USE_GBUFFER: opaque variants read the runtime `actually_calc_albedo`
    // cbuffer bit (`Kernel5PS.entries[5].z == 0u`); transparent variants
    // pin `false` at WGSL-assembly time (they have no albedo pass writing
    // the G-buffer). `let` (vs `const`) because the opaque substitution
    // is a runtime expression.
    let SL_USE_GBUFFER: bool = __SL_USE_GBUFFER__;
    var albedo: vec4<f32>;
    var bump_normal: vec3<f32>;
    var specular_mask: f32;
    if (SL_USE_GBUFFER) {
        // Opaque branch — Load G-buffer. RT0 is Rg11b10Ufloat (no alpha)
        // so spec_mask is stashed in RT1.w by entry_albedo.wgsl.
        let fp = vec2<i32>(in.clip_position.xy);
        let albedo_full = textureLoad(albedo_texture, fp, 0);
        let normal_packed = textureLoad(normal_texture, fp, 0);
        bump_normal = normalize(normal_packed.xyz * 2.0 - 1.0 + 1e-6 * normal);
        specular_mask = albedo_full.w;
        albedo = vec4<f32>(albedo_full.rgb, specular_mask);
    } else {
        // Transparent branch — recompute from material textures.
        let misc = vec4<f32>(0.0);
        var bump_normal_unnorm: vec3<f32>;
        calc_bumpmap(texcoord, in.fragment_to_camera_world, tangent, binormal, normal, &bump_normal_unnorm);
        calc_albedo(texcoord, &albedo, bump_normal_unnorm, misc);
        bump_normal = normalize(bump_normal_unnorm + 1e-6 * normal);
        calc_specular_mask(texcoord, albedo.w, &specular_mask);
    }

    let view_dot_normal = dot(view_dir, bump_normal);
    let view_reflect_dir = (view_dot_normal * bump_normal - view_dir) * 2.0 + view_dir;

    // HLSL static_per_pixel_ps lines 496-505:
    //   sample_lightprobe_texture(lightmap_texcoord, sh_coef[4], dom_dir, dom_int);
    //   prt_ravi_diff = (1, 1, 1, dot(N, dom_dir));
    //   pack_constants_texture_array_linear(sh_coef, sh_lighting_coefficients);
    var probe = sample_lightprobe_texture(in.lightmap_texcoord);

    // Atlas-empty fallback — WORKAROUND, not engine-faithful.
    //
    // Riverworld's `vahalla_waterfall.shader` (and ~16% of riverworld
    // atlas texels generally) lives in a region where the lightmap
    // baker authored BC3 alpha=0. `decode_bpp16_luvw` returns
    // `(uvw × 2 - 2) × (s0.a × s1.a × range)` = 0 because alpha=0 →
    // SH coefficients are zero → diffuse radiance is zero → output is
    // `out_rgb = 0 × albedo = 0` (the "white looks black" symptom).
    //
    // **Engine path trace (dllcache 13372, 2026-05-15)**: engine routes
    // transparent BSP clusters through `_entry_point_static_lighting_per_pixel`
    // hardcoded (see `c_structure_renderer::render_transparent_cluster_mesh_part
    // @ 0x180690fd0`). `render_cluster_mesh_part` only swaps to the SH
    // entry for `rmfl` foliage, missing-lightmap BSPs, or debug-force-default
    // — rmsh waterfall hits NONE. The cluster path has NO equivalent of
    // `select_instance_entry_point`'s single_probe fallback. Engine HLSL
    // `sample_lightprobe_texture` is byte-for-byte identical to ours.
    // With the same atlas data, engine should produce the same zero output.
    //
    // The fact that engine renders the waterfall visible strongly suggests
    // MCC's binary cache atlas data differs from what we extract from the
    // loose `.bitmap` tag — tool.exe's cache compile likely repairs the
    // zero-alpha regions. See `project_waterfall_atlas_empty_fallback_2026_05_15`
    // for the full trace, remaining hypotheses, and the queued investigation
    // (extract MCC binary cache atlas + decompile tool.exe lightmap baking).
    //
    // Until that resolves, this fallback substitutes the engine's
    // per-cluster ravi cbuffer (`engine_lighting_ps.ravi[]`, analytical
    // frame-default sky probe written by
    // `calculate_and_set_ravi_constants_internal`) when atlas SH DC is
    // below threshold. Produces visibly-correct output that approximately
    // matches engine.
    //
    // Previous history: added `f182018` (2026-05-07), removed
    // `d42d077` 8.2 cleanup (2026-05-10) on the grounds it wasn't
    // engine-faithful. Restored 2026-05-15 after dllcache trace
    // confirmed the bug isn't in the lighting code path but in the
    // atlas DATA, and that the engine-faithful fix lives in tool.exe.
    // Engine-faithful default: NO empty-atlas fallback. The engine's
    // lightmap_sampling_fx.hlsl + entry_points_fx.hlsl per-pixel path use
    // the decoded atlas probe DIRECTLY — there is no DC threshold anywhere
    // in the engine HLSL. The fallback below substituted the cluster
    // default sky-probe + sun when baked DC < 0.01, which cannot tell an
    // empty/unbaked atlas region from a legitimately-black BAKED shadow, so
    // it lit every deep shadow (ghosttown's upper walls/ceilings). OFF by
    // default; construct's empty loose-tag atlas is a separate DATA gap to
    // fix in the loader, not by overriding real baked shadows here.
    const LIGHTMAP_EMPTY_ATLAS_FALLBACK: bool = false;
    let _atlas_dc = probe.sh[0];
    if (LIGHTMAP_EMPTY_ATLAS_FALLBACK && (_atlas_dc.r + _atlas_dc.g + _atlas_dc.b) < 0.01) {
        // `engine_lighting_ps.ravi` layout (per `build_default_sh_array`
        // in spherical_harmonics_fx.wgsl, mirroring engine
        // `calculate_and_set_ravi_constants_internal @ 0x1806aa650`):
        //   ravi[0]        = DC (RGB in xyz)
        //   ravi[1..3].xyz = (sh[3], sh[1], -sh[2]) for R, G, B channels
        // Invert that packing back into LightprobeSample's sh[0..3] (per
        // `pack_constants_texture_array_linear` source layout: sh[1] = L1
        // y-axis, sh[2] = L1 z-axis, sh[3] = L1 x-axis).
        let r0 = engine_lighting_ps.ravi[0].xyz;
        let r1 = engine_lighting_ps.ravi[1];
        let r2 = engine_lighting_ps.ravi[2];
        let r3 = engine_lighting_ps.ravi[3];
        probe.sh[0] = r0;
        probe.sh[1] = vec3<f32>(r1.y, r2.y, r3.y);
        probe.sh[2] = vec3<f32>(-r1.z, -r2.z, -r3.z);
        probe.sh[3] = vec3<f32>(r1.x, r2.x, r3.x);
        probe.dir = normalize(dominant_light.direction.xyz);
        probe.intensity = dominant_light.intensity.xyz;
    }

    let dominant_light_dir = probe.dir;
    let dominant_light_intensity = probe.intensity;
    let sh_linear = pack_constants_texture_array_linear(probe);

    // HLSL umbrella line 348:
    //   diffuse_radiance = ravi_order_2_with_dominant_light(N, sh_lighting_coefficients,
    //                                                       light_direction, light_intensity);
    let diffuse_radiance_initial = ravi_order_2_with_dominant_light(
        bump_normal, sh_linear, dominant_light_dir, dominant_light_intensity);

    // HLSL umbrella lines 350-361: zero-pad to 10-vec4 for the area-specular
    // path inside CALC_MATERIAL — L2 stays zero.
    let zero = vec4<f32>(0.0);
    let sh = array<vec4<f32>, 10>(
        sh_linear[0], sh_linear[1], sh_linear[2], sh_linear[3],
        zero, zero, zero, zero, zero, zero,
    );

    let prt_ravi_diff = vec4<f32>(1.0, 1.0, 1.0, dot(normal, dominant_light_dir));

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
    // HLSL umbrella line 390:
    //   self_illum_radiance = calc_self_illumination_ps(...) * ILLUM_SCALE;
    // ILLUM_SCALE = g_alt_exposure.r; placeholder helper returns 1.0
    // until alt exposure plumbing lands.
    let self_illum_radiance = calc_self_illumination(texcoord, &albedo_for_illum, view_dir_in_tangent_space)
        * g_alt_exposure();

    // Per-variant blend_fx outputs — see entry_static_sh.wgsl for the
    // table, all substituted by render_methods/mod.rs at WGSL assembly.
    const BLEND_MULTIPLICATIVE_ENABLED: f32 = __BLEND_MULTIPLICATIVE_ENABLED__;
    const BLEND_MULTIPLICATIVE_FACTOR:  f32 = __BLEND_MULTIPLICATIVE_FACTOR__;
    const BLEND_FRESNEL_ENABLED: f32 = __BLEND_FRESNEL_ENABLED__;

    // Simple lights are now evaluated INSIDE the material model
    // (engine pattern — single_lobe_phong_fx.hlsl:111, cook_torrance_fx.hlsl:1084,
    // etc.) so the dynamic contribution is folded into `mat.diffuse_radiance`
    // and `mat.specular_color` with the material's authored spec_power.
    // No entry-shader-level simple_lights term to add.

    // Cascade shadow gating is NOT applied here. `mat.specular_color`
    // is computed from the dominant light extracted from the SH probe
    // (`dominant_light.direction/intensity`); the
    // probe is per-cluster lightmap-baked, so its directional energy
    // ALREADY reflects the cluster's sun occlusion. Multiplying again
    // by a cascade depth-test darkens correctly-lit metal in shadowed
    // clusters (the probe is dim there, specular is small, cascade
    // PCF nudges it toward zero — output reads as flat). Engine gates
    // cascade shadows on DYNAMIC objects only, where the lightmap
    // probe doesn't apply. We keep the cascade pipeline intact for
    // when dynamic-object rendering ships; static geometry is left
    // alone here to match dllcache.
    var out_rgb: vec3<f32>;
    var alpha_out: f32 = __ALPHA_CHANNEL_OUTPUT__;
    if (BLEND_MULTIPLICATIVE_ENABLED > 0.5) {
        out_rgb = (albedo_for_illum + self_illum_radiance) * BLEND_MULTIPLICATIVE_FACTOR;
    } else if (BLEND_FRESNEL_ENABLED > 0.5) {
        // glass BLEND_FRESNEL (entry_points_fx.hlsl:258) — diffuse
        // PREMULTIPLIED by albedo.w; reflections add on top; output alpha
        // = saturate(fresnel + albedo.w). See entry_static_sh.wgsl.
        out_rgb = mat.diffuse_radiance * albedo_for_illum * albedo.w
            + self_illum_radiance
            + envmap_radiance
            + mat.specular_color.xyz;
        out_rgb = (out_rgb * in.extinction + in.inscatter * BLEND_FOG_INSCATTER_SCALE) * g_exposure();
        alpha_out = saturate(mat.specular_color.w + albedo.w);
    } else {
        // HLSL umbrella composition (entry_points_fx.hlsl):
        //   final = diffuse_radiance × albedo + specular_color + self_illum + envmap
        // Material model already folded simple_lights into both
        // `mat.diffuse_radiance` and `mat.specular_color`.
        out_rgb = mat.diffuse_radiance * albedo_for_illum
            + mat.specular_color.xyz
            + self_illum_radiance
            + envmap_radiance;
        out_rgb = (out_rgb * in.extinction + in.inscatter * BLEND_FOG_INSCATTER_SCALE) * g_exposure();
    }
    // Underwater fog is applied by the separate `render_underwater_fog`
    // fullscreen post-pass (engine `c_water_renderer::render_underwater_fog
    // @ 0x180694080`). Don't apply here — would double-fog when underwater.

    out_rgb = out_rgb * __ALPHA_PREMULTIPLY__;

    // Engine `convert_to_render_target` clamps RGB ≥ 0 before RT write
    // (`render_target_fx.hlsl:29`). Negatives in RT1 propagate into the
    // bloom pyramid as black-hole pixels in composite.
    let accum = vec4<f32>(max(out_rgb, vec3<f32>(0.0)), alpha_out);
    return AccumPixel(accum, accum);
}
