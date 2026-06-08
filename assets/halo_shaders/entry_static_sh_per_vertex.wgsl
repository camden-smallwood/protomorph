// Halo `static_sh` PS body but the SH coefficients come from a
// SECONDARY VERTEX BUFFER instead of the cbuffer-bound averaged
// probe. Engine-faithful per-vertex SH path: the rasterizer
// interpolates the per-vertex `(DC, X, Y, -Z)` packing across the
// primitive, and the PS evaluates `ravi_order_2` at the fragment
// normal — gives spatially-varying ambient instead of the per-instance
// averaging bridge `static_sh.wgsl` uses.
//
// VS attribute slots 7-10 carry the SH stream
// (`PerVertexShVertex` — see `geometry/mod.rs`). Slot 0-6 are the
// usual `ModelVertex` attribs.
//
// PS body otherwise mirrors `entry_static_sh.wgsl` line for line. Only
// the `diffuse_radiance_initial` source line changes: SH is sampled
// from the interpolated vertex stream, not from `build_default_sh_array`.

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
    // Per-vertex SH stream — `(DC, X, Y, -Z)` per channel.
    @location(7) sh_r: vec4<f32>,
    @location(8) sh_g: vec4<f32>,
    @location(9) sh_b: vec4<f32>,
    @location(10) dominant_intensity: vec4<f32>,
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
    // Per-vertex SH — interpolated to fragment.
    @location(9) sh_r: vec4<f32>,
    @location(10) sh_g: vec4<f32>,
    @location(11) sh_b: vec4<f32>,
    @location(12) dominant_intensity: vec4<f32>,
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
    out.sh_r = in.sh_r;
    out.sh_g = in.sh_g;
    out.sh_b = in.sh_b;
    out.dominant_intensity = in.dominant_intensity;
    return out;
}

// Evaluate `ravi_order_2` per channel at `n` using the interpolated
// per-vertex `(DC, X, Y, -Z)` packing. Same coefficients as
// `bake.rs::ravi_eval_order2` so the bake math and the per-fragment
// runtime path agree.
fn pv_ravi_order2(n: vec3<f32>, sh: vec4<f32>) -> f32 {
    let dc = sh.x;
    let l1 = sh.yzw;
    return (0.886227 * dc + (-1.023328) * dot(n, l1)) * 0.31830989;
}

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

    // Per-vertex SH eval at the fragment's bump normal. The 4 vec4
    // SH array expected by `calc_material` mirrors how the cbuffer
    // path packs ravi constants:
    //   sh[0] = (R_DC, G_DC, B_DC, _)
    //   sh[1..3] = (X, Y, -Z) in each channel slot, padded order-2
    //              to order-3 with zero L2 bands.
    let diffuse_r = pv_ravi_order2(bump_normal, in.sh_r);
    let diffuse_g = pv_ravi_order2(bump_normal, in.sh_g);
    let diffuse_b = pv_ravi_order2(bump_normal, in.sh_b);
    let diffuse_radiance_initial = vec3<f32>(diffuse_r, diffuse_g, diffuse_b);

    // Build the 10-vec4 SH array `calc_material` expects. Order-2
    // SH (DC + 3 L1) fills sh[0..3]; the remaining 6 slots (L2 bands +
    // padding) stay zero — same shape `static_sh.wgsl` produces from
    // `build_default_sh_array` for cbuffer-bound averaged probes.
    // Channel-major packing matching engine `pack_constants_texture_array`
    // (spherical_harmonics_fx.hlsl:38-53):
    //   [0] = (DC.r, DC.g, DC.b, 0)               ← DC packed per-channel
    //   [1] = (X.r, Y.r, -Z.r, 0)                 ← RED channel's L1
    //   [2] = (X.g, Y.g, -Z.g, 0)                 ← GREEN channel's L1
    //   [3] = (X.b, Y.b, -Z.b, 0)                 ← BLUE channel's L1
    //
    // `area_specular` reads `dot(refl, sh[1..3].xyz)` expecting each slot
    // to carry ONE channel's full L1 (so x1.r/g/b come out per-channel).
    // The previous (buggy) packing was AXIS-MAJOR — `sh[1] = (X.r, X.g, X.b)`,
    // so `dot(refl, sh[1].xyz)` mixed all three channels' X-axis components
    // into the RED slot, producing per-cluster cross-channel garbage that
    // appeared as a rainbow on metals.
    //
    // `PerVertexShVertex::from_probe` (geometry/mod.rs:118) packs each
    // channel as `(DC, X, Y, -Z)`, so the per-channel L1 vec3 is just
    // `sh_*.yzw`.
    let zero4 = vec4<f32>(0.0);
    let sh = array<vec4<f32>, 10>(
        vec4<f32>(in.sh_r.x, in.sh_g.x, in.sh_b.x, 0.0),                // DC.rgb
        vec4<f32>(in.sh_r.y, in.sh_r.z, in.sh_r.w, 0.0),                // RED L1   (X, Y, -Z)
        vec4<f32>(in.sh_g.y, in.sh_g.z, in.sh_g.w, 0.0),                // GREEN L1
        vec4<f32>(in.sh_b.y, in.sh_b.z, in.sh_b.w, 0.0),                // BLUE L1
        zero4, zero4, zero4, zero4, zero4, zero4,
    );

    // Dominant light — luma-weighted L1 collapse. Engine
    // `entry_points_fx.hlsl:753-755` uses the SAME math with Rec.709
    // luma weights `(0.212656, 0.715158, 0.0721856)`. Our per-vertex
    // packing is `(DC, X, Y, -Z)` per channel (see VertexInput comment
    // line 25), so direct per-axis extraction matches engine semantics
    // — the swizzle hack `.wyz` in engine only works because their
    // `probe0_3_*` packs L1 in a different order. (Earlier attempt
    // verbatim-copied the engine swizzle and produced a wrong-axis
    // direction; reverted to the per-axis extraction with Rec.709 luma.)
    let l1_x = vec3<f32>(in.sh_r.y, in.sh_g.y, in.sh_b.y);
    let l1_y = vec3<f32>(in.sh_r.z, in.sh_g.z, in.sh_b.z);
    let l1_z = vec3<f32>(in.sh_r.w, in.sh_g.w, in.sh_b.w);
    let luma = vec3<f32>(0.212656, 0.715158, 0.0721856);
    let dom_raw = vec3<f32>(dot(l1_x, luma), dot(l1_y, luma), dot(l1_z, luma));
    let dom_len = max(length(dom_raw), 1e-6);
    let dominant_light_dir = -dom_raw / dom_len;
    let dominant_light_intensity = in.dominant_intensity.xyz;

    // Engine `entry_points_fx.hlsl:760` — `(1, 1, 1, dot(N, L))`.
    // `.y` is unused per HLSL comment but written as 1.0 not 0.0.
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
    let self_illum_radiance = calc_self_illumination(texcoord, &albedo_for_illum, view_dir_in_tangent_space)
        * g_alt_exposure();

    const BLEND_MULTIPLICATIVE_ENABLED: f32 = __BLEND_MULTIPLICATIVE_ENABLED__;
    const BLEND_MULTIPLICATIVE_FACTOR:  f32 = __BLEND_MULTIPLICATIVE_FACTOR__;
    const BLEND_FRESNEL_ENABLED: f32 = __BLEND_FRESNEL_ENABLED__;

    // Simple lights are evaluated INSIDE the material model now —
    // `mat.diffuse_radiance` + `mat.specular_color` already include them.

    // Cascade shadow gating reserved for dynamic objects.
    var out_rgb: vec3<f32>;
    var alpha_out: f32 = __ALPHA_CHANNEL_OUTPUT__;
    if (BLEND_MULTIPLICATIVE_ENABLED > 0.5) {
        out_rgb = (albedo_for_illum + self_illum_radiance) * BLEND_MULTIPLICATIVE_FACTOR;
    } else if (BLEND_FRESNEL_ENABLED > 0.5) {
        // glass BLEND_FRESNEL (entry_points_fx.hlsl:258) — diffuse
        // premultiplied by albedo.w; reflections additive; alpha =
        // saturate(fresnel + albedo.w). See entry_static_sh.wgsl.
        out_rgb = mat.diffuse_radiance * albedo_for_illum * albedo.w
            + self_illum_radiance
            + envmap_radiance
            + mat.specular_color.xyz;
        out_rgb = (out_rgb * in.extinction + in.inscatter * BLEND_FOG_INSCATTER_SCALE) * g_exposure();
        alpha_out = saturate(mat.specular_color.w + albedo.w);
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

    out_rgb = out_rgb * __ALPHA_PREMULTIPLY__;

    // Engine `convert_to_render_target` clamps RGB ≥ 0 before RT write
    // (`render_target_fx.hlsl:29`). Negatives in RT1 propagate into the
    // bloom pyramid as black-hole pixels in composite.
    let accum = vec4<f32>(max(out_rgb, vec3<f32>(0.0)), alpha_out);
    return AccumPixel(accum, accum);
}
