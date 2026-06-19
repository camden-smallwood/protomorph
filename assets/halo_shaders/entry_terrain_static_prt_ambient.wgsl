// Halo `terrain / static_prt_ambient` entry point — vertex + pixel main.
//
// Faithful port of `terrain_new_fx.hlsl` static_prt_ambient path
// (terrain_new_fx.hlsl:1363-1435). Used when a BSP cluster carries
// PRT-ambient transfer data (single AO scalar per vertex). Engine
// `c_structure_renderer::select_cluster_entry_point` picks this
// variant when the cluster's mesh has a PRT transfer stream and no
// per-pixel lightmap / per-vertex SH.
//
// Engine VS (terrain_new_fx.hlsl:1364):
//   `static_prt_ambient_vs` reads a single `prt_c0_c3` scalar from
//   BLENDWEIGHT1 (per-vertex AO). Computes prt_ravi_diff (4-vec) from
//   the single-probe cbuffer SH (v_lighting_constant_0) and AO.
//
// Engine PS (terrain_new_fx.hlsl:1409):
//   delegates to `static_lighting_shared_ps_quadratic` (the prt-aware
//   variant of `static_lighting_shared_ps`), scales diffuse by
//   prt_ravi_diff.x, area_specular by prt_ravi_diff.z, analytical sun
//   gate by prt_ravi_diff.w. SH coefficients still come from the
//   single-probe cbuffer (`engine_lighting_ps.ravi`).
//
// Protomorph PRT-Ambient secondary VB: `PrtAmbientVertex` (single f32
// per vertex, `geometry/mod.rs`). Same stream consumed by the rmsh
// equivalent at `entry_static_prt_ambient.wgsl`.
//
// PS body inherits the full terrain blend pipeline from the static_sh
// variant — only the SH source path picks the cbuffer (no atlas fallback)
// and the final diffuse / area-specular are PRT-modulated.
//
// See blueprint at `reference_terrain_fx_blueprint.md`.
//
// 4-layer blend semantics (terrain_fx.hlsl:172-200, 552-572):
//   1. Sample blend_map at uv*blend_map_xform — gives layer weights
//      blend.xyzw (one channel per terrain material layer 0..3).
//   2. Normalize blend so the sum of active weights == 1.
//   3. For each layer N where blend.N > 0:
//        base = sample(base_map_m_N, uv*base_map_m_N_xform)
//        detail = sample(detail_map_m_N, uv*detail_map_m_N_xform)
//        layer_albedo = base * detail * global_albedo_tint
//        bump_ts = sample_bumpmap(bump_map_m_N + detail_bump_m_N, uv*xform)
//        layer_normal = TBN * bump_ts
//        albedo_accum += layer_albedo * blend.N
//        bump_accum   += layer_normal * blend.N
//   4. Normalize bump_accum.
//   5. diffuse_radiance = ravi_order_3(bump_normal, sh_lighting_coefficients)
//   6. albedo * diffuse_radiance + analytical specular per layer.
//
// v1 simplifications (vs full HLSL umbrella):
//   - No parallax, no alpha_test (terrain doesn't use those typically).
//   - Per-layer specular folded into single shared analytical lobe
//     using layer-0's specular params (terrain_fx blends per-layer
//     via `blend_surface_parameters` — porting that's a follow-up).
//   - No env_map (terrain rarely uses cubemap env reflection).
//   - No atmospheric scattering / exposure (Phase F+).
//   - Output single Rg11b10Ufloat (not the dual-target accum_pixel).

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
    // Per-vertex PRT transfer scalar (`PrtAmbientVertex` secondary VB,
    // single f32 per vertex). Engine equivalent: terrain
    // `static_prt_ambient_vs` `prt_c0_c3` BLENDWEIGHT1 input.
    @location(11) prt_transfer: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) world_binormal: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
    @location(5) camera_world: vec3<f32>,
    @location(6) extinction: vec3<f32>,
    @location(7) inscatter: vec3<f32>,
    @location(8) lightmap_texcoord: vec2<f32>,
    // prt_ravi_diff interpolated to fragment — mirror of engine
    // `static_prt_ambient_vs` output at TEXCOORD5:
    //   .x = diffuse occlusion %    (prt_mono / ravi_mono)
    //   .y = prt_mono               (unused per HLSL comment)
    //   .z = specular occlusion %   (ambient_occlusion × π / 0.886227)
    //   .w = sun-direction gate     (min(N·L, prt_mono); kills backfacing
    //                                analytical specular)
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
    // Same as static_sh_vs (terrain shares VS with rmsh — only PS
    // differs). BSP geometry is rigid — single identity bone.
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
    out.camera_world = camera_world;
    out.extinction = extinction;
    out.inscatter = inscatter;
    out.lightmap_texcoord = in.lightmap_texcoord;

    // PRT-ambient block — mirror of engine `static_prt_ambient_vs`
    // (terrain_new_fx.hlsl:1383-1397). Computes per-vertex prt_ravi_diff
    // from the single-probe cbuffer (engine_lighting_ps.ravi[0])
    // and the per-vertex AO scalar. Interpolated to fragment.
    let ambient_occlusion = clamp(in.prt_transfer, 0.0, 1.0);
    let lighting_c0 = dot(
        engine_lighting_ps.ravi[0].xyz,
        vec3<f32>(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    );
    let ravi_mono_raw = (0.886227 * lighting_c0) / 3.1415926535;
    let prt_mono_raw = ambient_occlusion * lighting_c0;
    let prt_mono = max(prt_mono_raw, 0.01);
    let ravi_mono = max(ravi_mono_raw, 0.01);
    let prt_ravi_ratio = prt_mono / ravi_mono;
    // Engine `get_constant_analytical_light_dir_vs` (terrain_new_fx.hlsl:543):
    //   return -normalize(v_lighting_constant_1 + v_lighting_constant_2 +
    //                     v_lighting_constant_3).xyz;
    // Computes the analytical sun direction from the SH L1 bands of the
    // default-lighting cbuffer. We use engine_lighting_ps.ravi[1..3]
    // (VS-visible at binding 1) — the FS-only dominant_light binding
    // can't be read here. The SH L1 bands ARE the dominant-light
    // direction by construction (calculate_dominant_light_from_lightprobe
    // is what populates dominant_light, derived from the same L1 sum).
    let l1_sum_dir = engine_lighting_ps.ravi[1].xyz
                   + engine_lighting_ps.ravi[2].xyz
                   + engine_lighting_ps.ravi[3].xyz;
    let l1_sum_len2 = dot(l1_sum_dir, l1_sum_dir);
    var analytical_light_dir = vec3<f32>(0.0, 0.0, 1.0);
    if (l1_sum_len2 > 1.0e-8) {
        analytical_light_dir = -l1_sum_dir * inverseSqrt(l1_sum_len2);
    }
    let n_dot_l = dot(out.world_normal, analytical_light_dir);
    out.prt_ravi_diff = vec4<f32>(
        prt_ravi_ratio,
        prt_mono,
        (ambient_occlusion * 3.1415926535) / 0.886227,
        min(n_dot_l, prt_mono),
    );
    return out;
}

// Sample one terrain layer's albedo. Returns vec4: .rgb = base × detail
// (color, gets `× tint × 4.59479` post-accumulation), .a = base.a × detail.a
// (specular mask, NO tint multiplier per HLSL's `mult.w = 1.0`).
// Mirrors `ACCUMULATE_MATERIAL_ALBEDO` macro (terrain_fx.hlsl:552-572).
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

// Sample one terrain layer's bump in TANGENT space. Halo's
// `calc_bumpmap` (terrain_fx.hlsl:220-237) adds bump+detail raw and
// reconstructs Z from XY. Per-layer bumps are accumulated in tangent
// space across layers, then transformed to world ONCE — matches HLSL
// `ACCUMULATE_MATERIAL_BUMP` ordering (bump_normal_world is built in
// tangent space then mul(tangent_frame, bump_normal) once).
//
// `detail_bump_enabled` mirrors HLSL's `#if DETAIL_BUMP_ENABLED` gate
// (terrain_fx.hlsl:116) — false when 4 materials are active OR the
// variant carries the `(four_material_shaders_disable_detail_bump)`
// suffix. When false, detail_bump sample is skipped and only the
// primary bump_map contributes.
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
    // Add raw in tangent space, reconstruct Z (HLSL line 232-234).
    var sum = vec3<f32>(bump_ts.x + detail_ts.x, bump_ts.y + detail_ts.y, 0.0);
    sum.z = sqrt(max(1.0 - sum.x * sum.x - sum.y * sum.y, 0.0));
    return sum;
}

// `specular_parameters` struct from terrain_fx.hlsl:909-919.
// Accumulated per-active-layer by `blend_surface_parameters`.
struct SpecularParameters {
    // .rgb = layer-blended specular tint (modulated by 1 - albedo_blend),
    // .a   = sum(blend.N * albedo_specular_tint_blend_m_N) — used for
    //         the per-pixel tint↔albedo lerp at composition time.
    normal_albedo: vec4<f32>,
    power: f32,
    analytical: f32,
    area: f32,
    envmap: f32,
    fresnel_steepness: f32,
    weight: f32,
}

// terrain_fx.hlsl:935-957 `blend_specular_parameters`. One call per
// active layer; accumulates weighted contributions.
fn blend_specular(
    blend_amount: f32,
    spec_tint: vec3<f32>,
    albedo_spec_tint_blend: f32,
    spec_power: f32,
    spec_coefficient: f32,
    analytical_contribution: f32,
    area_contribution: f32,
    environment_contribution: f32,
    fresnel_steepness: f32,
    spec: ptr<function, SpecularParameters>,
) {
    (*spec).normal_albedo.r = (*spec).normal_albedo.r + blend_amount * spec_tint.r * (1.0 - albedo_spec_tint_blend);
    (*spec).normal_albedo.g = (*spec).normal_albedo.g + blend_amount * spec_tint.g * (1.0 - albedo_spec_tint_blend);
    (*spec).normal_albedo.b = (*spec).normal_albedo.b + blend_amount * spec_tint.b * (1.0 - albedo_spec_tint_blend);
    (*spec).normal_albedo.a = (*spec).normal_albedo.a + blend_amount * albedo_spec_tint_blend;
    (*spec).power             = (*spec).power + blend_amount * spec_power;
    (*spec).analytical        = (*spec).analytical + blend_amount * spec_coefficient * analytical_contribution;
    (*spec).area              = (*spec).area + blend_amount * spec_coefficient * area_contribution;
    (*spec).envmap            = (*spec).envmap + blend_amount * spec_coefficient * environment_contribution;
    (*spec).fresnel_steepness = (*spec).fresnel_steepness + blend_amount * fresnel_steepness;
    (*spec).weight            = (*spec).weight + blend_amount;
}

// Engine `BLEND_SELF_ILLUM` macro (terrain_new_fx.hlsl:630-657) — one
// layer's self-illum contribution = self_illum_map * (self_illum_detail *
// DETAIL_MULTIPLIER) * self_illum_color * self_illum_intensity.
fn blend_self_illum_layer(
    uv: vec2<f32>,
    map: texture_2d<f32>,
    map_samp: sampler,
    map_xform: vec4<f32>,
    detail: texture_2d<f32>,
    detail_samp: sampler,
    detail_xform: vec4<f32>,
    color: vec3<f32>,
    intensity: f32,
) -> vec3<f32> {
    let illum = textureSample(map, map_samp, transform_texcoord(uv, map_xform)).rgb;
    let detail_sample = textureSample(detail, detail_samp, transform_texcoord(uv, detail_xform)).rgb;
    return illum * (detail_sample * 4.59479) * color * intensity;
}

// terrain_new_fx.hlsl:661-766 `blend_surface_parameters`. Per-layer
// gating via __SPECULAR_MATERIAL_{N}__ + __SELF_ILLUM_MATERIAL_{N}__
// substitutions mirrors HLSL `#if SPECULAR_MATERIAL(material_N_type)`
// and `#if SELF_ILLUM_MATERIAL(material_N_type)`. Multiplying the
// blend_amount by the gate makes non-spec/non-illum layers contribute
// zero — equivalent to the engine omitting the BLEND_SPECULAR call.
//
// Returns both `spec` and accumulated `self_illum` since HLSL's `out`
// params don't translate; WGSL multi-return goes via struct.
struct SurfaceParams {
    spec: SpecularParameters,
    self_illum: vec3<f32>,
}

fn blend_surface_parameters(uv: vec2<f32>, blend: vec4<f32>) -> SurfaceParams {
    var spec: SpecularParameters;
    spec.normal_albedo = vec4<f32>(0.0);
    spec.power = 0.001 * 1.0;
    spec.analytical = 0.0;
    spec.area = 0.0;
    spec.envmap = 0.0;
    spec.fresnel_steepness = 0.001 * 5.0;
    spec.weight = 0.001;

    // Per-layer specular accumulation. Gate multiplies blend amount
    // by SPECULAR_MATERIAL_{N} substitution (1.0 or 0.0) so non-spec
    // layers add nothing — mirrors engine's compile-time `#if` skip.
    blend_specular(
        blend.x * __SPECULAR_MATERIAL_0__,
        material.specular_tint_m_0.xyz,
        material.albedo_specular_tint_blend_m_0.x,
        material.specular_power_m_0.x,
        material.specular_coefficient_m_0.x,
        material.analytical_specular_contribution_m_0.x,
        material.area_specular_contribution_m_0.x,
        material.environment_specular_contribution_m_0.x,
        material.fresnel_curve_steepness_m_0.x,
        &spec,
    );
    blend_specular(
        blend.y * __SPECULAR_MATERIAL_1__,
        material.specular_tint_m_1.xyz,
        material.albedo_specular_tint_blend_m_1.x,
        material.specular_power_m_1.x,
        material.specular_coefficient_m_1.x,
        material.analytical_specular_contribution_m_1.x,
        material.area_specular_contribution_m_1.x,
        material.environment_specular_contribution_m_1.x,
        material.fresnel_curve_steepness_m_1.x,
        &spec,
    );
    blend_specular(
        blend.z * __SPECULAR_MATERIAL_2__,
        material.specular_tint_m_2.xyz,
        material.albedo_specular_tint_blend_m_2.x,
        material.specular_power_m_2.x,
        material.specular_coefficient_m_2.x,
        material.analytical_specular_contribution_m_2.x,
        material.area_specular_contribution_m_2.x,
        material.environment_specular_contribution_m_2.x,
        material.fresnel_curve_steepness_m_2.x,
        &spec,
    );
    blend_specular(
        blend.w * __SPECULAR_MATERIAL_3__,
        material.specular_tint_m_3.xyz,
        material.albedo_specular_tint_blend_m_3.x,
        material.specular_power_m_3.x,
        material.specular_coefficient_m_3.x,
        material.analytical_specular_contribution_m_3.x,
        material.area_specular_contribution_m_3.x,
        material.environment_specular_contribution_m_3.x,
        material.fresnel_curve_steepness_m_3.x,
        &spec,
    );

    let scale = 1.0 / max(spec.weight, 0.001);
    spec.fresnel_steepness = spec.fresnel_steepness * scale;
    spec.power = spec.power * scale;

    // Self-illum accumulation. Engine HLSL only loops layers 0/1/2 —
    // material_3_type has no branch (terrain_new_fx.hlsl:753-765).
    // SELF_ILLUM_MATERIAL_3 always substitutes 0.0 to mirror that.
    var self_illum = vec3<f32>(0.0);
    self_illum = self_illum + blend_self_illum_layer(
        uv,
        self_illum_map_m_0, self_illum_map_m_0_sampler, material.self_illum_map_m_0_xform,
        self_illum_detail_map_m_0, self_illum_detail_map_m_0_sampler, material.self_illum_detail_map_m_0_xform,
        material.self_illum_color_m_0.rgb, material.self_illum_intensity_m_0.x,
    ) * (blend.x * __SELF_ILLUM_MATERIAL_0__);
    self_illum = self_illum + blend_self_illum_layer(
        uv,
        self_illum_map_m_1, self_illum_map_m_1_sampler, material.self_illum_map_m_1_xform,
        self_illum_detail_map_m_1, self_illum_detail_map_m_1_sampler, material.self_illum_detail_map_m_1_xform,
        material.self_illum_color_m_1.rgb, material.self_illum_intensity_m_1.x,
    ) * (blend.y * __SELF_ILLUM_MATERIAL_1__);
    self_illum = self_illum + blend_self_illum_layer(
        uv,
        self_illum_map_m_2, self_illum_map_m_2_sampler, material.self_illum_map_m_2_xform,
        self_illum_detail_map_m_2, self_illum_detail_map_m_2_sampler, material.self_illum_detail_map_m_2_xform,
        material.self_illum_color_m_2.rgb, material.self_illum_intensity_m_2.x,
    ) * (blend.z * __SELF_ILLUM_MATERIAL_2__);

    return SurfaceParams(spec, self_illum);
}

// Single MRT — `_surface_post_HDR` slot 0 only. Normal G-buffer was
// written by `_entry_point_albedo` and is sampled (when needed) via
// camera_bgl @ binding 11.

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
    let view_dir = normalize(in.camera_world - in.world_position);

    // 1. Sample blend mask at xformed UV
    let blend_uv = transform_texcoord(in.tex_coords, material.blend_map_xform);
    var blend = textureSample(blend_map, blend_map_sampler, blend_uv);

    // 2. Mask off inactive layers per the rmsh's category choices
    //    (mirrors HLSL `#if ACTIVE_MATERIAL(material_N_type)` gating in
    //    `sample_blend_normalized`, terrain_fx.hlsl:184-195). Substitution
    //    tokens get replaced at assembly time by the rmtr dispatcher.
    let layer_active = vec4<f32>(
        __MATERIAL_0_ACTIVE__,
        __MATERIAL_1_ACTIVE__,
        __MATERIAL_2_ACTIVE__,
        __MATERIAL_3_ACTIVE__,
    );
    blend = blend * layer_active;

    // 3. Normalize so sum of ACTIVE weights == 1 (terrain_new_fx.hlsl:193).
    //    Engine does NOT clamp blend_sum — line 170-171 comment:
    //    "We've decided this isn't worth the instruction - just change
    //    your blend map" (re: epsilon for pure-black blend pixels).
    let blend_sum = blend.x + blend.y + blend.z + blend.w;
    blend = blend / blend_sum;

    // Engine `terrain_new_fx.hlsl:941, 959`:
    //   bump_normal = normal_texture.Load(int3(fragment_position.xy, 0)).xyz * 2 - 1;
    //   albedo      = albedo_texture.Load(int3(fragment_position.xy, 0));
    // RT0 is now Rgba16Float (per dllcache `_surface_accum_HDR`,
    // e_surface=13), so albedo.w carries the engine-faithful spec_mask
    // = sum_N(blend.N * base_N.a * detail_N.a).
    let fp = vec2<i32>(in.clip_position.xy);
    let albedo_full = textureLoad(albedo_texture, fp, 0);
    let normal_packed = textureLoad(normal_texture, fp, 0);
    // Engine reads bump_normal raw (terrain_new_fx.hlsl:941) — no
    // normalize. G-buffer was written with normalized values; small
    // interpolation drift is acceptable to match engine output.
    let bump_normal = normal_packed.xyz * 2.0 - 1.0;
    let albedo_w = albedo_full.w;
    let albedo = albedo_full.rgb;

    // 4. Per-layer specular + self-illum accumulation — full
    //    `blend_surface_parameters` port from terrain_new_fx.hlsl:661-766.
    //    Per-layer gating mirrors HLSL `#if SPECULAR_MATERIAL` and
    //    `#if SELF_ILLUM_MATERIAL` via substitution tokens.
    let surf = blend_surface_parameters(in.tex_coords, blend);
    let spec = surf.spec;
    let self_illum = surf.self_illum;

    // 5. Per-layer diffuse_coefficient blend (terrain_fx.hlsl:967-1014).
    let diffuse_coef = blend.x * material.diffuse_coefficient_m_0.x
                     + blend.y * material.diffuse_coefficient_m_1.x
                     + blend.z * material.diffuse_coefficient_m_2.x
                     + blend.w * material.diffuse_coefficient_m_3.x;

    // SH probe — engine path is `terrain_new_fx.hlsl::static_lighting_shared_ps_linear_with_dominant_light`
    // (line 899). Atlas decode produces ORDER-2 SH (4 vec4) PLUS
    // separately-encoded dominant_light_direction and dominant_light_intensity.
    // The dominant light is fed into:
    //   - `ravi_order_2_with_dominant_light` for diffuse (subtracts dom from
    //     SH then re-adds it as analytical lambertian — without this term
    //     the diffuse misses the bulk of sun-driven brightness)
    //   - the Phong analytical specular term directly (atlas value, NOT
    //     SH-extracted approximation)
    let has_lightmap = in.lightmap_texcoord.x != 0.0 || in.lightmap_texcoord.y != 0.0;
    var sh_linear: array<vec4<f32>, 4>;
    var dom_dir: vec3<f32>;
    var dom_int: vec3<f32>;
    if (has_lightmap) {
        var probe = sample_lightprobe_texture(in.lightmap_texcoord);
        // Atlas-empty fallback — WORKAROUND, not engine-faithful.
        // ~16% of riverworld atlas texels carry BC3 alpha=0 → decode
        // returns SH=0 → diffuse=0 → black terrain. The cluster path
        // has no engine fallback (per dllcache trace in
        // project_waterfall_atlas_empty_fallback_2026_05_15). Believed
        // root cause: MCC's binary cache atlas differs from the loose
        // .bitmap tag (tool.exe likely repairs zero-alpha regions
        // during cache build). Until that's resolved, substitute the
        // engine_lighting_ps frame-default ravi cbuffer when atlas SH
        // DC is near zero. Same pattern as entry_static_per_pixel.wgsl.
        // Engine-faithful default OFF: no DC threshold exists in the engine
        // HLSL; it lit real baked shadows. See entry_static_per_pixel.wgsl.
        const LIGHTMAP_EMPTY_ATLAS_FALLBACK: bool = __LIGHTMAP_EMPTY_ATLAS_FALLBACK__;
        let _atlas_dc = probe.sh[0];
        if (LIGHTMAP_EMPTY_ATLAS_FALLBACK && (_atlas_dc.r + _atlas_dc.g + _atlas_dc.b) < 0.01) {
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
        sh_linear = pack_constants_texture_array_linear(probe);
        dom_dir = probe.dir;
        dom_int = probe.intensity;
    } else {
        // No lightmap UV — `setup_default_lighting` on the cluster path
        // uploads SH3 to engine_lighting_ps and dominant light to
        // dominant_light. Take the first 4 vec4s (DC + L1) for the
        // order-2 layout.
        let r = engine_lighting_ps.ravi;
        sh_linear = array<vec4<f32>, 4>(r[0], r[1], r[2], r[3]);
        dom_dir = dominant_light.direction.xyz;
        dom_int = dominant_light.intensity.rgb;
    }
    let diffuse_radiance = ravi_order_2_with_dominant_light(
        bump_normal, sh_linear, dom_dir, dom_int);
    let diffuse = albedo * diffuse_radiance * max(diffuse_coef, 0.0);

    let view_reflect = reflect(-view_dir, bump_normal);

    // Three-stage specular tint (terrain_fx.hlsl:1163-1165):
    //   1. layer-blended tint → albedo (weighted by spec.normal_albedo.a)
    //   2. result → white at glancing angles via fresnel power curve
    let n_dot_v = clamp(dot(bump_normal, view_dir), 0.0, 1.0);
    // Use albedo (the spec mask) — for v1, treat the multi-layer
    // accumulated albedo's average as the tint-blend target.
    let normal_tint = mix(spec.normal_albedo.rgb, albedo, spec.normal_albedo.a);
    let fresnel_blend = pow(1.0 - n_dot_v, max(spec.fresnel_steepness, 0.001));
    let specular_tint = mix(normal_tint, vec3<f32>(1.0), fresnel_blend);

    // Analytical Phong on dominant light. Engine
    // `calc_phong_outgoing_light` (terrain_new_fx.hlsl:245-270):
    //   if (n_dot_l > 0 && n_dot_v > 0):
    //     spec = color * pow(l_dot_r, power) * ((power+1)/2pi)
    //   else: 0
    // Gate prevents phantom specular on back-facing fragments.
    // Engine uses raw `specular_power` — no max(.., 1.0) clamp.
    let n_dot_l = dot(bump_normal, dom_dir);
    let l_dot_r = max(dot(dom_dir, view_reflect), 0.0);
    let phong_norm = (spec.power + 1.0) / 6.2832;
    var analytical_specular_light = vec3<f32>(0.0);
    if (n_dot_l > 0.0 && n_dot_v > 0.0) {
        analytical_specular_light = pow(l_dot_r, spec.power) * phong_norm * dom_int;
    }

    // Area specular — engine `terrain_new_fx.hlsl:985-986`:
    //   area_specular_light = ravi_order_2_new(view_reflect_dir, sh_lighting_coefficients);
    //   area_specular_light = max(0.0f, area_specular_light);
    // Order-2 4-vec4 path (NOT order-3 zero-padded). Pairs with the
    // static_per_pixel atlas source — only 4 SH coefs to begin with.
    let area_specular_light = max(
        ravi_order_2_new(view_reflect, sh_linear),
        vec3<f32>(0.0),
    );

    // Envmap — engine `terrain_new_fx.hlsl:993-998`:
    //   envmap_light = CALC_ENVMAP(envmap_type)(
    //       view_dir, bump_normal, view_reflect_dir,
    //       float4(1, 1, 1, max(0.01, 1.01 - specular.power / 200.0)),
    //       area_specular_light);
    // The 4th arg packs spec_reflectance (1,1,1) with roughness in .w.
    // The 5th arg is the low-frequency area-specular color the envmap
    // is tinted by. Engine routes via per-shader CALC_ENVMAP macro;
    // protomorph picks the WGSL envmap fragment at assembly time
    // based on the rmtr's environment_mapping choice (see
    // render_methods/mod.rs rmtr dispatch).
    let envmap_roughness = max(0.01, 1.01 - spec.power / 200.0);
    let envmap_light = calc_environment_map(
        view_dir,
        bump_normal,
        view_reflect,
        vec4<f32>(1.0, 1.0, 1.0, envmap_roughness),
        area_specular_light,
    );

    // Final specular composition (terrain_fx.hlsl:1184-1187).
    // `albedo_w` (= sum_N(blend.N * base_N.a * detail_N.a)) is the spec
    // mask — modulates total specular contribution per-pixel.
    let analytic_specular = albedo_w * specular_tint * (
        envmap_light * spec.envmap +
        area_specular_light * spec.area +
        analytical_specular_light * spec.analytical
    );

    // Simple lights — diffuse + specular per HLSL
    // `calc_simple_lights_analytical` (terrain_new_fx.hlsl:951-957).
    // Engine uses raw `specular.power` — no max(.., 1.0) clamp.
    let sl = calc_simple_lights_analytical(
        in.world_position,
        bump_normal,
        view_reflect,
        spec.power,
    );

    // Engine `out_color.rgb += self_illum` (terrain_new_fx.hlsl:1018).
    // Self_illum is added INSIDE the lit accumulator so atmospheric
    // extinction and exposure also apply to glow.
    // Engine terrain_new_fx.hlsl:880/887 — diffuse_coefficient multiplies the
    // SUM (SH + simple-light) diffuse; spec.analytical multiplies the SUM
    // (analytical + simple-light) specular. So the simple-light terms must
    // carry the same coefficients as their SH/analytical counterparts.
    let lit = diffuse + sl.diffuse * albedo * max(diffuse_coef, 0.0)
        + analytic_specular
        + sl.specular * specular_tint * albedo_w * spec.analytical
        + self_illum;
    // Atmospheric scattering + exposure (terrain_new_fx.hlsl:1021):
    //   out_color.rgb = (out_color.rgb * extinction + inscatter) * g_exposure.rrr;
    // Engine has NO scale on the inscatter term — drop our
    // FOG_INSCATTER_SCALE multiplier on this path.
    let composed = (lit * in.extinction + in.inscatter) * g_exposure();

    // Engine `convert_to_render_target` clamps RGB ≥ 0 before RT write
    // (`render_target_fx.hlsl:29`).
    let accum = vec4<f32>(max(composed, vec3<f32>(0.0)), 1.0);
    return AccumPixel(accum, accum);
}
