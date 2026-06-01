// Halo `terrain / static_per_vertex` entry point — vertex + pixel main.
//
// Faithful port of `terrain_fx.hlsl` static_per_vertex path
// (terrain_fx.hlsl:1228-1284). Used when a BSP cluster carries
// per-vertex lightprobes (lightmap.clusters[i].pervertex_block_index
// != -1) — the engine `c_structure_renderer::select_cluster_entry_point`
// picks this variant over `static_sh` (cbuffer probe) or
// `static_per_pixel` (atlas) per cluster.
//
// Engine VS (terrain_fx.hlsl:1228):
//   static_per_vertex_vs reads per-vertex SH from TEXCOORD3/4/5
//   (4 SH coefs per RGB channel = 3 × float4 = c0_3_r/g/b),
//   passes them through to PS as TEXCOORD5/6/7.
//
// Engine PS (terrain_fx.hlsl:1257-1283):
//   delegates to `static_lighting_shared_ps` whose `get_sh_coefficients`
//   for per_vertex (terrain_fx.hlsl:845-853):
//     L0_3[3] = {data.p0_3_r, data.p0_3_g, data.p0_3_b};
//     L4_7[3] = {0, 0, 0};   // L2 zeroed (per-vertex stream is order-2 only)
//     pack_constants(L0_3, L4_7, sh_lighting_coefficients);
//   Then `ravi_order_3(bump_normal, sh_lighting_coefficients)` for diffuse
//   (terrain_fx.hlsl:1128) — same order-3 evaluation as cbuffer/atlas paths,
//   just with the L2 coefs pinned to zero.
//
// Protomorph per-vertex SH stream (`PerVertexShVertex` in geometry/mod.rs):
//   secondary VB carries (DC, X, Y, -Z) per channel as 4-vec4
//   attributes (sh_r, sh_g, sh_b, dom_intensity). Z is negated to
//   match the cbuffer convention `_pack_constants_linear` emits.
//   At evaluation we re-pack (DC, X, Y, Z) → (DC, Y, Z, X) per HLSL
//   `pack_constants` (terrain_fx.hlsl uses pack_constants which puts
//   the linear bands in the L1 m=-1, m=0, m=+1 slots).
//
// PS body inherits the full terrain blend + specular pipeline from
// the `static_sh` variant (only the SH source differs — per the
// engine's `entry_point_data` indirection).
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
    // Per-vertex SH stream — secondary VB (`PerVertexShVertex` layout
    // from `geometry/mod.rs`). `(DC, X, Y, -Z)` per channel. Engine
    // equivalent: terrain `static_per_vertex_vs` `c0_3_r/g/b` inputs
    // at TEXCOORD3/4/5.
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
    @location(5) camera_world: vec3<f32>,
    @location(6) extinction: vec3<f32>,
    @location(7) inscatter: vec3<f32>,
    @location(8) lightmap_texcoord: vec2<f32>,
    // Per-vertex SH interpolated to fragment. Mirror of engine
    // `static_per_vertex_vs` outputs `probe0_3_r/g/b` at TEXCOORD5/6/7.
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
    // Pass per-vertex SH through to PS — rasterizer interpolates
    // linearly. Engine `static_per_vertex_vs` does the same pass-through
    // (terrain_fx.hlsl:1248-1250: probe0_3_r= c0_3_r; etc).
    out.sh_r = in.sh_r;
    out.sh_g = in.sh_g;
    out.sh_b = in.sh_b;
    out.dominant_intensity = in.dominant_intensity;
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

    // SH probe — engine `static_per_vertex_ps` →
    // `static_lighting_shared_ps` → `get_sh_coefficients` (terrain_fx.hlsl
    // :845-853):
    //   L0_3[3]= {data.p0_3_r, data.p0_3_g, data.p0_3_b};
    //   L4_7[3]= {0, 0, 0};   // L2 zeroed for per-vertex stream
    //   pack_constants(L0_3, L4_7, sh_lighting_coefficients);
    //
    // `pack_constants` (spherical_harmonics_fx.hlsl) builds the 10-coef
    // cbuffer layout from 4 channel-major SH coefs. The per-vertex
    // stream packs `(DC, X, Y, -Z)` per channel (PerVertexShVertex,
    // geometry/mod.rs:71), so we re-arrange into the cbuffer convention
    // `(DC, Y, Z, X)` per channel — the layout `ravi_order_2_new` /
    // `ravi_order_2_with_dominant_light` consume.
    //
    // Dominant direction synthesized from L1 bands per
    // `terrain_fx.hlsl:1138`:
    //   analytical_light_dir = -normalize(L1_R.xyz + L1_G.xyz + L1_B.xyz)
    // Dominant intensity per `terrain_fx.hlsl:1142-1145`:
    //   I_r = dot(-dir, L1_R * 0.488603) + L0.r * 0.28209479
    //   I_g = dot(-dir, L1_G * 0.488603) + L0.g * 0.28209479
    //   I_b = dot(-dir, L1_B * 0.488603) + L0.b * 0.28209479
    //   I *= PI
    // Engine reference: `pack_constants_linear`
    // (spherical_harmonics_fx.hlsl:30-36) produces per-channel L1
    // vector layout (NOT per-coef-per-RGB):
    //   lighting_constants[0] = (DC_r, DC_g, DC_b, 0)
    //   lighting_constants[1] = (R.L1_x, R.L1_y, -R.L1_z, 0)
    //   lighting_constants[2] = (G.L1_x, G.L1_y, -G.L1_z, 0)
    //   lighting_constants[3] = (B.L1_x, B.L1_y, -B.L1_z, 0)
    // Protomorph `PerVertexShVertex` stores `(DC, X, Y, -Z)` per
    // channel — W is PRE-NEGATED, so sh_r.yzw = (X, Y, -Z) maps
    // directly into lighting_constants[1]. No swizzle/sign flip.
    let pv_r = in.sh_r;
    let pv_g = in.sh_g;
    let pv_b = in.sh_b;
    let sh_linear = array<vec4<f32>, 4>(
        vec4<f32>(pv_r.x, pv_g.x, pv_b.x, 0.0),  // DC per channel
        vec4<f32>(pv_r.y, pv_r.z, pv_r.w, 0.0),  // R: (X, Y, -Z) direct
        vec4<f32>(pv_g.y, pv_g.z, pv_g.w, 0.0),  // G: (X, Y, -Z) direct
        vec4<f32>(pv_b.y, pv_b.z, pv_b.w, 0.0),  // B: (X, Y, -Z) direct
    );
    // Dominant-light direction (terrain_fx.hlsl:1138):
    //   analytical_light_dir = -normalize(L1[1].xyz + L1[2].xyz + L1[3].xyz)
    // With the per-channel layout above, .xyz of [1]/[2]/[3] is each
    // channel's (X, Y, -Z) vector. Sum of those: produces an averaged
    // 3-vector that approximates the dominant direction (engine treats
    // each channel's L1 as carrying the same dominant direction with
    // intensity-weighted magnitude — averaging is the legit shortcut).
    // We need canonical (X, Y, Z) so undo the -Z storage convention:
    //   canonical L1 per channel = (L1.x, L1.y, -L1.z)
    let canon_r = vec3<f32>(pv_r.y, pv_r.z, -pv_r.w);
    let canon_g = vec3<f32>(pv_g.y, pv_g.z, -pv_g.w);
    let canon_b = vec3<f32>(pv_b.y, pv_b.z, -pv_b.w);
    let l1_sum = canon_r + canon_g + canon_b;
    let l1_sum_len2 = dot(l1_sum, l1_sum);
    var dom_dir = vec3<f32>(0.0, 0.0, 1.0);
    if (l1_sum_len2 > 1.0e-8) {
        dom_dir = -l1_sum * inverseSqrt(l1_sum_len2);
    }
    // Dominant intensity per terrain_fx.hlsl:1142-1145.
    //   I_r = dot(-dom_dir, L1_R * 0.488603) + DC_r * 0.28209479
    //   I_g = dot(-dom_dir, L1_G * 0.488603) + DC_g * 0.28209479
    //   I_b = dot(-dom_dir, L1_B * 0.488603) + DC_b * 0.28209479
    //   I *= π
    let i_r = dot(-dom_dir, canon_r * 0.488603) + pv_r.x * 0.28209479;
    let i_g = dot(-dom_dir, canon_g * 0.488603) + pv_g.x * 0.28209479;
    let i_b = dot(-dom_dir, canon_b * 0.488603) + pv_b.x * 0.28209479;
    let dom_int = max(vec3<f32>(i_r, i_g, i_b) * 3.1415926535, vec3<f32>(0.0));
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
    let lit = diffuse + sl.diffuse * albedo
        + analytic_specular
        + sl.specular * specular_tint * albedo_w
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
