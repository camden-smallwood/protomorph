// Port of `environment_mapping_fx.hlsl` — `dynamic` variant (line 73).
//
// HLSL:
//   float3 calc_environment_map_dynamic_ps(
//       in float3 view_dir, in float3 normal, in float3 reflect_dir,
//       in float4 specular_reflectance_and_roughness,
//       in float3 low_frequency_specular_color)
//   {
//       reflect_dir.y = -reflect_dir.y;
//       // PC path: lod from screen-space derivatives + roughness scale.
//       float grad_x = length(ddx(reflect_dir));
//       float grad_y = length(ddy(reflect_dir));
//       float base_lod = 6.0 * sqrt(max(grad_x, grad_y)) - 0.6;
//       float lod = max(base_lod, specular_reflectance_and_roughness.w * env_roughness_scale * 4);
//       float4 reflection_0 = sampleCUBElod(dynamic_environment_map_0, reflect_dir, lod);
//       float4 reflection_1 = sampleCUBElod(dynamic_environment_map_1, reflect_dir, lod);
//       float3 reflection = (reflection_0.rgb * reflection_0.a * 256) * dynamic_environment_blend.rgb
//                         + (reflection_1.rgb * reflection_1.a * 256) * (1 - dynamic_environment_blend.rgb);
//       return reflection * specular_reflectance_and_roughness.xyz
//            * env_tint_color * low_frequency_specular_color;
//   }
//
// **NOTE:** Halo's runtime updates `dynamic_environment_map_0/1` per
// cluster (previous + current cubemap probes for temporal blending).
// Until Phase G2 lands, the per-variant texture binder maps both to
// the 1×1 black-cube fallback, so `reflection ≈ 0` for now — but the
// math + bindings are HLSL-correct.

fn calc_environment_map_dynamic_ps(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let _u_normal = normal;

    var rd = reflect_dir;
    rd.y = -rd.y;

    // PC LOD path (HLSL `#ifdef pc`).
    let grad_x = length(dpdx(rd));
    let grad_y = length(dpdy(rd));
    let base_lod = 6.0 * sqrt(max(grad_x, grad_y)) - 0.6;
    let lod = max(base_lod, specular_reflectance_and_roughness.w * material.env_roughness_scale.x * 4.0);

    let reflection_0 = textureSampleLevel(dynamic_environment_map_0, dynamic_environment_map_0_sampler, rd, lod);
    let reflection_1 = textureSampleLevel(dynamic_environment_map_1, dynamic_environment_map_1_sampler, rd, lod);

    // `dynamic_environment_blend` is engine-managed (MiscPS cbuffer,
    // `hlsl_constant_persist_fx.hlsl:104`) — sourced from the renderer's
    // per-frame env-probe blend driver, NOT per-material data. Previously
    // baked as a per-material baseline at `(0.5,0.5,0.5,0)` to match the
    // engine default until the cbuffer was wired through.
    let blend = engine_misc_ps.dynamic_environment_blend.xyz;
    let reflection = (reflection_0.rgb * reflection_0.a * 256.0) * blend
                   + (reflection_1.rgb * reflection_1.a * 256.0) * (vec3<f32>(1.0) - blend);

    // `__ENV_REFLECTION_GAIN__` is a DIAGNOSTIC A/B multiplier (default
    // 1.0), substituted from `PROTOMORPH_ENV_GAIN` at WGSL assembly. Used
    // to localize the pre-existing reflection-magnitude deficit on
    // env-mapped surfaces (guardian glass/floor read matte vs MCC) — find
    // the gain that matches MCC, then fix the real upstream term
    // (ambient SH / env-cube level / exposure). Not engine-faithful; 1.0 = off.
    return (__ENV_REFLECTION_GAIN__) * reflection
         * specular_reflectance_and_roughness.xyz
         * material.env_tint_color.xyz
         * low_frequency_specular_color;
}

fn calc_environment_map(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    return calc_environment_map_dynamic_ps(view_dir, normal, reflect_dir, specular_reflectance_and_roughness, low_frequency_specular_color);
}
