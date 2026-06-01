// Port of `environment_mapping_fx.hlsl` — `none` variant.
//
// HLSL signature (verbatim):
//   float3 calc_environment_map_none_ps(
//       in float3 view_dir, in float3 normal, in float3 reflect_dir,
//       in float4 specular_reflectance_and_roughness,
//       in float3 low_frequency_specular_color)
//   { return float3(0.0f, 0.0f, 0.0f); }

fn calc_environment_map_none_ps(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    let _u0 = view_dir;
    let _u1 = normal;
    let _u2 = reflect_dir;
    let _u3 = specular_reflectance_and_roughness;
    let _u4 = low_frequency_specular_color;
    return vec3<f32>(0.0);
}

// HLSL `CALC_ENVMAP(envmap_type)` macro alias for this variant. WGSL
// has no preprocessor, so the assembler includes one variant file
// per assembly and the umbrella calls `calc_environment_map`. The
// wrapper compiles to a single function call (naga inlines it).
fn calc_environment_map(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    return calc_environment_map_none_ps(view_dir, normal, reflect_dir, specular_reflectance_and_roughness, low_frequency_specular_color);
}
