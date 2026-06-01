// Port of `environment_mapping_fx.hlsl` — `per_pixel` variant
// (environment_mapping_fx.hlsl:41).
//
// HLSL:
//   float3 calc_environment_map_per_pixel_ps(
//       in float3 view_dir, in float3 normal, in float3 reflect_dir,
//       in float4 specular_reflectance_and_roughness,
//       in float3 low_frequency_specular_color)
//   {
//       reflect_dir.y = -reflect_dir.y;
//       float4 reflection = sampleCUBE(environment_map, reflect_dir);
//       return reflection.rgb * specular_reflectance_and_roughness.xyz
//            * low_frequency_specular_color * env_tint_color * reflection.a;
//   }

fn calc_environment_map_per_pixel_ps(
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
    let reflection = textureSample(environment_map, environment_map_sampler, rd);
    return reflection.rgb
         * specular_reflectance_and_roughness.xyz
         * low_frequency_specular_color
         * material.env_tint_color.xyz
         * reflection.a;
}

// HLSL `CALC_ENVMAP(envmap_type)` macro alias. The umbrella calls
// `calc_environment_map`; this variant's wrapper resolves it to
// the per_pixel PS function. naga inlines.
fn calc_environment_map(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    return calc_environment_map_per_pixel_ps(view_dir, normal, reflect_dir, specular_reflectance_and_roughness, low_frequency_specular_color);
}
