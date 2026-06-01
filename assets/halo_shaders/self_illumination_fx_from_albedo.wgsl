// Port of `self_illumination_fx.hlsl` — `from_albedo` variant
// (line 92). HLSL function: `calc_self_illumination_from_albedo_ps`.
// (rmdf option name: `from_diffuse`.)
//
// HLSL:
//   float3 calc_self_illumination_from_albedo_ps(
//       in float2 texcoord, inout float3 albedo, in float3 view_dir)
//   {
//       float3 self_illum = albedo * self_illum_color.xyz * self_illum_intensity;
//       albedo = float3(0.f, 0.f, 0.f);
//       return self_illum;
//   }

fn calc_self_illumination_from_albedo_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_uv = texcoord;
    let _u_view = view_dir;
    let self_illum = (*albedo) * material.self_illum_color.xyz * material.self_illum_intensity.x;
    *albedo = vec3<f32>(0.0);
    return self_illum;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_from_albedo_ps(texcoord, albedo, view_dir);
}
