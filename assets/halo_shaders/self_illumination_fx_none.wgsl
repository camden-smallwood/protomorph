// Port of `self_illumination_fx.hlsl` — `none` variant (line 4).
//
// HLSL:
//   float3 calc_self_illumination_none_ps(
//       in float2 texcoord, inout float3 albedo_times_light, in float3 view_dir)
//   { return float3(0.0f, 0.0f, 0.0f); }

fn calc_self_illumination_none_ps(
    texcoord: vec2<f32>,
    albedo_times_light: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_uv = texcoord;
    let _u_view = view_dir;
    let _u_albedo = albedo_times_light;
    return vec3<f32>(0.0);
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_none_ps(texcoord, albedo, view_dir);
}
