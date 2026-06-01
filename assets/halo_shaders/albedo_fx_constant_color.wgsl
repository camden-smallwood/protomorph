// Port of `albedo_fx.hlsl` — `constant_color` variant (line 65).
//
// HLSL:
//   void calc_albedo_constant_color_ps(
//       in float2 texcoord, out float4 albedo,
//       in float3 normal, in float4 misc)
//   {
//       albedo = albedo_color;
//       apply_pc_albedo_modifier(albedo, normal);
//   }

fn calc_albedo_constant_color_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_uv = texcoord;
    let _u_normal = normal;
    let _u_misc = misc;
    *albedo = material.albedo_color;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_constant_color_ps(texcoord, albedo, normal, misc);
}
