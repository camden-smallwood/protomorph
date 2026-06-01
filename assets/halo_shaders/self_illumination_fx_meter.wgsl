// Port of `self_illumination_fx.hlsl` — `meter` variant (line 129).
//
// HLSL:
//   float3 calc_self_illumination_meter_ps(
//       in float2 texcoord, inout float3 albedo, in float3 view_dir)
//   {
//       float4 sample = sample2D(meter_map, transform_texcoord(texcoord, meter_map_xform));
//       return (sample.x >= 0.5f)
//           ? (meter_value >= sample.w)
//               ? meter_color_on.xyz
//               : meter_color_off.xyz
//           : float3(0, 0, 0);
//   }
//
// Channel semantics (engine):
//   sample.x (RED)    — mask channel, binary 0.5 threshold; pixel is part
//                        of the meter only when red >= 0.5.
//   sample.w (ALPHA)  — per-pixel meter level threshold; pixel is "ON"
//                        when meter_value >= sample.w.

fn calc_self_illumination_meter_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_albedo = albedo;
    let _u_view = view_dir;

    let sample = textureSample(meter_map, meter_map_sampler,
        transform_texcoord(texcoord, material.meter_map_xform),
    );

    if (sample.x < 0.5) {
        return vec3<f32>(0.0);
    }
    if (material.meter_value.x >= sample.w) {
        return material.meter_color_on.xyz;
    }
    return material.meter_color_off.xyz;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_meter_ps(texcoord, albedo, view_dir);
}
