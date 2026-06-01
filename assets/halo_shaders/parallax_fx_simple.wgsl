// Port of `parallax_fx.hlsl` — `simple` variant (line 15).
//
// HLSL:
//   void calc_parallax_simple_ps(
//       in float2 texcoord, in float3 view_dir,
//       out float2 parallax_texcoord)
//   {
//       texcoord = transform_texcoord(texcoord, height_map_xform);
//       float height = (sample(height_map, texcoord).g - 0.5) * height_scale;
//       parallax_texcoord = texcoord + height * view_dir.xy;
//       parallax_texcoord = (parallax_texcoord - height_map_xform.zw) / height_map_xform.xy;
//   }

fn calc_parallax_simple_ps(
    texcoord: vec2<f32>,
    view_dir: vec3<f32>,
    parallax_texcoord: ptr<function, vec2<f32>>,
) {
    let xformed = transform_texcoord(texcoord, material.height_map_xform);
    let height = (textureSample(height_map, height_map_sampler, xformed).g - 0.5) * material.height_scale.x;
    var p = xformed + height * view_dir.xy;
    p = (p - material.height_map_xform.zw) / material.height_map_xform.xy;
    *parallax_texcoord = p;
}

fn calc_parallax(texcoord: vec2<f32>, view_dir: vec3<f32>, parallax_texcoord: ptr<function, vec2<f32>>) {
    calc_parallax_simple_ps(texcoord, view_dir, parallax_texcoord);
}
