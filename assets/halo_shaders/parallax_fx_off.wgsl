// Port of `parallax_fx.hlsl` — `off` variant (line 7).
//
// HLSL:
//   void calc_parallax_off_ps(
//       in float2 texcoord, in float3 view_dir,
//       out float2 parallax_texcoord)
//   { parallax_texcoord = texcoord; }

fn calc_parallax_off_ps(
    texcoord: vec2<f32>,
    view_dir: vec3<f32>,
    parallax_texcoord: ptr<function, vec2<f32>>,
) {
    let _u_view = view_dir;
    *parallax_texcoord = texcoord;
}

fn calc_parallax(texcoord: vec2<f32>, view_dir: vec3<f32>, parallax_texcoord: ptr<function, vec2<f32>>) {
    calc_parallax_off_ps(texcoord, view_dir, parallax_texcoord);
}
