// `warp` category — `from_texture` option. Port of warp_fx.hlsl:
//
//   void calc_warp_from_texture_ps(texcoord, view_dir, out parallax_texcoord):
//     parallax_texcoord = texcoord
//       + sample(warp_map, xform(texcoord, warp_map_xform)).xy
//         * float2(warp_amount_x, warp_amount_y);
//
// warp and parallax are mutually exclusive producers of the working
// texcoord — the engine #defines `calc_parallax_ps` to the warp function
// when the warp category is set. So this defines `calc_parallax` (the
// slot the entry point calls); the assembler emits it INSTEAD of the
// parallax fragment when warp != none. `view_dir` is unused by this
// option (matches the HLSL).
fn calc_warp_from_texture_ps(
    texcoord: vec2<f32>,
    view_dir: vec3<f32>,
    parallax_texcoord: ptr<function, vec2<f32>>,
) {
    let _u_view = view_dir;
    let w = textureSample(
        warp_map, warp_map_sampler,
        transform_texcoord(texcoord, material.warp_map_xform),
    ).xy;
    *parallax_texcoord = texcoord
        + w * vec2<f32>(material.warp_amount_x.x, material.warp_amount_y.x);
}

fn calc_parallax(
    texcoord: vec2<f32>,
    view_dir: vec3<f32>,
    parallax_texcoord: ptr<function, vec2<f32>>,
) {
    calc_warp_from_texture_ps(texcoord, view_dir, parallax_texcoord);
}
