// Port of `parallax_fx.hlsl` — `interpolated` variant (line 47).
//
// Two-tap parallax with linear interpolation between heightmap samples.
// HLSL:
//   void calc_parallax_interpolated_ps(
//       in float2 texcoord, in float3 view_dir,
//       out float2 parallax_texcoord)
//   {
//       texcoord = transform_texcoord(texcoord, height_map_xform);
//       float cur_height = 0;
//       float h1 = (sample(height_map, texcoord).g - 0.5) * height_scale;
//       float diff = h1 - cur_height;
//       float2 step_offset = diff * view_dir.xy;
//       parallax_texcoord = texcoord + step_offset;
//       cur_height = diff * view_dir.z;
//       float h2 = (sample(height_map, parallax_texcoord).g - 0.5) * height_scale;
//       diff = h2 - cur_height;
//       if (sign(diff) != sign(h1 - cur_height)) {
//           float pct = h1 / (cur_height - h2 + h1);
//           parallax_texcoord = texcoord + pct * step_offset;
//       } else {
//           parallax_texcoord = parallax_texcoord + diff * view_dir.xy;
//       }
//       parallax_texcoord = (parallax_texcoord - height_map_xform.zw) / height_map_xform.xy;
//   }

fn calc_parallax_interpolated_ps(
    texcoord: vec2<f32>,
    view_dir: vec3<f32>,
    parallax_texcoord: ptr<function, vec2<f32>>,
) {
    let xformed = transform_texcoord(texcoord, material.height_map_xform);

    // First tap at the unperturbed UV.
    var cur_height: f32 = 0.0;
    let h1 = (textureSample(height_map, height_map_sampler, xformed).g - 0.5) * material.height_scale.x;
    var diff = h1 - cur_height;
    let step_offset = diff * view_dir.xy;
    var p = xformed + step_offset;
    cur_height = diff * view_dir.z;

    // Second tap after stepping along view_dir.
    let h2 = (textureSample(height_map, height_map_sampler, p).g - 0.5) * material.height_scale.x;
    diff = h2 - cur_height;

    // HLSL: `if (sign(diff) != sign(h1 - cur_height))`. Note `h1 - cur_height`
    // here re-computes against the (now updated) cur_height — equivalent to
    // the original height_difference (h1 - 0) only on first iter; on this
    // line cur_height has been overwritten, so the test is "did we cross
    // zero between samples?".
    let h1_minus_cur = h1 - cur_height;
    if (sign(diff) != sign(h1_minus_cur)) {
        // Engine `parallax_fx.hlsl:67` is `float pct = h1 / (cur_height
        // - h2 + h1);` — raw division. Works on Xenon/DX9 where NaN
        // semantics differ, but at grazing camera angles our WGSL/Metal
        // path can hit `denom ≈ 0` (smooth surface, h2 ≈ cur_height +
        // h1) which propagates NaN through to the texcoord and wraps
        // the bitmap sampler. Clamp to a tiny epsilon to keep the
        // parallax sample bounded; difference vs engine is imperceptible
        // when denom isn't degenerate.
        let denom = cur_height - h2 + h1;
        let pct = h1 / select(denom, 1e-6, abs(denom) < 1e-6);
        p = xformed + pct * step_offset;
    } else {
        p = p + diff * view_dir.xy;
    }

    *parallax_texcoord = (p - material.height_map_xform.zw) / material.height_map_xform.xy;
}

fn calc_parallax(texcoord: vec2<f32>, view_dir: vec3<f32>, parallax_texcoord: ptr<function, vec2<f32>>) {
    calc_parallax_interpolated_ps(texcoord, view_dir, parallax_texcoord);
}
