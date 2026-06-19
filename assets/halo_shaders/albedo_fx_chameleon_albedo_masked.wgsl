// `albedo_fx.hlsl::calc_albedo_chameleon_albedo_masked_ps` — engine-faithful port.
//
// Verbatim from Halo Online `albedo.fx`. Third chameleon variant; rmop
// params:
//   base_map / albedo_color           — primary base (tinted, rgba)
//   base_masked_map / albedo_masked_color — secondary base (tinted, rgba)
//   chameleon_mask_map                — selects base vs base_masked
//   chameleon_color0..3, offsets, fresnel_power — view-angle shift
//
// Engine structure (was previously mis-blended): the chameleon shift is
// applied to ONLY the masked base, then the two bases are lerp'd by the
// mask — NOT applied to the combined diffuse:
//   base        = sample(base_map)        * albedo_color
//   base_masked = sample(base_masked_map) * albedo_masked_color
//   base_masked.rgb *= calc_chameleon(N, V)
//   albedo = lerp(base, base_masked, mask)
// `view_dir` (world-space) arrives via `misc.xyz`; angle factor is the
// engine's N·V (not the old `1 - n.z` stand-in).

// `calc_chameleon` (albedo.fx) — verbatim; see albedo_fx_chameleon.wgsl.
fn calc_chameleon(normal: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let dp = pow(max(dot(normal, view_dir), 0.0), material.chameleon_fresnel_power.x);
    let off1 = material.chameleon_color_offset1.x;
    let off2 = material.chameleon_color_offset2.x;

    var col0 = material.chameleon_color0.rgb;
    var col1 = material.chameleon_color1.rgb;
    var lrp = dp * (1.0 / off1);
    if (dp > off1) {
        col0 = material.chameleon_color1.rgb;
        col1 = material.chameleon_color2.rgb;
        lrp = (dp - off1) * (1.0 / (off2 - off1));
    }
    if (dp > off2) {
        col0 = material.chameleon_color2.rgb;
        col1 = material.chameleon_color3.rgb;
        lrp = (dp - off2) * (1.0 / (1.0 - off2));
    }
    return mix(col0, col1, lrp);
}

fn calc_albedo_chameleon_albedo_masked_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform)) * material.albedo_color;
    let base_masked = textureSample(base_masked_map, base_masked_map_sampler, transform_texcoord(texcoord, material.base_masked_map_xform)) * material.albedo_masked_color;
    // HLSL albedo_fx.hlsl:386 — mask sampled through transform_texcoord.
    let mask = textureSample(chameleon_mask_map, chameleon_mask_map_sampler,
        transform_texcoord(texcoord, material.chameleon_mask_map_xform)).r;

    // Chameleon tints ONLY the masked base, then lerp the two bases by mask.
    // HLSL passes the raw (already-unit) normal — no normalize.
    let masked_rgb = base_masked.rgb * calc_chameleon(normal, misc.xyz);
    (*albedo) = mix(base, vec4<f32>(masked_rgb, base_masked.a), mask);
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_chameleon_albedo_masked_ps(texcoord, albedo, normal, misc);
}
