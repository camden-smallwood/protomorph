// `albedo_fx.hlsl::calc_albedo_chameleon_masked_ps` — engine-faithful port.
//
// Verbatim from Halo Online `albedo.fx`. Variant of `chameleon` that
// gates the paint shift by a mask (`chameleon_mask_map`): mask=1 → full
// chameleon, mask=0 → white tint (straight `base × detail`). Now uses
// the engine's true camera-angle factor (see `calc_chameleon`), not the
// old `1 - n.z` stand-in. `view_dir` (world-space) arrives via `misc.xyz`.

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

fn calc_albedo_chameleon_masked_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));
    let mask = textureSample(chameleon_mask_map, chameleon_mask_map_sampler, texcoord).r;

    // color = lerp(1.0, calc_chameleon(N, V), mask)
    let color = mix(vec3<f32>(1.0), calc_chameleon(normalize(normal), misc.xyz), mask);

    (*albedo).r = base.r * (detail.r * DETAIL_MULTIPLIER) * color.r;
    (*albedo).g = base.g * (detail.g * DETAIL_MULTIPLIER) * color.g;
    (*albedo).b = base.b * (detail.b * DETAIL_MULTIPLIER) * color.b;
    (*albedo).a = base.a * detail.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_chameleon_masked_ps(texcoord, albedo, normal, misc);
}
