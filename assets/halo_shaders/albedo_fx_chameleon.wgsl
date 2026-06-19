// `albedo_fx.hlsl::calc_albedo_chameleon_ps` — engine-faithful port.
//
// Ported verbatim from the Halo Online shader source
// (`ShaderGenerator/halo_online_shaders/albedo.fx`), the authoritative
// source these s3d/community shaders were authored against. The MCC
// `gpix` corpus omits the body (compiled inline), but the HO source
// matched our prior reverse-engineering everywhere except the angle
// factor: the engine uses true camera-angle `pow(max(N·V, 0), power)`,
// NOT the old `1 - n.z` world-up stand-in (which shifted with surface
// curvature but not with the camera — visibly wrong for the MP-armor
// paint effect).
//
// The chameleon rmop (`shaders/shader_options/albedo_chameleon`)
// declares: base_map, detail_map, chameleon_color0..3 (4 colors),
// chameleon_color_offset1/2 (piecewise positions in [0,1]),
// chameleon_fresnel_power.
//
// `view_dir` (world-space fragment→camera) is threaded in via `misc.xyz`
// from the entry point: the engine passes it as a separate `calc_albedo_*_ps`
// argument; protomorph's reduced `calc_albedo` interface reuses the
// otherwise-unused `misc` channel for it.

// `calc_chameleon` (albedo.fx) — verbatim. dp = pow(max(N·V, 0), power)
// is the camera-angle factor; piecewise lerp across the 4 colors at
// offset1/offset2. Shared by all chameleon albedo variants.
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

fn calc_albedo_chameleon_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));

    // HLSL passes the raw (already-unit) normal — no normalize.
    let color = calc_chameleon(normal, misc.xyz);

    (*albedo).r = base.r * (detail.r * DETAIL_MULTIPLIER) * color.r;
    (*albedo).g = base.g * (detail.g * DETAIL_MULTIPLIER) * color.g;
    (*albedo).b = base.b * (detail.b * DETAIL_MULTIPLIER) * color.b;
    (*albedo).a = base.a * detail.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_chameleon_ps(texcoord, albedo, normal, misc);
}
