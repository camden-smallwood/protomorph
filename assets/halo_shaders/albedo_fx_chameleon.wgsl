// `albedo_fx.hlsl::calc_albedo_chameleon_ps` — STAND-IN.
//
// The chameleon rmop (`shaders/shader_options/albedo_chameleon`)
// declares: base_map, detail_map, chameleon_color0..3 (4 colors),
// chameleon_color_offset1, chameleon_color_offset2 (2 piecewise
// positions in [0,1]), chameleon_fresnel_power. No corresponding
// `calc_albedo_chameleon_ps` body exists in the MCC HLSL corpus we
// extracted — Bungie either inlined the body or it was added in a
// later MCC update that the extractor missed. This file is a
// **best-effort port**, not engine-verbatim.
//
// The engine effect is view-angle-dependent paint shift between 4
// colors (used by Master Chief's MP armor). Without an authoritative
// HLSL anchor, we approximate the angle factor via the bumped surface
// normal's alignment with world-up: `factor = pow(1 - n.z, fresnel_power)`.
// Bumped normal varies per pixel so the color shift is visible across
// curved surfaces; it is NOT camera-angle correct (engine uses N·V).
//
// Color blend: piecewise lerp across the 4 colors with the 2 offsets:
//   factor ∈ [0, offset1)  → lerp(color0, color1)
//   factor ∈ [offset1, offset2) → lerp(color1, color2)
//   factor ∈ [offset2, 1]  → lerp(color2, color3)
//
// Final: `albedo = base.rgb * (detail.rgb * DETAIL_MULTIPLIER) * chameleon_color`.
//
// When the engine chameleon HLSL surfaces or someone reverse-engineers
// it, swap this body for the verbatim port.

fn calc_albedo_chameleon_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_misc = misc;
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));

    // Stand-in fresnel factor — engine would use 1 - dot(N, V). Without
    // view direction in this function's signature, approximate via
    // 1 - n.z (alignment with world-up axis).
    let n = normalize(normal);
    let cos_factor = saturate(1.0 - abs(n.z));
    let power = max(material.chameleon_fresnel_power.x, 1.0e-3);
    let factor = pow(cos_factor, power);

    let off1 = clamp(material.chameleon_color_offset1.x, 0.0, 1.0);
    let off2 = clamp(material.chameleon_color_offset2.x, off1, 1.0);

    var chameleon_color: vec3<f32>;
    if (factor < off1) {
        let t = factor / max(off1, 1.0e-3);
        chameleon_color = mix(material.chameleon_color0.rgb, material.chameleon_color1.rgb, t);
    } else if (factor < off2) {
        let t = (factor - off1) / max(off2 - off1, 1.0e-3);
        chameleon_color = mix(material.chameleon_color1.rgb, material.chameleon_color2.rgb, t);
    } else {
        let t = (factor - off2) / max(1.0 - off2, 1.0e-3);
        chameleon_color = mix(material.chameleon_color2.rgb, material.chameleon_color3.rgb, t);
    }

    (*albedo).r = base.r * (detail.r * DETAIL_MULTIPLIER) * chameleon_color.r;
    (*albedo).g = base.g * (detail.g * DETAIL_MULTIPLIER) * chameleon_color.g;
    (*albedo).b = base.b * (detail.b * DETAIL_MULTIPLIER) * chameleon_color.b;
    (*albedo).a = base.a * detail.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_chameleon_ps(texcoord, albedo, normal, misc);
}
