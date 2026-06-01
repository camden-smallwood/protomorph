// `albedo_fx.hlsl::calc_albedo_chameleon_masked_ps` — STAND-IN.
//
// Variant of `chameleon` that gates the color-shift effect by a mask
// (`chameleon_mask_map`). Mask=1 → full chameleon paint shift;
// mask=0 → straight `base × detail` (no chameleon tint). Same caveat
// as `albedo_fx_chameleon.wgsl` — no engine HLSL function body is
// present in the extracted MCC corpus; approximation uses the bumped
// normal's world-up alignment in place of the engine's N·V fresnel.

fn calc_albedo_chameleon_masked_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_misc = misc;
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let detail = textureSample(detail_map, detail_map_sampler, transform_texcoord(texcoord, material.detail_map_xform));
    let mask_sample = textureSample(chameleon_mask_map, chameleon_mask_map_sampler, texcoord).r;

    // Stand-in fresnel — see `albedo_fx_chameleon.wgsl` for caveats.
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

    // Mask gates the chameleon shift — outside the mask, paint is
    // the unaltered base × detail product (white tint).
    let tint = mix(vec3<f32>(1.0), chameleon_color, mask_sample);

    (*albedo).r = base.r * (detail.r * DETAIL_MULTIPLIER) * tint.r;
    (*albedo).g = base.g * (detail.g * DETAIL_MULTIPLIER) * tint.g;
    (*albedo).b = base.b * (detail.b * DETAIL_MULTIPLIER) * tint.b;
    (*albedo).a = base.a * detail.a;
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_chameleon_masked_ps(texcoord, albedo, normal, misc);
}
