// `albedo_fx.hlsl::calc_albedo_chameleon_albedo_masked_ps` — STAND-IN.
//
// Third chameleon variant. Distinct rmop param set vs `chameleon` /
// `chameleon_masked`:
//   base_map         — primary base
//   albedo_color     — tint for base_map sample
//   base_masked_map  — secondary base (used where mask=1)
//   albedo_masked_color — tint for base_masked_map sample
//   chameleon_mask_map  — gates the chameleon paint shift
//   chameleon_color0..3, offsets, fresnel_power — view-angle shift
//
// Effect: blend two diffuse textures (base + base_masked) by the
// chameleon mask, then apply the chameleon color shift on top.
//
// Same caveats as `albedo_fx_chameleon.wgsl` — no engine HLSL function
// body is in the extracted MCC corpus; approximation uses bumped
// normal's world-up alignment in place of N·V fresnel.

fn calc_albedo_chameleon_albedo_masked_ps(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec4<f32>>,
    normal: vec3<f32>,
    misc: vec4<f32>,
) {
    let _u_misc = misc;
    let base = textureSample(base_map, base_map_sampler, transform_texcoord(texcoord, material.base_map_xform));
    let base_masked = textureSample(base_masked_map, base_masked_map_sampler, transform_texcoord(texcoord, material.base_masked_map_xform));
    let mask_sample = textureSample(chameleon_mask_map, chameleon_mask_map_sampler, texcoord).r;

    // Blend base_map vs base_masked_map by the mask, each tinted.
    let base_tinted = base.rgb * material.albedo_color.rgb;
    let masked_tinted = base_masked.rgb * material.albedo_masked_color.rgb;
    let diffuse_rgb = mix(base_tinted, masked_tinted, mask_sample);

    // Stand-in fresnel — see `albedo_fx_chameleon.wgsl`.
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

    // Chameleon shift gated by mask: outside mask = unaltered diffuse.
    let final_tint = mix(vec3<f32>(1.0), chameleon_color, mask_sample);

    (*albedo).r = diffuse_rgb.r * final_tint.r;
    (*albedo).g = diffuse_rgb.g * final_tint.g;
    (*albedo).b = diffuse_rgb.b * final_tint.b;
    // Alpha: linear blend follows the diffuse choice.
    (*albedo).a = mix(base.a * material.albedo_color.a, base_masked.a * material.albedo_masked_color.a, mask_sample);
}

fn calc_albedo(texcoord: vec2<f32>, albedo: ptr<function, vec4<f32>>, normal: vec3<f32>, misc: vec4<f32>) {
    calc_albedo_chameleon_albedo_masked_ps(texcoord, albedo, normal, misc);
}
