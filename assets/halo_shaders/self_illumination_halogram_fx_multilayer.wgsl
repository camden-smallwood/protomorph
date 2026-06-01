// Port of self_illumination_halogram_fx.hlsl:14-47 — multilayer variant.
//
// Parallax depth-layered self-illumination: samples self_illum_map
// `layers_of_4 × 4` times along the view direction, exponentially
// decaying by depth_darken per sample. Final result is
// `pow(accum, layer_contrast) * self_illum_color.rgb * self_illum_intensity`.
//
// HLSL:
//     texcoord = transform_texcoord(texcoord, self_illum_map_xform);
//     float2 offset = view_dir.xy * self_illum_map_xform.xy
//                   * float2(texcoord_aspect_ratio, 1.0)
//                   * layer_depth / (layers_of_4 * 4);
//
//     float4 accum = 0; float depth_intensity = 1;
//     for (int x = 0; x < layers_of_4; x++) {
//         // 4-tap unrolled inner loop:
//         accum += depth_intensity * sample(self_illum_map, texcoord);
//         texcoord -= offset; depth_intensity *= depth_darken;
//         // ... ×3 more
//     }
//     accum /= (layers_of_4 * 4);
//     result.rgb = pow(accum.rgb, layer_contrast)
//                * self_illum_color * self_illum_intensity;
//
// `albedo` is `inout` in HLSL but never written to in this function —
// pure unused parameter. Mirror as `_albedo` reference.

fn calc_self_illumination_multilayer_ps(
    texcoord_in: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let _u_albedo = albedo;

    var texcoord = transform_texcoord(texcoord_in, material.self_illum_map_xform);

    // `layers_of_4 = 0` would divide-by-zero; clamp to 1 minimum.
    // Engine authors typically set 1-4 (= 4-16 total samples).
    let layers = max(material.layers_of_4.x, 1.0);
    let total_samples = layers * 4.0;

    let offset = view_dir.xy
        * material.self_illum_map_xform.xy
        * vec2<f32>(material.texcoord_aspect_ratio.x, 1.0)
        * material.layer_depth.x
        / total_samples;

    var accum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var depth_intensity = 1.0;

    let layer_count = i32(layers);
    for (var x: i32 = 0; x < layer_count; x = x + 1) {
        // 4-tap unrolled inner — matches HLSL lines 31-38.
        accum = accum + depth_intensity * textureSample(self_illum_map, self_illum_map_sampler, texcoord);
        texcoord = texcoord - offset;
        depth_intensity = depth_intensity * material.depth_darken.x;

        accum = accum + depth_intensity * textureSample(self_illum_map, self_illum_map_sampler, texcoord);
        texcoord = texcoord - offset;
        depth_intensity = depth_intensity * material.depth_darken.x;

        accum = accum + depth_intensity * textureSample(self_illum_map, self_illum_map_sampler, texcoord);
        texcoord = texcoord - offset;
        depth_intensity = depth_intensity * material.depth_darken.x;

        accum = accum + depth_intensity * textureSample(self_illum_map, self_illum_map_sampler, texcoord);
        texcoord = texcoord - offset;
        depth_intensity = depth_intensity * material.depth_darken.x;
    }

    accum = accum / total_samples;

    let rgb = pow(accum.rgb, vec3<f32>(material.layer_contrast.x))
        * material.self_illum_color.rgb
        * material.self_illum_intensity.x;

    return rgb;
}

fn calc_self_illumination(
    texcoord: vec2<f32>,
    albedo: ptr<function, vec3<f32>>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    return calc_self_illumination_multilayer_ps(texcoord, albedo, view_dir);
}
