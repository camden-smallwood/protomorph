// Engine `decal_fx.hlsl::sample_diffuse` — albedo=vector_alpha branch
// (line 241-262).
//
// Distance-field / "vector" alpha decals. The vector_map's green channel
// stores a signed-distance field (≥0.5 = inside, <0.5 = outside);
// `vector_sharpness` controls the falloff slope, `antialias_tweak`
// scales the screen-space-derivative-based AA factor.
//
// HLSL on PC:
//   color           = sampleBiasGlobal2D(base_map, transform_texcoord(uv, base_map_xform)).rgb;
//   vector_distance = sampleBiasGlobal2D(vector_map, uv).g;
//   scale           = antialias_tweak / 0.001;        // PC fast path
//   scale           = max(scale, 1.0);
//   vector_alpha    = saturate((vector_distance - 0.5) * min(scale, vector_sharpness) + 0.5);
//   return float4(color * vector_alpha, vector_alpha);
//
// The non-PC path uses `getGradients` (xenos-only intrinsic) to derive
// `scale` from per-texel screen-space rate of change; PC bypasses
// (`scale /= 0.001f`) which makes `scale` always saturate against
// `vector_sharpness` in practice. Engine-faithful to mirror the PC
// path here.
//
// rmt2 packs scalar PARAMs as vec4 — `.x` for `antialias_tweak` /
// `vector_sharpness`.

fn sample_diffuse(texcoord_tile: vec2<f32>, texcoord: vec2<f32>, palette_v: f32) -> vec4<f32> {
    let _u_tile = texcoord_tile;
    let _u_palette = palette_v;

    let color = textureSample(base_map, base_map_sampler,
        transform_texcoord(texcoord, material.base_map_xform),
    ).rgb;
    let vector_distance = textureSample(vector_map, vector_map_sampler, texcoord).g;

    // HLSL line 247-249: PC path — `scale = antialias_tweak / 0.001`.
    var scale = material.antialias_tweak.x / 0.001;
    scale = max(scale, 1.0);

    // HLSL line 259: vector_alpha = saturate((d - 0.5) * min(scale, sharp) + 0.5).
    let vector_alpha = saturate(
        (vector_distance - 0.5) * min(scale, material.vector_sharpness.x) + 0.5
    );

    return vec4<f32>(color * vector_alpha, vector_alpha);
}
