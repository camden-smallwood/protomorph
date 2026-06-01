// Shared math + texture helpers used by multiple shader fragments.
// Mirrors snippets from `utilities_fx.hlsl`, `texture_xform_fx.hlsl`,
// and `bump_mapping_fx.hlsl`.

// `transform_texcoord(uv, xform)` — multiply by xform.xy (scale),
// add xform.zw (offset). Mirrors Halo's most common UV transform.
// Phase 2A: identity transform (xform = vec4(1, 1, 0, 0)) is the only
// one we emit; rotation variants come later.
fn transform_texcoord(uv: vec2<f32>, xform: vec4<f32>) -> vec2<f32> {
    return uv * xform.xy + xform.zw;
}

// `sample_bumpmap(tex, tex_sampler, uv)` — sample a tangent-space normal map and
// reconstruct Z from XY. Halo BC5/DXN stores X+Y as SNORM; the texture
// is bound as `Bc5RgSnorm`, so wgpu hands us .rg already in [-1, 1].
// We reconstruct Z = sqrt(1 - X² - Y²), matching Halo's HLSL behavior.
fn sample_bumpmap(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> vec3<f32> {
    let xy = textureSample(tex, samp, uv).xy;
    let xy_sq = xy.x * xy.x + xy.y * xy.y;
    let z = sqrt(max(0.0, 1.0 - xy_sq));
    return normalize(vec3<f32>(xy.x, xy.y, z));
}

// sRGB gamma-correction multiplier for detail maps (2^2.2). Mirrors
// Halo's `DETAIL_MULTIPLIER` constant in albedo_fx.hlsl.
const DETAIL_MULTIPLIER: f32 = 4.59479;
