// Port of `lightmap_sampling_fx.hlsl` — atlas decode and SH packing
// shared by every entry point that consumes the per-pixel lightmap
// (`entry_terrain_static_sh`, `entry_static_sh`, ...). Mirrors the
// HLSL file 1:1 in name and structure (per
// `feedback_wgsl_must_mirror_hlsl.md`).
//
// `lightprobe_atlas`, `lightprobe_intensity_atlas`, `lightmap_compress`
// and `lightmap_atlas_sampler` come from `engine_bindings.wgsl`.

// `decode_bpp16_luvw` (lightmap_sampling_fx.hlsl:15-23). Reconstructs
// one SH coefficient (vec3) from a pair of BC3 atlas samples that
// encode (L, direction) halves — alpha = L scale, RGB = direction.
fn decode_bpp16_luvw(s0: vec4<f32>, s1: vec4<f32>, l_range: f32) -> vec3<f32> {
    let l = s0.a * s1.a * l_range;
    let uvw = s0.xyz + s1.xyz;
    return (uvw * 2.0 - 2.0) * l;
}

// SH coefficients + dominant light, the result of one
// `sample_lightprobe_texture` call. Bundled as a struct because WGSL
// can't return multiple `out` params.
struct LightprobeSample {
    sh: array<vec3<f32>, 4>,
    dir: vec3<f32>,
    intensity: vec3<f32>,
}

// Port of `sample_lightprobe_texture` (lightmap_sampling_fx.hlsl:25),
// order-2 path. Riverworld's atlas has 8 slices for 4 SH coefs +
// 2 slices on the intensity atlas for dominant light.
//   sh_coefficients[0]: DC (L=0, m=0)
//   sh_coefficients[1]: L=1, m=-1 (y axis)
//   sh_coefficients[2]: L=1, m=0  (z axis)
//   sh_coefficients[3]: L=1, m=+1 (x axis)
fn sample_lightprobe_texture(uv: vec2<f32>) -> LightprobeSample {
    // Halo binds these atlases with `_sampler_filter_lightprobe_texture_array`
    // which dllcache's d3d11_sampler_state_cache::get maps to
    // D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT — i.e. bilinear with point mip.
    // (Atlases are mip-less so mip behavior is moot.)
    let s0 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 0);
    let s1 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 1);
    let s2 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 2);
    let s3 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 3);
    let s4 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 4);
    let s5 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 5);
    let s6 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 6);
    let s7 = textureSample(lightprobe_atlas, lightmap_atlas_sampler, uv, 7);
    let i0 = textureSample(lightprobe_intensity_atlas, lightmap_atlas_sampler, uv, 0);
    let i1 = textureSample(lightprobe_intensity_atlas, lightmap_atlas_sampler, uv, 1);

    // Compression-scalar layout matches dllcache's `submit_extern_texture`
    // case 11 pack (3-per-vec4) AND the HLSL accessors in
    // `lightmap_sampling_fx.hlsl:64-68`:
    //   sh[0]    ← constant_0.x
    //   sh[1]    ← constant_0.y
    //   sh[2]    ← constant_0.z
    //   sh[3]    ← constant_1.x
    //   intensity ← constant_1.y
    var out: LightprobeSample;
    out.sh[0]    = decode_bpp16_luvw(s0, s1, lightmap_compress.constant_0.x);
    out.sh[1]    = decode_bpp16_luvw(s2, s3, lightmap_compress.constant_0.y);
    out.sh[2]    = decode_bpp16_luvw(s4, s5, lightmap_compress.constant_0.z);
    out.sh[3]    = decode_bpp16_luvw(s6, s7, lightmap_compress.constant_1.x);
    out.intensity = decode_bpp16_luvw(i0, i1, lightmap_compress.constant_1.y);

    // Extract dominant light direction from the SH L1 coefficients
    // (lightmap_sampling_fx.hlsl:65-68). ITU-R BT.709 luminance weights.
    let dr = vec3<f32>(-out.sh[3].r, -out.sh[1].r, out.sh[2].r);
    let dg = vec3<f32>(-out.sh[3].g, -out.sh[1].g, out.sh[2].g);
    let db = vec3<f32>(-out.sh[3].b, -out.sh[1].b, out.sh[2].b);
    let dir = dr * 0.212656 + dg * 0.715158 + db * 0.0721856;
    let len = length(dir);
    out.dir = select(vec3<f32>(0.0, 0.0, 1.0), dir / len, len > 1e-6);
    return out;
}

// `pack_constants_texture_array_linear` — port of
// spherical_harmonics_fx.hlsl:55. Packs the 4 sh_coef vec3s (DC + 3 L1)
// into the 4-vec4 ORDER-2 layout consumed by
// `ravi_order_2_with_dominant_light` and the linear umbrella. Halo's
// basis convention negates the z component.
//   [0] = (sh[0].rgb, 0)
//   [1] = (sh[3].r, sh[1].r, -sh[2].r, 0)
//   [2] = (sh[3].g, sh[1].g, -sh[2].g, 0)
//   [3] = (sh[3].b, sh[1].b, -sh[2].b, 0)
fn pack_constants_texture_array_linear(probe: LightprobeSample) -> array<vec4<f32>, 4> {
    let r0 = probe.sh[0];
    let r1 = probe.sh[1];
    let r2 = probe.sh[2];
    let r3 = probe.sh[3];
    return array<vec4<f32>, 4>(
        vec4<f32>(r0.r, r0.g, r0.b, 0.0),
        vec4<f32>(r3.r, r1.r, -r2.r, 0.0),
        vec4<f32>(r3.g, r1.g, -r2.g, 0.0),
        vec4<f32>(r3.b, r1.b, -r2.b, 0.0),
    );
}

// Port of `pack_constants` (spherical_harmonics_fx.hlsl:14, order-2 path).
// Takes the 4 sh_coef vec3s and packs into the 10-vec4 ravi cbuffer the
// existing `ravi_order_3` consumes. L2 portion (cbuf[4..9]) stays zero.
fn pack_lightprobe_to_ravi(probe: LightprobeSample) -> array<vec4<f32>, 10> {
    let r0 = probe.sh[0];
    let r1 = probe.sh[1];
    let r2 = probe.sh[2];
    let r3 = probe.sh[3];
    return array<vec4<f32>, 10>(
        vec4<f32>(r0.r, r0.g, r0.b, 0.0),                 // [0] DC, RGB packed
        vec4<f32>(r3.r, r1.r, -r2.r, 0.0),                // [1] L1 R (x, y, -z)
        vec4<f32>(r3.g, r1.g, -r2.g, 0.0),                // [2] L1 G
        vec4<f32>(r3.b, r1.b, -r2.b, 0.0),                // [3] L1 B
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),   // [4..6] L2 first half — unused at order-2
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),   // [7..9] L2 diag — unused
    );
}
