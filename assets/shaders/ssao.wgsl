// Port of `ssao_hlsl.hlsl` `default_ps` (engine shader 95).
//
// 8-tap kernel SSAO. Reads:
//   - depth (slot 0) — linear eye-space distance in world units
//   - normal MRT (slot 1) — world-space normal encoded as `(n*0.5+0.5, 1)`
//   - 4×4 noise (slot 2) — random rotation table from `InitTextureSSAONoise`
//
// Per-pixel:
//   1. Skip if depth > 2000 world units (sky, very far geometry).
//   2. Decode normal, transform world → view via `PS_REG_SSAO_MV_*`.
//   3. Sample 8 random-rotated kernel offsets in view-space, project
//      to screen, sample depth at offset, compute "is this sample
//      occluded by something closer?" via depth-ratio test.
//   4. Soft-blend valid samples; output AO factor `lerp(1, ao², intensity)`.
//
// The compile-time depth linearization comes from outside — we sample
// the raw depth buffer and convert via `depth_constants` (1/z_near or
// 1/z_far depending on format). For our reverse-Z infinite far, raw
// depth d maps to eye-distance via `eye_d = 1 / (d * z_near_inv)`.
//
// Cbuffer layout (engine slot → field):
//   0x4E0000 (VS) TEXCOORD_SCALE  = (w/4, h/4, 0, 0)
//   0x4E0001 (VS) VS_FRUSTUM_SCALE
//   0x4F0000 (PS) SSAO_PARAMS     = (radius, 16.0, 1/sample_z_threshold, intensity)
//   0x4F0001 (PS) PS_FRUSTUM_SCALE = (tan(fov/2)*aspect*2, tan(fov/2)*2, 1/x, 1/y)
//   0x4F0002 (PS) PS_REG_SSAO_MV_1/2/3 — 3 rows of view matrix (world→view)

struct SsaoParams {
    /// `(w/4, h/4, 0, 0)` of accum_HDR — for noise sampling.
    texcoord_scale: vec4<f32>,
    /// `(tan(fov/2)*aspect*2, tan(fov/2)*2, 1/x, 1/y)`. The .xy form
    /// the view-frustum corner; .zw is `perspK = 1/(2*tan(fov/2)*aspect)` etc.
    frustum_scale: vec4<f32>,
    /// `(radius, 16, 1/sample_z_threshold, intensity)`.
    ssao_params: vec4<f32>,
    /// 3 rows of view matrix for world→view normal transform.
    mv_row_0: vec4<f32>,
    mv_row_1: vec4<f32>,
    mv_row_2: vec4<f32>,
    /// `(1/z_near, ?, ?, ?)` — for linearizing raw depth to eye distance.
    depth_constants: vec4<f32>,
}

// Depth surface is a Depth32Float texture — declare as
// `texture_depth_2d` to match the bind-group layout's `Depth` sample
// type (validation fails on `texture_2d<f32>` for depth views).
// `textureSample` on a depth texture returns f32 (the depth value)
// instead of vec4 — call-sites are adjusted to drop the `.r` swizzle.
@group(0) @binding(0) var depth_sampler_tex: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var normal_sampler_tex: texture_2d<f32>;
@group(0) @binding(3) var normal_sampler: sampler;
@group(0) @binding(4) var noise_sampler_tex: texture_2d<f32>;
@group(0) @binding(5) var noise_sampler: sampler;
@group(0) @binding(6) var<uniform> u: SsaoParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tex_coord: vec4<f32>,    // .xy = full-res UV, .zw = noise UV
    @location(1) view_vec: vec3<f32>,     // unprojected view ray for this pixel
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0),
    );
    let uv = uvs[idx];
    var out: VsOut;
    out.clip = vec4<f32>(positions[idx], 0.0, 1.0);
    // Engine VS: viewVec = (uv - 0.5, 1.0) * (frustum_scale.xy, 1.0).
    // The frustum_scale.xy passed to VS is the same as ssao_params for
    // PS — we use only the PS-side frustum here. Since we collapse VS
    // and PS frustum_scale into a single uniform, use it consistently.
    out.view_vec = vec3<f32>((uv.x - 0.5) * u.frustum_scale.x,
                              (uv.y - 0.5) * u.frustum_scale.y,
                              1.0);
    out.tex_coord = vec4<f32>(uv.x, uv.y, uv.x * u.texcoord_scale.x, uv.y * u.texcoord_scale.y);
    return out;
}

/// Linearize raw depth-buffer depth (reverse-Z, infinite far) into
/// eye-space distance in world units. With `Mat4::perspective_infinite_reverse_rh`,
/// raw_d = z_near / world_z, so world_z = z_near / raw_d.
fn linearize_depth(raw: f32) -> f32 {
    let z_near_inv = u.depth_constants.x;
    if (raw <= 0.0) {
        return 1.0e6;  // far / sky
    }
    return 1.0 / (raw * z_near_inv);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let raw_depth = textureSample(depth_sampler_tex, depth_sampler, in.tex_coord.xy);
    let lin_depth = linearize_depth(raw_depth);

    // Engine line 56-58: skip far pixels (sky etc.).
    if (lin_depth > 2000.0) {
        return vec4<f32>(1.0);
    }

    // 8-tap kernel directions (engine lines 61-71). Pre-normalized,
    // pre-scaled radii.
    let kernel0 = normalize(vec3<f32>( 1.0, -1.0, 0.5)) * 0.10;
    let kernel1 = normalize(vec3<f32>(-1.0, -1.0, 0.5)) * 0.09;
    let kernel2 = normalize(vec3<f32>( 1.0,  1.0, 0.5)) * 0.08;
    let kernel3 = normalize(vec3<f32>(-1.0,  1.0, 0.5)) * 0.07;
    let kernel4 = normalize(vec3<f32>(-1.0,  0.0, 1.0)) * 0.06;
    let kernel5 = normalize(vec3<f32>( 1.0,  0.0, 1.0)) * 0.05;
    let kernel6 = normalize(vec3<f32>( 0.0, -1.0, 1.0)) * 0.045;
    let kernel7 = normalize(vec3<f32>( 0.0,  1.0, 1.0)) * 0.04;

    // Decode + transform normal to view-space (engine lines 74-83).
    var n2 = textureSample(normal_sampler_tex, normal_sampler, in.tex_coord.xy).xyz;
    n2 = n2 * 2.0 - vec3<f32>(1.0);
    let n_view = vec3<f32>(
        dot(n2, u.mv_row_0.xyz),
        dot(n2, u.mv_row_1.xyz),
        dot(n2, u.mv_row_2.xyz),
    );

    // Random rotation from noise (engine lines 99-115).
    let rot = textureSample(noise_sampler_tex, noise_sampler, in.tex_coord.zw);

    // Reflect the kernel against view-space normal + per-pixel rotation.
    // Engine: arrKernel[i].xy is `dot(constK.xy, rot.xy/zw)`, then z-bias,
    // then reflect against normalize(N + (0,0,-1)).
    let n_reflect = normalize(n_view + vec3<f32>(0.0, 0.0, -1.0));
    var rotated_kernel: array<vec3<f32>, 8>;
    rotated_kernel[0] = reflect(vec3<f32>(dot(kernel0.xy, rot.xy), dot(kernel0.xy, rot.zw), kernel0.z), n_reflect);
    rotated_kernel[1] = reflect(vec3<f32>(dot(kernel1.xy, rot.xy), dot(kernel1.xy, rot.zw), kernel1.z), n_reflect);
    rotated_kernel[2] = reflect(vec3<f32>(dot(kernel2.xy, rot.xy), dot(kernel2.xy, rot.zw), kernel2.z), n_reflect);
    rotated_kernel[3] = reflect(vec3<f32>(dot(kernel3.xy, rot.xy), dot(kernel3.xy, rot.zw), kernel3.z), n_reflect);
    rotated_kernel[4] = reflect(vec3<f32>(dot(kernel4.xy, rot.xy), dot(kernel4.xy, rot.zw), kernel4.z), n_reflect);
    rotated_kernel[5] = reflect(vec3<f32>(dot(kernel5.xy, rot.xy), dot(kernel5.xy, rot.zw), kernel5.z), n_reflect);
    rotated_kernel[6] = reflect(vec3<f32>(dot(kernel6.xy, rot.xy), dot(kernel6.xy, rot.zw), kernel6.z), n_reflect);
    rotated_kernel[7] = reflect(vec3<f32>(dot(kernel7.xy, rot.xy), dot(kernel7.xy, rot.zw), kernel7.z), n_reflect);

    // Sample-radius scaling (engine lines 121-127).
    var sample_scale = vec3<f32>(u.ssao_params.x);  // base radius
    sample_scale = sample_scale * (1.4 + lin_depth / 10.0);
    sample_scale = sample_scale * saturate(lin_depth * 0.5);

    let depth_range_scale = u.ssao_params.z / sample_scale.z;
    let depth_test_softness = 64.0 / sample_scale.z;

    // Eye position reconstruction (engine line 132).
    let eye_position = lin_depth * in.view_vec / in.view_vec.z;

    // perspK = (frustum_scale.zw) — the 2D camera scale for projecting
    // view-space samples to UV.
    let persp_k = u.frustum_scale.zw;

    var sky_access = vec4<f32>(0.0);
    var sum: f32 = 0.0;

    // Two iterations of 4 samples each (engine lines 145-176).
    for (var iter: i32 = 0; iter < 2; iter = iter + 1) {
        var sample0 = rotated_kernel[iter * 4 + 0] * sample_scale + eye_position;
        sample0.x = sample0.x / sample0.z;
        sample0.y = sample0.y / sample0.z;
        sample0.x = sample0.x * persp_k.x + 0.5;
        sample0.y = sample0.y * persp_k.y + 0.5;
        let d0 = linearize_depth(textureSample(depth_sampler_tex, depth_sampler, sample0.xy)) - sample0.z;

        var sample1 = rotated_kernel[iter * 4 + 1] * sample_scale + eye_position;
        sample1.x = sample1.x / sample1.z;
        sample1.y = sample1.y / sample1.z;
        sample1.x = sample1.x * persp_k.x + 0.5;
        sample1.y = sample1.y * persp_k.y + 0.5;
        let d1 = linearize_depth(textureSample(depth_sampler_tex, depth_sampler, sample1.xy)) - sample1.z;

        var sample2 = rotated_kernel[iter * 4 + 2] * sample_scale + eye_position;
        sample2.x = sample2.x / sample2.z;
        sample2.y = sample2.y / sample2.z;
        sample2.x = sample2.x * persp_k.x + 0.5;
        sample2.y = sample2.y * persp_k.y + 0.5;
        let d2 = linearize_depth(textureSample(depth_sampler_tex, depth_sampler, sample2.xy)) - sample2.z;

        var sample3 = rotated_kernel[iter * 4 + 3] * sample_scale + eye_position;
        sample3.x = sample3.x / sample3.z;
        sample3.y = sample3.y / sample3.z;
        sample3.x = sample3.x * persp_k.x + 0.5;
        sample3.y = sample3.y * persp_k.y + 0.5;
        let d3 = linearize_depth(textureSample(depth_sampler_tex, depth_sampler, sample3.xy)) - sample3.z;

        let v_distance = vec4<f32>(d0, d1, d2, d3);
        let v_distance_scaled = v_distance * depth_range_scale;
        let f_range_is_valid = vec4<f32>(1.0)
            - (saturate(-v_distance_scaled) + saturate(abs(v_distance_scaled))) * 0.5;

        let w = max(f_range_is_valid, vec4<f32>(0.5));
        sky_access = sky_access
            + mix(vec4<f32>(1.0), saturate(v_distance * depth_test_softness + vec4<f32>(1.0)),
                  saturate(f_range_is_valid * 2.0))
              * w;
        sum = sum + dot(w, vec4<f32>(1.0));
    }

    let color = saturate(dot(sky_access, vec4<f32>(1.0 / sum)));
    let ssao = color * color;

    // Intensity falloff (engine lines 182-184).
    let intensity = u.ssao_params.w * saturate(2.0 - lin_depth / 1000.0);
    return vec4<f32>(mix(vec3<f32>(1.0), vec3<f32>(ssao), intensity), 1.0);
}
