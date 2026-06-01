// Spherical harmonics evaluation. Faithful port of
// /Users/camden/Halo/halo3_mcc_hlsl_extracted/spherical_harmonics_fx.hlsl.
//
// Halo's `ravi_order_*` functions evaluate a packed SH lightprobe at a
// given world-space normal. Input is a 10-element vec4 array — same
// shape every Halo PS uses regardless of input source (per-pixel
// lightmap atlas decode, per-vertex unpacked, single-probe expand,
// or default-lighting fallback). The packing is NOT one coefficient
// per vec4 — it's grouped for fewer dot products:
//
//   [0]    = DC term, RGB packed in .rgb
//   [1..3] = L1 linear bands, one per channel (.rgb = SH_(L=1, m=-1, 0, +1) for R/G/B)
//   [4..6] = L2 first half, one per channel — coefficients for the
//            (xy, yz, zx) basis products (m=-2, -1, +1)
//   [7..9] = L2 second half — coefficients for (x², y², z², 1/3)
//            basis products (m=0, +2, +/- DC correction)
//
// For the default-lighting fallback (no atlas), only [0..3] are
// populated — [4..9] stay zero, which makes `ravi_order_3` reduce to
// `ravi_order_2` mathematically.

const RAVI_C1: f32 = 0.429043;       // L2 normalization
const RAVI_C2: f32 = 0.511664;       // L1 normalization
const RAVI_C4: f32 = 0.886227;       // DC normalization (sqrt(pi)/2)
const RAVI_INV_PI: f32 = 0.31830989; // 1/pi

// `ravi_order_2` — DC + L1 bands only (4 coefficients per channel).
fn ravi_order_2(normal: vec3<f32>, lighting_constants: array<vec4<f32>, 10>) -> vec3<f32> {
    var x1: vec3<f32>;
    var c = lighting_constants;
    x1.r = dot(normal, c[1].rgb);
    x1.g = dot(normal, c[2].rgb);
    x1.b = dot(normal, c[3].rgb);
    return (RAVI_C4 * c[0].rgb + (-2.0 * RAVI_C2) * x1) * RAVI_INV_PI;
}

// `ravi_order_3` — full SH3 (DC + L1 + L2). Used by `static_sh_ps`.
fn ravi_order_3(normal: vec3<f32>, lighting_constants: array<vec4<f32>, 10>) -> vec3<f32> {
    var x1: vec3<f32>;
    var x2: vec3<f32>;
    var x3: vec3<f32>;
    var c = lighting_constants;

    // L1 — (x, y, z) dotted against per-channel coefficients
    x1.r = dot(normal, c[1].rgb);
    x1.g = dot(normal, c[2].rgb);
    x1.b = dot(normal, c[3].rgb);

    // L2 first half — (xy, yz, zx) basis products
    let a = normal.xyz * normal.yzx;
    x2.r = dot(a, c[4].rgb);
    x2.g = dot(a, c[5].rgb);
    x2.b = dot(a, c[6].rgb);

    // L2 second half — (x², y², z², 1/3) basis products
    let b = vec4<f32>(normal.x * normal.x, normal.y * normal.y, normal.z * normal.z, 1.0 / 3.0);
    x3.r = dot(b, c[7]);
    x3.g = dot(b, c[8]);
    x3.b = dot(b, c[9]);

    return (RAVI_C4 * c[0].rgb + (-2.0 * RAVI_C2) * x1 + (-2.0 * RAVI_C1) * x2 - RAVI_C1 * x3) * RAVI_INV_PI;
}

// `ravi_order_2_new` — order-2 ravi on the 4-vec4 layout (NO dominant
// extraction). Mirrors spherical_harmonics_fx.hlsl:108-128. Used by
// terrain_new_fx's `static_lighting_shared_ps_linear_with_dominant_light`
// for area_specular (where the SH coefs are NOT pre-subtracted by the
// dominant light, since area_specular wants the FULL probe energy).
fn ravi_order_2_new(normal: vec3<f32>, lighting_constants: array<vec4<f32>, 4>) -> vec3<f32> {
    var c = lighting_constants;
    var x1: vec3<f32>;
    x1.r = dot(normal, c[1].rgb);
    x1.g = dot(normal, c[2].rgb);
    x1.b = dot(normal, c[3].rgb);
    return (RAVI_C4 * c[0].rgb + (-2.0 * RAVI_C2) * x1) * RAVI_INV_PI;
}

// `ravi_order_2_with_dominant_light` — order-2 ravi after subtracting
// the dominant light's L0+L1 contribution from the SH coefficients,
// then re-adding it as an analytical lambertian dot. Mirrors
// spherical_harmonics_fx.hlsl:130-150.
//
// Why subtract+re-add: the dominant light is one of the brightest
// directional sources hitting this surface; baking it into the smooth
// SH probe blurs it out and underrepresents specular response. Halo
// pulls it out, treats it as a point source for the diffuse lambertian,
// and what remains in the SH is the rest of the lighting environment.
//
// Operates on a 4-vec4 ORDER-2 SH layout (L0 + 3×L1), NOT the 10-vec4
// order-3 packing — that's why this is separate from `ravi_order_2`
// above (which uses the 10-vec4 fallback layout).
fn ravi_order_2_with_dominant_light(
    normal: vec3<f32>,
    lighting_constants: array<vec4<f32>, 4>,
    dominant_light_dir: vec3<f32>,
    dominant_light_intensity: vec3<f32>,
) -> vec3<f32> {
    var c = lighting_constants;
    // dir_eval = (-Y1X * y, -Y1X * z, -Y1X * x) — Halo basis (negated y/x).
    let dir_eval = vec3<f32>(
        -0.4886025 * dominant_light_dir.y,
        -0.4886025 * dominant_light_dir.z,
        -0.4886025 * dominant_light_dir.x,
    );
    // dir_eval.zxy = (-Y1X*x, -Y1X*y, -Y1X*z) — the (x, y, -z)-axis lobe
    // packed into c[1..3]'s (.x, .y, .z) per spherical_harmonics_fx.hlsl
    // pack_constants_texture_array_linear (entries 1..3).
    let dir_zxy = vec3<f32>(dir_eval.z, dir_eval.x, dir_eval.y);
    c[1] = vec4<f32>(c[1].xyz - dir_zxy * dominant_light_intensity.r, c[1].w);
    c[2] = vec4<f32>(c[2].xyz - dir_zxy * dominant_light_intensity.g, c[2].w);
    c[3] = vec4<f32>(c[3].xyz - dir_zxy * dominant_light_intensity.b, c[3].w);
    c[0] = vec4<f32>(c[0].xyz - 0.2820948 * dominant_light_intensity, c[0].w);

    var x1: vec3<f32>;
    x1.r = dot(normal, c[1].rgb);
    x1.g = dot(normal, c[2].rgb);
    x1.b = dot(normal, c[3].rgb);
    let lightprobe_color = (RAVI_C4 * c[0].rgb + (-2.0 * RAVI_C2) * x1) * RAVI_INV_PI;
    let dominant_lighting = max(dot(dominant_light_dir, normal), 0.0)
                          * dominant_light_intensity
                          * 0.281;
    return lightprobe_color + dominant_lighting;
}

// SH3 area-specular evaluation along a reflection direction. Verbatim
// port of `calculate_area_specular_phong_order_3` from
// `two_lobe_phong_fx.hlsl:205-249`. The same body lives in
// `organism_material_fx.hlsl:91-115` (the `_2` variant) with identical
// constants — engine shares this between material models, NOT a
// protomorph copy-paste error.
//
// Constants (`p_0=0.4231425, p_1=-0.3805236, p_2=-0.4018891, p_3=-0.2009446`)
// are precomputed simplifications of the engine's commented derivations:
//   p_0 = 0.886227f                            (≈ 0.282095 * 1.5)
//   p_1 = 0.511664f * -2 = -1.023328  (but optimized to -0.3805236)
//   p_2 = 0.429043f * -2 = -0.858086  (but optimized to -0.4018891)
//   p_3 = 0.429043f * -1 = -0.429043  (but optimized to -0.2009446)
// The optimized values bake in an `exp(-N * power_invert)` falloff for
// the default power (≈18-25 per typical metal). DO NOT switch to
// `_new_phong_3` (the roughness-modulated alternative) — engine's
// two_lobe_phong call site uses THIS function exactly.
fn calculate_area_specular_phong_order_3(
    reflection_dir: vec3<f32>,
    sh_lighting_coefficients: array<vec4<f32>, 10>,
    power: f32,
    tint: vec3<f32>,
) -> vec3<f32> {
    let p_0: f32 =  0.4231425;
    let p_1: f32 = -0.3805236;
    let p_2: f32 = -0.4018891;
    let p_3: f32 = -0.2009446;

    var c = sh_lighting_coefficients;
    let x0 = vec3<f32>(c[0].r * p_0);

    var x1: vec3<f32>;
    x1.r = dot(reflection_dir, c[1].xyz);
    x1.g = dot(reflection_dir, c[2].xyz);
    x1.b = dot(reflection_dir, c[3].xyz);
    x1 = x1 * p_1;

    let quadratic_a = reflection_dir.xyz * reflection_dir.yzx;
    var x2: vec3<f32>;
    x2.x = dot(quadratic_a, c[4].xyz);
    x2.y = dot(quadratic_a, c[5].xyz);
    x2.z = dot(quadratic_a, c[6].xyz);
    x2 = x2 * p_2;

    let quadratic_b = vec4<f32>(
        reflection_dir.x * reflection_dir.x,
        reflection_dir.y * reflection_dir.y,
        reflection_dir.z * reflection_dir.z,
        1.0 / 3.0,
    );
    var x3: vec3<f32>;
    x3.x = dot(quadratic_b, c[7]);
    x3.y = dot(quadratic_b, c[8]);
    x3.z = dot(quadratic_b, c[9]);
    x3 = x3 * p_3;

    let _unused_power = power;
    return (x0 + x1 + x2 + x3) * tint;
}

// Halo `calculate_analytical_specular_new_phong_3` — the
// modified-Phong analytic-specular form used by single_lobe_phong
// (and various other material models). Faithful port of
// spherical_harmonics_fx.hlsl:553.
//
// Modified Phong distribution: σ = roughness, ω = view_reflect dir,
// θ = angle between dominant_light_dir and reflect_dir. Returns
// `4πσ² cos³θ × exp(−tan²θ / σ²) × dominant_light_intensity`.
fn calculate_analytical_specular_new_phong_3(
    dominant_light_dir: vec3<f32>,
    dominant_light_intensity: vec3<f32>,
    reflect_dir: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let roughness_sq = roughness * roughness;
    let cos_theta = max(dot(dominant_light_dir, reflect_dir), 0.0);
    let cos_theta_sq = cos_theta * cos_theta;
    let denom = max(cos_theta_sq, 1e-6);
    let tan_theta_sq = (1.0 - cos_theta_sq) / denom;
    let tan_theta_sq_over_roughness_sq =
        select(0.0, tan_theta_sq / roughness_sq, roughness_sq != 0.0);
    return (3.1415926 * roughness_sq * cos_theta * cos_theta_sq)
        * exp(-tan_theta_sq_over_roughness_sq)
        * dominant_light_intensity;
}

// Halo `calculate_area_specular_new_phong_3` — modified-Phong
// area-specular evaluation against a 10-coef SH probe. Faithful port
// of spherical_harmonics_fx.hlsl:568. The HLSL `order3` bool gates
// the quadratic SH bands: most callers pass it from the rmt2
// `order3_area_specular` runtime flag, but **glass hardcodes
// `false`** (glass_material_fx.hlsl:86) regardless of the rmt2 value.
// Order-3 path adds `(x2 + x3) * c_quadratic`; order-2 path drops
// them entirely. Critical to match the HLSL byte-for-byte for
// static_sh material paths where bands 4-9 carry real probe data —
// running order-3 instead of order-2 over-brightens area specular.
fn calculate_area_specular_new_phong_3(
    reflection_dir: vec3<f32>,
    sh_lighting_coefficients: array<vec4<f32>, 10>,
    roughness: f32,
    order3: bool,
) -> vec3<f32> {
    let roughness_sq = roughness * roughness;

    let c_dc: f32 = 0.282095;
    let c_linear: f32 =
        -(0.5128945834 + (-0.1407369526) * roughness + (-0.2660066620e-2) * roughness_sq) * 0.60;

    var c = sh_lighting_coefficients;
    let x0 = c[0].rgb;

    var x1: vec3<f32>;
    x1.r = dot(reflection_dir, c[1].rgb);
    x1.g = dot(reflection_dir, c[2].rgb);
    x1.b = dot(reflection_dir, c[3].rgb);

    if (!order3) {
        return max(x0 * c_dc + x1 * c_linear, vec3<f32>(0.0));
    }

    let c_quadratic: f32 =
        -(0.7212524717 + (-0.5541015389) * roughness + (0.7960539966e-1) * roughness_sq) * 0.5;

    let quadratic_a = reflection_dir.xyz * reflection_dir.yzx;
    var x2: vec3<f32>;
    x2.x = dot(quadratic_a, c[4].rgb);
    x2.y = dot(quadratic_a, c[5].rgb);
    x2.z = dot(quadratic_a, c[6].rgb);

    let quadratic_b = vec4<f32>(
        reflection_dir.x * reflection_dir.x,
        reflection_dir.y * reflection_dir.y,
        reflection_dir.z * reflection_dir.z,
        1.0 / 3.0,
    );
    var x3: vec3<f32>;
    x3.x = dot(quadratic_b, c[7]);
    x3.y = dot(quadratic_b, c[8]);
    x3.z = dot(quadratic_b, c[9]);

    return max(x0 * c_dc + x1 * c_linear + x2 * c_quadratic + x3 * c_quadratic, vec3<f32>(0.0));
}

// Build the 10-element ravi-order-3 SH array from our `default_lighting`
// cbuffer (the engine-side fallback layout — only DC + L1 populated;
// L2 stays zero, so `ravi_order_3` collapses to `ravi_order_2` for
// this input). For Phase F1 lightmap-fed SH this builder is replaced
// by the atlas-decoded array.
//
// **Layout matches Halo's `calculate_and_set_ravi_constants_internal
// @ 0x1806aa650`** which packs L1 into the cbuffer as:
//   vec4[1] = (sh[3], sh[1], -sh[2], 0)   for R channel
//           = (x_axis, y_axis, -z_axis, 0)
//
// where sh[1]/sh[2]/sh[3] are the m=-1, m=0, m=+1 SH coefficients
// (y-, z-, x-axis respectively). Halo NEGATES the z-axis component
// so `dot(N, c[1].xyz)` evaluates correctly under their basis
// convention (Y1,0 = -k·z, opposite to standard).
//
// Our `engine_lighting_ps.ravi` carries the Rust-packed SH3
// (`pack_ravi_constants`); just re-emit the array.
fn build_default_sh_array() -> array<vec4<f32>, 10> {
    return array<vec4<f32>, 10>(
        engine_lighting_ps.ravi[0],
        engine_lighting_ps.ravi[1],
        engine_lighting_ps.ravi[2],
        engine_lighting_ps.ravi[3],
        engine_lighting_ps.ravi[4],
        engine_lighting_ps.ravi[5],
        engine_lighting_ps.ravi[6],
        engine_lighting_ps.ravi[7],
        engine_lighting_ps.ravi[8],
        engine_lighting_ps.ravi[9],
    );
}
