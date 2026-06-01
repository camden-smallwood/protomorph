// Port of `cook_torrance_fx.hlsl` — `calc_material_cook_torrance_ps`
// orchestrator + analytical-specular + sh_glossy_ct_3 helpers + the
// `sh_rotate_023` + base-shape rotation utilities.
//
// HLSL hierarchy:
//   calc_material_cook_torrance_ps(...)        // line 1700
//     → calc_material_cook_torrance_base(...)  // line 1023
//         → calc_material_analytic_specular_cook_torrance_ps(...)  // line 234
//         → calculate_ambientness(...)         // shared
//         → calc_simple_lights_analytical(...) // shared
//         → sh_glossy_ct_3(...) | sh_glossy_ct_2(...)  // line 922 / 854
//
// Constants (cook_torrance_fx.hlsl):
//   c_view_z_shift     = 0.5/32.0
//   c_roughness_shift  = 0.0
//   SQRT3              = 1.73205080756 (from spherical_harmonics_fx.hlsl)
//
// Cbuffer fields used (declared in cook_torrance_fx.hlsl PARAMs):
//   fresnel_color, roughness, albedo_blend, specular_tint,
//   analytical_anti_shadow_control + the shared umbrella scalars
//   (specular_coefficient, area_/analytical_/environment_map_specular_contribution,
//   diffuse_coefficient).
//
// Texture bindings used:
//   g_sampler_cc0236 — pre-integrated BRDF LUT (rasterizer_globals.material_textures[0])
//   g_sampler_dd0236 — pre-integrated BRDF LUT (rasterizer_globals.material_textures[2])
//   g_sampler_c78d78 — pre-integrated BRDF LUT (rasterizer_globals.material_textures[1])
//   material_texture — per-pixel material parameter override (when use_material_texture=true)
//
// **Status:** math is HLSL-faithful. LUT bitmaps are still bound to
// the white-fallback view until the rasterizer_globals → bitmap-load
// pipeline lands — visually wrong (no specular distribution shape)
// but no panics, and the shader infrastructure is ready.

const SQRT3: f32 = 1.73205080756;
const c_view_z_shift: f32 = 0.015625; // 0.5 / 32.0
const c_roughness_shift: f32 = 0.0;

struct MaterialOutput {
    envmap_specular_reflectance_and_roughness: vec4<f32>,
    envmap_area_specular_only: vec3<f32>,
    specular_color: vec4<f32>,
    diffuse_radiance: vec3<f32>,
}

// Halo `calculate_ambientness` in spherical_harmonics_fx.hlsl. Returns
// a [0, 1] scalar — how much of the SH probe's energy comes from the
// dominant-light direction.
fn calculate_ambientness(
    sh_lighting_coefficients: array<vec4<f32>, 4>,
    dominant_light_intensity: vec3<f32>,
    dominant_light_dir: vec3<f32>,
) -> f32 {
    let dir_eval = vec3<f32>(
        -0.4886025 * dominant_light_dir.y,
        -0.4886025 * dominant_light_dir.z,
        -0.4886025 * dominant_light_dir.x,
    );
    let temp = vec4<f32>(
        sh_lighting_coefficients[2].xyz - dir_eval.zxy * dominant_light_intensity.y,
        sh_lighting_coefficients[0].y - 0.2820948 * dominant_light_intensity.y,
    );
    let num = dot(temp, temp);
    let denom_xyz = sh_lighting_coefficients[2].xyz;
    let denom = dot(denom_xyz, denom_xyz) + sh_lighting_coefficients[0].y * sh_lighting_coefficients[0].y;
    let ambientness = select(0.0, num / denom, num > 0.0);
    return min(ambientness, 1.0);
}

// HLSL: float3 sh_rotate_023(int irgb, float3 rotate_x, float3 rotate_z, float4 sh_0, float4 sh_312[3])
// (cook_torrance_fx.hlsl line ~835). Selects the per-channel SH
// component using `irgb` (0 = R, 1 = G, 2 = B) and projects against
// the local view-tangent frame.
fn sh_rotate_023(
    irgb: i32,
    rotate_x: vec3<f32>,
    rotate_z: vec3<f32>,
    sh_0: vec4<f32>,
    sh_312: array<vec4<f32>, 3>,
) -> vec3<f32> {
    let sh312_irgb = sh_312[irgb];
    let sh0_irgb = select(
        select(sh_0.z, sh_0.y, irgb == 1),
        sh_0.x,
        irgb == 0,
    );
    return vec3<f32>(
        sh0_irgb,
        -dot(rotate_z, sh312_irgb.xyz),
        dot(rotate_x, sh312_irgb.xyz),
    );
}

// HLSL: void sh_glossy_ct_2(view_dir, rotate_z, sh_0, sh_312,
//                           roughness, r_dot_l, power,
//                           out specular_part, out schlick_part)
// (cook_torrance_fx.hlsl line 854). Order-2 variant — drops the
// L2 (`sh_457` / `sh_8866`) basis projections so only the DC + L1
// SH terms contribute. Used when the rmt2's `order3_area_specular`
// bool is false. Same `sh_rotate_023` per-channel construction.
fn sh_glossy_ct_2(
    view_dir: vec3<f32>,
    rotate_z: vec3<f32>,
    sh_0: vec4<f32>,
    sh_312: array<vec4<f32>, 3>,
    roughness: f32,
    r_dot_l: f32,
    power: f32,
    specular_part: ptr<function, vec3<f32>>,
    schlick_part: ptr<function, vec3<f32>>,
) {
    let rotate_x = normalize(view_dir - dot(view_dir, rotate_z) * rotate_z);

    let t_roughness = max(roughness, 0.05);
    let view_lookup = vec2<f32>(
        pow(max(dot(view_dir, rotate_x), 0.0001), power) + c_view_z_shift,
        t_roughness + c_roughness_shift,
    );

    let c_value = textureSample(g_sampler_cc0236, g_sampler_cc0236_sampler, view_lookup);
    let d_value = textureSample(g_sampler_dd0236, g_sampler_dd0236_sampler, view_lookup);

    // R channel — order-2 puts 0 in the quadratic slot (HLSL line 888).
    var sh_local = vec4<f32>(
        sh_rotate_023(0, rotate_x, rotate_z, sh_0, sh_312),
        0.0,
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    let sp_r = dot(c_value, sh_local);
    let sk_r = dot(d_value, sh_local);

    // G channel
    sh_local = vec4<f32>(
        sh_rotate_023(1, rotate_x, rotate_z, sh_0, sh_312),
        0.0,
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    let sp_g = dot(c_value, sh_local);
    let sk_g = dot(d_value, sh_local);

    // B channel
    sh_local = vec4<f32>(
        sh_rotate_023(2, rotate_x, rotate_z, sh_0, sh_312),
        0.0,
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    let sp_b = dot(c_value, sh_local);
    let sk_b = dot(d_value, sh_local);

    *specular_part = vec3<f32>(sp_r, sp_g, sp_b);
    *schlick_part = vec3<f32>(sk_r, sk_g, sk_b) * 0.01;
}

// HLSL: void sh_glossy_ct_3(view_dir, rotate_z, sh_0, sh_312, sh_457, sh_8866,
//                           roughness, r_dot_l, power,
//                           out specular_part, out schlick_part)
// (cook_torrance_fx.hlsl line 922).
fn sh_glossy_ct_3(
    view_dir: vec3<f32>,
    rotate_z: vec3<f32>,
    sh_0: vec4<f32>,
    sh_312: array<vec4<f32>, 3>,
    sh_457: array<vec4<f32>, 3>,
    sh_8866: array<vec4<f32>, 3>,
    roughness: f32,
    r_dot_l: f32,
    power: f32,
    specular_part: ptr<function, vec3<f32>>,
    schlick_part: ptr<function, vec3<f32>>,
) {
    // build local frame
    let rotate_x = normalize(view_dir - dot(view_dir, rotate_z) * rotate_z);
    let rotate_y = cross(rotate_z, rotate_x);

    let t_roughness = max(roughness, 0.05);
    let view_lookup = vec2<f32>(
        pow(max(dot(view_dir, rotate_x), 0.0001), power) + c_view_z_shift,
        t_roughness + c_roughness_shift,
    );

    // bases: 0, 2, 3, 6 — packed in cc0236 (specular) + dd0236 (Schlick).
    var c_value = textureSample(g_sampler_cc0236, g_sampler_cc0236_sampler, view_lookup);
    let d_value = textureSample(g_sampler_dd0236, g_sampler_dd0236_sampler, view_lookup);

    var quadratic_a: vec4<f32>;
    var quadratic_b: vec4<f32>;
    var sh_local: vec4<f32>;

    quadratic_a = vec4<f32>(rotate_z.yzx * rotate_z.xyz * (-SQRT3), 0.0);
    quadratic_b = vec4<f32>(rotate_z.xyz * rotate_z.xyz, 1.0 / 3.0) * 0.5 * (-SQRT3);

    // R channel
    sh_local = vec4<f32>(
        sh_rotate_023(0, rotate_x, rotate_z, sh_0, sh_312),
        dot(quadratic_a.xyz, sh_457[0].xyz) + dot(quadratic_b, sh_8866[0]),
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    var sp_r = dot(c_value, sh_local);
    var sk_r = dot(d_value, sh_local);

    // G channel
    sh_local = vec4<f32>(
        sh_rotate_023(1, rotate_x, rotate_z, sh_0, sh_312),
        dot(quadratic_a.xyz, sh_457[1].xyz) + dot(quadratic_b, sh_8866[1]),
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    var sp_g = dot(c_value, sh_local);
    var sk_g = dot(d_value, sh_local);

    // B channel
    sh_local = vec4<f32>(
        sh_rotate_023(2, rotate_x, rotate_z, sh_0, sh_312),
        dot(quadratic_a.xyz, sh_457[2].xyz) + dot(quadratic_b, sh_8866[2]),
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    var sp_b = dot(c_value, sh_local);
    var sk_b = dot(d_value, sh_local);

    var sp = vec3<f32>(sp_r, sp_g, sp_b);
    var sk = vec3<f32>(sk_r, sk_g, sk_b);

    // basis 7 (c78d78 LUT)
    c_value = textureSample(g_sampler_c78d78, g_sampler_c78d78_sampler, view_lookup);
    let qa7 = rotate_x.xyz * rotate_z.yzx + rotate_x.yzx * rotate_z.xyz;
    let qb7 = rotate_x.xyz * rotate_z.xyz;
    let sh7 = vec3<f32>(
        dot(qa7, sh_457[0].xyz) + dot(qb7, sh_8866[0].xyz),
        dot(qa7, sh_457[1].xyz) + dot(qb7, sh_8866[1].xyz),
        dot(qa7, sh_457[2].xyz) + dot(qb7, sh_8866[2].xyz),
    ) * r_dot_l;
    sp = sp + c_value.x * sh7;
    sk = sk + c_value.z * sh7;

    // basis 8
    let qa8 = rotate_x.xyz * rotate_x.yzx - rotate_y.yzx * rotate_y.xyz;
    let qb8 = 0.5 * (rotate_x.xyz * rotate_x.xyz - rotate_y.xyz * rotate_y.xyz);
    let sh8 = vec3<f32>(
        -dot(qa8, sh_457[0].xyz) - dot(qb8, sh_8866[0].xyz),
        -dot(qa8, sh_457[1].xyz) - dot(qb8, sh_8866[1].xyz),
        -dot(qa8, sh_457[2].xyz) - dot(qb8, sh_8866[2].xyz),
    ) * r_dot_l;
    sp = sp + c_value.y * sh8;
    sk = sk + c_value.w * sh8;

    *specular_part = sp;
    *schlick_part = sk * 0.01;
}

// HLSL: void calc_material_analytic_specular_cook_torrance_ps(
//     view_dir, normal_dir, view_reflect_dir, light_dir, light_irradiance,
//     diffuse_albedo_color, texcoord, vertex_n_dot_l, surface_normal, misc,
//     out spatially_varying_material_parameters,
//     out specular_fresnel_color, out specular_albedo_color, out analytic_specular_radiance)
// (cook_torrance_fx.hlsl line 234).
struct CtAnalyticOutput {
    spatially_varying_material_parameters: vec4<f32>,
    specular_fresnel_color: vec3<f32>,
    specular_albedo_color: vec3<f32>,
    analytic_specular_radiance: vec3<f32>,
}

fn calc_material_analytic_specular_cook_torrance_ps(
    view_dir: vec3<f32>,
    normal_dir: vec3<f32>,
    view_reflect_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_irradiance: vec3<f32>,
    diffuse_albedo_color: vec3<f32>,
    vertex_n_dot_l: f32,
) -> CtAnalyticOutput {
    let _u_reflect = view_reflect_dir;

    // material_parameters = (specular_coefficient, albedo_blend, env_contrib, roughness)
    // (the `use_material_texture=true` per-pixel override path is
    // skipped in v1 — material_texture sampling lands when we wire
    // the per-pixel BRDF map.)
    let mp = vec4<f32>(
        material.specular_coefficient.x,
        material.albedo_blend.x,
        material.environment_map_specular_contribution.x,
        material.roughness.x,
    );

    let specular_albedo_color = diffuse_albedo_color * mp.y + material.fresnel_color.xyz * (1.0 - mp.y);

    let n_dot_l = dot(normal_dir, light_dir);
    let n_dot_v = dot(normal_dir, view_dir);
    let min_dot = min(n_dot_l, n_dot_v);

    var analytic = vec3<f32>(0.00001);
    var fresnel = specular_albedo_color;

    if (min_dot > 0.0) {
        let half_vector = normalize(view_dir + light_dir);
        let n_dot_h = dot(normal_dir, half_vector);
        let v_dot_h = dot(view_dir, half_vector);

        let geometry_term = 2.0 * n_dot_h * min_dot / (saturate(v_dot_h) + 0.00001);

        // Cook-Torrance fresnel via index-of-refraction substitution.
        let f0 = min(specular_albedo_color, vec3<f32>(0.999));
        let sqrt_f0 = sqrt(f0);
        let n_ior = (1.0 + sqrt_f0) / (1.0 - sqrt_f0);
        let g = sqrt(n_ior * n_ior + v_dot_h * v_dot_h - 1.0);
        let gpc = g + v_dot_h;
        let gmc = g - v_dot_h;
        let r = (v_dot_h * gpc - 1.0) / (v_dot_h * gmc + 1.0);
        fresnel = 0.5 * (gmc * gmc / (gpc * gpc + 0.00001)) * (1.0 + r * r);

        // Beckmann-like distribution.
        let t_roughness = max(mp.w, 0.05);
        let m_squared = t_roughness * t_roughness;
        let cosine_alpha_squared = n_dot_h * n_dot_h;
        let distribution =
            exp((cosine_alpha_squared - 1.0) / (m_squared * cosine_alpha_squared))
            / (m_squared * cosine_alpha_squared * cosine_alpha_squared + 0.00001);

        analytic = distribution * saturate(geometry_term)
                 / (3.14159265 * n_dot_v + 0.00001) * fresnel;
        analytic = min(analytic, vec3<f32>(vertex_n_dot_l + 1.0)) * light_irradiance;
    }

    return CtAnalyticOutput(
        mp,
        fresnel,
        specular_albedo_color,
        analytic,
    );
}

// HLSL: void calc_material_cook_torrance_base(
//     view_dir, fragment_to_camera_world, view_normal, view_reflect_dir_world,
//     sh_lighting_coefficients[10], view_light_dir, light_color, albedo_color,
//     specular_mask, texcoord, prt_ravi_diff, tangent_frame, misc, spec_tint,
//     out envmap_specular_reflectance_and_roughness,
//     out envmap_area_specular_only,
//     out specular_color,
//     inout diffuse_radiance)
// (cook_torrance_fx.hlsl line 1023).
//
// `calc_material_cook_torrance_ps` (line 1700) just calls this with
// `spec_tint = specular_tint` from the cbuffer.
fn calc_material(
    view_dir: vec3<f32>,
    fragment_to_camera_world: vec3<f32>,
    surface_normal: vec3<f32>,
    view_reflect_dir: vec3<f32>,
    sh_lighting_coefficients: array<vec4<f32>, 10>,
    analytical_light_dir: vec3<f32>,
    analytical_light_intensity: vec3<f32>,
    diffuse_reflectance: vec3<f32>,
    specular_mask: f32,
    texcoord: vec2<f32>,
    prt_ravi_diff: vec4<f32>,
    diffuse_radiance_in: vec3<f32>,
    fragment_position_world: vec3<f32>,
) -> MaterialOutput {
    let _u_view_world = fragment_to_camera_world;
    let _u_uv = texcoord;

    let analytic = calc_material_analytic_specular_cook_torrance_ps(
        view_dir,
        surface_normal,
        view_reflect_dir,
        analytical_light_dir,
        analytical_light_intensity,
        diffuse_reflectance,
        prt_ravi_diff.w,
    );
    let svmp = analytic.spatially_varying_material_parameters;
    let effective_reflectance = analytic.specular_albedo_color;
    var specular_analytical = analytic.analytic_specular_radiance;

    // Anti-shadow attenuation (HLSL line 1073).
    if (material.analytical_anti_shadow_control.x > 0.0) {
        let sh_temp = array<vec4<f32>, 4>(
            sh_lighting_coefficients[0],
            sh_lighting_coefficients[1],
            sh_lighting_coefficients[2],
            sh_lighting_coefficients[3],
        );
        let ambientness = calculate_ambientness(
            sh_temp, analytical_light_intensity, analytical_light_dir,
        );
        let mult = pow(1.0 - ambientness, material.analytical_anti_shadow_control.x * 100.0);
        specular_analytical = specular_analytical * mult;
    }

    // Simple lights — HLSL line 1094 passes
    // `GET_MATERIAL_SPECULAR_POWER(cook_torrance)(svmp.a)` =
    // `0.27291 * pow(roughness, -2.1973)` when roughness != 0, else 0.
    // (svmp.w mirrors `roughness` here — material_texture override
    // path not yet wired.)
    // Engine `cook_torrance_fx.hlsl:1087/1357/1600` gates simple_lights
    // on `if (!no_dynamic_lights)`; the else branch zeroes both lights.
    const NO_DYNAMIC_LIGHTS: bool = __NO_DYNAMIC_LIGHTS__;
    var simple_diffuse = vec3<f32>(0.0);
    var simple_specular = vec3<f32>(0.0);
    if (!NO_DYNAMIC_LIGHTS) {
        let spec_power = select(
            0.27291 * pow(svmp.w, -2.1973),
            0.0,
            svmp.w == 0.0,
        );
        let simple = calc_simple_lights_analytical(
            fragment_position_world,
            surface_normal,
            view_reflect_dir,
            spec_power,
        );
        simple_diffuse = simple.diffuse;
        simple_specular = simple.specular;
    }

    // Area specular — HLSL line 1112-1147 branches between
    // `sh_glossy_ct_3` (uses L0+L1+L2 = 10 SH coefs) and
    // `sh_glossy_ct_2` (uses L0+L1 only = 4 coefs) on the rmt2 bool
    // `order3_area_specular`. Substituted per-variant.
    const ORDER3_AREA_SPECULAR: bool = __ORDER3_AREA_SPECULAR__;
    let r_dot_l = max(dot(analytical_light_dir, view_reflect_dir), 0.0) * 0.65 + 0.35;
    let sh_312 = array<vec4<f32>, 3>(
        sh_lighting_coefficients[1],
        sh_lighting_coefficients[2],
        sh_lighting_coefficients[3],
    );

    var specular_part: vec3<f32>;
    var schlick_part: vec3<f32>;
    if (ORDER3_AREA_SPECULAR) {
        let sh_457 = array<vec4<f32>, 3>(
            sh_lighting_coefficients[4],
            sh_lighting_coefficients[5],
            sh_lighting_coefficients[6],
        );
        let sh_8866 = array<vec4<f32>, 3>(
            sh_lighting_coefficients[7],
            sh_lighting_coefficients[8],
            sh_lighting_coefficients[9],
        );
        sh_glossy_ct_3(
            view_dir,
            surface_normal,
            sh_lighting_coefficients[0],
            sh_312,
            sh_457,
            sh_8866,
            svmp.w,
            r_dot_l,
            1.0,
            &specular_part,
            &schlick_part,
        );
    } else {
        sh_glossy_ct_2(
            view_dir,
            surface_normal,
            sh_lighting_coefficients[0],
            sh_312,
            svmp.w,
            r_dot_l,
            1.0,
            &specular_part,
            &schlick_part,
        );
    }

    let sh_glossy = specular_part * effective_reflectance
                  + (vec3<f32>(1.0) - effective_reflectance) * schlick_part;

    let envmap_area = sh_glossy * prt_ravi_diff.z * material.specular_tint.xyz;
    let envmap_specular_reflectance_xyz =
        vec3<f32>(svmp.z * specular_mask * svmp.x);
    let envmap_specular_reflectance_and_roughness =
        vec4<f32>(envmap_specular_reflectance_xyz, svmp.w);

    // Specular composition (HLSL line 1155):
    let combined =
          (simple_specular * effective_reflectance + specular_analytical)
              * material.analytical_specular_contribution.x
        + max(sh_glossy, vec3<f32>(0.0)) * material.area_specular_contribution.x;
    var specular_color =
        vec4<f32>(specular_mask * svmp.x * material.specular_tint.xyz * combined, 0.0);
    specular_color = specular_color * prt_ravi_diff.z;

    // Diffuse modulation (HLSL line 1163-1170; we don't sample
    // material_texture, so diffuse_adjusted = diffuse_coefficient).
    let diffuse_adjusted = material.diffuse_coefficient.x;
    var diffuse_radiance = diffuse_radiance_in * prt_ravi_diff.x;
    diffuse_radiance = (simple_diffuse + diffuse_radiance) * diffuse_adjusted;

    return MaterialOutput(
        envmap_specular_reflectance_and_roughness,
        envmap_area,
        specular_color,
        diffuse_radiance,
    );
}
