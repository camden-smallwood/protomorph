// Halo `material_model / two_lobe_phong` — bit-for-bit faithful port of
// `calc_material_two_lobe_phong_ps` in
// /Users/camden/Halo/halo3_mcc_hlsl_extracted/two_lobe_phong_fx.hlsl:283.
//
// Body matches HLSL line for line. Drop-in for the umbrella's
// CALC_MATERIAL(material_type) call site.
//
// Caller contract (matches the HLSL out/inout form):
//   - Inputs `view_dir`, `fragment_to_camera_world`, `surface_normal`,
//     `view_reflect_dir`, `sh_lighting_coefficients[10]`,
//     `analytical_light_dir`, `analytical_light_intensity`,
//     `diffuse_reflectance`, `specular_mask`, `texcoord`, `prt_ravi_diff`,
//     `diffuse_radiance` (input — ravi_order_3 already evaluated)
//   - Outputs (returned via `MaterialOutput`):
//       envmap_specular_reflectance_and_roughness: vec4
//       envmap_area_specular_only:                 vec3
//       specular_color:                            vec4
//       diffuse_radiance:                          vec3 (modulated)
//
// Stubs preserved as zero-returns so the equation shape is intact:
//   - `calc_simple_lights_analytical` (no dynamic-light cbuffer yet).
//   - `tangent_frame` / `misc` / `fragment_to_camera_world` are unused
//     by the HLSL body for two_lobe_phong (they're pass-through to
//     simple_lights), so we omit the `tangent_frame` argument here.

struct MaterialOutput {
    envmap_specular_reflectance_and_roughness: vec4<f32>,
    envmap_area_specular_only: vec3<f32>,
    specular_color: vec4<f32>,
    diffuse_radiance: vec3<f32>,
}

// Halo `calculate_fresnel` in two_lobe_phong_fx.hlsl:29 — derives the
// blended specular power and tint from the artist's two-lobe controls.
struct FresnelResult {
    power: f32,
    tint: vec3<f32>,
}
fn calculate_fresnel_two_lobe(
    view_dir: vec3<f32>,
    normal_dir: vec3<f32>,
    albedo_color: vec3<f32>,
) -> FresnelResult {
    let n_dot_v = max(dot(normal_dir, view_dir), 0.0);
    let fresnel_blend = pow(1.0 - n_dot_v, material.fresnel_curve_steepness.x);
    let power = mix(material.normal_specular_power.x, material.glancing_specular_power.x, fresnel_blend);
    var tint = mix(material.normal_specular_tint.xyz, material.glancing_specular_tint.xyz, fresnel_blend);
    tint = mix(tint, albedo_color, material.albedo_specular_tint_blend.x);
    return FresnelResult(power, tint);
}

// Halo `calculate_area_specular_phong_order_2` (4-coef SH path).
// Verbatim port of `two_lobe_phong_fx.hlsl:251-277` (and identical
// body in `organism_material_fx.hlsl:91-118`). Engine literally
// shares this function across material models — not a port mistake.
//
// Constants (`p_0=0.4231425, p_1=-0.3805236`) bake in the standard
// Phong falloff for default power values. DO NOT switch to
// `_new_phong_2` — engine's two_lobe_phong call site uses THIS
// function exactly (verified 2026-05-17 via HLSL extract).
fn calculate_area_specular_phong_order_2(
    reflection_dir: vec3<f32>,
    sh_lighting_coefficients: array<vec4<f32>, 4>,
    power: f32,
    tint: vec3<f32>,
) -> vec3<f32> {
    let p_0: f32 =  0.4231425;
    let p_1: f32 = -0.3805236;

    let x0 = vec3<f32>(sh_lighting_coefficients[0].r * p_0);

    var x1: vec3<f32>;
    x1.r = dot(reflection_dir, sh_lighting_coefficients[1].xyz);
    x1.g = dot(reflection_dir, sh_lighting_coefficients[2].xyz);
    x1.b = dot(reflection_dir, sh_lighting_coefficients[3].xyz);
    x1 = x1 * p_1;

    let _unused_power = power;
    return (x0 + x1) * tint;
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

// Halo `calc_simple_lights_analytical` lives in
// `simple_lights_fx.wgsl` (prepended to the variant by
// `render_methods/mod.rs`) — the local stub here was removed so the
// real implementation can compile in.

// Halo `calc_material_analytic_specular_two_lobe_phong_ps` in
// two_lobe_phong_fx.hlsl:100. Outputs analytic specular radiance plus
// the fresnel tint, per-light albedo, and material parameters fed
// into the rest of the umbrella.
struct AnalyticSpecularOutput {
    material_parameters: vec4<f32>,
    specular_fresnel_color: vec3<f32>,
    specular_albedo_color: vec3<f32>,
    analytic_specular_radiance: vec3<f32>,
}
fn calc_material_analytic_specular_two_lobe_phong(
    view_dir: vec3<f32>,
    normal_dir: vec3<f32>,
    view_reflect_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_irradiance: vec3<f32>,
    diffuse_albedo_color: vec3<f32>,
    texcoord: vec2<f32>,
) -> AnalyticSpecularOutput {
    let fresnel = calculate_fresnel_two_lobe(view_dir, normal_dir, diffuse_albedo_color);
    let power_or_roughness = fresnel.power;
    let specular_fresnel_color = fresnel.tint;

    // HLSL: specular_albedo_color = normal_specular_tint
    let specular_albedo_color = material.normal_specular_tint.xyz;

    // HLSL: material_parameters.rgb = (specular_coefficient,
    //   albedo_specular_tint_blend, environment_map_specular_contribution)
    //       material_parameters.a   = power_or_roughness
    let material_parameters = vec4<f32>(
        material.specular_coefficient.x,
        material.albedo_specular_tint_blend.x,
        material.environment_map_specular_contribution.x,
        power_or_roughness,
    );

    let l_dot_r = dot(light_dir, view_reflect_dir);
    var analytic_specular_radiance: vec3<f32>;
    if (l_dot_r > 0.0) {
        analytic_specular_radiance =
            pow(l_dot_r, power_or_roughness) *
            ((power_or_roughness + 1.0) / 6.2832) *
            specular_fresnel_color *
            light_irradiance;
        let _unused_texcoord = texcoord;
    } else {
        analytic_specular_radiance = vec3<f32>(0.0);
    }

    return AnalyticSpecularOutput(
        material_parameters,
        specular_fresnel_color,
        specular_albedo_color,
        analytic_specular_radiance,
    );
}

// Halo `calc_material_two_lobe_phong_ps` — bit-for-bit port. The HLSL
// uses out/inout params; WGSL wraps them into `MaterialOutput`.
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
    let _u_frag_cam = fragment_to_camera_world;
    let analytic = calc_material_analytic_specular_two_lobe_phong(
        view_dir,
        surface_normal,
        view_reflect_dir,
        analytical_light_dir,
        analytical_light_intensity,
        diffuse_reflectance,
        texcoord,
    );

    var analytic_specular_radiance = analytic.analytic_specular_radiance;
    let material_parameters = analytic.material_parameters;
    let specular_fresnel_color = analytic.specular_fresnel_color;

    // Anti-shadow attenuation (HLSL line 334).
    if (material.analytical_anti_shadow_control.x > 0.0) {
        // Build sh_temp[4] as the HLSL does — the first four coefficients
        // of sh_lighting_coefficients[10].
        let sh_temp = array<vec4<f32>, 4>(
            sh_lighting_coefficients[0],
            sh_lighting_coefficients[1],
            sh_lighting_coefficients[2],
            sh_lighting_coefficients[3],
        );
        let ambientness = calculate_ambientness(
            sh_temp, analytical_light_intensity, analytical_light_dir,
        );
        let ambient_multiplier = pow(1.0 - ambientness, material.analytical_anti_shadow_control.x * 100.0);
        analytic_specular_radiance = analytic_specular_radiance * ambient_multiplier;
    }

    // Simple lights — engine line 348 passes `fragment_position_world`
    // (Camera_Position_PS - fragment_to_camera_world). Engine
    // `two_lobe_phong_fx.hlsl:346/465` gates on `if (!no_dynamic_lights)`;
    // the else branch zeroes both lights.
    const NO_DYNAMIC_LIGHTS: bool = __NO_DYNAMIC_LIGHTS__;
    var simple_light_diffuse_light = vec3<f32>(0.0);
    var simple_light_specular_light = vec3<f32>(0.0);
    if (!NO_DYNAMIC_LIGHTS) {
        let simple = calc_simple_lights_analytical(
            fragment_position_world,
            surface_normal,
            view_reflect_dir,
            material_parameters.w,
        );
        simple_light_diffuse_light = simple.diffuse;
        simple_light_specular_light = simple.specular;
    }

    // Area specular — HLSL line 365-384 branches between
    // `calculate_area_specular_phong_order_3` (10 SH coefs) and
    // `calculate_area_specular_phong_order_2` (4 coefs) on the rmt2
    // bool `order3_area_specular`. Substituted per-variant.
    const ORDER3_AREA_SPECULAR: bool = __ORDER3_AREA_SPECULAR__;
    var area_specular_radiance: vec3<f32>;
    if (ORDER3_AREA_SPECULAR) {
        area_specular_radiance = calculate_area_specular_phong_order_3(
            view_reflect_dir,
            sh_lighting_coefficients,
            material_parameters.w,
            specular_fresnel_color,
        );
    } else {
        let sh_4 = array<vec4<f32>, 4>(
            sh_lighting_coefficients[0],
            sh_lighting_coefficients[1],
            sh_lighting_coefficients[2],
            sh_lighting_coefficients[3],
        );
        area_specular_radiance = calculate_area_specular_phong_order_2(
            view_reflect_dir,
            sh_4,
            material_parameters.w,
            specular_fresnel_color,
        );
    }

    // Specular composition (HLSL line 387):
    //   specular_color.xyz = specular_mask × material_parameters.r × (
    //     (simple_light_specular + max(analytic_specular, 0)) × analytical_specular_contribution +
    //     max(area_specular_radiance × area_specular_contribution, 0))
    let combined =
        (simple_light_specular_light + max(analytic_specular_radiance, vec3<f32>(0.0)))
            * material.analytical_specular_contribution.x
        + max(area_specular_radiance * material.area_specular_contribution.x, vec3<f32>(0.0));
    var specular_color = vec4<f32>(specular_mask * material_parameters.x * combined, 0.0);

    // Modulate with prt — HLSL line 394: `specular_color *= prt_ravi_diff.z`.
    specular_color = specular_color * prt_ravi_diff.z;

    // Envmap outputs (HLSL line 397-399):
    let envmap_area_specular_only = area_specular_radiance * prt_ravi_diff.z;
    let envmap_specular_reflectance_xyz =
        vec3<f32>(material_parameters.z * specular_mask * material_parameters.x);
    let envmap_roughness = max(0.01, 1.01 - material_parameters.w / 200.0);
    let envmap_specular_reflectance_and_roughness =
        vec4<f32>(envmap_specular_reflectance_xyz, envmap_roughness);

    // Diffuse modulation (HLSL line 403-404):
    //   diffuse_radiance = prt_ravi_diff.x × diffuse_radiance
    //   diffuse_radiance = (simple_light_diffuse + diffuse_radiance) × diffuse_coefficient
    var diffuse_radiance = prt_ravi_diff.x * diffuse_radiance_in;
    diffuse_radiance = (simple_light_diffuse_light + diffuse_radiance) * material.diffuse_coefficient.x;

    return MaterialOutput(
        envmap_specular_reflectance_and_roughness,
        envmap_area_specular_only,
        specular_color,
        diffuse_radiance,
    );
}
