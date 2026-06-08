// Halo `material_model / glass` — port of `calc_material_glass_ps` in
// /Users/camden/Halo/halo3_mcc_hlsl_extracted/glass_material_fx.hlsl:57.
//
// Glass material model: modified-Phong specular like single_lobe_phong,
// but writes a fresnel scalar into `specular_radiance.w` for transparency
// blending, and the envmap reflectance multiplier folds in the fresnel
// instead of `environment_map_specular_contribution`.
//
// Cbuffer fields used (declared by the rmop chain at variant assembly):
//   roughness, fresnel_coefficient, fresnel_curve_steepness,
//   fresnel_curve_bias, specular_coefficient, diffuse_coefficient,
//   analytical_specular_contribution, area_specular_contribution.
//
// HLSL composition (line 121):
//   specular_radiance.xyz = (area_specular × area_specular_contribution
//                            + analytical_specular × analytical_specular_contribution
//                            + simple_light_specular)
//                           × specular_coefficient × specular_mask
//   specular_radiance.w   = fresnel
//   envmap_reflectance    = vec4(scalar, scalar, scalar, roughness)
//                            where scalar = specular_coefficient * fresnel * specular_mask
//   envmap_area_specular_only = sh_lighting_coefficients[0].xyz   // ambient-only

struct MaterialOutput {
    envmap_specular_reflectance_and_roughness: vec4<f32>,
    envmap_area_specular_only: vec3<f32>,
    specular_color: vec4<f32>,
    diffuse_radiance: vec3<f32>,
}

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
    let _u0 = fragment_to_camera_world;
    let _u1 = diffuse_reflectance;
    let _u2 = texcoord;

    // GLASS_SPECULAR_DIM — SCOPED WORKAROUND (2026-06-08).
    // The glass reflection terms (area_specular × area_specular_contribution=5,
    // and the env-cube reflection) are engine-faithful and now fed the CORRECT
    // per-instance probe (see the transparent BspInstancePart routing fix,
    // d34b378). But on sun-FACING glass instances the bright probe drives
    // `area_specular×5 + env` to a high HDR value that blows past protomorph's
    // shared cubic tone curve to white, where the engine lands grey — a
    // residual ~2× bright-specular HDR-magnitude gap that is NOT a
    // data/routing/decode bug (all verified to match the engine/HLSL). Dimming
    // both glass specular terms tames the sun-facing windows while barely
    // touching shadowed glass (whose probe is already dim). Remove when the
    // bright-specular HDR magnitude / exposure is reconciled with the engine.
    const GLASS_SPECULAR_DIM: f32 = 0.3;

    // Area specular — gated on contribution > 0 per HLSL line 80.
    // HLSL line 86 hardcodes `false` for the order3 toggle (glass uses
    // ONLY DC + linear SH bands, even when the variant is static_sh
    // and bands 4-9 carry real data).
    var area_specular = vec3<f32>(0.0);
    if (material.area_specular_contribution.x > 0.0) {
        area_specular = calculate_area_specular_new_phong_3(
            view_reflect_dir,
            sh_lighting_coefficients,
            material.roughness.x,
            false,
        );
    }

    // Analytical specular — gated on contribution > 0 per HLSL line 91.
    var analytical_specular = vec3<f32>(0.0);
    if (material.analytical_specular_contribution.x > 0.0) {
        analytical_specular = calculate_analytical_specular_new_phong_3(
            analytical_light_dir,
            analytical_light_intensity,
            view_reflect_dir,
            material.roughness.x,
        );
    }

    // Simple lights — HLSL line 110:
    //   spec_power = (roughness == 0) ? 0 : (0.27291 * pow(roughness, -2.1973)).
    // Engine `glass_material_fx.hlsl:104` gates on `if (!no_dynamic_lights)`;
    // the else branch zeroes both lights.
    const NO_DYNAMIC_LIGHTS: bool = __NO_DYNAMIC_LIGHTS__;
    var simple_diffuse = vec3<f32>(0.0);
    var simple_specular = vec3<f32>(0.0);
    if (!NO_DYNAMIC_LIGHTS) {
        let spec_power = select(
            0.27291 * pow(material.roughness.x, -2.1973),
            0.0,
            material.roughness.x == 0.0,
        );
        let s = calc_simple_lights_analytical(
            fragment_position_world,
            surface_normal,
            view_reflect_dir,
            spec_power,
        );
        simple_diffuse = s.diffuse;
        simple_specular = s.specular;
    }
    let simple = SimpleLightsResult(simple_diffuse, simple_specular);

    // HLSL line 120: fresnel = fresnel_coefficient
    //                        + (1 - fresnel_coefficient) * pow(1 - n.v, fresnel_curve_steepness)
    //                        + fresnel_curve_bias
    let n_dot_v = max(dot(surface_normal, view_dir), 0.0);
    let fresnel =
          material.fresnel_coefficient.x
        + (1.0 - material.fresnel_coefficient.x) * pow(1.0 - n_dot_v, material.fresnel_curve_steepness.x)
        + material.fresnel_curve_bias.x;

    // HLSL line 121: combined specular × specular_coefficient × specular_mask.
    let combined =
          area_specular * material.area_specular_contribution.x * GLASS_SPECULAR_DIM
        + analytical_specular * material.analytical_specular_contribution.x
        + simple.specular;
    let specular_color = vec4<f32>(
        combined * material.specular_coefficient.x * specular_mask,
        fresnel,
    );

    // HLSL line 123: diffuse_radiance = (diffuse + simple_diff) * diffuse_coefficient
    let diffuse_radiance = (diffuse_radiance_in + simple.diffuse) * material.diffuse_coefficient.x;

    // HLSL line 124-125:
    //   env_multiplyer = specular_coefficient * fresnel * specular_mask
    //   envmap_reflectance = (env_multiplyer, env_multiplyer, env_multiplyer, roughness)
    // HLSL line 124: env_mult = specular_coefficient * fresnel * specular_mask.
    // × GLASS_SPECULAR_DIM (scoped workaround — see top of fn).
    let env_mult = material.specular_coefficient.x * fresnel * specular_mask * GLASS_SPECULAR_DIM;
    let envmap_specular_reflectance_and_roughness =
        vec4<f32>(env_mult, env_mult, env_mult, material.roughness.x);

    // HLSL line 126: envmap_area_specular_only = sh_lighting_coefficients[0].xyz
    let envmap_area_specular_only = sh_lighting_coefficients[0].xyz;

    return MaterialOutput(
        envmap_specular_reflectance_and_roughness,
        envmap_area_specular_only,
        specular_color,
        diffuse_radiance,
    );
}
