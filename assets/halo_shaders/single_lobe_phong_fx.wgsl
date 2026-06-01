// Halo `material_model / single_lobe_phong` — faithful port of
// `calc_material_single_lobe_phong_ps` in
// /Users/camden/Halo/halo3_mcc_hlsl_extracted/single_lobe_phong_fx.hlsl:67.
//
// Modified-Phong distribution parameterized by `roughness` (rmop
// param). Single specular tint `specular_tint`. No fresnel two-lobe
// blend — that's two_lobe_phong's job.
//
// HLSL composition (line 130):
//   specular_radiance.xyz =
//       (area_specular × area_specular_contribution
//        + analytical_specular × analytical_specular_contribution
//        + simple_light_specular)
//       × specular_coefficient × specular_mask × specular_tint
//
// Cbuffer fields used (declared by the rmop chain at variant assembly):
//   roughness, specular_tint, specular_coefficient, diffuse_coefficient,
//   analytical_specular_contribution, area_specular_contribution,
//   environment_map_specular_contribution.

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
    let _u0 = view_dir;
    let _u1 = fragment_to_camera_world;
    let _u3 = diffuse_reflectance;
    let _u4 = texcoord;

    // Area specular — modified-Phong. HLSL passes the rmt2
    // `order3_area_specular` runtime bool (single_lobe_phong_fx.hlsl:96).
    // Substituted per-variant from the resolved rmop chain.
    const ORDER3_AREA_SPECULAR: bool = __ORDER3_AREA_SPECULAR__;
    let area_specular = calculate_area_specular_new_phong_3(
        view_reflect_dir,
        sh_lighting_coefficients,
        material.roughness.x,
        ORDER3_AREA_SPECULAR,
    );

    // Analytical specular — gated on contribution > 0 per HLSL.
    var analytical_specular = vec3<f32>(0.0);
    if (material.analytical_specular_contribution.x > 0.0) {
        analytical_specular = calculate_analytical_specular_new_phong_3(
            analytical_light_dir,
            analytical_light_intensity,
            view_reflect_dir,
            material.roughness.x,
        );
    }

    // Simple lights — HLSL line 120: spec_power = specular_power_from_roughness()
    // = 0.27291 * pow(roughness, -2.1973), or 0 when roughness == 0.
    // Engine `single_lobe_phong_fx.hlsl:114-128` gates on
    // `if (!no_dynamic_lights)`; the else branch zeroes both lights.
    const NO_DYNAMIC_LIGHTS: bool = __NO_DYNAMIC_LIGHTS__;
    var simple_diffuse = vec3<f32>(0.0);
    var simple_specular = vec3<f32>(0.0);
    if (!NO_DYNAMIC_LIGHTS) {
        let spec_power = select(
            0.27291 * pow(material.roughness.x, -2.1973),
            0.0,
            material.roughness.x == 0.0,
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
    let simple = SimpleLightsResult(simple_diffuse, simple_specular);

    // HLSL composition (line 130):
    let combined =
          area_specular * material.area_specular_contribution.x
        + analytical_specular * material.analytical_specular_contribution.x
        + simple.specular;
    let specular_color = vec4<f32>(
        combined * material.specular_coefficient.x * specular_mask * material.specular_tint.xyz,
        0.0,
    );

    // HLSL line 133: diffuse_radiance = (diffuse_radiance_in + simple_diff) * diffuse_coefficient
    let diffuse_radiance = (diffuse_radiance_in + simple.diffuse) * material.diffuse_coefficient.x;

    // HLSL line 134-136: envmap reflectance is scalar
    //   = specular_coefficient × specular_mask × env_map_spec_contribution.
    let env_mult = material.specular_coefficient.x
                 * specular_mask
                 * material.environment_map_specular_contribution.x;
    let envmap_specular_reflectance_and_roughness = vec4<f32>(env_mult, env_mult, env_mult, material.roughness.x);
    let envmap_area_specular_only = area_specular * prt_ravi_diff.z;

    return MaterialOutput(
        envmap_specular_reflectance_and_roughness,
        envmap_area_specular_only,
        specular_color,
        diffuse_radiance,
    );
}
