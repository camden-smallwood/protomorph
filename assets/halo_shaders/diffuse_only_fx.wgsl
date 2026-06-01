// Halo `material_model / diffuse_only` — faithful port of
// `calc_material_diffuse_only_ps` in
// /Users/camden/Halo/halo3_mcc_hlsl_extracted/diffuse_only_fx.hlsl:53.
//
// No specular at all. `envmap_specular_reflectance_and_roughness`
// returns (1, 1, 1, 0) per HLSL — the umbrella's `CALC_ENVMAP` will
// multiply it through; for `env_mapping/none` that's still zero.
// `envmap_area_specular_only` is the SH DC term × constant — Halo's
// "ambient cubemap" approximation.

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
    // Mark unused inputs (HLSL signature parity).
    let _u0 = view_dir;
    let _u1 = fragment_to_camera_world;
    let _u4 = analytical_light_dir;
    let _u5 = analytical_light_intensity;
    let _u6 = diffuse_reflectance;
    let _u7 = specular_mask;
    let _u8 = texcoord;

    // HLSL `diffuse_only_fx.hlsl:74-91` — gate simple_lights on
    // `if (!no_dynamic_lights)`. Both branches multiply the SH diffuse by
    // `prt_ravi_diff.x` (AO); the gated branch additionally adds the
    // analytical simple_lights diffuse.
    const NO_DYNAMIC_LIGHTS: bool = __NO_DYNAMIC_LIGHTS__;
    var diffuse_radiance: vec3<f32>;
    if (!NO_DYNAMIC_LIGHTS) {
        let simple = calc_simple_lights_analytical(
            fragment_position_world,
            surface_normal,
            view_reflect_dir,
            1.0,
        );
        diffuse_radiance = simple.diffuse + diffuse_radiance_in * prt_ravi_diff.x;
    } else {
        diffuse_radiance = diffuse_radiance_in * prt_ravi_diff.x;
    }

    // HLSL line 94: envmap_specular_reflectance_and_roughness = (1,1,1,0).
    // HLSL line 95: envmap_area_specular_only = 0.282094815 * sh[0].xyz.
    let envmap_area = 0.282094815 * sh_lighting_coefficients[0].xyz;

    return MaterialOutput(
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        envmap_area,
        vec4<f32>(0.0),
        diffuse_radiance,
    );
}
