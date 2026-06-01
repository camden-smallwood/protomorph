// Halo `material_model / none` — line-for-line port of
// `calc_material_none_ps` in
// /Users/camden/Halo/halo3_mcc_hlsl_extracted/material_model_none_fx.hlsl:38-62.
//
// All material outputs zeroed: diffuse_radiance, specular_radiance,
// envmap_area_specular_only. Used by halogram materials where the
// only visible contribution should come from self_illumination.
//
// `envmap_specular_reflectance_and_roughness = (1, 1, 1, 0)` matches
// the HLSL — convention: with envmap_type=none in the umbrella, this
// gets multiplied through into a zero result anyway.

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
    let _u2 = surface_normal;
    let _u3 = view_reflect_dir;
    let _u4 = sh_lighting_coefficients;
    let _u5 = analytical_light_dir;
    let _u6 = analytical_light_intensity;
    let _u7 = diffuse_reflectance;
    let _u8 = specular_mask;
    let _u9 = texcoord;
    let _u10 = prt_ravi_diff;
    let _u11 = diffuse_radiance_in;
    let _u12 = fragment_position_world;

    return MaterialOutput(
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec3<f32>(0.0),
        vec4<f32>(0.0),
        vec3<f32>(0.0),
    );
}
