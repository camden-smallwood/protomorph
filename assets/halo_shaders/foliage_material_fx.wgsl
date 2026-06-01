// Port of `foliage_material_fx.hlsl` — `calc_material_foliage_ps`.
//
// Structurally identical to `diffuse_only`: pure diffuse, no specular,
// `envmap_specular_reflectance_and_roughness = (1,1,1,0)`,
// `envmap_area_specular_only = 0.282094815 × sh[0].xyz`.
//
// The HLSL writes:
//   diffuse_radiance = diffuse_radiance × prt_ravi_diff.x;
//   diffuse_radiance = simple_light_diffuse_light + diffuse_radiance;
//   specular_radiance = 0;
//
// Material-model signature still uses the legacy `MaterialOutput`
// struct return rather than HLSL's 4 out/inout params — that
// refactor is shared across all material models and will land
// alongside two_lobe_phong's signature port.

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
    let _u4 = analytical_light_dir;
    let _u5 = analytical_light_intensity;
    let _u6 = diffuse_reflectance;
    let _u7 = specular_mask;
    let _u8 = texcoord;

    // HLSL `foliage_material_fx.hlsl:75-91` — gate simple_lights on
    // `if (!no_dynamic_lights)`. Both branches multiply the SH diffuse
    // by `prt_ravi_diff.x`; the gated branch additionally adds the
    // analytical simple_lights diffuse. spec_power = 1.0f, specular_radiance
    // = 0 always (foliage has no specular).
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

    let envmap_area = 0.282094815 * sh_lighting_coefficients[0].xyz;

    return MaterialOutput(
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        envmap_area,
        vec4<f32>(0.0),
        diffuse_radiance,
    );
}
