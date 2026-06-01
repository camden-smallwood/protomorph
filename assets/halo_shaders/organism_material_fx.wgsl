// Port of `organism_material_fx.hlsl::calc_material_organism_ps`
// (line 121-346). Faithful translation of the "sub-translucent
// organism" material model — used by Flood biomass on isolation /
// other flood maps. Combines analytic + area Phong specular with
// rim, ambient, subsurface scattering, and view-aligned transparency
// FORCE_BRANCH blocks gated by their coefficient > 0.01.
//
// Cbuffer PARAMs (declared by rmop chain at variant assembly):
//   diffuse_tint, diffuse_coefficient
//   area_specular_coefficient, analytical_specular_coefficient
//   specular_tint, specular_power
//   environment_map_coefficient, environment_map_tint
//   fresnel_curve_steepness                  (declared, used by upstream env eval)
//   rim_coefficient, rim_tint, rim_power,
//   rim_start, rim_maps_transition_ratio
//   ambient_coefficient, ambient_tint
//   subsurface_coefficient, subsurface_tint,
//   subsurface_propagation_bias, subsurface_normal_detail
//   transparence_coefficient, transparence_tint,
//   transparence_normal_bias, transparence_normal_detail
//   final_tint
//   analytical_anti_shadow_control           (declared, currently unused)
//
// Texture PARAMs (resolved per-variant by material_bindings):
//   specular_map, occlusion_parameter_map,
//   subsurface_map, transparence_map
//
// **Engine-faithful caveats:**
// - The HLSL's `calculate_area_specular_phong_order_2` is a custom
//   organism-only helper (organism_material_fx.hlsl:91-115) that drops
//   the DC (`s0 = x1` not `x0 + x1`). Ported verbatim below as
//   `organism_area_specular_phong_order_2`.
// - The `calc_simple_lights_analytical_diffuse_translucent` call inside
//   the transparency block exists in the engine shared HLSL —
//   protomorph approximates with the standard `calc_simple_lights_analytical`
//   diffuse output for v1.
// - PARAMs are rmt2-packed as vec4 — scalar fields read via `.x`.

struct MaterialOutput {
    envmap_specular_reflectance_and_roughness: vec4<f32>,
    envmap_area_specular_only: vec3<f32>,
    specular_color: vec4<f32>,
    diffuse_radiance: vec3<f32>,
}

// Engine `calc_material_analytic_specular` (organism_material_fx.hlsl:71-88).
// Standard normalized Phong: `(L·R)^p * (p+1)/(2π) * I`, clamped to 0.
fn organism_analytic_specular(
    view_reflect_dir: vec3<f32>,
    analytical_light_dir: vec3<f32>,
    analytical_light_intensity: vec3<f32>,
    power_or_roughness: f32,
) -> vec3<f32> {
    let l_dot_r = dot(analytical_light_dir, view_reflect_dir);
    if (l_dot_r > 0.0) {
        return pow(l_dot_r, power_or_roughness)
             * ((power_or_roughness + 1.0) / 6.2832)
             * analytical_light_intensity;
    }
    return vec3<f32>(0.0);
}

// Engine `calculate_area_specular_phong_order_2`
// (organism_material_fx.hlsl:91-115). Order-2 variant that DROPS the DC
// term — the HLSL writes `s0 = x1` (line 114) not `x0 + x1`.
fn organism_area_specular_phong_order_2(
    reflection_dir: vec3<f32>,
    sh: array<vec4<f32>, 10>,
) -> vec3<f32> {
    let p_1: f32 = -0.3805236;
    // L1 projection per channel — sh[1..3] hold L1 bands packed as RGB.
    var x1: vec3<f32>;
    x1.r = dot(reflection_dir, sh[1].xyz);
    x1.g = dot(reflection_dir, sh[2].xyz);
    x1.b = dot(reflection_dir, sh[3].xyz);
    return x1 * p_1;
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
    let _u_diff_refl = diffuse_reflectance;
    let _u_spec_mask = specular_mask;

    // HLSL line 140: `surface_normal = tangent_frame[2]` — protomorph
    // already passes the bumped normal in as `surface_normal`. The
    // HLSL distinguishes `bump_normal` vs `tangent_frame[2]`; in our
    // pipeline both arrive as the same input. Subsurface/transparence
    // uses `lerp(surface_normal, bump_normal, *_normal_detail)` —
    // collapses to `surface_normal` when they're identical.
    let bump_normal = surface_normal;

    // HLSL line 143-144: sample specular map for power-or-roughness.
    let specular_map_color = textureSample(specular_map, specular_map_sampler, texcoord);
    let power_or_roughness = specular_map_color.a * material.specular_power.x;

    // HLSL line 147-162: simple-lights branch gated on no_dynamic_lights.
    const NO_DYNAMIC_LIGHTS: bool = __NO_DYNAMIC_LIGHTS__;
    var simple_lights_bump_diffuse = vec3<f32>(0.0);
    var simple_lights_bump_specular = vec3<f32>(0.0);
    if (!NO_DYNAMIC_LIGHTS) {
        let spec_factor = power_or_roughness * dot(specular_map_color.rgb, specular_map_color.rgb);
        let simple = calc_simple_lights_analytical(
            fragment_position_world,
            bump_normal,
            view_reflect_dir,
            spec_factor,
        );
        simple_lights_bump_diffuse = simple.diffuse;
        // HLSL line 161: lower power magic — 0.33 attenuation.
        simple_lights_bump_specular = simple.specular * 0.33;
    }

    // HLSL line 164-170: diffuse color = (simple + SH) × coef × tint.
    let diffuse_color =
        (simple_lights_bump_diffuse + diffuse_radiance_in)
        * material.diffuse_coefficient.x
        * material.diffuse_tint.xyz;

    // HLSL line 172-198: analytic + area Phong specular.
    let analytic_specular_radiance = organism_analytic_specular(
        view_reflect_dir,
        analytical_light_dir,
        analytical_light_intensity,
        power_or_roughness,
    );
    var area_specular_radiance = organism_area_specular_phong_order_2(
        view_reflect_dir,
        sh_lighting_coefficients,
    );
    area_specular_radiance = max(area_specular_radiance, vec3<f32>(0.0));

    var specular_color =
        analytic_specular_radiance * material.analytical_specular_coefficient.x
        + area_specular_radiance * material.area_specular_coefficient.x
        + simple_lights_bump_specular * material.analytical_specular_coefficient.x;
    specular_color = specular_color * material.specular_tint.xyz * specular_map_color.rgb;

    // HLSL line 200-205: environment params.
    let envmap_area = vec3<f32>(prt_ravi_diff.z);
    let envmap_spec_rr = vec4<f32>(
        material.environment_map_tint.xyz
            * material.environment_map_coefficient.x
            * specular_map_color.rgb,
        1.0,
    );

    // HLSL line 209-211: occlusion params (xy channel).
    let occlusion_map_value = textureSample(occlusion_parameter_map, occlusion_parameter_map_sampler, texcoord).xy;
    let ambient_occlusion = occlusion_map_value.x;
    // visibility_occlusion currently unused downstream — HLSL only
    // reads it back into the diffuse pipeline through SH evaluation
    // upstream, which protomorph applies in the calling entry.
    let _u_vis_occlusion = occlusion_map_value.y;

    // HLSL line 213-228: rim lighting FORCE_BRANCH on rim_coefficient > 0.01.
    var rim_color_specular = vec3<f32>(0.0);
    var rim_color_diffuse = vec3<f32>(0.0);
    if (material.rim_coefficient.x > 0.01) {
        let v_dot_n = dot(view_dir, bump_normal);
        var rim_ratio = saturate(1.0 - v_dot_n);
        rim_ratio = saturate((rim_ratio - material.rim_start.x) / max(1.0 - material.rim_start.x, 0.001));
        rim_ratio = pow(rim_ratio, material.rim_power.x);
        let rim_color = analytical_light_intensity * rim_ratio
            * material.rim_tint.xyz * material.rim_coefficient.x;
        rim_color_specular = rim_color * material.rim_maps_transition_ratio.x * specular_map_color.rgb;
        rim_color_diffuse = rim_color * (1.0 - material.rim_maps_transition_ratio.x);
    }

    // HLSL line 230-240: ambient FORCE_BRANCH on ambient_coefficient > 0.01.
    var ambient_color = vec3<f32>(0.0);
    if (material.ambient_coefficient.x > 0.01) {
        ambient_color = ambient_occlusion
            * sh_lighting_coefficients[0].xyz
            * material.ambient_tint.xyz
            * material.ambient_coefficient.x;
    }

    // HLSL line 242-283: subsurface FORCE_BRANCH on subsurface_coefficient > 0.01.
    var subsurface_color = vec3<f32>(0.0);
    if (material.subsurface_coefficient.x > 0.01) {
        let subsurface_map_color = textureSample(subsurface_map, subsurface_map_sampler, texcoord);
        let subsurface_normal = normalize(
            mix(surface_normal, bump_normal, material.subsurface_normal_detail.x)
            + analytical_light_dir
                * material.subsurface_propagation_bias.x
                * subsurface_map_color.w
        );
        var area_radiance_subsurface = organism_area_specular_phong_order_2(
            subsurface_normal,
            sh_lighting_coefficients,
        );
        area_radiance_subsurface = max(area_radiance_subsurface, vec3<f32>(0.0));
        // HLSL line 262-275: simple-lights subsurface (diffuse only kept,
        // simple_lights_subsurface_specular discarded per source comment
        // around line 267).
        var simple_lights_subsurface_diffuse = vec3<f32>(0.0);
        if (!NO_DYNAMIC_LIGHTS) {
            let s_sub = calc_simple_lights_analytical(
                fragment_position_world,
                surface_normal,
                vec3<f32>(0.0, 0.0, 1.0),
                power_or_roughness,
            );
            simple_lights_subsurface_diffuse = s_sub.diffuse;
        }
        subsurface_color =
            (area_radiance_subsurface + simple_lights_subsurface_diffuse)
            * material.subsurface_tint.xyz
            * material.subsurface_coefficient.x
            * subsurface_map_color.xyz
            * ambient_occlusion;
    }

    // HLSL line 285-327: transparency FORCE_BRANCH on transparence_coefficient > 0.01.
    var transparence_color = vec3<f32>(0.0);
    if (material.transparence_coefficient.x > 0.01) {
        let transparence_map_color = textureSample(transparence_map, transparence_map_sampler, texcoord);
        var area_radiance_transparence = organism_area_specular_phong_order_2(
            -view_dir,
            sh_lighting_coefficients,
        );
        area_radiance_transparence = max(area_radiance_transparence, vec3<f32>(0.0));
        // HLSL line 301-305: `calc_simple_lights_analytical_diffuse_translucent`.
        // v1 approximation: standard analytical diffuse (no -view eval).
        var dynamic_radiance_transparence = vec3<f32>(0.0);
        if (!NO_DYNAMIC_LIGHTS) {
            let t_sl = calc_simple_lights_analytical(
                fragment_position_world,
                -view_dir,
                view_reflect_dir,
                0.0,
            );
            dynamic_radiance_transparence = t_sl.diffuse;
        }
        // HLSL line 308-319: normal_bias from view alignment to bump.
        let transparence_normal = normalize(
            mix(surface_normal, bump_normal, material.transparence_normal_detail.x)
        );
        var normal_weight = dot(view_dir, transparence_normal);
        if (material.transparence_normal_bias.x < 0.0) {
            normal_weight = 1.0 - normal_weight;
        }
        let normal_bias = saturate(
            1.0 - normal_weight
                * abs(material.transparence_normal_bias.x * transparence_map_color.w)
        );
        transparence_color =
            (area_radiance_transparence + dynamic_radiance_transparence)
            * normal_bias
            * material.transparence_tint.xyz
            * material.transparence_coefficient.x
            * transparence_map_color.xyz;
    }

    // HLSL line 329-345: output assembly + final tint.
    var output_color_xyz =
        rim_color_specular + specular_color + subsurface_color + transparence_color;
    var diffuse_radiance_out = diffuse_color + rim_color_diffuse + ambient_color;
    output_color_xyz = output_color_xyz * material.final_tint.xyz;
    diffuse_radiance_out = diffuse_radiance_out * material.final_tint.xyz;

    return MaterialOutput(
        envmap_spec_rr,
        envmap_area,
        vec4<f32>(output_color_xyz, 0.0),
        diffuse_radiance_out,
    );
}
