// Port of `cook_torrance_fx.hlsl` — `cook_torrance_pbr_maps` variant.
// Mirrors `calc_material_cook_torrance_pbr_maps_ps` (HLSL line 1535) +
// `calc_material_analytic_specular_cook_torrance_pbr_maps_ps` (line 386).
//
// HLSL hierarchy (cook_torrance_fx.hlsl):
//   calc_material_cook_torrance_pbr_maps_ps(...)        // line 1535
//     → calc_material_analytic_specular_cook_torrance_pbr_maps_ps(...)  // line 386
//     → calculate_ambientness(...)         // shared
//     → calc_simple_lights_analytical(...) // shared (stubbed in v1)
//     → sh_glossy_ct_3(...)                // line 922 (shared with cook_torrance)
//
// **Differences vs `cook_torrance` (cbuffer `specular_tint`):**
//   - `spec_tint` is sampled from `spec_tint_map` (HLSL line 1555) — a
//     per-pixel tint texture, NOT the cbuffer constant. v1 falls back
//     to the cbuffer `specular_tint` (same as cook_torrance v1).
//   - `spatially_varying_material_parameters.r` (specular coefficient)
//     and `.a` (roughness) are sampled from `material_texture` (HLSL
//     line 406, `.xxyy` swizzle). v1 falls back to the cbuffer
//     `specular_coefficient` / `roughness` (same as cook_torrance with
//     `use_material_texture=false`).
//   - The HLSL `if (use_material_texture)` branch present in cook_torrance
//     does NOT exist here — pbr_maps unconditionally samples. Once the
//     material_texture binding is wired (TODO), pbr_maps must always
//     sample, but cook_torrance is gated.
//
// **Status:** under v1 cbuffer fallbacks the math is identical to
// cook_torrance. Real per-pixel material_texture / spec_tint_map sampling
// requires wiring those bindings — separate task. Until then weapons /
// equipment using this shader render with cbuffer-uniform spec params,
// no per-pixel PBR detail.

const SQRT3: f32 = 1.73205080756;
const c_view_z_shift: f32 = 0.015625; // 0.5 / 32.0
const c_roughness_shift: f32 = 0.0;

struct MaterialOutput {
    envmap_specular_reflectance_and_roughness: vec4<f32>,
    envmap_area_specular_only: vec3<f32>,
    specular_color: vec4<f32>,
    diffuse_radiance: vec3<f32>,
}

// `calculate_ambientness` (spherical_harmonics_fx.hlsl). Verbatim port
// of cook_torrance_fx.wgsl's copy.
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

// `sh_rotate_023` (cook_torrance_fx.hlsl line ~835).
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

// `sh_glossy_ct_2` (cook_torrance_fx.hlsl line 854) — order-2 variant
// used when `order3_area_specular` is false. Drops the L2 basis
// projections; only DC + L1 contribute.
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

    var sh_local = vec4<f32>(sh_rotate_023(0, rotate_x, rotate_z, sh_0, sh_312), 0.0);
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    let sp_r = dot(c_value, sh_local);
    let sk_r = dot(d_value, sh_local);

    sh_local = vec4<f32>(sh_rotate_023(1, rotate_x, rotate_z, sh_0, sh_312), 0.0);
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    let sp_g = dot(c_value, sh_local);
    let sk_g = dot(d_value, sh_local);

    sh_local = vec4<f32>(sh_rotate_023(2, rotate_x, rotate_z, sh_0, sh_312), 0.0);
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    let sp_b = dot(c_value, sh_local);
    let sk_b = dot(d_value, sh_local);

    *specular_part = vec3<f32>(sp_r, sp_g, sp_b);
    *schlick_part = vec3<f32>(sk_r, sk_g, sk_b) * 0.01;
}

// `sh_glossy_ct_3` (cook_torrance_fx.hlsl line 922) — verbatim port
// of cook_torrance_fx.wgsl's copy.
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
    let rotate_x = normalize(view_dir - dot(view_dir, rotate_z) * rotate_z);
    let rotate_y = cross(rotate_z, rotate_x);

    let t_roughness = max(roughness, 0.05);
    let view_lookup = vec2<f32>(
        pow(max(dot(view_dir, rotate_x), 0.0001), power) + c_view_z_shift,
        t_roughness + c_roughness_shift,
    );

    var c_value = textureSample(g_sampler_cc0236, g_sampler_cc0236_sampler, view_lookup);
    let d_value = textureSample(g_sampler_dd0236, g_sampler_dd0236_sampler, view_lookup);

    var quadratic_a: vec4<f32>;
    var quadratic_b: vec4<f32>;
    var sh_local: vec4<f32>;

    quadratic_a = vec4<f32>(rotate_z.yzx * rotate_z.xyz * (-SQRT3), 0.0);
    quadratic_b = vec4<f32>(rotate_z.xyz * rotate_z.xyz, 1.0 / 3.0) * 0.5 * (-SQRT3);

    sh_local = vec4<f32>(
        sh_rotate_023(0, rotate_x, rotate_z, sh_0, sh_312),
        dot(quadratic_a.xyz, sh_457[0].xyz) + dot(quadratic_b, sh_8866[0]),
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    var sp_r = dot(c_value, sh_local);
    var sk_r = dot(d_value, sh_local);

    sh_local = vec4<f32>(
        sh_rotate_023(1, rotate_x, rotate_z, sh_0, sh_312),
        dot(quadratic_a.xyz, sh_457[1].xyz) + dot(quadratic_b, sh_8866[1]),
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    var sp_g = dot(c_value, sh_local);
    var sk_g = dot(d_value, sh_local);

    sh_local = vec4<f32>(
        sh_rotate_023(2, rotate_x, rotate_z, sh_0, sh_312),
        dot(quadratic_a.xyz, sh_457[2].xyz) + dot(quadratic_b, sh_8866[2]),
    );
    sh_local = sh_local * vec4<f32>(1.0, r_dot_l, r_dot_l, r_dot_l);
    var sp_b = dot(c_value, sh_local);
    var sk_b = dot(d_value, sh_local);

    var sp = vec3<f32>(sp_r, sp_g, sp_b);
    var sk = vec3<f32>(sk_r, sk_g, sk_b);

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

struct CtAnalyticOutput {
    spatially_varying_material_parameters: vec4<f32>,
    specular_fresnel_color: vec3<f32>,
    specular_albedo_color: vec3<f32>,
    analytic_specular_radiance: vec3<f32>,
}

// `calc_material_analytic_specular_cook_torrance_pbr_maps_ps`
// (cook_torrance_fx.hlsl line 386). pbr_maps difference vs cook_torrance:
// material params .r/.g come from `material_texture` per-pixel; .y/.z
// stay cbuffer (`albedo_blend` / `environment_map_specular_contribution`).
fn calc_material_analytic_specular_cook_torrance_pbr_maps_ps(
    view_dir: vec3<f32>,
    normal_dir: vec3<f32>,
    view_reflect_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_irradiance: vec3<f32>,
    diffuse_albedo_color: vec3<f32>,
    vertex_n_dot_l: f32,
    texcoord: vec2<f32>,
) -> CtAnalyticOutput {
    let _u_reflect = view_reflect_dir;

    // HLSL line 406-409 (engine-faithful per-pixel sample):
    //   mp = sample(material_texture, texcoord).xxyy
    //      → (tex.r, tex.r, tex.g, tex.g)
    //   mp.y = albedo_blend                              (overwrite g)
    //   mp.z = environment_map_specular_contribution     (overwrite b)
    //   → mp = (tex.r, albedo_blend, env_contrib, tex.g)
    let mtex = textureSample(material_texture, material_texture_sampler, texcoord);
    let mp = vec4<f32>(
        mtex.r,
        material.albedo_blend.x,
        material.environment_map_specular_contribution.x,
        mtex.g,
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

        let f0 = min(specular_albedo_color, vec3<f32>(0.999));
        let sqrt_f0 = sqrt(f0);
        let n_ior = (1.0 + sqrt_f0) / (1.0 - sqrt_f0);
        let g = sqrt(n_ior * n_ior + v_dot_h * v_dot_h - 1.0);
        let gpc = g + v_dot_h;
        let gmc = g - v_dot_h;
        let r = (v_dot_h * gpc - 1.0) / (v_dot_h * gmc + 1.0);
        fresnel = 0.5 * (gmc * gmc / (gpc * gpc + 0.00001)) * (1.0 + r * r);

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

// `calc_material_cook_torrance_pbr_maps_ps` (cook_torrance_fx.hlsl
// line 1535). pbr_maps difference vs cook_torrance: `spec_tint` is
// sampled from `spec_tint_map` (line 1555); v1 fallback uses cbuffer
// `specular_tint`. Otherwise structurally identical to cook_torrance.
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

    // HLSL line 1555: spec_tint = sample2D(spec_tint_map, texcoord).xyz.
    let spec_tint = textureSample(spec_tint_map, spec_tint_map_sampler, texcoord).xyz;

    let analytic = calc_material_analytic_specular_cook_torrance_pbr_maps_ps(
        view_dir,
        surface_normal,
        view_reflect_dir,
        analytical_light_dir,
        analytical_light_intensity,
        diffuse_reflectance,
        prt_ravi_diff.w,
        texcoord,
    );
    let svmp = analytic.spatially_varying_material_parameters;
    let effective_reflectance = analytic.specular_albedo_color;
    var specular_analytical = analytic.analytic_specular_radiance;

    // Anti-shadow attenuation (HLSL line 1586).
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

    // Simple lights — HLSL line 1607 passes
    // `GET_MATERIAL_SPECULAR_POWER(cook_torrance)(svmp.a)` =
    // `0.27291 * pow(roughness, -2.1973)` when roughness != 0, else 0.
    // Engine `cook_torrance_fx.hlsl:1600` gates on `if (!no_dynamic_lights)`;
    // the else branch zeroes both lights.
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

    // Area specular — HLSL line 1631/1650 branches between sh_glossy_ct_3
    // (10 SH coefs) and sh_glossy_ct_2 (4 coefs) on the rmt2 bool
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

    // HLSL line 1664: spec_tint multiplies envmap_area_specular_only.
    let envmap_area = sh_glossy * prt_ravi_diff.z * spec_tint;
    let envmap_specular_reflectance_xyz =
        vec3<f32>(svmp.z * specular_mask * svmp.x);
    let envmap_specular_reflectance_and_roughness =
        vec4<f32>(envmap_specular_reflectance_xyz, svmp.w);

    // HLSL line 1668: specular_color uses spec_tint (sampled, not cbuffer).
    let combined =
          (simple_specular * effective_reflectance + specular_analytical)
              * material.analytical_specular_contribution.x
        + max(sh_glossy, vec3<f32>(0.0)) * material.area_specular_contribution.x;
    var specular_color =
        vec4<f32>(specular_mask * svmp.x * spec_tint * combined, 0.0);
    specular_color = specular_color * prt_ravi_diff.z;

    // Engine `cook_torrance_fx.hlsl:1677-1680`:
    //   if (use_material_texture) diffuse_adjusted = 1.0 - svmp.r;
    //   else                      diffuse_adjusted = diffuse_coefficient;
    // For pbr_maps the variant unconditionally samples `material_texture`
    // (line 1671), so `use_material_texture` is effectively always true.
    // `svmp.r` = sampled specular_coefficient; `1 - r` gives the
    // metalness-style diffuse falloff that defines pbr_maps materials.
    let diffuse_adjusted = 1.0 - svmp.x;
    var diffuse_radiance = diffuse_radiance_in * prt_ravi_diff.x;
    diffuse_radiance = (simple_diffuse + diffuse_radiance) * diffuse_adjusted;

    return MaterialOutput(
        envmap_specular_reflectance_and_roughness,
        envmap_area,
        specular_color,
        diffuse_radiance,
    );
}
