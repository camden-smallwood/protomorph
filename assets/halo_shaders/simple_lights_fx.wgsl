// Port of `simple_lights_fx.hlsl` (177 lines). Provides per-fragment
// dynamic-light evaluation for up to 8 simple point/spot lights from
// the `simple_lights` cbuffer.
//
// Cbuffer layout — 8 lights × 5 vec4 + 1 vec4 count header (656 bytes
// total). See `crate::halo::lighting::GpuSimpleLights`.
//
//   count.x          = simple_light_count
//   lights[i].pos    = (position.xyz, light_source_size)
//   lights[i].dir    = (inv_direction.xyz, sphere_percentage)
//   lights[i].color  = (color.xyz, smoothness)
//   lights[i].falloff= (distance_scale, cone_scale, distance_offset, cone_offset)
//   lights[i].bound  = (bounding_radius2, _, _, _)
//
// Used by:
//   - decorator DynamicLights / WindDynamicLights / WavyDynamicLights /
//     ShadedDynamicLights (entry_decorator.wgsl, VS-side)
//   - foliage_fx.wgsl (PS-side via `calc_simple_lights_analytical`)
//   - cook_torrance / two_lobe_phong / single_lobe_phong PS material
//     models that have a simple-light specular term

struct SimpleLight {
    position_size: vec4<f32>,
    inv_direction_sphere: vec4<f32>,
    color_smooth: vec4<f32>,
    falloff_scale_offset: vec4<f32>,
    bounding_radius2: vec4<f32>,
};

struct SimpleLights {
    count: vec4<f32>,
    lights: array<SimpleLight, 8>,
};

@group(0) @binding(8) var<uniform> simple_lights: SimpleLights;

/// `calculate_simple_light` (simple_lights_fx.hlsl:18). Computes the
/// per-light radiance + normalized fragment-to-light vector. Returns
/// a 4-tuple via struct (WGSL has no out-params for vector returns).
struct SimpleLightSample {
    light_radiance: vec3<f32>,
    fragment_to_light: vec3<f32>,
};

fn calculate_simple_light(
    light_index: u32,
    fragment_position_world: vec3<f32>,
) -> SimpleLightSample {
    let l = simple_lights.lights[light_index];
    let light_position = l.position_size.xyz;
    let light_size = l.position_size.w;
    let inv_dir = l.inv_direction_sphere.xyz;
    let sphere = l.inv_direction_sphere.w;
    let color = l.color_smooth.xyz;
    let smooth_exp = l.color_smooth.w;
    let falloff_scale = l.falloff_scale_offset.xy;
    let falloff_offset = l.falloff_scale_offset.zw;

    var to_light = light_position - fragment_position_world;
    let dist2 = dot(to_light, to_light);
    let inv_dist = inverseSqrt(max(dist2, 1e-12));
    to_light = to_light * inv_dist;

    // Faithful port of `simple_lights_fx.hlsl:46-50`:
    //   falloff.x = 1 / (size + dist²)
    //   falloff.y = dot(to_light, inv_direction)
    //   falloff   = max(0.0001, falloff × scale + offset)
    //   falloff.y = pow(falloff.y, smooth) + sphere
    //   combined  = saturate(falloff.x) × saturate(falloff.y)
    var falloff = vec2<f32>(
        1.0 / (light_size + dist2),
        dot(to_light, inv_dir),
    );
    falloff = max(vec2<f32>(0.0001), falloff * falloff_scale + falloff_offset);
    falloff.y = pow(falloff.y, smooth_exp) + sphere;
    let combined = clamp(falloff.x, 0.0, 1.0) * clamp(falloff.y, 0.0, 1.0);

    return SimpleLightSample(
        color * combined,
        to_light,
    );
}

/// `calc_simple_lights_analytical` (simple_lights_fx.hlsl:55). Diffuse
/// + specular accumulation across all 8 simple lights. Each light is
/// distance-culled against `bounding_radius2`.
struct SimpleLightsResult {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

fn calc_simple_lights_analytical(
    fragment_position_world: vec3<f32>,
    surface_normal: vec3<f32>,
    view_reflect_dir_world: vec3<f32>,
    specular_power: f32,
) -> SimpleLightsResult {
    var diffuse = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);
    let count = u32(simple_lights.count.x);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let l = simple_lights.lights[i];
        let to_light_test = l.position_size.xyz - fragment_position_world;
        let dist2 = dot(to_light_test, to_light_test);
        if (dist2 >= l.bounding_radius2.x) {
            continue;
        }
        let s = calculate_simple_light(i, fragment_position_world);

        // Diffuse: cosine lobe with 5% ambient floor (engine line 95).
        let cosine = dot(surface_normal, s.fragment_to_light);
        diffuse = diffuse + s.light_radiance * max(0.05, cosine);

        // Specular: clamped Phong lobe (engine line 99). `safe_pow`
        // there is `pow(max(x, 0), n)` — directly here.
        let r_dot_l = max(0.0, dot(s.fragment_to_light, view_reflect_dir_world));
        specular = specular + s.light_radiance * pow(r_dot_l, specular_power);
    }
    specular = specular * specular_power;

    return SimpleLightsResult(diffuse, specular);
}

/// `calc_simple_lights_analytical_diffuse_translucent` (engine line 136).
/// Diffuse-only path used by foliage / decorator dynamic variants.
/// Engine line 169 commented out the cosine modulation — we mirror that
/// (no surface_normal × to_light cosine; just radiance accumulation).
fn calc_simple_lights_analytical_diffuse_translucent(
    fragment_position_world: vec3<f32>,
    surface_normal: vec3<f32>,
    translucency: vec3<f32>,
) -> vec3<f32> {
    var diffuse = vec3<f32>(0.0);
    let count = u32(simple_lights.count.x);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let l = simple_lights.lights[i];
        let to_light_test = l.position_size.xyz - fragment_position_world;
        let dist2 = dot(to_light_test, to_light_test);
        if (dist2 >= l.bounding_radius2.x) {
            continue;
        }
        let s = calculate_simple_light(i, fragment_position_world);
        // Engine line 169 has the cosine commented out — pure radiance
        // accumulation. We mirror the engine's choice exactly.
        diffuse = diffuse + s.light_radiance;
    }
    return diffuse;
}

/// `calc_diffuse_lobe` (engine line 113). Saturated cosine lobe.
fn calc_diffuse_lobe(
    fragment_normal: vec3<f32>,
    fragment_to_light: vec3<f32>,
) -> f32 {
    return clamp(dot(fragment_normal, fragment_to_light), 0.0, 1.0);
}
