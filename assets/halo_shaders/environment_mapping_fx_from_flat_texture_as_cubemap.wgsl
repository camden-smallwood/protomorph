// Port of `environment_mapping_fx.hlsl` —
// `from_flat_texture_as_cubemap` variant (line 156-202 + 204-217).
//
// Identical math to `from_flat_texture` for computing the fisheye
// projection (envmap_dir + radius + texcoord), but the final sample
// is via a cubemap. HLSL:
//
//   envmap_dir.x = dot(reflect_dir, flat_envmap_matrix_x.xyz);
//   envmap_dir.y = dot(reflect_dir, flat_envmap_matrix_y.xyz);
//   envmap_dir.z = dot(reflect_dir, flat_envmap_matrix_z.xyz);
//   radius   = sqrt((envmap_dir.z + 1) / hemisphere_percentage);
//   texcoord = envmap_dir.xy * radius / length(envmap_dir.xy);
//   texcoord = (1 + texcoord) * 0.5;
//   cube_texcoord = float3(
//       1.0f,
//       -((texcoord.y * 2.0f) - 1.0f),
//       -((texcoord.x * 2.0f) - 1.0f));
//   reflection = sampleCUBE(flat_environment_map, cube_texcoord);
//   return reflection * specular_reflectance_and_roughness.xyz
//                     * env_tint_color;
//
// Used when the rmop's `flat_environment_map` parameter resolves to a
// cube bitmap (some authoring tools bake the fisheye into a cubemap
// face and let the engine project into it via the synthesized cube
// direction). The assembler routes here when `flat_environment_map`'s
// resolved bitm tag reports `is_cube() == true`, regardless of whether
// the rmt2's option literal is `from_flat_texture` or
// `from_flat_texture_as_cubemap` — the binding type is source-of-truth.

fn calc_environment_map_from_flat_texture_as_cubemap_ps(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let _u_normal = normal;
    let _u_lfsc = low_frequency_specular_color;

    // Fisheye projection — HLSL line 167-175.
    let envmap_dir = vec3<f32>(
        dot(reflect_dir, material.flat_envmap_matrix_x.xyz),
        dot(reflect_dir, material.flat_envmap_matrix_y.xyz),
        dot(reflect_dir, material.flat_envmap_matrix_z.xyz),
    );
    let radius = sqrt((envmap_dir.z + 1.0) / material.hemisphere_percentage.x);
    let xy_len = length(envmap_dir.xy);
    var texcoord = envmap_dir.xy * radius / max(xy_len, 1.0e-6);
    texcoord = (vec2<f32>(1.0) + texcoord) * 0.5;

    // HLSL line 178-181: synthesize a +X-facing cube direction whose
    // y/z components encode the 2D texcoord (mapped from [0,1] →
    // [-1,1] with engine sign flip on both axes).
    let cube_texcoord = vec3<f32>(
        1.0,
        -((texcoord.y * 2.0) - 1.0),
        -((texcoord.x * 2.0) - 1.0),
    );
    let reflection = textureSample(flat_environment_map, flat_environment_map_sampler, cube_texcoord);

    return reflection.rgb
         * specular_reflectance_and_roughness.xyz
         * material.env_tint_color.xyz;
}

fn calc_environment_map(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    return calc_environment_map_from_flat_texture_as_cubemap_ps(
        view_dir, normal, reflect_dir,
        specular_reflectance_and_roughness,
        low_frequency_specular_color,
    );
}
