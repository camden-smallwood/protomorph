// Port of `environment_mapping_fx.hlsl` — `from_flat_texture` variant
// (line 156-202).
//
// Fisheye-projection environment map sampled from a flat 2D texture
// (a hemispherical-paraboloid encoding). Used by maps that bake an
// outdoor sky/cubemap into a single 2D bitmap for low-cost reflection.
//
// HLSL:
//   envmap_dir.x = dot(reflect_dir, flat_envmap_matrix_x.xyz);
//   envmap_dir.y = dot(reflect_dir, flat_envmap_matrix_y.xyz);
//   envmap_dir.z = dot(reflect_dir, flat_envmap_matrix_z.xyz);
//   radius   = sqrt((envmap_dir.z + 1) / hemisphere_percentage);
//   texcoord = envmap_dir.xy * radius / length(envmap_dir.xy);
//   texcoord = (1 + texcoord) * 0.5;
//   reflection = sample2D(flat_environment_map, texcoord);
//   return reflection * specular_reflectance_and_roughness.xyz
//                     * env_tint_color;
//
// The `env_bloom_override` block is the E3 bloom-tap hack — we don't
// route it into the bloom pass yet (no `BLOOM_OVERRIDE` consumer);
// skipping is engine-correct for the SDR/non-E3 path.
//
// Rmt2 packs scalar PARAMs as vec4 — read `.x` for `hemisphere_percentage`.

fn calc_environment_map_from_flat_texture_ps(
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    reflect_dir: vec3<f32>,
    specular_reflectance_and_roughness: vec4<f32>,
    low_frequency_specular_color: vec3<f32>,
) -> vec3<f32> {
    let _u_view = view_dir;
    let _u_normal = normal;
    let _u_lfsc = low_frequency_specular_color;

    // HLSL line 167-169: rotate reflect_dir into envmap space.
    let envmap_dir = vec3<f32>(
        dot(reflect_dir, material.flat_envmap_matrix_x.xyz),
        dot(reflect_dir, material.flat_envmap_matrix_y.xyz),
        dot(reflect_dir, material.flat_envmap_matrix_z.xyz),
    );

    // HLSL line 171: radius for fisheye projection.
    let radius = sqrt((envmap_dir.z + 1.0) / material.hemisphere_percentage.x);

    // HLSL line 174: normalize xy then scale by radius.
    let xy_len = length(envmap_dir.xy);
    var texcoord = envmap_dir.xy * radius / max(xy_len, 1.0e-6);
    // HLSL line 175: 0..1 texture space.
    texcoord = (vec2<f32>(1.0) + texcoord) * 0.5;

    let reflection = textureSample(flat_environment_map, flat_environment_map_sampler, texcoord);

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
    return calc_environment_map_from_flat_texture_ps(
        view_dir, normal, reflect_dir,
        specular_reflectance_and_roughness,
        low_frequency_specular_color,
    );
}
