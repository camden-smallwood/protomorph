// Geometry pass — G-buffer output with normal mapping + skeletal skinning
// Copy of geometry.wgsl with linear blend skinning via bone matrices

// Group 0: Camera uniforms
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Group 1: Model uniform (dynamic offset)
struct ModelUniforms {
    model: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model_u: ModelUniforms;

// Group 2: Material
struct MaterialProps {
    diffuse_color: vec3<f32>,
    bump_scaling: f32,
    ambient_color: vec3<f32>,
    ambient_amount: f32,
    specular_color: vec3<f32>,
    specular_amount: f32,
    emissive_color: vec3<f32>,
    emissive_intensity: f32,
    roughness: f32,
    fresnel_f0: f32,
    metallic: f32,
    _pad3: f32,
};
@group(2) @binding(0) var t_diffuse: texture_2d<f32>;
@group(2) @binding(1) var t_normal: texture_2d<f32>;
@group(2) @binding(2) var t_specular: texture_2d<f32>;
@group(2) @binding(3) var t_emissive: texture_2d<f32>;
@group(2) @binding(4) var t_opacity: texture_2d<f32>;
@group(2) @binding(5) var s_material: sampler;
@group(2) @binding(6) var<uniform> material: MaterialProps;

// Group 3: Bone matrices (storage buffer with dynamic offset)
@group(3) @binding(0) var<storage, read> node_matrices: array<mat4x4<f32>>;

// Vertex input — compressed (32 bytes)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,     // Snorm8x2 → vec2<f32>, oct-encoded
    @location(2) tex_coords: vec2<f32>,     // Float16x2 → vec2<f32>
    @location(3) tangent_sign: vec4<f32>,   // Snorm8x4 → vec4<f32>, xyz=tangent, w=bitangent sign
    @location(4) node_indices: vec4<u32>,   // Uint8x4 → vec4<u32>
    @location(5) node_weights: vec4<f32>,   // Unorm8x4 → vec4<f32>
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) tbn_col0: vec3<f32>,
    @location(2) tbn_col1: vec3<f32>,
    @location(3) tbn_col2: vec3<f32>,
};

fn oct_decode_vtx(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Linear blend skinning
    var skin_matrix = mat4x4<f32>(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    skin_matrix += node_matrices[in.node_indices[0]] * in.node_weights[0];
    skin_matrix += node_matrices[in.node_indices[1]] * in.node_weights[1];
    skin_matrix += node_matrices[in.node_indices[2]] * in.node_weights[2];
    skin_matrix += node_matrices[in.node_indices[3]] * in.node_weights[3];

    let weight_sum = in.node_weights[0] + in.node_weights[1] + in.node_weights[2] + in.node_weights[3];
    if (weight_sum < 0.0001) {
        skin_matrix = mat4x4(vec4(1.,0.,0.,0.), vec4(0.,1.,0.,0.), vec4(0.,0.,1.,0.), vec4(0.,0.,0.,1.));
    }

    let skinned_model = model_u.model * skin_matrix;
    let world_pos = skinned_model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;

    // Decode compressed normal and reconstruct TBN
    let obj_normal = oct_decode_vtx(in.normal_oct);
    let obj_tangent = normalize(in.tangent_sign.xyz);
    let obj_bitangent = cross(obj_normal, obj_tangent) * sign(in.tangent_sign.w);

    let normal_matrix = mat3x3<f32>(
        skinned_model[0].xyz, skinned_model[1].xyz, skinned_model[2].xyz,
    );
    out.tbn_col0 = normalize(normal_matrix * obj_tangent);
    out.tbn_col1 = normalize(normal_matrix * obj_bitangent);
    out.tbn_col2 = normalize(normal_matrix * obj_normal);

    return out;
}

// G-buffer outputs (identical to geometry.wgsl)
struct GBufferOutput {
    @location(0) normal: vec2<f32>,           // Rg16Float: octahedral-encoded world normal
    @location(1) albedo_specular: vec4<f32>,  // Rgba8UnormSrgb: rgb=albedo, a=specular (baked)
    @location(2) material_out: vec4<f32>,     // Rgba8Unorm: r=ambient, g=metallic, b=roughness, a=fresnel_f0
    @location(3) emissive_out: vec4<f32>,     // Rg11b10Ufloat: rgb=emissive HDR color
};

// Octahedral encoding: unit vec3 → vec2 in [-1,1]
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let sum = abs(n.x) + abs(n.y) + abs(n.z);
    var p = n.xy / sum;
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * sign(p);
    }
    return p;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;
    let uv = in.tex_coords;

    // Opacity discard
    let opacity = textureSample(t_opacity, s_material, uv).r;
    if (opacity < 0.1) {
        discard;
    }

    // Reconstruct TBN matrix
    let tbn = mat3x3<f32>(in.tbn_col0, in.tbn_col1, in.tbn_col2);

    // G-buffer 0: octahedral-encoded world normal
    let normal_sample = textureSample(t_normal, s_material, uv).rgb * 2.0 - 1.0;
    let scaled = normal_sample * vec3<f32>(1.0, 1.0, 1.0 / max(material.bump_scaling, 0.001));
    let world_normal = normalize(tbn * scaled);
    out.normal = oct_encode(world_normal);

    // G-buffer 1: albedo + baked specular (spec_tex * spec_amount)
    let diffuse_tex = textureSample(t_diffuse, s_material, uv);
    let spec_tex = textureSample(t_specular, s_material, uv);
    out.albedo_specular = vec4<f32>(
        material.diffuse_color * diffuse_tex.rgb,
        spec_tex.r * material.specular_amount
    );

    // G-buffer 2: material properties
    out.material_out = vec4<f32>(
        material.ambient_amount,
        material.metallic,
        material.roughness,
        material.fresnel_f0
    );

    // G-buffer 3: emissive HDR
    let emissive_tex = textureSample(t_emissive, s_material, uv).rgb;
    out.emissive_out = vec4<f32>(
        min(emissive_tex * material.emissive_color * material.emissive_intensity, vec3(500.0)),
        1.0
    );

    return out;
}
