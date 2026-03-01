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
    specular_shininess: f32,
    _pad1: f32,
    _pad2: f32,
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

// Vertex input — skinned (extends rigid with bone indices + weights)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) node_indices: vec4<u32>,
    @location(6) node_weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) tbn_col0: vec3<f32>,
    @location(3) tbn_col1: vec3<f32>,
    @location(4) tbn_col2: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Linear blend skinning: weighted sum of up to 4 bone transforms
    var skin_matrix = mat4x4<f32>(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
    );
    skin_matrix += node_matrices[in.node_indices[0]] * in.node_weights[0];
    skin_matrix += node_matrices[in.node_indices[1]] * in.node_weights[1];
    skin_matrix += node_matrices[in.node_indices[2]] * in.node_weights[2];
    skin_matrix += node_matrices[in.node_indices[3]] * in.node_weights[3];

    // Fallback for unskinned vertices (all weights zero)
    let weight_sum = in.node_weights[0] + in.node_weights[1] + in.node_weights[2] + in.node_weights[3];
    if (weight_sum < 0.0001) {
        skin_matrix = mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
    }

    let skinned_model = model_u.model * skin_matrix;
    let world_pos = skinned_model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;

    // Compute TBN matrix in world space using the skinned model matrix
    let normal_matrix = mat3x3<f32>(
        skinned_model[0].xyz,
        skinned_model[1].xyz,
        skinned_model[2].xyz,
    );
    let t = normalize(normal_matrix * in.tangent);
    let b = normalize(normal_matrix * in.bitangent);
    let n = normalize(normal_matrix * in.normal);

    out.tbn_col0 = t;
    out.tbn_col1 = b;
    out.tbn_col2 = n;

    return out;
}

// G-buffer outputs (identical to geometry.wgsl)
struct GBufferOutput {
    @location(0) position_depth: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo_specular: vec4<f32>,
    @location(3) material_out: vec4<f32>,
};

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

    // G-buffer 0: world position + clip depth
    out.position_depth = vec4<f32>(in.world_position, in.clip_position.z);

    // G-buffer 1: normal mapping + emissive luminance
    let normal_sample = textureSample(t_normal, s_material, uv).rgb * 2.0 - 1.0;
    let scaled = normal_sample * vec3<f32>(1.0, 1.0, 1.0 / max(material.bump_scaling, 0.001));
    let emissive_tex = textureSample(t_emissive, s_material, uv).rgb;
    let emissive = emissive_tex * material.emissive_color * material.emissive_intensity;
    let emissive_luma = dot(emissive, vec3<f32>(0.2126, 0.7152, 0.0722));
    out.normal = vec4<f32>(normalize(tbn * scaled), emissive_luma);

    // G-buffer 2: albedo + specular
    let diffuse_tex = textureSample(t_diffuse, s_material, uv);
    let spec_tex = textureSample(t_specular, s_material, uv);
    out.albedo_specular = vec4<f32>(
        material.diffuse_color * diffuse_tex.rgb,
        spec_tex.r
    );

    // G-buffer 3: material properties
    out.material_out = vec4<f32>(
        material.ambient_amount,
        material.specular_amount,
        material.specular_shininess / 256.0,
        0.0
    );

    return out;
}
