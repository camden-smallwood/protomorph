// Depth pre-pass (skinned) — writes depth only, with alpha-test for cutout geometry
// Uses the same bind groups as the full geometry skinned pass.

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

// Group 2: Material (only opacity texture + sampler used)
@group(2) @binding(4) var t_opacity: texture_2d<f32>;
@group(2) @binding(5) var s_material: sampler;

// Group 3: Bone matrices (storage buffer with dynamic offset)
@group(3) @binding(0) var<storage, read> node_matrices: array<mat4x4<f32>>;

// Vertex input — same layout as geometry_skinned.wgsl (must match VertexSkinned)
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
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

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
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) {
    let opacity = textureSample(t_opacity, s_material, in.tex_coords).r;
    if (opacity < 0.1) {
        discard;
    }
}
