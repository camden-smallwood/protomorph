// Depth-only vertex shader for skinned meshes (shadow pass) — dual quaternion skinning

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
};

struct ModelUniforms {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var<storage, read> node_matrices: array<mat4x4<f32>>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) node_indices: vec4<u32>,
    @location(2) node_weights: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    var skinned = vec4(0.0);
    for (var i = 0u; i < 4u; i++) {
        skinned += node_weights[i] * (node_matrices[node_indices[i]] * vec4(position, 1.0));
    }
    let weight_sum = node_weights[0] + node_weights[1] + node_weights[2] + node_weights[3];
    if (weight_sum < 0.0001) {
        skinned = vec4(position, 1.0);
    }

    var out = camera.projection * camera.view * model.model * skinned;
    // Shadow pancaking: clamp near-plane to avoid clipping casters behind the light
    out.z = max(out.z, 0.0);
    return out;
}
