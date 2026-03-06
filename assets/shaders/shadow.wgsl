// Depth-only vertex shader for rigid meshes (shadow pass)

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
};

struct ModelUniforms {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    var out = camera.projection * camera.view * model.model * vec4(position, 1.0);
    // Shadow pancaking: clamp near-plane to avoid clipping casters behind the light
    out.z = max(out.z, 0.0);
    return out;
}
