// Forward emissive pass — additively blends emissive color into the lighting buffer
// Runs after deferred lighting, uses Equal depth compare (reuses pre-pass depth)
// Uses the same bind groups as the geometry pass so no extra setup needed.

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

// Group 2: Material (emissive texture + sampler + material props)
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
    _pad2: f32,
    _pad3: f32,
};
@group(2) @binding(3) var t_emissive: texture_2d<f32>;
@group(2) @binding(4) var t_opacity: texture_2d<f32>;
@group(2) @binding(5) var s_material: sampler;
@group(2) @binding(6) var<uniform> material: MaterialProps;

// Vertex input — same layout as geometry.wgsl (must match VertexRigid)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model_u.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;

    // Alpha-test to match depth prepass
    let opacity = textureSample(t_opacity, s_material, uv).r;
    if (opacity < 0.1) {
        discard;
    }

    let emissive_tex = textureSample(t_emissive, s_material, uv).rgb;
    let emissive = emissive_tex * material.emissive_color * material.emissive_intensity;

    // Clamp to Rg11b10Ufloat safe range
    return vec4<f32>(min(emissive, vec3(500.0)), 1.0);
}
