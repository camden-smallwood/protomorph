// Simplified forward-lit shader for env probe cubemap rendering (rigid)
// Diffuse-only with single dominant light direction

// Group 0: Camera (dynamic offset)
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Group 1: Model (dynamic offset)
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

// Dominant light direction (approximation for probe capture)
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.0, 0.5, 0.866);

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model_u.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;

    let normal_matrix = mat3x3<f32>(
        model_u.model[0].xyz,
        model_u.model[1].xyz,
        model_u.model[2].xyz,
    );
    out.world_normal = normalize(normal_matrix * in.normal);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;

    // Opacity discard
    let opacity = textureSample(t_opacity, s_material, uv).r;
    if (opacity < 0.1) {
        discard;
    }

    let diffuse_tex = textureSample(t_diffuse, s_material, uv).rgb;
    let albedo = material.diffuse_color * diffuse_tex;

    let n = normalize(in.world_normal);
    let n_dot_l = max(dot(n, LIGHT_DIR), 0.0);

    let ambient = material.ambient_color * material.ambient_amount * 0.3;
    let color = albedo * (ambient + n_dot_l);

    return vec4<f32>(color, 1.0);
}
