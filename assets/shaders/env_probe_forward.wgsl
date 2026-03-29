// Simplified forward-lit shader for env probe cubemap rendering (skinned)
// Diffuse-only with single dominant light direction + linear blend skinning

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

// Group 3: Bone matrices
@group(3) @binding(0) var<storage, read> node_matrices: array<mat4x4<f32>>;

const LIGHT_DIR: vec3<f32> = vec3<f32>(0.0, 0.5, 0.866);

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
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

    var sm = mat4x4<f32>(vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    sm += node_matrices[in.node_indices[0]] * in.node_weights[0];
    sm += node_matrices[in.node_indices[1]] * in.node_weights[1];
    sm += node_matrices[in.node_indices[2]] * in.node_weights[2];
    sm += node_matrices[in.node_indices[3]] * in.node_weights[3];
    let ws = in.node_weights[0] + in.node_weights[1] + in.node_weights[2] + in.node_weights[3];
    if (ws < 0.0001) {
        sm = mat4x4(vec4(1.,0.,0.,0.), vec4(0.,1.,0.,0.), vec4(0.,0.,1.,0.), vec4(0.,0.,0.,1.));
    }

    let skinned_model = model_u.model * sm;
    let world_pos = skinned_model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;

    let obj_normal = oct_decode_vtx(in.normal_oct);
    let normal_matrix = mat3x3<f32>(
        skinned_model[0].xyz, skinned_model[1].xyz, skinned_model[2].xyz,
    );
    out.world_normal = normalize(normal_matrix * obj_normal);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;

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
