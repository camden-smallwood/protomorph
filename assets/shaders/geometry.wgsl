// Geometry pass — G-buffer output with normal mapping
// Port of C's model.vs + geometry.fs
// Compact 4 MRT to fit within Metal's 32 bytes-per-sample limit:
//   Rgba16Float(8) + Rgba16Float(8) + Rgba8UnormSrgb(4) + Rgba8Unorm(4) = 24 bytes

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

// Vertex input
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
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

    let world_pos = model_u.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.projection * camera.view * world_pos;
    out.tex_coords = in.tex_coords;

    // Compute TBN matrix in world space (matching C's normal_matrix approach)
    let normal_matrix = mat3x3<f32>(
        model_u.model[0].xyz,
        model_u.model[1].xyz,
        model_u.model[2].xyz,
    );
    let t = normalize(normal_matrix * in.tangent);
    let b = normalize(normal_matrix * in.bitangent);
    let n = normalize(normal_matrix * in.normal);

    out.tbn_col0 = t;
    out.tbn_col1 = b;
    out.tbn_col2 = n;

    return out;
}

// G-buffer outputs (4 MRT — 8+8+4+4 = 24 bytes, within Metal's 32-byte limit)
// Metal aligns each attachment to 8 bytes for per-sample budget, so max 4 attachments = 32 bytes.
struct GBufferOutput {
    @location(0) position_depth: vec4<f32>,   // Rgba16Float: xyz=world_pos, w=clip_z
    @location(1) normal: vec4<f32>,           // Rgba16Float: xyz=normal, w=emissive_luminance (HDR)
    @location(2) albedo_specular: vec4<f32>,  // Rgba8UnormSrgb: rgb=albedo, a=specular_tex
    @location(3) material_out: vec4<f32>,     // Rgba8Unorm: r=ambient, g=spec_amount, b=shininess/256, a=unused
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

    // G-buffer 0: world position + clip depth (for SSAO)
    out.position_depth = vec4<f32>(in.world_position, in.clip_position.z);

    // G-buffer 1: normal mapping (matching C geometry.fs) + emissive luminance in w
    let normal_sample = textureSample(t_normal, s_material, uv).rgb * 2.0 - 1.0;
    let scaled = normal_sample * vec3<f32>(1.0, 1.0, 1.0 / max(material.bump_scaling, 0.001));
    let emissive_tex = textureSample(t_emissive, s_material, uv).rgb;
    let emissive = emissive_tex * material.emissive_color * material.emissive_intensity;
    let emissive_luma = dot(emissive, vec3<f32>(0.2126, 0.7152, 0.0722));
    out.normal = vec4<f32>(normalize(tbn * scaled), emissive_luma);

    // G-buffer 2: albedo (diffuse_color * diffuse_texture) + specular in alpha
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
        material.roughness,
        material.fresnel_f0
    );

    return out;
}
