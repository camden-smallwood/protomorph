use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ModelUniforms {
    pub model: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuadVertex {
    pub position: [f32; 2],
    pub texcoord: [f32; 2],
}

impl QuadVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub const QUAD_VERTICES: [QuadVertex; 6] = [
    QuadVertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] },
    QuadVertex { position: [-1.0, -1.0], texcoord: [0.0, 1.0] },
    QuadVertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] },
    QuadVertex { position: [-1.0,  1.0], texcoord: [0.0, 0.0] },
    QuadVertex { position: [ 1.0, -1.0], texcoord: [1.0, 1.0] },
    QuadVertex { position: [ 1.0,  1.0], texcoord: [1.0, 0.0] },
];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsaoParams {
    pub kernel_samples: [[f32; 4]; 32],
    pub strength: f32,
    pub falloff: f32,
    pub radius: f32,
    pub noise_scale_x: f32,
    pub noise_scale_y: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BloomDownsampleParams {
    pub threshold: f32,
    pub knee: f32,
    pub texel_size: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UpsampleParams {
    pub filter_radius: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CompositeParams {
    pub bloom_strength: f32,
    pub exposure: f32,
    pub saturation: f32,
    pub grain_intensity: f32,
}

pub const BLOOM_MIP_COUNT: usize = 5;

pub const SHADOW_MAP_SIZE: u32 = 1024;
pub const SPOT_SHADOW_MAP_SIZE: u32 = 2048;
pub const MAX_POINT_SHADOW_CASTERS: usize = 2;
pub const MAX_SPOT_SHADOW_CASTERS: usize = 2;
pub const CSM_CASCADE_COUNT: usize = 3;
pub const CSM_MAP_SIZE: u32 = 2048;
pub const SHADOW_CAMERA_SLOTS: usize =
    MAX_POINT_SHADOW_CASTERS * 6 + MAX_SPOT_SHADOW_CASTERS + CSM_CASCADE_COUNT;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuShadowData {
    pub point_params: [[f32; 4]; MAX_POINT_SHADOW_CASTERS],
    pub spot_view_proj: [[[f32; 4]; 4]; MAX_SPOT_SHADOW_CASTERS],
    pub spot_params: [[f32; 4]; MAX_SPOT_SHADOW_CASTERS],
    pub cascade_view_proj: [[[f32; 4]; 4]; CSM_CASCADE_COUNT],
    pub cascade_splits: [f32; 4],
    pub cascade_texel_sizes: [f32; 4], // world-space texel size per cascade, [3] = pad
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuAtmosphereData {
    pub sun_direction: [f32; 3],
    pub atmosphere_enable: f32,
    pub rayleigh_coefficients: [f32; 3],
    pub rayleigh_height_scale: f32,
    pub mie_coefficient: f32,
    pub mie_height_scale: f32,
    pub mie_g: f32,
    pub max_fog_thickness: f32,
    pub inscatter_scale: f32,
    pub reference_height: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuGodRayParams {
    pub sun_screen_pos: [f32; 2],
    pub density: f32,
    pub weight: f32,
    pub decay: f32,
    pub exposure: f32,
    pub num_samples: f32,
    pub sun_visible: f32,
    pub sun_color: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSHCoefficients {
    pub coefficients: [[f32; 4]; 9], // 9 L2 basis functions, each (R, G, B, pad)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSkyParams {
    pub inverse_view_projection: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub _pad: f32,
}

pub const ENV_PROBE_SIZE: u32 = 128;
pub const ENV_PROBE_MIP_COUNT: u32 = 6; // 128 -> 4

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuEnvProbeData {
    pub probe_position: [f32; 3],
    pub env_roughness_scale: f32,
    pub env_specular_contribution: f32,
    pub env_mip_count: f32,
    pub env_intensity: f32,
    pub env_diffuse_intensity: f32,
}
