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
    pub _pad: [f32; 2],
}

pub const BLOOM_MIP_COUNT: usize = 5;

pub const SHADOW_MAP_SIZE: u32 = 1024;
pub const MAX_POINT_SHADOW_CASTERS: usize = 2;
pub const MAX_SPOT_SHADOW_CASTERS: usize = 2;
pub const SHADOW_CAMERA_SLOTS: usize = MAX_POINT_SHADOW_CASTERS * 6 + MAX_SPOT_SHADOW_CASTERS;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuShadowData {
    pub point_params: [[f32; 4]; MAX_POINT_SHADOW_CASTERS],
    pub spot_view_proj: [[[f32; 4]; 4]; MAX_SPOT_SHADOW_CASTERS],
    pub spot_params: [[f32; 4]; MAX_SPOT_SHADOW_CASTERS],
}
