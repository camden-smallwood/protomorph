use crate::{
    gpu_types::{GpuAtmosphereData, GpuSkyParams},
    renderer::{
        create_fullscreen_pipeline,
        helpers::{sampler_entry, tex_entry, uniform_entry},
        shared::SharedResources,
    },
};
use glam::{Mat4, Vec3};

pub struct SkyPass {
    pipeline: wgpu::RenderPipeline,
    bgl0: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bgl1: wgpu::BindGroupLayout,
    bind_group0: wgpu::BindGroup,
    bind_group1: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
}

impl SkyPass {
    pub fn new(shared: &SharedResources, position_depth_view: &wgpu::TextureView) -> Self {
        let device = &shared.device;

        let bgl0 = create_bgl0(device);
        let bgl1 = create_bgl1(device);
        let pipeline = create_pipeline(device, &bgl0, &bgl1);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_params"),
            size: size_of::<GpuSkyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group0 = create_bind_group0(
            device,
            &bgl0,
            position_depth_view,
            &shared.nearest_sampler,
        );

        let bind_group1 = create_bind_group1(
            device,
            &bgl1,
            &params_buffer,
            &shared.atmosphere_buffer,
        );

        Self {
            pipeline,
            bgl0,
            bgl1,
            bind_group0,
            bind_group1,
            params_buffer,
        }
    }

    pub fn resize(&mut self, shared: &SharedResources, position_depth_view: &wgpu::TextureView) {
        self.bind_group0 = create_bind_group0(
            &shared.device,
            &self.bgl0,
            position_depth_view,
            &shared.nearest_sampler,
        );
        // bind_group1 doesn't change on resize (buffers stay the same)
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        lighting_base_view: &wgpu::TextureView,
        view: Mat4,
        projection: Mat4,
        camera_position: Vec3,
    ) {
        let inverse_vp = (projection * view).inverse();

        let params = GpuSkyParams {
            inverse_view_projection: inverse_vp.to_cols_array_2d(),
            camera_position: camera_position.into(),
            _pad: 0.0,
        };

        shared
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: lighting_base_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group0, &[]);
        rpass.set_bind_group(1, &self.bind_group1, &[]);
        rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Bind group layouts + creation
// ---------------------------------------------------------------------------

fn create_bgl0(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let unfilterable = wgpu::TextureSampleType::Float { filterable: false };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sky_bgl0"),
        entries: &[
            tex_entry(0, unfilterable),
            sampler_entry(1, wgpu::SamplerBindingType::NonFiltering),
        ],
    })
}

fn create_bgl1(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sky_bgl1"),
        entries: &[
            uniform_entry(
                0,
                size_of::<GpuSkyParams>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
            uniform_entry(
                1,
                size_of::<GpuAtmosphereData>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
        ],
    })
}

fn create_bind_group0(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    position_depth_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky_bg0"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(position_depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_bind_group1(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params_buffer: &wgpu::Buffer,
    atmosphere_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky_bg1"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: atmosphere_buffer.as_entire_binding(),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn create_pipeline(
    device: &wgpu::Device,
    bgl0: &wgpu::BindGroupLayout,
    bgl1: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    create_fullscreen_pipeline(
        device,
        wgpu::include_wgsl!("../../assets/shaders/sky.wgsl"),
        &[bgl0, bgl1],
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        "sky_pipeline",
    )
}
