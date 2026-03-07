use crate::{
    renderer::{
        create_fullscreen_pipeline,
        helpers::{color_clear_attach, depth_tex_entry, sampler_entry, tex_entry, uniform_entry},
        shared::{IntermediateTargets, SharedResources},
    },
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

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

pub struct GodRaysPass {
    // Radial blur pass
    ray_pipeline: wgpu::RenderPipeline,
    ray_bgl: wgpu::BindGroupLayout,
    ray_bind_group: wgpu::BindGroup,

    // Composite pass
    composite_pipeline: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    composite_bind_group: wgpu::BindGroup,

    // Params buffer
    params_buffer: wgpu::Buffer,
}

impl GodRaysPass {
    pub fn new(
        shared: &SharedResources,
        depth_view: &wgpu::TextureView,
        intermediates: &IntermediateTargets,
    ) -> Self {
        let device = &shared.device;

        // --- Radial blur pass ---
        let ray_bgl = create_ray_bgl(device);
        let ray_pipeline = create_ray_pipeline(device, &ray_bgl);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("god_ray_params"),
            size: size_of::<GpuGodRayParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ray_bind_group = create_ray_bind_group(
            device,
            &ray_bgl,
            depth_view,
            &shared.nearest_sampler,
            &params_buffer,
        );

        // --- Composite pass ---
        let composite_bgl = create_composite_bgl(device);
        let composite_pipeline = create_composite_pipeline(device, &composite_bgl);
        let composite_bind_group = create_composite_bind_group(
            device,
            &composite_bgl,
            &intermediates.god_rays_view,
            &shared.bloom_sampler,
        );

        Self {
            ray_pipeline,
            ray_bgl,
            ray_bind_group,
            composite_pipeline,
            composite_bgl,
            composite_bind_group,
            params_buffer,
        }
    }

    pub fn resize(
        &mut self,
        shared: &SharedResources,
        depth_view: &wgpu::TextureView,
        intermediates: &IntermediateTargets,
    ) {
        self.ray_bind_group = create_ray_bind_group(
            &shared.device,
            &self.ray_bgl,
            depth_view,
            &shared.nearest_sampler,
            &self.params_buffer,
        );
        self.composite_bind_group = create_composite_bind_group(
            &shared.device,
            &self.composite_bgl,
            &intermediates.god_rays_view,
            &shared.bloom_sampler,
        );
    }

    pub fn record_trace(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        god_rays_view: &wgpu::TextureView,
        view: Mat4,
        projection: Mat4,
        sun_direction: Vec3,
        sun_color: Vec3,
    ) {
        // Compute sun screen position
        let vp = projection * view;
        let sun_world = sun_direction * 1000.0;
        let clip = vp * Vec4::new(sun_world.x, sun_world.y, sun_world.z, 1.0);

        let (sun_visible, ndc) = if clip.w > 0.0 {
            let n = clip.truncate() / clip.w;
            let screen_dist = (n.x * n.x + n.y * n.y).sqrt();
            let fade = (1.0 - (screen_dist - 0.5).max(0.0) / 1.0).max(0.0);
            (fade, n)
        } else {
            (0.0f32, Vec3::ZERO)
        };
        let sun_screen = [ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5];

        let params = GpuGodRayParams {
            sun_screen_pos: sun_screen,
            density: 1.0,
            weight: 0.01,
            decay: 1.0,
            exposure: 0.25,
            num_samples: 32.0,
            sun_visible,
            sun_color: (sun_color * 1.0).into(),
            _pad: 0.0,
        };

        shared
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("god_rays_pass"),
            color_attachments: &[Some(color_clear_attach(god_rays_view))],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.ray_pipeline);
        rpass.set_bind_group(0, &self.ray_bind_group, &[]);
        rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }

    pub fn record_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        lighting_base_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("god_rays_composite_pass"),
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

        rpass.set_pipeline(&self.composite_pipeline);
        rpass.set_bind_group(0, &self.composite_bind_group, &[]);
        rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Bind group layouts + creation
// ---------------------------------------------------------------------------

fn create_ray_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("god_ray_bgl"),
        entries: &[
            depth_tex_entry(0), // t_depth (Depth32Float)
            sampler_entry(1, wgpu::SamplerBindingType::NonFiltering),
            uniform_entry(
                2,
                size_of::<GpuGodRayParams>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
        ],
    })
}

fn create_ray_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("god_ray_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

fn create_composite_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("god_rays_composite_bgl"),
        entries: &[
            tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
            sampler_entry(1, wgpu::SamplerBindingType::Filtering),
        ],
    })
}

fn create_composite_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    god_rays_view: &wgpu::TextureView,
    filtering_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("god_rays_composite_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(god_rays_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(filtering_sampler),
            },
        ],
    })
}

fn create_composite_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    create_fullscreen_pipeline(
        device,
        wgpu::include_wgsl!("../../assets/shaders/god_rays_composite.wgsl"),
        &[bgl],
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rg11b10Ufloat,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        })],
        "god_rays_composite_pipeline",
    )
}

fn create_ray_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    create_fullscreen_pipeline(
        device,
        wgpu::include_wgsl!("../../assets/shaders/god_rays.wgsl"),
        &[bgl],
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rg11b10Ufloat,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        "god_rays_pipeline",
    )
}

