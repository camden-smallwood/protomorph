use crate::{
    lights::LightType,
    renderer::{
        color_clear_attach, create_fullscreen_pipeline, depth_tex_entry, sampler_entry,
        shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
        tex_entry, uniform_entry,
    },
};
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

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

// ===========================================================================
// God Rays Trace Pass
// ===========================================================================

pub struct GodRaysTracePass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
}

impl GodRaysTracePass {
    pub fn new(
        shared: &SharedResources,
        depth_view: &wgpu::TextureView,
    ) -> Self {
        let device = &shared.device;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("god_ray_bgl"),
            entries: &[
                depth_tex_entry(0),
                sampler_entry(1, wgpu::SamplerBindingType::NonFiltering),
                uniform_entry(
                    2,
                    size_of::<GpuGodRayParams>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
            ],
        });
        let pipeline = create_fullscreen_pipeline(
            device,
            wgpu::include_wgsl!("../../assets/shaders/god_rays.wgsl"),
            &[&bgl],
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rg11b10Ufloat,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "god_rays_pipeline",
        );

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("god_ray_params"),
            size: size_of::<GpuGodRayParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = Self::create_bind_group(
            device,
            &bgl,
            depth_view,
            &shared.nearest_sampler,
            &params_buffer,
        );

        Self {
            pipeline,
            bgl,
            bind_group,
            params_buffer,
        }
    }

    fn create_bind_group(
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
}

impl RenderPass for GodRaysTracePass {
    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        _intermediates: &IntermediateTargets,
    ) {
        self.bind_group = Self::create_bind_group(
            &shared.device,
            &self.bgl,
            &gbuffer.depth_view,
            &shared.nearest_sampler,
            &self.params_buffer,
        );
    }

    fn prepare(&mut self, ctx: &FrameContext) {
        let game = ctx.game;

        let (sun_dir, sun_color) = {
            let mut sd = Vec3::new(0.0, 0.0, 1.0);
            let mut sc = Vec3::ZERO;
            for (_idx, light) in game.lights.iter() {
                if !light.hidden && light.light_type == LightType::Directional {
                    sd = -light.direction.normalize();
                    sc = light.diffuse_color;
                    break;
                }
            }
            (sd, sc)
        };

        let vp = game.camera.projection * game.camera.view;
        let sun_world = sun_dir * 1000.0;
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

        ctx.shared
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("god_rays_pass"),
            color_attachments: &[Some(color_clear_attach(&ctx.intermediates.god_rays_view))],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

// ===========================================================================
// God Rays Composite Pass
// ===========================================================================

pub struct GodRaysCompositePass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl GodRaysCompositePass {
    pub fn new(
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
    ) -> Self {
        let device = &shared.device;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("god_rays_composite_bgl"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let pipeline = create_fullscreen_pipeline(
            device,
            wgpu::include_wgsl!("../../assets/shaders/god_rays_composite.wgsl"),
            &[&bgl],
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
        );
        let bind_group = Self::create_bind_group(
            device,
            &bgl,
            &intermediates.god_rays_view,
            &shared.bloom_sampler,
        );

        Self {
            pipeline,
            bgl,
            bind_group,
        }
    }

    fn create_bind_group(
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
}

impl RenderPass for GodRaysCompositePass {
    fn resize(
        &mut self,
        shared: &SharedResources,
        _gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.bind_group = Self::create_bind_group(
            &shared.device,
            &self.bgl,
            &intermediates.god_rays_view,
            &shared.bloom_sampler,
        );
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("god_rays_composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.intermediates.lighting_base_view,
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
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

