use crate::renderer::{
    create_fullscreen_pipeline, depth_tex_entry, sampler_entry, tex_entry, uniform_entry,
    shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CompositeParams {
    prev_view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    camera_position: [f32; 3],
    quarter_texel_w: f32,
    quarter_texel_h: f32,
    frame_index: u32,
    _pad: [f32; 2],
}

pub struct CloudCompositePass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
}

impl CloudCompositePass {
    pub fn new(
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) -> Self {
        let device = &shared.device;
        let filterable = wgpu::TextureSampleType::Float { filterable: true };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud_composite_bgl"),
            entries: &[
                tex_entry(0, filterable),                                           // cloud quarter-res
                tex_entry(1, filterable),                                           // cloud history
                depth_tex_entry(2),                                                 // depth
                sampler_entry(3, wgpu::SamplerBindingType::Filtering),              // bilinear
                sampler_entry(4, wgpu::SamplerBindingType::NonFiltering),           // nearest
                uniform_entry(5, size_of::<CompositeParams>() as u64, wgpu::ShaderStages::FRAGMENT, false),
                tex_entry(6, filterable),                                           // pre-cloud scene copy
                tex_entry(7, filterable),                                           // god rays
            ],
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud_composite_params"),
            contents: bytemuck::bytes_of(&CompositeParams {
                prev_view_projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
                inverse_view_projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
                camera_position: [0.0; 3],
                quarter_texel_w: 1.0,
                quarter_texel_h: 1.0,
                frame_index: 0,
                _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = Self::create_bind_group(
            device,
            &bgl,
            &intermediates.cloud_raymarch_view,
            &intermediates.cloud_history_views[1],
            &gbuffer.depth_view,
            &shared.bloom_sampler,
            &shared.nearest_sampler,
            &params_buffer,
            &intermediates.water_copy_view,
            &intermediates.god_rays_view,
        );

        // Two color targets:
        //   0: scene (lighting_base) — premultiplied alpha blend
        //   1: history buffer — straight overwrite
        let pipeline = create_fullscreen_pipeline(
            device,
            wgpu::include_wgsl!("../../assets/shaders/cloud_composite.wgsl"),
            &[&bgl],
            &[
                // Target 0: scene composite — shader computes final color (non-linear transmittance)
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg11b10Ufloat,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // Target 1: history buffer — straight overwrite
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            "cloud_composite_pipeline",
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
        cloud_view: &wgpu::TextureView,
        history_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        filter_sampler: &wgpu::Sampler,
        nearest_sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
        scene_copy_view: &wgpu::TextureView,
        god_rays_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud_composite_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(history_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(filter_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(nearest_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(scene_copy_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(god_rays_view) },
            ],
        })
    }

}

impl RenderPass for CloudCompositePass {
    fn prepare(&mut self, ctx: &FrameContext) {
        let game = ctx.game;
        let vp = game.camera.projection * game.camera.view;
        let inverse_vp = vp.inverse();
        let (qw, qh) = ctx.intermediates.cloud_quarter_size;

        // Swap which history buffer we read from based on frame parity
        let read_idx = (ctx.frame_index as usize + 1) % 2;
        self.bind_group = Self::create_bind_group(
            &ctx.shared.device,
            &self.bgl,
            &ctx.intermediates.cloud_raymarch_view,
            &ctx.intermediates.cloud_history_views[read_idx],
            &ctx.gbuffer.depth_view,
            &ctx.shared.bloom_sampler,
            &ctx.shared.nearest_sampler,
            &self.params_buffer,
            &ctx.intermediates.water_copy_view,
            &ctx.intermediates.god_rays_view,
        );

        let params = CompositeParams {
            prev_view_projection: ctx.prev_view_projection.to_cols_array_2d(),
            inverse_view_projection: inverse_vp.to_cols_array_2d(),
            camera_position: game.camera.position.into(),
            quarter_texel_w: 1.0 / qw as f32,
            quarter_texel_h: 1.0 / qh as f32,
            frame_index: ctx.frame_index,
            _pad: [0.0; 2],
        };

        ctx.shared
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        // Copy pre-cloud scene to water_copy for non-linear transmittance reading
        encoder.copy_texture_to_texture(
            ctx.intermediates.lighting_base_texture.as_image_copy(),
            ctx.intermediates.water_copy_texture.as_image_copy(),
            wgpu::Extent3d {
                width: ctx.shared.config.width,
                height: ctx.shared.config.height,
                depth_or_array_layers: 1,
            },
        );

        let write_idx = ctx.frame_index as usize % 2;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud_composite_pass"),
            color_attachments: &[
                // Attachment 0: scene — load existing, blend clouds on top
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.lighting_base_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                // Attachment 1: history buffer — clear + overwrite
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.intermediates.cloud_history_views[write_idx],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }

    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.bind_group = Self::create_bind_group(
            &shared.device,
            &self.bgl,
            &intermediates.cloud_raymarch_view,
            &intermediates.cloud_history_views[1],
            &gbuffer.depth_view,
            &shared.bloom_sampler,
            &shared.nearest_sampler,
            &self.params_buffer,
            &intermediates.water_copy_view,
            &intermediates.god_rays_view,
        );
    }
}
