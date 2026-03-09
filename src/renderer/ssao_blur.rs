use crate::renderer::{
    create_fullscreen_pipeline, depth_tex_entry, sampler_entry,
    shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
    tex_entry,
};

pub struct SsaoBlurPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl SsaoBlurPass {
    pub fn new(
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) -> Self {
        let filterable = wgpu::TextureSampleType::Float { filterable: true };

        let bgl = shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_blur_bgl"),
            entries: &[
                tex_entry(0, filterable),     // t_ssao (Rgba16Float, filterable)
                depth_tex_entry(1),           // t_depth (Depth32Float)
                sampler_entry(2, wgpu::SamplerBindingType::NonFiltering),
            ],
        });

        let pipeline = create_fullscreen_pipeline(
            &shared.device,
            wgpu::include_wgsl!("../../assets/shaders/ssao_blur.wgsl"),
            &[&bgl],
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "ssao_blur_pipeline",
        );

        let bind_group = Self::create_bind_group(
            &shared.device,
            &bgl,
            &intermediates.ssao_view,
            &gbuffer.depth_view,
            &shared.nearest_sampler,
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
        ssao_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_blur_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

impl RenderPass for SsaoBlurPass {
    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.bind_group = Self::create_bind_group(
            &shared.device,
            &self.bgl,
            &intermediates.ssao_view,
            &gbuffer.depth_view,
            &shared.nearest_sampler,
        );
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao_blur_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.intermediates.ssao_blur_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
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


