use crate::renderer::{
    create_fullscreen_pipeline, sampler_entry,
    shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
    tex_entry,
};

pub struct FxaaPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl FxaaPass {
    pub fn new(
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let filterable = wgpu::TextureSampleType::Float { filterable: true };

        let bgl =
            shared
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fxaa_bgl"),
                    entries: &[
                        tex_entry(0, filterable),
                        sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                    ],
                });

        let pipeline = create_fullscreen_pipeline(
            &shared.device,
            wgpu::include_wgsl!("../../assets/shaders/fxaa.wgsl"),
            &[&bgl],
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "fxaa_pipeline",
        );

        let bind_group = Self::create_bind_group(
            &shared.device,
            &bgl,
            &intermediates.post_composite_view,
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
        input_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

impl RenderPass for FxaaPass {
    fn resize(
        &mut self,
        shared: &SharedResources,
        _gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.bind_group = FxaaPass::create_bind_group(
            &shared.device,
            &self.bgl,
            &intermediates.post_composite_view,
            &shared.bloom_sampler,
        );
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fxaa_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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


