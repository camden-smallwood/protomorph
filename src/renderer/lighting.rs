use crate::renderer::{
    color_clear_attach, create_fullscreen_pipeline, depth_tex_entry, sampler_entry,
    shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedBindGroup, SharedResources},
    tex_entry,
};

pub struct LightingPass {
    pipeline: wgpu::RenderPipeline,
    gbuffer_bgl: wgpu::BindGroupLayout,
    gbuffer_bind_group: wgpu::BindGroup,
    shadow_bind_group: SharedBindGroup,
    env_probe_bind_group: SharedBindGroup,
}

impl LightingPass {
    pub fn new(
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
        shadow_bgl: &wgpu::BindGroupLayout,
        env_probe_bgl: &wgpu::BindGroupLayout,
        shadow_bind_group: SharedBindGroup,
        env_probe_bind_group: SharedBindGroup,
    ) -> Self {
        let unfilterable = wgpu::TextureSampleType::Float {
            filterable: false,
        };

        let filterable = wgpu::TextureSampleType::Float {
            filterable: true,
        };

        let gbuffer_bgl =
            shared
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("lighting_gbuffer_bgl"),
                    entries: &[
                        depth_tex_entry(0),
                        tex_entry(1, unfilterable),
                        tex_entry(2, filterable),
                        tex_entry(3, filterable),
                        tex_entry(4, filterable),
                        sampler_entry(5, wgpu::SamplerBindingType::NonFiltering),
                    ],
                });

        let pipeline = create_fullscreen_pipeline(
            &shared.device,
            wgpu::include_wgsl!("../../assets/shaders/lighting.wgsl"),
            &[
                &gbuffer_bgl,
                &shared.lighting_uniforms_bgl,
                shadow_bgl,
                env_probe_bgl,
            ],
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rg11b10Ufloat,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "lighting_pipeline",
        );

        let gbuffer_bind_group = Self::create_gbuffer_bind_group(
            &shared.device,
            &gbuffer_bgl,
            gbuffer,
            intermediates,
            &shared.nearest_sampler,
        );

        Self {
            pipeline,
            gbuffer_bgl,
            gbuffer_bind_group,
            shadow_bind_group,
            env_probe_bind_group,
        }
    }

    fn create_gbuffer_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_gbuffer_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.albedo_specular_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&intermediates.ssao_blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }
}

impl RenderPass for LightingPass {
    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.gbuffer_bind_group = Self::create_gbuffer_bind_group(
            &shared.device,
            &self.gbuffer_bgl,
            gbuffer,
            intermediates,
            &shared.nearest_sampler,
        );
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let shadow_bg = self.shadow_bind_group.borrow();
        let env_probe_bg = self.env_probe_bind_group.borrow();

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting_pass"),
            color_attachments: &[Some(color_clear_attach(&ctx.intermediates.lighting_base_view))],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
        rpass.set_bind_group(1, &ctx.shared.lighting_uniforms_bind_group, &[]);
        rpass.set_bind_group(2, &*shadow_bg, &[]);
        rpass.set_bind_group(3, &*env_probe_bg, &[]);
        rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}


