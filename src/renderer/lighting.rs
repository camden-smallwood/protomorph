use std::mem::size_of;

use crate::{
    lights::GpuLightingUniforms,
    renderer::{
        color_clear_attach, create_fullscreen_pipeline, depth_tex_entry, sampler_entry, tex_entry,
        uniform_entry,
        env_probe::{GpuAtmosphereData, GpuEnvProbeData, GpuSHCoefficients, GpuSkyParams},
        shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedBindGroup, SharedResources},
    },
};

// ===========================================================================
// Deferred Lighting Pass — single fullscreen quad reads GBuffer, computes all
// lighting (ambient, env probe, SH, direct + shadow, SSAO) in one draw.
// ===========================================================================

pub struct DeferredLightingPass {
    pipeline: wgpu::RenderPipeline,
    gbuffer_bgl: wgpu::BindGroupLayout,
    lighting_bgl: wgpu::BindGroupLayout,
    gbuffer_bind_group: wgpu::BindGroup,
    shadow_bind_group: SharedBindGroup,
    lighting_bind_group: wgpu::BindGroup,
    env_cube_view: wgpu::TextureView,
}

impl DeferredLightingPass {
    pub fn new(
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
        shadow_bgl: &wgpu::BindGroupLayout,
        shadow_bind_group: SharedBindGroup,
        env_cube_view: wgpu::TextureView,
        env_filtering_sampler: &wgpu::Sampler,
    ) -> Self {
        let filterable = wgpu::TextureSampleType::Float { filterable: true };
        let unfilterable = wgpu::TextureSampleType::Float { filterable: false };

        // --- Group 0: GBuffer + SSAO ---
        let gbuffer_bgl = shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deferred_lighting_gbuffer_bgl"),
            entries: &[
                tex_entry(0, unfilterable),                                    // normal
                tex_entry(1, filterable),                                      // albedo_specular
                tex_entry(2, filterable),                                      // material
                depth_tex_entry(3),                                            // depth
                tex_entry(4, filterable),                                      // ssao (raw, blurred inline)
                sampler_entry(5, wgpu::SamplerBindingType::NonFiltering),      // nearest_sampler
                sampler_entry(6, wgpu::SamplerBindingType::Filtering),         // filtering_sampler
                tex_entry(7, filterable),                                      // emissive
            ],
        });

        // --- Group 2: Lighting + Env Probe + Atmosphere ---
        let lighting_bgl = shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deferred_lighting_lighting_bgl"),
            entries: &[
                uniform_entry(0, size_of::<GpuLightingUniforms>() as u64, wgpu::ShaderStages::FRAGMENT, false),
                uniform_entry(1, size_of::<GpuAtmosphereData>() as u64, wgpu::ShaderStages::FRAGMENT, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: filterable,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                sampler_entry(3, wgpu::SamplerBindingType::Filtering),
                uniform_entry(4, size_of::<GpuEnvProbeData>() as u64, wgpu::ShaderStages::FRAGMENT, false),
                uniform_entry(5, size_of::<GpuSHCoefficients>() as u64, wgpu::ShaderStages::FRAGMENT, false),
                uniform_entry(6, size_of::<GpuSkyParams>() as u64, wgpu::ShaderStages::FRAGMENT, false),
            ],
        });

        // --- Pipeline ---
        let pipeline = create_fullscreen_pipeline(
            &shared.device,
            wgpu::include_wgsl!("../../assets/shaders/lighting.wgsl"),
            &[&gbuffer_bgl, shadow_bgl, &lighting_bgl],
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rg11b10Ufloat,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "deferred_lighting_pipeline",
        );

        // --- Bind groups ---
        let gbuffer_bind_group = Self::create_gbuffer_bind_group(
            &shared.device, &gbuffer_bgl, gbuffer, intermediates, shared,
        );
        let lighting_bind_group = Self::create_lighting_bind_group(
            &shared.device, &lighting_bgl, shared, &env_cube_view, env_filtering_sampler,
        );

        Self {
            pipeline,
            gbuffer_bgl,
            lighting_bgl,
            gbuffer_bind_group,
            shadow_bind_group,
            lighting_bind_group,
            env_cube_view,
        }
    }

    fn create_gbuffer_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
        shared: &SharedResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_lighting_gbuffer_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&gbuffer.albedo_specular_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffer.material_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffer.depth_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&intermediates.ssao_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&shared.nearest_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&shared.filtering_sampler) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&gbuffer.emissive_view) },
            ],
        })
    }

    fn create_lighting_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        shared: &SharedResources,
        env_cube_view: &wgpu::TextureView,
        env_filtering_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_lighting_lighting_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shared.lighting_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shared.atmosphere_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(env_cube_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(env_filtering_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: shared.env_probe_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: shared.sh_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: shared.sky_params_buffer.as_entire_binding() },
            ],
        })
    }
}

impl RenderPass for DeferredLightingPass {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("deferred_lighting_pass"),
            color_attachments: &[Some(color_clear_attach(&ctx.intermediates.lighting_base_view))],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
        let bg = self.shadow_bind_group.borrow();
        rpass.set_bind_group(1, &*bg, &[]);
        rpass.set_bind_group(2, &self.lighting_bind_group, &[]);
        rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }

    fn resize(&mut self, shared: &SharedResources, gbuffer: &GBuffer, intermediates: &IntermediateTargets) {
        self.gbuffer_bind_group = Self::create_gbuffer_bind_group(
            &shared.device, &self.gbuffer_bgl, gbuffer, intermediates, shared,
        );
        self.lighting_bind_group = Self::create_lighting_bind_group(
            &shared.device, &self.lighting_bgl, shared, &self.env_cube_view, &shared.filtering_sampler,
        );
    }
}
