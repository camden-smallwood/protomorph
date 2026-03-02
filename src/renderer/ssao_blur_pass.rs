use crate::renderer::{
    create_fullscreen_pipeline,
    helpers::{sampler_entry, tex_entry},
    shared::{GBuffer, IntermediateTargets, SharedResources},
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
        let bgl = create_ssao_blur_bgl(&shared.device);
        let pipeline = create_ssao_blur_pipeline(&shared.device, &bgl);

        let bind_group = create_ssao_blur_bind_group(
            &shared.device,
            &bgl,
            &intermediates.ssao_view,
            &gbuffer.position_depth_view,
            &shared.nearest_sampler,
        );

        Self {
            pipeline,
            bgl,
            bind_group,
        }
    }

    pub fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.bind_group = create_ssao_blur_bind_group(
            &shared.device,
            &self.bgl,
            &intermediates.ssao_view,
            &gbuffer.position_depth_view,
            &shared.nearest_sampler,
        );
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        ssao_blur_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao_blur_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ssao_blur_view,
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
        rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Bind group layout + creation
// ---------------------------------------------------------------------------

fn create_ssao_blur_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let filterable = wgpu::TextureSampleType::Float { filterable: true };
    let unfilterable = wgpu::TextureSampleType::Float { filterable: false };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ssao_blur_bgl"),
        entries: &[
            tex_entry(0, filterable),     // t_ssao (R8Unorm, filterable)
            tex_entry(1, unfilterable),   // t_position_depth (Rgba16Float, unfilterable)
            sampler_entry(2, wgpu::SamplerBindingType::NonFiltering),
        ],
    })
}

fn create_ssao_blur_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    ssao_view: &wgpu::TextureView,
    position_depth_view: &wgpu::TextureView,
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
                resource: wgpu::BindingResource::TextureView(position_depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn create_ssao_blur_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    create_fullscreen_pipeline(
        device,
        wgpu::include_wgsl!("../../assets/shaders/ssao_blur.wgsl"),
        &[bgl],
        &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        "ssao_blur_pipeline",
    )
}
