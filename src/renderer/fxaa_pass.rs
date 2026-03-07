use crate::renderer::{
    create_fullscreen_pipeline, sampler_entry,
    shared::{IntermediateTargets, SharedResources},
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
        let bgl = create_fxaa_bgl(&shared.device);
        let pipeline = create_fxaa_pipeline(&shared.device, &bgl, surface_format);

        let bind_group = create_fxaa_bind_group(
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

    pub fn resize(&mut self, shared: &SharedResources, intermediates: &IntermediateTargets) {
        self.bind_group = create_fxaa_bind_group(
            &shared.device,
            &self.bgl,
            &intermediates.post_composite_view,
            &shared.bloom_sampler,
        );
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        surface_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fxaa_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
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
        rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Bind group layout + creation
// ---------------------------------------------------------------------------

fn create_fxaa_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let filterable = wgpu::TextureSampleType::Float { filterable: true };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fxaa_bgl"),
        entries: &[
            tex_entry(0, filterable),
            sampler_entry(1, wgpu::SamplerBindingType::Filtering),
        ],
    })
}

fn create_fxaa_bind_group(
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

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn create_fxaa_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    create_fullscreen_pipeline(
        device,
        wgpu::include_wgsl!("../../assets/shaders/fxaa.wgsl"),
        &[bgl],
        &[Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        "fxaa_pipeline",
    )
}
