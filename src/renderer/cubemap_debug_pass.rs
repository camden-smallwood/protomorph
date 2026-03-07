use crate::renderer::{
    create_fullscreen_pipeline, sampler_entry, shared::SharedResources, tex_entry, uniform_entry,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DebugParams {
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    _pad: f32,
}

pub struct CubemapDebugPass {
    pipeline: wgpu::RenderPipeline,
    bind_groups: [wgpu::BindGroup; 6],
}

impl CubemapDebugPass {
    pub fn new(
        shared: &SharedResources,
        face_views: [&wgpu::TextureView; 6],
        filtering_sampler: &wgpu::Sampler,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let device = &shared.device;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cubemap_debug_bgl"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                uniform_entry(
                    2,
                    size_of::<DebugParams>() as u64,
                    wgpu::ShaderStages::VERTEX,
                    false,
                ),
            ],
        });

        let pipeline = create_fullscreen_pipeline(
            device,
            wgpu::include_wgsl!("../../assets/shaders/cubemap_debug.wgsl"),
            &[&bgl],
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "cubemap_debug_pipeline",
        );

        let scale = 0.12;
        let y = -0.75;

        let bind_groups: [wgpu::BindGroup; 6] = std::array::from_fn(|face| {
            let x = -0.75 + face as f32 * 0.27;
            let params = DebugParams {
                offset_x: x,
                offset_y: y,
                scale,
                _pad: 0.0,
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("cubemap_debug_params_{face}")),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("cubemap_debug_bg_{face}")),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(face_views[face]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(filtering_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            })
        });

        Self {
            pipeline,
            bind_groups,
        }
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        surface_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cubemap_debug_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
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
        rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));

        for face in 0..6 {
            rpass.set_bind_group(0, &self.bind_groups[face], &[]);
            rpass.draw(0..6, 0..1);
        }
    }
}
