use crate::renderer::{
    color_clear_attach, sampler_entry,
    shared::{IntermediateTargets, QuadVertex, SharedResources},
    tex_entry, uniform_entry,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const BLOOM_MIP_COUNT: usize = 4;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BloomDownsampleParams {
    pub threshold: f32,
    pub knee: f32,
    pub texel_size: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BloomUpsampleParams {
    pub filter_radius: f32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BloomCompositeParams {
    pub bloom_strength: f32,
    pub exposure: f32,
    pub saturation: f32,
    pub grain_intensity: f32,
}

pub struct BloomPass {
    prefilter_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    upsample_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    downsample_bgl: wgpu::BindGroupLayout,
    upsample_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,

    params_buffers: Vec<wgpu::Buffer>,
    upsample_params_buffer: wgpu::Buffer,
    composite_params_buffer: wgpu::Buffer,

    prefilter_bind_group: wgpu::BindGroup,
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    upsample_bind_groups: Vec<wgpu::BindGroup>,
    composite_bind_group: wgpu::BindGroup,
}

impl BloomPass {
    pub fn new(
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let downsample_bgl = create_bloom_downsample_bgl(&shared.device);
        let upsample_bgl = create_bloom_upsample_bgl(&shared.device);
        let composite_bgl = create_bloom_composite_bgl(&shared.device);

        let prefilter_pipeline = create_bloom_prefilter_pipeline(&shared.device, &downsample_bgl);
        let downsample_pipeline = create_bloom_downsample_pipeline(&shared.device, &downsample_bgl);
        let upsample_pipeline = create_bloom_upsample_pipeline(&shared.device, &upsample_bgl);
        let composite_pipeline = create_bloom_composite_pipeline(&shared.device, &composite_bgl, surface_format);

        let params_buffers: Vec<wgpu::Buffer> = (0..BLOOM_MIP_COUNT)
            .map(|i| {
                shared
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("bloom_params_{i}")),
                        contents: bytemuck::bytes_of(&BloomDownsampleParams {
                            threshold: 0.8,
                            knee: 0.3,
                            texel_size: [0.0, 0.0],
                        }),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
            })
            .collect();

        let upsample_params_buffer =
            shared
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("upsample_params"),
                    contents: bytemuck::bytes_of(&BloomUpsampleParams {
                        filter_radius: 0.008,
                        _pad: [0.0; 3],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let composite_params_buffer =
            shared
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("composite_params"),
                    contents: bytemuck::bytes_of(&BloomCompositeParams {
                        bloom_strength: 0.15,
                        exposure: 2.0,
                        saturation: 0.85,
                        grain_intensity: 0.0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let prefilter_bind_group = create_bloom_downsample_bind_group(
            &shared.device,
            &downsample_bgl,
            &intermediates.lighting_base_view,
            &shared.bloom_sampler,
            &params_buffers[0],
        );

        let downsample_bind_groups: Vec<wgpu::BindGroup> = (0..BLOOM_MIP_COUNT - 1)
            .map(|i| {
                create_bloom_downsample_bind_group(
                    &shared.device,
                    &downsample_bgl,
                    &intermediates.bloom_mip_views[i],
                    &shared.bloom_sampler,
                    &params_buffers[i + 1],
                )
            })
            .collect();

        let upsample_bind_groups: Vec<wgpu::BindGroup> = (0..BLOOM_MIP_COUNT - 1)
            .map(|i| {
                create_bloom_upsample_bind_group(
                    &shared.device,
                    &upsample_bgl,
                    &intermediates.bloom_mip_views[i + 1],
                    &shared.bloom_sampler,
                    &upsample_params_buffer,
                )
            })
            .collect();

        let composite_bind_group = create_bloom_composite_bind_group(
            &shared.device,
            &composite_bgl,
            &intermediates.lighting_base_view,
            &intermediates.bloom_mip_views[0],
            &shared.bloom_sampler,
            &composite_params_buffer,
        );

        update_bloom_params(
            &shared.queue,
            &params_buffers,
            shared.config.width,
            shared.config.height,
            &intermediates.bloom_mip_sizes,
        );

        Self {
            prefilter_pipeline,
            downsample_pipeline,
            upsample_pipeline,
            composite_pipeline,
            downsample_bgl,
            upsample_bgl,
            composite_bgl,
            params_buffers,
            upsample_params_buffer,
            composite_params_buffer,
            prefilter_bind_group,
            downsample_bind_groups,
            upsample_bind_groups,
            composite_bind_group,
        }
    }

    pub fn resize(&mut self, shared: &SharedResources, intermediates: &IntermediateTargets) {
        self.prefilter_bind_group = create_bloom_downsample_bind_group(
            &shared.device,
            &self.downsample_bgl,
            &intermediates.lighting_base_view,
            &shared.bloom_sampler,
            &self.params_buffers[0],
        );

        self.downsample_bind_groups = (0..BLOOM_MIP_COUNT - 1)
            .map(|i| {
                create_bloom_downsample_bind_group(
                    &shared.device,
                    &self.downsample_bgl,
                    &intermediates.bloom_mip_views[i],
                    &shared.bloom_sampler,
                    &self.params_buffers[i + 1],
                )
            })
            .collect();

        self.upsample_bind_groups = (0..BLOOM_MIP_COUNT - 1)
            .map(|i| {
                create_bloom_upsample_bind_group(
                    &shared.device,
                    &self.upsample_bgl,
                    &intermediates.bloom_mip_views[i + 1],
                    &shared.bloom_sampler,
                    &self.upsample_params_buffer,
                )
            })
            .collect();

        self.composite_bind_group = create_bloom_composite_bind_group(
            &shared.device,
            &self.composite_bgl,
            &intermediates.lighting_base_view,
            &intermediates.bloom_mip_views[0],
            &shared.bloom_sampler,
            &self.composite_params_buffer,
        );

        update_bloom_params(
            &shared.queue,
            &self.params_buffers,
            shared.config.width,
            shared.config.height,
            &intermediates.bloom_mip_sizes,
        );
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        intermediates: &IntermediateTargets,
        surface_view: &wgpu::TextureView,
    ) {
        // Prefilter
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_prefilter"),
                color_attachments: &[Some(color_clear_attach(&intermediates.bloom_mip_views[0]))],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            rpass.set_pipeline(&self.prefilter_pipeline);
            rpass.set_bind_group(0, &self.prefilter_bind_group, &[]);
            rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }

        // Downsample
        for i in 0..BLOOM_MIP_COUNT - 1 {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_downsample"),
                color_attachments: &[Some(color_clear_attach(
                    &intermediates.bloom_mip_views[i + 1],
                ))],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            rpass.set_pipeline(&self.downsample_pipeline);
            rpass.set_bind_group(0, &self.downsample_bind_groups[i], &[]);
            rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }

        // Upsample
        for i in (0..BLOOM_MIP_COUNT - 1).rev() {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &intermediates.bloom_mip_views[i],
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

            rpass.set_pipeline(&self.upsample_pipeline);
            rpass.set_bind_group(0, &self.upsample_bind_groups[i], &[]);
            rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }

        // Composite + tone map
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_composite"),
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

            rpass.set_pipeline(&self.composite_pipeline);
            rpass.set_bind_group(0, &self.composite_bind_group, &[]);
            rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
    }
}

// ---------------------------------------------------------------------------
// Bind group layouts
// ---------------------------------------------------------------------------

fn create_bloom_downsample_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let filterable = wgpu::TextureSampleType::Float {
        filterable: true,
    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom_downsample_bgl"),
        entries: &[
            tex_entry(0, filterable),
            sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            uniform_entry(
                2,
                size_of::<BloomDownsampleParams>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
        ],
    })
}

fn create_bloom_upsample_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let filterable = wgpu::TextureSampleType::Float {
        filterable: true,
    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom_upsample_bgl"),
        entries: &[
            tex_entry(0, filterable),
            sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            uniform_entry(
                2,
                size_of::<BloomUpsampleParams>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
        ],
    })
}

fn create_bloom_composite_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let filterable = wgpu::TextureSampleType::Float {
        filterable: true,
    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom_composite_bgl"),
        entries: &[
            tex_entry(0, filterable),
            tex_entry(1, filterable),
            sampler_entry(2, wgpu::SamplerBindingType::Filtering),
            uniform_entry(
                3,
                size_of::<BloomCompositeParams>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
        ],
    })
}

// ---------------------------------------------------------------------------
// Bind group creation
// ---------------------------------------------------------------------------

fn create_bloom_downsample_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bloom_downsample_bg"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_bloom_upsample_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bloom_upsample_bg"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_bloom_composite_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    base_view: &wgpu::TextureView,
    bloom_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bloom_composite_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(base_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(bloom_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

fn update_bloom_params(
    queue: &wgpu::Queue,
    buffers: &[wgpu::Buffer],
    screen_w: u32,
    screen_h: u32,
    mip_sizes: &[(u32, u32)],
) {
    let params0 = BloomDownsampleParams {
        threshold: 0.8,
        knee: 0.3,
        texel_size: [1.0 / screen_w as f32, 1.0 / screen_h as f32],
    };

    queue.write_buffer(&buffers[0], 0, bytemuck::bytes_of(&params0));

    for i in 0..BLOOM_MIP_COUNT - 1 {
        let (mw, mh) = mip_sizes[i];

        let params = BloomDownsampleParams {
            threshold: 0.0,
            knee: 0.0,
            texel_size: [1.0 / mw as f32, 1.0 / mh as f32],
        };

        queue.write_buffer(&buffers[i + 1], 0, bytemuck::bytes_of(&params));
    }
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

fn create_bloom_prefilter_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!(
        "../../assets/shaders/bloom_downsample.wgsl"
    ));

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bloom_prefilter_layout"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bloom_prefilter_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[QuadVertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_prefilter"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_bloom_downsample_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!(
        "../../assets/shaders/bloom_downsample.wgsl"
    ));

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bloom_downsample_layout"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bloom_downsample_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[QuadVertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_downsample"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_bloom_upsample_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!(
        "../../assets/shaders/bloom_upsample.wgsl"
    ));

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bloom_upsample_layout"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bloom_upsample_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[QuadVertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
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
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_bloom_composite_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!(
        "../../assets/shaders/bloom.wgsl"
    ));
    
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("bloom_composite_pipeline_layout")),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bloom_composite_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[QuadVertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}
