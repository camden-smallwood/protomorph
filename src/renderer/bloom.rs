use crate::{
    lights::LightType,
    renderer::{
        color_clear_attach, create_fullscreen_pipeline, depth_tex_entry, sampler_entry,
        shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, QuadVertex, SharedResources},
        tex_entry, uniform_entry,
    },
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
    pub vignette_strength: f32,
    pub sun_screen_x: f32,
    pub sun_screen_y: f32,
    pub sun_visible: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FlareParams {
    sun_screen_pos: [f32; 2],
    sun_visible: f32,
    aspect_ratio: f32,
    chroma_shift: f32,
    _flare_pad: [f32; 3],
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

    // Lens flare (drawn in same render pass as bloom composite)
    flare_pipeline: wgpu::RenderPipeline,
    flare_bgl: wgpu::BindGroupLayout,
    flare_bind_group: wgpu::BindGroup,
    flare_params_buffer: wgpu::Buffer,
}

impl BloomPass {
    pub fn new(
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let filterable = wgpu::TextureSampleType::Float {
            filterable: true,
        };

        let downsample_bgl =
            shared
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                });

        let upsample_bgl =
            shared
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                });

        let composite_bgl =
            shared
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        depth_tex_entry(4),
                        sampler_entry(5, wgpu::SamplerBindingType::NonFiltering),
                    ],
                });

        let prefilter_pipeline = {
            let shader = shared.device.create_shader_module(wgpu::include_wgsl!(
                "../../assets/shaders/bloom_downsample.wgsl"
            ));

            let layout =
                shared
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("bloom_prefilter_layout"),
                        bind_group_layouts: &[&downsample_bgl],
                        immediate_size: 0,
                    });

            shared
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                            format: wgpu::TextureFormat::Rg11b10Ufloat,
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
        };

        let downsample_pipeline = {
            let shader = shared.device.create_shader_module(wgpu::include_wgsl!(
                "../../assets/shaders/bloom_downsample.wgsl"
            ));

            let layout =
                shared
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("bloom_downsample_layout"),
                        bind_group_layouts: &[&downsample_bgl],
                        immediate_size: 0,
                    });

            shared
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                            format: wgpu::TextureFormat::Rg11b10Ufloat,
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
        };

        let upsample_pipeline = {
            let shader = shared.device.create_shader_module(wgpu::include_wgsl!(
                "../../assets/shaders/bloom_upsample.wgsl"
            ));

            let layout =
                shared
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("bloom_upsample_layout"),
                        bind_group_layouts: &[&upsample_bgl],
                        immediate_size: 0,
                    });

            shared
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        };

        let composite_pipeline = {
            let shader = shared.device.create_shader_module(wgpu::include_wgsl!(
                "../../assets/shaders/bloom.wgsl"
            ));

            let layout =
                shared
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("bloom_composite_pipeline_layout")),
                        bind_group_layouts: &[&composite_bgl],
                        immediate_size: 0,
                    });

            shared
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        };

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
                        exposure: 1.5,
                        saturation: 0.85,
                        grain_intensity: 0.0,
                        vignette_strength: 0.3,
                        sun_screen_x: 0.0,
                        sun_screen_y: 0.0,
                        sun_visible: 0.0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let prefilter_bind_group = Self::create_downsample_bind_group(
            &shared.device,
            &downsample_bgl,
            &intermediates.lighting_base_view,
            &shared.bloom_sampler,
            &params_buffers[0],
        );

        let downsample_bind_groups: Vec<wgpu::BindGroup> = (0..BLOOM_MIP_COUNT - 1)
            .map(|i| {
                Self::create_downsample_bind_group(
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
                Self::create_upsample_bind_group(
                    &shared.device,
                    &upsample_bgl,
                    &intermediates.bloom_mip_views[i + 1],
                    &shared.bloom_sampler,
                    &upsample_params_buffer,
                )
            })
            .collect();

        let composite_bind_group = Self::create_composite_bind_group(
            &shared.device,
            &composite_bgl,
            &intermediates.lighting_base_view,
            &intermediates.bloom_mip_views[0],
            &shared.bloom_sampler,
            &composite_params_buffer,
            &gbuffer.depth_view,
            &shared.nearest_sampler,
        );

        Self::update_params(
            &shared.queue,
            &params_buffers,
            shared.config.width,
            shared.config.height,
            &intermediates.bloom_mip_sizes,
        );

        // --- Lens flare ---
        let flare_bgl = shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens_flare_bgl"),
            entries: &[
                tex_entry(0, filterable),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                depth_tex_entry(2),
                sampler_entry(3, wgpu::SamplerBindingType::NonFiltering),
                uniform_entry(4, size_of::<FlareParams>() as u64, wgpu::ShaderStages::FRAGMENT, false),
            ],
        });
        let flare_params_buffer = shared.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lens_flare_params"),
            contents: bytemuck::bytes_of(&FlareParams {
                sun_screen_pos: [0.5, 0.5], sun_visible: 0.0, aspect_ratio: 1.333,
                chroma_shift: 0.005, _flare_pad: [0.0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let flare_bind_group = Self::create_flare_bind_group(
            &shared.device, &flare_bgl, &intermediates.bloom_mip_views[2],
            &shared.bloom_sampler, &gbuffer.depth_view, &shared.nearest_sampler, &flare_params_buffer,
        );
        let flare_pipeline = create_fullscreen_pipeline(
            &shared.device,
            wgpu::include_wgsl!("../../assets/shaders/lens_flare.wgsl"),
            &[&flare_bgl],
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "lens_flare_pipeline",
        );

        Self {
            prefilter_pipeline, downsample_pipeline, upsample_pipeline, composite_pipeline,
            downsample_bgl, upsample_bgl, composite_bgl,
            params_buffers, upsample_params_buffer, composite_params_buffer,
            prefilter_bind_group, downsample_bind_groups, upsample_bind_groups, composite_bind_group,
            flare_pipeline, flare_bgl, flare_bind_group, flare_params_buffer,
        }
    }

    fn create_downsample_bind_group(
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

    fn create_upsample_bind_group(
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

    fn create_composite_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        base_view: &wgpu::TextureView,
        bloom_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
        depth_view: &wgpu::TextureView,
        nearest_sampler: &wgpu::Sampler,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
            ],
        })
    }

    fn create_flare_bind_group(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        bloom_view: &wgpu::TextureView, bloom_sampler: &wgpu::Sampler,
        depth_view: &wgpu::TextureView, nearest_sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens_flare_bg"), layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(bloom_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(bloom_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(nearest_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        })
    }

    fn update_params(
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
}

impl RenderPass for BloomPass {
    fn prepare(&mut self, ctx: &FrameContext) {
        // Project sun to screen — match god_rays exactly
        let sun_dir = -ctx.game.sky.sun_direction; // direction TO the sun
        let sun_world = sun_dir * 1000.0; // same as god_rays
        let vp = ctx.game.camera.projection * ctx.game.camera.view;
        let clip = vp * glam::Vec4::new(sun_world.x, sun_world.y, sun_world.z, 1.0);

        let (sun_x, sun_y, sun_visible) = if clip.w > 0.0 {
            let ndc = clip.truncate() / clip.w;
            let screen_dist = (ndc.x * ndc.x + ndc.y * ndc.y).sqrt();
            let fade = (1.0 - (screen_dist - 0.5).max(0.0) / 1.0).max(0.0);
            let uv_x = ndc.x * 0.5 + 0.5;
            let uv_y = -ndc.y * 0.5 + 0.5;
            (uv_x, uv_y, fade)
        } else {
            (0.0, 0.0, 0.0)
        };

        let params = BloomCompositeParams {
            bloom_strength: 0.15,
            exposure: 1.5,
            saturation: 0.85,
            grain_intensity: 0.0,
            vignette_strength: if ctx.game.enable_vignette { 0.3 } else { 0.0 },
            sun_screen_x: sun_x,
            sun_screen_y: sun_y,
            sun_visible,
        };
        ctx.shared.queue.write_buffer(
            &self.composite_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        // Lens flare params
        let game = ctx.game;
        let (flare_dir, _) = {
            let mut sd = glam::Vec3::new(0.0, 0.0, 1.0);
            let mut sc = glam::Vec3::ZERO;
            for (_idx, light) in game.lights.iter() {
                if !light.hidden && light.light_type == LightType::Directional {
                    sd = -light.direction.normalize();
                    sc = light.diffuse_color;
                    break;
                }
            }
            (sd, sc)
        };
        let flare_clip = vp * glam::Vec4::new(flare_dir.x * 1000.0, flare_dir.y * 1000.0, flare_dir.z * 1000.0, 1.0);
        let (flare_screen, flare_visible) = if flare_clip.w > 0.0 {
            let n = flare_clip.truncate() / flare_clip.w;
            let max_ndc = n.x.abs().max(n.y.abs());
            let t = ((1.0 - max_ndc).max(0.0) / 0.2).clamp(0.0, 1.0);
            ([n.x * 0.5 + 0.5, -n.y * 0.5 + 0.5], t * t * (3.0 - 2.0 * t))
        } else {
            ([0.5f32, 0.5], 0.0)
        };
        ctx.shared.queue.write_buffer(
            &self.flare_params_buffer, 0,
            bytemuck::bytes_of(&FlareParams {
                sun_screen_pos: flare_screen, sun_visible: flare_visible,
                aspect_ratio: ctx.shared.config.width as f32 / ctx.shared.config.height as f32,
                chroma_shift: 0.005, _flare_pad: [0.0; 3],
            }),
        );
    }

    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.prefilter_bind_group = Self::create_downsample_bind_group(
            &shared.device,
            &self.downsample_bgl,
            &intermediates.lighting_base_view,
            &shared.bloom_sampler,
            &self.params_buffers[0],
        );

        self.downsample_bind_groups = (0..BLOOM_MIP_COUNT - 1)
            .map(|i| {
                Self::create_downsample_bind_group(
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
                Self::create_upsample_bind_group(
                    &shared.device,
                    &self.upsample_bgl,
                    &intermediates.bloom_mip_views[i + 1],
                    &shared.bloom_sampler,
                    &self.upsample_params_buffer,
                )
            })
            .collect();

        self.composite_bind_group = Self::create_composite_bind_group(
            &shared.device,
            &self.composite_bgl,
            &intermediates.lighting_base_view,
            &intermediates.bloom_mip_views[0],
            &shared.bloom_sampler,
            &self.composite_params_buffer,
            &gbuffer.depth_view,
            &shared.nearest_sampler,
        );

        self.flare_bind_group = Self::create_flare_bind_group(
            &shared.device, &self.flare_bgl, &intermediates.bloom_mip_views[2],
            &shared.bloom_sampler, &gbuffer.depth_view, &shared.nearest_sampler, &self.flare_params_buffer,
        );

        Self::update_params(
            &shared.queue,
            &self.params_buffers,
            shared.config.width,
            shared.config.height,
            &intermediates.bloom_mip_sizes,
        );
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let intermediates = ctx.intermediates;
        let shared = ctx.shared;

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
                    view: &ctx.intermediates.post_composite_view,
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

            // Lens flare (same render pass, additive)
            rpass.set_pipeline(&self.flare_pipeline);
            rpass.set_bind_group(0, &self.flare_bind_group, &[]);
            rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
    }
}

