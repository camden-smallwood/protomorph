//! Light-volume (`rmlv`) strip render — Track R1 increment 2.
//!
//! Draws each light volume as a stack of roll-locked billboard quads (one
//! instance per profile) into the lighting-base color target, additively.
//! Stateless: the CPU rebuilds the profile instances every frame
//! ([`crate::halo::effects::EffectStore::frame_advance`]); this module just
//! uploads them and issues per-strip instanced draws of `light_volume.wgsl`.
//!
//! First pass: single color target, no depth (volumes draw on top). Depth
//! occlusion + the second (bloom) MRT are follow-ups.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use blam_tags::render_method::AlphaBlendMode;

use crate::halo::effects::{LightVolumeDraw, LightVolumeProfileGpu};
use crate::halo::render::shared::SharedResources;

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct PreparedDraw {
    blend: AlphaBlendMode,
    group1: wgpu::BindGroup,
    first: u32,
    count: u32,
}

pub struct LightVolumeGpu {
    shader: wgpu::ShaderModule,
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<AlphaBlendMode, wgpu::RenderPipeline>,
    sampler: wgpu::Sampler,
    fallback_view: wgpu::TextureView,
    frame_buf: wgpu::Buffer,
    profile_buf: wgpu::Buffer,
    profile_cap: u64,
    group0: Option<wgpu::BindGroup>,
    /// Cached `base_map` texture views, keyed by bitmap path (registered once
    /// at load via [`Self::register_texture`]).
    textures: HashMap<String, wgpu::TextureView>,
    prepared: Vec<PreparedDraw>,
    /// Per-draw uniform buffers, kept alive for the frame's bind groups.
    draw_bufs: Vec<wgpu::Buffer>,
}

impl LightVolumeGpu {
    pub fn new(shared: &SharedResources) -> Self {
        let device = &shared.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("light_volume_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/halo_shaders/light_volume.wgsl").into(),
            ),
        });
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_volume_group0"),
            entries: &[
                // camera
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // frame (exposure)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // profiles (storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // scene depth (hard occlusion)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_volume_group1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    // Read in BOTH stages: VS (exposure branch) + FS (multiply).
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light_volume_pl"),
            bind_group_layouts: &[&group0_layout, &group1_layout],
            immediate_size: 0,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("light_volume_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // 1×1 white fallback (for a missing base_map).
        let fallback = device.create_texture_with_data(
            &shared.queue,
            &wgpu::TextureDescriptor {
                label: Some("light_volume_fallback"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255u8, 255, 255, 255],
        );
        let fallback_view = fallback.create_view(&Default::default());
        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_volume_frame"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let profile_cap = 256u64 * std::mem::size_of::<LightVolumeProfileGpu>() as u64;
        let profile_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_volume_profiles"),
            size: profile_cap,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            shader,
            group0_layout,
            group1_layout,
            pipeline_layout,
            pipelines: HashMap::new(),
            sampler,
            fallback_view,
            frame_buf,
            profile_buf,
            profile_cap,
            group0: None,
            textures: HashMap::new(),
            prepared: Vec::new(),
            draw_bufs: Vec::new(),
        }
    }

    /// Cache a `base_map` view by bitmap path (called at load for each unique
    /// light-volume base_map).
    pub fn register_texture(&mut self, path: String, view: wgpu::TextureView) {
        self.textures.insert(path, view);
    }

    pub fn has_texture(&self, path: &str) -> bool {
        self.textures.contains_key(path)
    }

    fn ensure_pipeline(&mut self, shared: &SharedResources, blend: AlphaBlendMode) {
        if self.pipelines.contains_key(&blend) {
            return;
        }
        let pipeline = shared.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("light_volume_pipeline"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    blend: crate::halo::gpu_particle::blend_state(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.insert(blend, pipeline);
    }

    /// Upload this frame's profiles + build the per-strip bind groups.
    /// `illum_exposure` = engine V_ILLUM_EXPOSURE (additive self-illum).
    pub fn prepare(
        &mut self,
        shared: &SharedResources,
        depth_view: &wgpu::TextureView,
        profiles: &[LightVolumeProfileGpu],
        draws: &[LightVolumeDraw],
        illum_exposure: f32,
    ) {
        self.prepared.clear();
        self.draw_bufs.clear();
        if profiles.is_empty() || draws.is_empty() {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(profiles);
        if bytes.len() as u64 > self.profile_cap {
            self.profile_cap = (bytes.len() as u64).next_power_of_two();
            self.profile_buf = shared.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("light_volume_profiles"),
                size: self.profile_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        shared.queue.write_buffer(&self.profile_buf, 0, bytes);
        shared
            .queue
            .write_buffer(&self.frame_buf, 0, bytemuck::cast_slice(&[illum_exposure, 0.0, 0.0, 0.0]));
        self.group0 = Some(shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_volume_group0_bg"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shared.camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.frame_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.profile_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
            ],
        }));
        for d in draws {
            self.ensure_pipeline(shared, d.blend_mode);
            let is_multiply = (d.blend_mode == AlphaBlendMode::Multiply) as u32;
            let draw_buf = shared.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("light_volume_draw"),
                contents: bytemuck::cast_slice(&[is_multiply, 0u32, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let tex = self.textures.get(&d.base_map).unwrap_or(&self.fallback_view);
            let group1 = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("light_volume_group1_bg"),
                layout: &self.group1_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(tex) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: draw_buf.as_entire_binding() },
                ],
            });
            self.draw_bufs.push(draw_buf);
            self.prepared.push(PreparedDraw {
                blend: d.blend_mode,
                group1,
                first: d.first,
                count: d.count,
            });
        }
    }

    /// Draw the prepared strips into `target` (the lighting-base view). No-op
    /// when nothing was prepared this frame.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        if self.prepared.is_empty() {
            return;
        }
        let Some(group0) = self.group0.as_ref() else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("light_volume_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for p in &self.prepared {
            let Some(pipe) = self.pipelines.get(&p.blend) else { continue };
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, group0, &[]);
            pass.set_bind_group(1, &p.group1, &[]);
            pass.draw(0..4, p.first..p.first + p.count);
        }
    }
}
