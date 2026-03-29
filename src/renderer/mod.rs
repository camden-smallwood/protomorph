mod bloom;
mod cloud_composite;
mod cloud_noise;
mod cloud_raymarch;
mod final_composite;
pub mod env_probe;
mod god_rays;
mod lighting;
mod geometry;
mod shadow;
mod shared;
mod ssao;

mod water;

use crate::{
    animation::MAXIMUM_NUMBER_OF_MODEL_NODES,
    camera::CameraUniforms,
    dds::{create_dds_texture, load_dds_from_file},
    game::GameState,
    materials::{GpuMaterialProps, MaterialTextureUsage},
    models::{ModelData, ModelMeshPart, ModelUniforms},
    objects::ObjectIndex,
    renderer::env_probe::GpuSkyParams,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use shared::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MAX_OBJECTS: usize = 256;

// ---------------------------------------------------------------------------
// GPU-side model types (visible to pass modules)
// ---------------------------------------------------------------------------

pub(crate) struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub parts: Vec<ModelMeshPart>,
}

pub(crate) struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
}

pub(crate) struct GpuModel {
    pub meshes: Vec<GpuMesh>,
    pub materials: Vec<GpuMaterial>,
}

// ---------------------------------------------------------------------------
// Renderer orchestrator
// ---------------------------------------------------------------------------

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    shared: SharedResources,
    gbuffer: GBuffer,
    intermediates: IntermediateTargets,

    passes: Vec<Box<dyn RenderPass>>,

    models: Vec<GpuModel>,

    // Preallocated per-frame scratch buffers
    render_list: Vec<(ObjectIndex, usize)>,
    node_matrix_staging: Vec<u8>,

    // Temporal reprojection state
    prev_vp: glam::Mat4,
    frame_counter: u32,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE,
            ..Default::default()
        }))
        .expect("Failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shared = SharedResources::new(device, queue, config);

        let gbuffer = GBuffer::new(&shared.device, shared.config.width, shared.config.height);
        let intermediates = IntermediateTargets::new(
            &shared.device,
            shared.config.width,
            shared.config.height,
            shared.config.format,
        );

        // Generate cloud noise textures (one-shot compute dispatch)
        let cloud_textures = cloud_noise::generate(&shared);

        // Construct producer passes first for cross-pass wiring
        let shadow_pass = shadow::ShadowPass::new(&shared);
        let env_probe_pass = env_probe::EnvProbePass::new(&shared);

        // Wire deferred lighting pass
        let deferred_lighting_pass = lighting::DeferredLightingPass::new(
            &shared,
            &gbuffer,
            &intermediates,
            &shadow_pass.bgl,
            shadow_pass.bind_group.clone(),
            env_probe_pass.create_cube_view(),
            env_probe_pass.filtering_sampler(),
        );
        let water_pass = water::WaterPass::new(
            &shared,
            &gbuffer,
            &intermediates,
            env_probe_pass.create_cube_view(),
            &shadow_pass.bgl,
            shadow_pass.bind_group.clone(),
        );

        // Build final composite BEFORE env_probe_pass is moved (needs face views by reference)
        let final_composite_pass = final_composite::FinalCompositePass::new(
            &shared,
            &intermediates,
            shared.config.format,
            std::array::from_fn(|i| env_probe_pass.face_view(i)),
            env_probe_pass.filtering_sampler(),
        );

        // Build cloud passes (raymarch takes ownership of noise textures)
        let cloud_raymarch_pass = cloud_raymarch::CloudRaymarchPass::new(
            &shared,
            &gbuffer,
            &intermediates,
            cloud_textures,
        );
        let cloud_composite_pass = cloud_composite::CloudCompositePass::new(
            &shared,
            &gbuffer,
            &intermediates,
        );

        // Build passes vec in execution order
        let passes: Vec<Box<dyn RenderPass>> = vec![
            Box::new(shadow_pass),
            Box::new(env_probe_pass),
            Box::new(geometry::GBufferPass::new(&shared)),
            Box::new(cloud_raymarch_pass),
            Box::new(ssao::SsaoPass::new(&shared, &gbuffer)),
            Box::new(god_rays::GodRaysTracePass::new(&shared, &gbuffer.depth_view, &intermediates.cloud_raymarch_view)),
            Box::new(deferred_lighting_pass),
            Box::new(cloud_composite_pass),
            Box::new(water_pass),
            Box::new(bloom::BloomPass::new(&shared, &gbuffer, &intermediates, shared.config.format)),
            Box::new(final_composite_pass),
        ];

        Self {
            surface,
            shared,
            gbuffer,
            intermediates,
            passes,
            models: Vec::new(),
            render_list: Vec::with_capacity(MAX_OBJECTS),
            node_matrix_staging: vec![
                0u8;
                MAXIMUM_NUMBER_OF_MODEL_NODES
                    * std::mem::size_of::<[[f32; 4]; 4]>()
            ],
            prev_vp: glam::Mat4::IDENTITY,
            frame_counter: 0,
        }
    }

    pub fn load_model(&mut self, path: &str) -> (usize, ModelData) {
        self.load_model_with_uv_scale(path, 1.0)
    }

    pub fn load_model_with_uv_scale(
        &mut self,
        path: &str,
        uv_scale: f32,
    ) -> (usize, ModelData) {
        let model = ModelData::from_file_with_uv_scale(path, uv_scale);

        let materials: Vec<GpuMaterial> = model
            .materials
            .iter()
            .map(|mat| {
                let load_tex = |usage: MaterialTextureUsage,
                                fallback: &wgpu::TextureView|
                 -> wgpu::TextureView {
                    if let Some(path) = mat.find_texture(usage) {
                        let dds = load_dds_from_file(path);
                        let linear = matches!(
                            usage,
                            MaterialTextureUsage::Normal
                                | MaterialTextureUsage::Specular
                                | MaterialTextureUsage::Opacity
                                | MaterialTextureUsage::Height
                        );
                        let texture = create_dds_texture(
                            &self.shared.device,
                            &self.shared.queue,
                            &dds,
                            linear,
                        );
                        texture.create_view(&wgpu::TextureViewDescriptor::default())
                    } else {
                        fallback.clone()
                    }
                };

                let diffuse_view = load_tex(
                    MaterialTextureUsage::Diffuse,
                    &self.shared.fallback_textures.white_view,
                );
                let normal_view = load_tex(
                    MaterialTextureUsage::Normal,
                    &self.shared.fallback_textures.default_normal_view,
                );
                let specular_view = load_tex(
                    MaterialTextureUsage::Specular,
                    &self.shared.fallback_textures.white_view,
                );
                let emissive_view = load_tex(
                    MaterialTextureUsage::Emissive,
                    &self.shared.fallback_textures.black_view,
                );
                let opacity_view = load_tex(
                    MaterialTextureUsage::Opacity,
                    &self.shared.fallback_textures.white_view,
                );

                let gpu_props = GpuMaterialProps::from_material(mat);
                let props_buffer =
                    self.shared
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("material_props"),
                            contents: bytemuck::bytes_of(&gpu_props),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let bind_group = self
                    .shared
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("material_bg"),
                        layout: &self.shared.material_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&diffuse_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&normal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&specular_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&emissive_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&opacity_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Sampler(
                                    &self.shared.filtering_sampler,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: props_buffer.as_entire_binding(),
                            },
                        ],
                    });

                GpuMaterial { bind_group }
            })
            .collect();

        let meshes: Vec<GpuMesh> = model
            .meshes
            .iter()
            .map(|mesh| {
                let verts = &mesh.vertices;
                let vertex_buffer = self.shared.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("vertex_buffer"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                );
                let index_buffer =
                    self.shared
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("index_buffer"),
                            contents: bytemuck::cast_slice(&mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                GpuMesh {
                    vertex_buffer,
                    index_buffer,
                    parts: mesh.parts.clone(),
                }
            })
            .collect();

        let index = self.models.len();
        self.models.push(GpuModel {
            meshes,
            materials,
        });
        (index, model)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.shared.config.width = width;
        self.shared.config.height = height;
        self.surface.configure(&self.shared.device, &self.shared.config);

        self.gbuffer = GBuffer::new(&self.shared.device, width, height);
        self.intermediates = IntermediateTargets::new(
            &self.shared.device,
            width,
            height,
            self.shared.config.format,
        );

        for pass in &mut self.passes {
            pass.resize(&self.shared, &self.gbuffer, &self.intermediates);
        }
    }

    pub fn render(&mut self, game: &GameState) {
        // Text prepare happens later with frame_ctx

        // Build sorted render list
        self.render_list.clear();
        self.render_list.extend(
            game.objects
                .iter()
                .filter_map(|(idx, obj)| obj.model_index.map(|m| (idx, m))),
        );
        let mut render_list = std::mem::take(&mut self.render_list);

        // Upload camera uniforms
        let cam_uniforms = CameraUniforms {
            view: game.camera.view.to_cols_array_2d(),
            projection: game.camera.projection.to_cols_array_2d(),
        };
        self.shared.queue.write_buffer(
            &self.shared.camera_buffer,
            0,
            bytemuck::bytes_of(&cam_uniforms),
        );

        // Upload model uniforms + bone matrices
        for (i, &(obj_idx, _model_idx)) in render_list.iter().enumerate() {
            let obj = game.objects.get(obj_idx);
            let model_u = ModelUniforms {
                model: obj.model_matrix().to_cols_array_2d(),
            };
            let offset = (i * self.shared.model_stride) as u64;
            self.shared.queue.write_buffer(
                &self.shared.model_buffer,
                offset,
                bytemuck::bytes_of(&model_u),
            );

            if let Some(anim_mgr) = &obj.animations {
                if anim_mgr.matrices_dirty {
                    let mat_count = anim_mgr.node_matrices.len();
                    let byte_len = mat_count * std::mem::size_of::<[[f32; 4]; 4]>();
                    for (j, mat) in anim_mgr.node_matrices.iter().enumerate() {
                        let offset = j * 64;
                        let cols = mat.to_cols_array();
                        self.node_matrix_staging[offset..offset + 64]
                            .copy_from_slice(bytemuck::cast_slice(&cols));
                    }
                    let nm_offset = (i * self.shared.node_matrices_stride) as u64;
                    self.shared.queue.write_buffer(
                        &self.shared.node_matrices_buffer,
                        nm_offset,
                        &self.node_matrix_staging[..byte_len],
                    );
                }
            }
        }

        self.shared.queue.write_buffer(
            &self.shared.atmosphere_buffer,
            0,
            bytemuck::bytes_of(&game.atmosphere_data()),
        );

        // Upload sky params for merged lighting+sky pass
        {
            let inverse_vp = (game.camera.projection * game.camera.view).inverse();
            let sky_params = GpuSkyParams {
                inverse_view_projection: inverse_vp.to_cols_array_2d(),
                camera_position: game.camera.position.into(),
                _pad: 0.0,
            };
            self.shared.queue.write_buffer(
                &self.shared.sky_params_buffer,
                0,
                bytemuck::bytes_of(&sky_params),
            );
        }

        // Acquire surface
        let output = match self.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface
                    .configure(&self.shared.device, &self.shared.config);
                self.render_list = render_list;
                return;
            }
            Err(e) => {
                eprintln!("Surface error: {e}");
                self.render_list = render_list;
                return;
            }
        };
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let frame_ctx = shared::FrameContext {
            shared: &self.shared,
            gbuffer: &self.gbuffer,
            intermediates: &self.intermediates,
            surface_view: &surface_view,
            models: &self.models,
            render_list: &render_list,
            game,
            frame_index: self.frame_counter,
            prev_view_projection: self.prev_vp,
        };

        // Prepare phase — mutable pass access for uniform uploads
        for pass in &mut self.passes {
            if pass.is_enabled(&frame_ctx) {
                pass.prepare(&frame_ctx);
            }
        }

        // Record phase — encode GPU commands
        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        for pass in self.passes.iter() {
            if pass.is_enabled(&frame_ctx) {
                pass.record(&mut encoder, &frame_ctx);
            }
        }

        self.shared.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Post-submit phase — cleanup
        for pass in &mut self.passes {
            pass.post_submit();
        }

        // Save current VP for next frame's temporal reprojection
        self.prev_vp = game.camera.projection * game.camera.view;
        self.frame_counter = self.frame_counter.wrapping_add(1);

        // Return preallocated buffers
        render_list.clear();
        self.render_list = render_list;
    }
}

// ---------------------------------------------------------------------------
// Shared pipeline helper (used by pass modules)
// ---------------------------------------------------------------------------

fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    shader_source: wgpu::ShaderModuleDescriptor,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    color_targets: &[Option<wgpu::ColorTargetState>],
    label: &str,
) -> wgpu::RenderPipeline {
    create_fullscreen_pipeline_with_depth(device, shader_source, bind_group_layouts, color_targets, None, label)
}

fn create_fullscreen_pipeline_with_depth(
    device: &wgpu::Device,
    shader_source: wgpu::ShaderModuleDescriptor,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    color_targets: &[Option<wgpu::ColorTargetState>],
    depth_stencil: Option<wgpu::DepthStencilState>,
    label: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader_source);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts,
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
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
            targets: color_targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Shared draw loop — used by all geometry passes
// ---------------------------------------------------------------------------

/// Draw all visible models. Caller must set pipeline and group 0 (camera) before calling.
/// Groups 1 (model) and 3 (node matrices) are set per-object with dynamic offsets.
/// If `bind_materials` is true, sets group 2 per mesh part from material bind groups.
/// Otherwise group 2 is assumed pre-set by the caller (forward lighting passes).
pub(crate) fn draw_models<'a>(
    rpass: &mut wgpu::RenderPass<'a>,
    shared: &'a SharedResources,
    ctx: &'a FrameContext<'a>,
    bind_materials: bool,
) {
    for (obj_slot, &(_obj_idx, model_idx)) in ctx.render_list.iter().enumerate() {
        let model_offset = (obj_slot * shared.model_stride) as u32;
        let nm_offset = (obj_slot * shared.node_matrices_stride) as u32;
        let gpu_model = &ctx.models[model_idx];

        rpass.set_bind_group(1, &shared.model_bind_group, &[model_offset]);
        rpass.set_bind_group(3, &shared.node_matrices_bind_group, &[nm_offset]);

        for mesh in &gpu_model.meshes {
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            for part in &mesh.parts {
                if bind_materials {
                    if let Some(material) = gpu_model.materials.get(part.material_index) {
                        rpass.set_bind_group(2, &material.bind_group, &[]);
                    }
                }
                rpass.draw_indexed(part.index_start..part.index_start + part.index_count, 0, 0..1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

pub fn color_clear_attach(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
            store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
    }
}

pub fn create_1x1_texture(device: &wgpu::Device, queue: &wgpu::Queue, rgba: [u8; 4], label: &str, format: wgpu::TextureFormat) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &rgba,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    texture
}

pub fn create_screen_texture(device: &wgpu::Device, w: u32, h: u32, format: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

pub fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub fn depth_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub fn sampler_entry(binding: u32, sampler_type: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(sampler_type),
        count: None,
    }
}

pub fn uniform_entry(binding: u32, size: u64, visibility: wgpu::ShaderStages, has_dynamic_offset: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}
