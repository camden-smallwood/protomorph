mod bloom;
mod cubemap_debug;
pub mod env_probe;
mod fxaa;
mod geometry;
mod god_rays;
mod lighting;
mod shadow;
mod shared;
mod ssao_blur;
mod ssao;
mod text;

use crate::{
    animation::MAXIMUM_NUMBER_OF_MODEL_NODES,
    camera::CameraUniforms,
    dds::{create_dds_texture, load_dds_from_file},
    game::GameState,
    materials::{GpuMaterialProps, MaterialTextureUsage},
    models::{ModelData, ModelMeshPart, ModelUniforms, VertexType},
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
    pub vertex_type: VertexType,
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

        // Construct producer passes first for cross-pass wiring
        let shadow_pass = shadow::ShadowPass::new(&shared);
        let env_probe_pass = env_probe::EnvProbePass::new(&shared);

        // Wire dependent passes using Rc::clone of shared bind groups
        let lighting_pass = lighting::LightingPass::new(
            &shared,
            &gbuffer,
            &intermediates,
            &shadow_pass.bgl,
            &env_probe_pass.bgl,
            shadow_pass.bind_group.clone(),
            env_probe_pass.bind_group.clone(),
        );
        let cubemap_debug_pass = cubemap_debug::CubemapDebugPass::new(
            &shared,
            std::array::from_fn(|i| env_probe_pass.face_view(i)),
            env_probe_pass.filtering_sampler(),
            shared.config.format,
        );

        // Build passes vec in execution order
        let passes: Vec<Box<dyn RenderPass>> = vec![
            Box::new(shadow_pass),
            Box::new(env_probe_pass),
            Box::new(geometry::DepthPrepass::new(&shared)),
            Box::new(geometry::GBufferPass::new(&shared)),
            Box::new(ssao::SsaoPass::new(&shared, &gbuffer)),
            Box::new(ssao_blur::SsaoBlurPass::new(&shared, &gbuffer, &intermediates)),
            Box::new(lighting_pass),
            Box::new(geometry::EmissiveForwardPass::new(&shared)),
            Box::new(god_rays::GodRaysTracePass::new(&shared, &gbuffer.depth_view)),
            Box::new(god_rays::GodRaysCompositePass::new(&shared, &intermediates)),
            Box::new(bloom::BloomPass::new(&shared, &intermediates, shared.config.format)),
            Box::new(fxaa::FxaaPass::new(&shared, &intermediates, shared.config.format)),
            Box::new(cubemap_debug_pass),
            Box::new(text::TextPass::new(&shared)),
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
        }
    }

    pub fn load_model(&mut self, path: &str) -> (usize, ModelData) {
        let model = ModelData::from_file(path);

        let materials: Vec<GpuMaterial> = model
            .materials
            .iter()
            .map(|mat| {
                let load_tex = |usage: MaterialTextureUsage,
                                fallback: &wgpu::TextureView|
                 -> wgpu::TextureView {
                    if let Some(path) = mat.find_texture(usage) {
                        println!("Loading texture ({:?}): {}", usage, path);
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
                let vertex_buffer =
                    match mesh.vertex_type {
                        VertexType::Rigid => self.shared.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("vertex_buffer_rigid"),
                                contents: bytemuck::cast_slice(&mesh.rigid_vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        ),
                        VertexType::Skinned => self.shared.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("vertex_buffer_skinned"),
                                contents: bytemuck::cast_slice(&mesh.skinned_vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        ),
                    };
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
                    vertex_type: mesh.vertex_type,
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
        self.render_list.sort_by_key(|&(_, model_idx)| {
            let gpu_model = &self.models[model_idx];
            gpu_model
                .meshes
                .iter()
                .any(|m| m.vertex_type == VertexType::Skinned) as u8
        });
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

        self.shared.queue.write_buffer(
            &self.shared.atmosphere_buffer,
            0,
            bytemuck::bytes_of(&game.atmosphere),
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
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
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
