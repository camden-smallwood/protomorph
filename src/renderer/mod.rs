mod bloom_pass;
mod cubemap_debug_pass;
mod env_probe_pass;
mod fxaa_pass;
mod geometry_pass;
mod god_rays_pass;
mod helpers;
mod lighting_pass;
mod shadow_pass;
mod shared;
mod ssao_blur_pass;
mod ssao_pass;
mod text_pass;

use crate::{
    animation::MAXIMUM_NUMBER_OF_MODEL_NODES,
    dds::{create_dds_texture, load_dds_from_file},
    game::GameState,
    gpu_types::*,
    lights::GpuLightingUniforms,
    materials::{GpuMaterialProps, MaterialTextureUsage},
    model::{ModelData, VertexType},
    objects::ObjectIndex,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use shared::*;

// ---------------------------------------------------------------------------
// GPU-side model types (visible to pass modules)
// ---------------------------------------------------------------------------

pub(crate) struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub parts: Vec<crate::model::ModelMeshPart>,
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

    shadow_pass: shadow_pass::ShadowPass,
    env_probe_pass: env_probe_pass::EnvProbePass,
    geometry_pass: geometry_pass::GeometryPass,
    ssao_pass: ssao_pass::SsaoPass,
    ssao_blur_pass: ssao_blur_pass::SsaoBlurPass,
    lighting_pass: lighting_pass::LightingPass,
    god_rays_pass: god_rays_pass::GodRaysPass,
    bloom_pass: bloom_pass::BloomPass,
    fxaa_pass: fxaa_pass::FxaaPass,
    cubemap_debug_pass: cubemap_debug_pass::CubemapDebugPass,
    text_pass: text_pass::TextPass,

    models: Vec<GpuModel>,

    // Preallocated per-frame scratch buffers
    render_list: Vec<(ObjectIndex, usize)>,
    point_shadow_casters: Vec<(usize, usize)>,
    spot_shadow_casters: Vec<(usize, usize)>,
    shadow_assignments: Vec<(usize, i32)>,
    node_matrix_staging: Vec<u8>,
}

pub const MAX_OBJECTS: usize = 256;

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

        let gbuffer = create_gbuffer(&shared.device, shared.config.width, shared.config.height);
        let intermediates = create_intermediates(
            &shared.device,
            shared.config.width,
            shared.config.height,
            shared.config.format,
        );

        let shadow_pass = shadow_pass::ShadowPass::new(&shared);
        let env_probe_pass = env_probe_pass::EnvProbePass::new(&shared);
        let geometry_pass = geometry_pass::GeometryPass::new(&shared);
        let ssao_pass = ssao_pass::SsaoPass::new(&shared, &gbuffer);
        let ssao_blur_pass =
            ssao_blur_pass::SsaoBlurPass::new(&shared, &gbuffer, &intermediates);
        let lighting_pass = lighting_pass::LightingPass::new(
            &shared,
            &gbuffer,
            &intermediates,
            &shadow_pass.bgl,
            &env_probe_pass.bgl,
        );
        let god_rays_pass = god_rays_pass::GodRaysPass::new(
            &shared,
            &gbuffer.depth_view,
            &intermediates,
        );
        let bloom_pass = bloom_pass::BloomPass::new(&shared, &intermediates, shared.config.format);
        let fxaa_pass =
            fxaa_pass::FxaaPass::new(&shared, &intermediates, shared.config.format);
        let cubemap_debug_pass = cubemap_debug_pass::CubemapDebugPass::new(
            &shared,
            std::array::from_fn(|i| env_probe_pass.face_view(i)),
            env_probe_pass.filtering_sampler(),
            shared.config.format,
        );
        let text_pass = text_pass::TextPass::new(&shared);

        Self {
            surface,
            shared,
            gbuffer,
            intermediates,
            shadow_pass,
            env_probe_pass,
            geometry_pass,
            ssao_pass,
            ssao_blur_pass,
            lighting_pass,
            god_rays_pass,
            bloom_pass,
            fxaa_pass,
            cubemap_debug_pass,
            text_pass,
            models: Vec::new(),
            render_list: Vec::with_capacity(MAX_OBJECTS),
            point_shadow_casters: Vec::with_capacity(MAX_POINT_SHADOW_CASTERS),
            spot_shadow_casters: Vec::with_capacity(MAX_SPOT_SHADOW_CASTERS),
            shadow_assignments: Vec::with_capacity(
                MAX_POINT_SHADOW_CASTERS + MAX_SPOT_SHADOW_CASTERS,
            ),
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
        self.surface
            .configure(&self.shared.device, &self.shared.config);

        self.gbuffer = create_gbuffer(&self.shared.device, width, height);
        self.intermediates = create_intermediates(
            &self.shared.device,
            width,
            height,
            self.shared.config.format,
        );

        self.ssao_pass.resize(&self.shared, &self.gbuffer);
        self.ssao_blur_pass
            .resize(&self.shared, &self.gbuffer, &self.intermediates);
        self.lighting_pass
            .resize(&self.shared, &self.gbuffer, &self.intermediates);
        self.god_rays_pass
            .resize(&self.shared, &self.gbuffer.depth_view, &self.intermediates);
        self.bloom_pass.resize(&self.shared, &self.intermediates);
        self.fxaa_pass.resize(&self.shared, &self.intermediates);
    }

    pub fn render(&mut self, game: &GameState) {
        // Prepare text
        self.text_pass.prepare(
            &self.shared,
            game.fps_counter.display_fps,
            self.shared.config.width,
            self.shared.config.height,
            game.debug_cubemap,
        );

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

        // Shadow preparation
        let mut point_shadow_casters = std::mem::take(&mut self.point_shadow_casters);
        let mut spot_shadow_casters = std::mem::take(&mut self.spot_shadow_casters);
        let mut shadow_assignments = std::mem::take(&mut self.shadow_assignments);
        self.shadow_pass.prepare(
            &self.shared,
            &game.lights,
            game.camera.view,
            game.camera.projection,
            &mut point_shadow_casters,
            &mut spot_shadow_casters,
            &mut shadow_assignments,
        );

        // Upload lighting uniforms (with shadow assignments)
        let light_uniforms = GpuLightingUniforms::from_scene(
            game.camera.position,
            game.camera.forward,
            &game.lights,
            &shadow_assignments,
            game.enable_specular_occlusion,
        );
        self.shared.queue.write_buffer(
            &self.shared.lighting_buffer,
            0,
            bytemuck::bytes_of(&light_uniforms),
        );

        self.shared.queue.write_buffer(
            &self.shared.atmosphere_buffer,
            0,
            bytemuck::bytes_of(&game.atmosphere),
        );

        // Acquire surface
        let output = match self.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface
                    .configure(&self.shared.device, &self.shared.config);
                self.render_list = render_list;
                self.point_shadow_casters = point_shadow_casters;
                self.spot_shadow_casters = spot_shadow_casters;
                self.shadow_assignments = shadow_assignments;
                return;
            }
            Err(e) => {
                eprintln!("Surface error: {e}");
                self.render_list = render_list;
                self.point_shadow_casters = point_shadow_casters;
                self.spot_shadow_casters = spot_shadow_casters;
                self.shadow_assignments = shadow_assignments;
                return;
            }
        };
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.shared
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        // === Shadow depth passes ===
        let has_directional_shadow = game
            .lights
            .iter()
            .any(|(_, l)| {
                !l.hidden
                    && l.casts_shadow
                    && l.light_type == crate::lights::LightType::Directional
            });
        self.shadow_pass.record(
            &mut encoder,
            &self.shared,
            &self.models,
            &render_list,
            &point_shadow_casters,
            &spot_shadow_casters,
            has_directional_shadow,
        );

        // === Env probe cubemap pass ===
        // Extract sun parameters for SH computation
        let (sun_dir_to_sun, sun_diffuse, sun_ambient) = {
            let mut sd = glam::Vec3::new(0.0, 0.0, 1.0);
            let mut sc = glam::Vec3::ZERO;
            let mut sa = glam::Vec3::ZERO;
            for (_idx, light) in game.lights.iter() {
                if !light.hidden && light.light_type == crate::lights::LightType::Directional {
                    sd = -light.direction.normalize(); // direction TO the sun
                    sc = light.diffuse_color;
                    sa = light.ambient_color;
                    break;
                }
            }
            (sd, sc, sa)
        };
        self.env_probe_pass.record(
            &mut encoder,
            &self.shared,
            &self.models,
            &render_list,
            game.camera.position,
            sun_dir_to_sun,
            sun_diffuse,
            sun_ambient,
            game.debug_cubemap_colors,
        );

        // === Depth pre-pass (populates depth buffer, zero overdraw in G-buffer) ===
        self.geometry_pass.record_depth_prepass(
            &mut encoder,
            &self.shared,
            &self.gbuffer,
            &self.models,
            &render_list,
        );

        // === Geometry pass (depth Equal, no depth write — only visible fragments shade) ===
        self.geometry_pass.record(
            &mut encoder,
            &self.shared,
            &self.gbuffer,
            &self.models,
            &render_list,
        );

        // === SSAO pass ===
        self.ssao_pass
            .record(&mut encoder, &self.shared, &self.intermediates.ssao_view);

        // === SSAO blur pass ===
        self.ssao_blur_pass
            .record(&mut encoder, &self.shared, &self.intermediates.ssao_blur_view);

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

        // === Lighting + Sky pass (merged — sky fills depth==0 pixels) ===
        self.lighting_pass.record(
            &mut encoder,
            &self.shared,
            &self.intermediates.lighting_base_view,
            &self.shadow_pass.bind_group,
            &self.env_probe_pass.bind_group,
        );

        // === Forward emissive pass (additive onto lighting buffer) ===
        self.geometry_pass.record_emissive(
            &mut encoder,
            &self.shared,
            &self.gbuffer,
            &self.intermediates.lighting_base_view,
            &self.models,
            &render_list,
        );

        // === God rays trace (writes to god_rays_view) ===
        self.god_rays_pass.record_trace(
            &mut encoder,
            &self.shared,
            &self.intermediates.god_rays_view,
            game.camera.view,
            game.camera.projection,
            sun_dir_to_sun,
            sun_diffuse,
        );

        // === God rays composite (additive onto lighting) ===
        self.god_rays_pass.record_composite(
            &mut encoder,
            &self.shared,
            &self.intermediates.lighting_base_view,
        );

        // === Bloom pass (writes to intermediate, not surface) ===
        self.bloom_pass.record(
            &mut encoder,
            &self.shared,
            &self.intermediates,
            &self.intermediates.post_composite_view,
        );

        // === FXAA pass (reads intermediate, writes to surface) ===
        self.fxaa_pass
            .record(&mut encoder, &self.shared, &surface_view);

        // === Cubemap debug overlay ===
        if game.debug_cubemap {
            self.cubemap_debug_pass
                .record(&mut encoder, &self.shared, &surface_view);
        }

        // === Text overlay ===
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
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
            self.text_pass.record(&mut rpass);
        }

        self.shared.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.text_pass.post_submit();

        // Return preallocated buffers
        render_list.clear();
        self.render_list = render_list;
        point_shadow_casters.clear();
        self.point_shadow_casters = point_shadow_casters;
        spot_shadow_casters.clear();
        self.spot_shadow_casters = spot_shadow_casters;
        shadow_assignments.clear();
        self.shadow_assignments = shadow_assignments;
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
