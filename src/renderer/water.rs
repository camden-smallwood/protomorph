use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{
    dds::{create_dds_texture, load_dds_from_file},
    lights::LightType,
    renderer::{
        depth_tex_entry, sampler_entry, tex_entry, uniform_entry,
        shared::{
            FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedBindGroup,
            SharedResources,
        },
    },
};

// ---------------------------------------------------------------------------
// Note on read-only depth attachment
// ---------------------------------------------------------------------------
// The water pass uses a read-only depth-stencil attachment (depth_ops: None)
// so the live depth buffer can be sampled in the fragment shader without a copy.
// Requirements: depth_write_enabled=false, depth_ops=None, texture has TEXTURE_BINDING.
// Supported on Vulkan, Metal, DX12 (not GLES/WebGL2).
// Trade-off: water surface depth is not written, but no subsequent passes need it.

// ---------------------------------------------------------------------------
// Water grid vertex
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaterVertex {
    position: [f32; 3],
}

impl WaterVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<WaterVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// ---------------------------------------------------------------------------
// Water uniforms (uploaded each frame)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaterUniforms {
    inverse_view_projection: [[f32; 4]; 4],
    camera_position: [f32; 3],
    time: f32,
    water_height: f32,
    absorption_depth: f32,
    edge_softness: f32,
    specular_roughness: f32,
    deep_water_color: [f32; 3],
    refraction_strength: f32,
    sun_direction: [f32; 3],
    sun_intensity: f32,
    sun_color: [f32; 3],
    _pad: f32,
}

// ---------------------------------------------------------------------------
// Grid mesh generation
// ---------------------------------------------------------------------------

const GRID_SIZE: u32 = 256;
const GRID_EXTENT: f32 = 40.0; // half-extent, so 80x80 pre-scale, 40x40 post 0.5 scale

fn generate_grid() -> (Vec<WaterVertex>, Vec<u32>) {
    let vertex_count = (GRID_SIZE + 1) * (GRID_SIZE + 1);
    let mut vertices = Vec::with_capacity(vertex_count as usize);

    for y in 0..=GRID_SIZE {
        for x in 0..=GRID_SIZE {
            let fx = (x as f32 / GRID_SIZE as f32) * 2.0 - 1.0;
            let fy = (y as f32 / GRID_SIZE as f32) * 2.0 - 1.0;
            vertices.push(WaterVertex {
                position: [fx * GRID_EXTENT, fy * GRID_EXTENT, 0.0],
            });
        }
    }

    let quad_count = GRID_SIZE * GRID_SIZE;
    let mut indices = Vec::with_capacity((quad_count * 6) as usize);

    for y in 0..GRID_SIZE {
        for x in 0..GRID_SIZE {
            let stride = GRID_SIZE + 1;
            let tl = y * stride + x;
            let tr = tl + 1;
            let bl = (y + 1) * stride + x;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, tr, bl, bl, tr, br]);
        }
    }

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// WaterPass
// ---------------------------------------------------------------------------

pub struct WaterPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    scene_bgl: wgpu::BindGroupLayout,
    scene_bind_group: wgpu::BindGroup,
    shadow_bind_group: SharedBindGroup,
    cubemap_view: wgpu::TextureView,
    bump0_view: wgpu::TextureView,
    bump1_view: wgpu::TextureView,
}

impl WaterPass {
    pub fn new(
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
        cubemap_view: wgpu::TextureView,
        shadow_bgl: &wgpu::BindGroupLayout,
        shadow_bind_group: SharedBindGroup,
    ) -> Self {
        let device = &shared.device;

        // --- Grid mesh ---
        let (vertices, indices) = generate_grid();
        let index_count = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // --- Water normal map textures ---
        let dv = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());

        let bump0_dds = load_dds_from_file("assets/textures/water/bump0.dds");
        let bump0_view = dv(&create_dds_texture(device, &shared.queue, &bump0_dds, true));

        let bump1_dds = load_dds_from_file("assets/textures/water/bump1.dds");
        let bump1_view = dv(&create_dds_texture(device, &shared.queue, &bump1_dds, true));

        // --- Uniform buffer ---
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water_uniform_buffer"),
            size: size_of::<WaterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water_uniform_bgl"),
                entries: &[uniform_entry(
                    0,
                    size_of::<WaterUniforms>() as u64,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    false,
                )],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_uniform_bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // --- Scene textures bind group (fully procedural — no water textures) ---
        let filterable = wgpu::TextureSampleType::Float { filterable: true };

        let scene_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water_scene_bgl"),
                entries: &[
                    depth_tex_entry(0),
                    tex_entry(1, filterable),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    sampler_entry(3, wgpu::SamplerBindingType::NonFiltering),
                    sampler_entry(4, wgpu::SamplerBindingType::Filtering),
                    tex_entry(5, filterable),
                    tex_entry(6, filterable),
                ],
            });

        let scene_bind_group = Self::create_scene_bind_group(
            device,
            &scene_bgl,
            &gbuffer.depth_view,
            intermediates,
            &cubemap_view,
            &bump0_view,
            &bump1_view,
            &shared.nearest_sampler,
            &shared.filtering_sampler,
        );

        // --- Pipeline ---
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/water.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water_pipeline_layout"),
            bind_group_layouts: &[&shared.camera_bgl, &uniform_bgl, &scene_bgl, shadow_bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[WaterVertex::layout()],
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
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_buffer,
            uniform_bind_group,
            scene_bgl,
            scene_bind_group,
            shadow_bind_group,
            cubemap_view,
            bump0_view,
            bump1_view,
        }
    }

    fn create_scene_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        depth_view: &wgpu::TextureView,
        intermediates: &IntermediateTargets,
        cubemap_view: &wgpu::TextureView,
        bump0_view: &wgpu::TextureView,
        bump1_view: &wgpu::TextureView,
        nearest_sampler: &wgpu::Sampler,
        filtering_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_scene_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &intermediates.water_copy_view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(cubemap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(bump0_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(bump1_view),
                },
            ],
        })
    }
}

impl RenderPass for WaterPass {
    fn prepare(&mut self, ctx: &FrameContext) {
        let game = ctx.game;
        let inverse_vp = (game.camera.projection * game.camera.view).inverse();

        // Find directional light (sun) for specular
        let mut sun_dir = [0.0f32; 3];
        let mut sun_intensity = 0.0f32;
        let mut sun_color = [1.0f32; 3];
        for (_idx, light) in game.lights.iter() {
            if light.light_type == LightType::Directional {
                sun_dir = light.direction.into();
                sun_intensity = light.diffuse_color.length();
                let c = light.diffuse_color.normalize_or_zero();
                sun_color = c.into();
                break;
            }
        }

        let uniforms = WaterUniforms {
            inverse_view_projection: inverse_vp.to_cols_array_2d(),
            camera_position: game.camera.position.into(),
            time: game.total_time,
            water_height: 0.5,
            absorption_depth: 2.0,
            edge_softness: 0.15,
            specular_roughness: 0.08,
            deep_water_color: [0.01, 0.1, 0.15],
            refraction_strength: 0.06,
            sun_direction: sun_dir,
            sun_intensity,
            sun_color,
            _pad: 0.0,
        };

        ctx.shared.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let size = wgpu::Extent3d {
            width: ctx.shared.config.width,
            height: ctx.shared.config.height,
            depth_or_array_layers: 1,
        };

        // Copy lighting buffer → water_copy for refraction sampling
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ctx.intermediates.lighting_base_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &ctx.intermediates.water_copy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );

        let shadow_bg = self.shadow_bind_group.borrow();

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("water_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.intermediates.lighting_base_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.gbuffer.depth_view,
                depth_ops: None, // Read-only: enables sampling depth in fragment shader
                stencil_ops: None,
            }),
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &ctx.shared.camera_bind_group, &[]);
        rpass.set_bind_group(1, &self.uniform_bind_group, &[]);
        rpass.set_bind_group(2, &self.scene_bind_group, &[]);
        rpass.set_bind_group(3, &*shadow_bg, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        self.scene_bind_group = Self::create_scene_bind_group(
            &shared.device,
            &self.scene_bgl,
            &gbuffer.depth_view,
            intermediates,
            &self.cubemap_view,
            &self.bump0_view,
            &self.bump1_view,
            &shared.nearest_sampler,
            &shared.filtering_sampler,
        );
    }
}
