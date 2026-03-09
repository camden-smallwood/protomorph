use crate::{
    camera::CameraUniforms,
    lights::LightType,
    models::{VertexRigid, VertexSkinned, VertexType},
    renderer::{
        create_fullscreen_pipeline,
        shared::{FrameContext, QuadVertex, RenderPass, SharedBindGroup, SharedResources},
        uniform_entry,
    },
};
use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::rc::Rc;
use glam::{Mat4, Vec3};

// Each face renders its matching world direction — no swizzle needed at sample time.
const FACE_DIRECTIONS: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::new(0.0, -1.0, 0.0)),
    (Vec3::NEG_X, Vec3::new(0.0, -1.0, 0.0)),
    (Vec3::Y, Vec3::new(0.0, 0.0, 1.0)),
    (Vec3::NEG_Y, Vec3::new(0.0, 0.0, -1.0)),
    (Vec3::NEG_Z, Vec3::new(0.0, -1.0, 0.0)),
    (Vec3::Z, Vec3::new(0.0, -1.0, 0.0)),
];

const UPDATE_INTERVAL: u64 = 1;

pub const ENV_PROBE_SIZE: u32 = 128;
pub const ENV_PROBE_MIP_COUNT: u32 = 6; // 128 -> 4

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuEnvProbeData {
    pub probe_position: [f32; 3],
    pub env_roughness_scale: f32,
    pub env_specular_contribution: f32,
    pub env_mip_count: f32,
    pub env_intensity: f32,
    pub env_diffuse_intensity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuAtmosphereData {
    pub sun_direction: [f32; 3],
    pub atmosphere_enable: f32,
    pub rayleigh_coefficients: [f32; 3],
    pub rayleigh_height_scale: f32,
    pub mie_coefficient: f32,
    pub mie_height_scale: f32,
    pub mie_g: f32,
    pub max_fog_thickness: f32,
    pub inscatter_scale: f32,
    pub reference_height: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSHCoefficients {
    pub coefficients: [[f32; 4]; 9], // 9 L2 basis functions, each (R, G, B, pad)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSkyParams {
    pub inverse_view_projection: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub _pad: f32,
}

#[allow(dead_code)]
pub struct EnvProbePass {
    // Sky background pipeline for cubemap faces
    sky_pipeline: wgpu::RenderPipeline,
    sky_params_buffers: [wgpu::Buffer; 6],
    sky_bind_groups: [wgpu::BindGroup; 6],

    // Forward rendering pipeline (rigid + skinned)
    forward_pipeline: wgpu::RenderPipeline,
    forward_skinned_pipeline: wgpu::RenderPipeline,

    // Mip downsample pipeline
    downsample_pipeline: wgpu::RenderPipeline,
    downsample_bgl: wgpu::BindGroupLayout,
    downsample_bind_groups: Vec<Vec<wgpu::BindGroup>>, // [face][mip-1]

    // Cubemap texture + views
    cubemap_texture: wgpu::Texture,
    cubemap_cube_view: wgpu::TextureView,
    face_mip_views: Vec<Vec<wgpu::TextureView>>, // [face][mip]

    // Depth texture for rendering (reused across faces)
    depth_view: wgpu::TextureView,

    // Camera for 6 faces (dynamic offset)
    camera_bgl: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    camera_stride: usize,
    camera_bind_group: wgpu::BindGroup,

    // Env probe uniform for lighting pass (group 3)
    probe_buffer: wgpu::Buffer,
    sh_buffer: wgpu::Buffer,
    pub bgl: wgpu::BindGroupLayout,
    pub bind_group: SharedBindGroup,

    // Filtering sampler for cubemap
    filtering_sampler: wgpu::Sampler,

    frame_counter: u64,
    probe_data: GpuEnvProbeData,
}

impl EnvProbePass {
    pub fn new(shared: &SharedResources) -> Self {
        let device = &shared.device;

        // --- Camera BGL + buffer (same pattern as shadow_pass) ---
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env_probe_camera_bgl"),
            entries: &[uniform_entry(
                0,
                size_of::<CameraUniforms>() as u64,
                wgpu::ShaderStages::VERTEX,
                true,
            )],
        });

        let min_alignment = device.limits().min_uniform_buffer_offset_alignment as usize;
        let camera_size = size_of::<CameraUniforms>();
        let camera_stride = ((camera_size + min_alignment - 1) / min_alignment) * min_alignment;
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("env_probe_camera_buffer"),
            size: (6 * camera_stride) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("env_probe_camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(camera_size as u64),
                }),
            }],
        });

        // --- Cubemap texture ---
        let cubemap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("env_probe_cubemap"),
            size: wgpu::Extent3d {
                width: ENV_PROBE_SIZE,
                height: ENV_PROBE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: ENV_PROBE_MIP_COUNT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let cubemap_cube_view = cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("env_probe_cube_view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // Per-face per-mip views for rendering
        let mut face_mip_views: Vec<Vec<wgpu::TextureView>> = Vec::with_capacity(6);
        for face in 0..6u32 {
            let mut mip_views = Vec::with_capacity(ENV_PROBE_MIP_COUNT as usize);
            for mip in 0..ENV_PROBE_MIP_COUNT {
                mip_views.push(cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("env_face{face}_mip{mip}")),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                }));
            }
            face_mip_views.push(mip_views);
        }

        // --- Depth texture (reused for each face) ---
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("env_probe_depth"),
            size: wgpu::Extent3d {
                width: ENV_PROBE_SIZE,
                height: ENV_PROBE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Forward pipelines ---
        let forward_pipeline = {
            let shader = device.create_shader_module(wgpu::include_wgsl!(
                "../../assets/shaders/env_probe_forward.wgsl"
            ));

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("env_forward_layout"),
                bind_group_layouts: &[&camera_bgl, &shared.model_bgl, &shared.material_bgl],
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("env_forward_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[VertexRigid::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let forward_skinned_pipeline = {
            let shader = device.create_shader_module(wgpu::include_wgsl!(
                "../../assets/shaders/env_probe_forward_skinned.wgsl"
            ));

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("env_forward_skinned_layout"),
                bind_group_layouts: &[&camera_bgl, &shared.model_bgl, &shared.material_bgl, &shared.node_matrices_bgl],
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("env_forward_skinned_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[VertexSkinned::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // --- Sky background pipeline ---
        let sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env_probe_sky_bgl"),
            entries: &[
                uniform_entry(
                    0,
                    size_of::<GpuSkyParams>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
                uniform_entry(
                    1,
                    size_of::<GpuAtmosphereData>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
            ],
        });

        let sky_pipeline = create_fullscreen_pipeline(
            device,
            wgpu::include_wgsl!("../../assets/shaders/env_probe_sky.wgsl"),
            &[&sky_bgl],
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            "env_probe_sky_pipeline",
        );

        let sky_params_buffers: [wgpu::Buffer; 6] = std::array::from_fn(|face| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("env_probe_sky_params_{face}")),
                size: size_of::<GpuSkyParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let sky_bind_groups: [wgpu::BindGroup; 6] = std::array::from_fn(|face| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("env_probe_sky_bg_{face}")),
                layout: &sky_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sky_params_buffers[face].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: shared.atmosphere_buffer.as_entire_binding(),
                    },
                ],
            })
        });

        // --- Mip downsample ---
        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env_downsample_bgl"),
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
            ],
        });

        let downsample_pipeline = {
            let shader = device.create_shader_module(wgpu::include_wgsl!(
                "../../assets/shaders/env_mip_downsample.wgsl"
            ));

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("env_downsample_layout"),
                bind_group_layouts: &[&downsample_bgl],
                immediate_size: 0,
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("env_downsample_pipeline"),
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

        // Filtering sampler for cubemap mip sampling
        let filtering_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env_filtering_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Create downsample bind groups: for each face, for mip 1..MIP_COUNT,
        // bind the source (face, mip-1) view
        let mut downsample_bind_groups: Vec<Vec<wgpu::BindGroup>> = Vec::with_capacity(6);
        for face in 0..6usize {
            let mut face_bgs = Vec::with_capacity((ENV_PROBE_MIP_COUNT - 1) as usize);
            for mip in 1..ENV_PROBE_MIP_COUNT as usize {
                face_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("env_downsample_f{face}_m{mip}")),
                    layout: &downsample_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &face_mip_views[face][mip - 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&filtering_sampler),
                        },
                    ],
                }));
            }
            downsample_bind_groups.push(face_bgs);
        }

        // --- Probe uniform buffer + BGL for lighting pass group 3 ---
        let probe_data = GpuEnvProbeData {
            probe_position: [0.0, 0.0, 1.0],
            env_roughness_scale: 1.0,
            env_specular_contribution: 0.5,
            env_mip_count: ENV_PROBE_MIP_COUNT as f32,
            env_intensity: 1.0,
            env_diffuse_intensity: 0.3,
        };

        let probe_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("env_probe_buffer"),
            size: size_of::<GpuEnvProbeData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sh_coefficients_buffer"),
            size: size_of::<GpuSHCoefficients>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env_probe_lighting_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                uniform_entry(
                    2,
                    size_of::<GpuEnvProbeData>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
                uniform_entry(
                    3,
                    size_of::<GpuSHCoefficients>() as u64,
                    wgpu::ShaderStages::FRAGMENT,
                    false,
                ),
            ],
        });

        let bind_group = Rc::new(RefCell::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("env_probe_lighting_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cubemap_cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&filtering_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: probe_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sh_buffer.as_entire_binding(),
                },
            ],
        })));

        Self {
            sky_pipeline,
            sky_params_buffers,
            sky_bind_groups,
            forward_pipeline,
            forward_skinned_pipeline,
            downsample_pipeline,
            downsample_bgl,
            downsample_bind_groups,
            cubemap_texture,
            cubemap_cube_view,
            face_mip_views,
            depth_view,
            camera_bgl,
            camera_buffer,
            camera_stride,
            camera_bind_group,
            probe_buffer,
            sh_buffer,
            bgl,
            bind_group,
            filtering_sampler,
            frame_counter: 0,
            probe_data,
        }
    }

    pub fn face_view(&self, face: usize) -> &wgpu::TextureView {
        &self.face_mip_views[face][0]
    }

    pub fn filtering_sampler(&self) -> &wgpu::Sampler {
        &self.filtering_sampler
    }

    fn should_update(&self) -> bool {
        self.frame_counter % UPDATE_INTERVAL == 0
    }

    /// Compute L2 SH coefficients analytically from sun + ambient.
    /// Stores raw SH projection: L * Y_lm(d) — no diffuse transfer baked in.
    /// Transfer coefficients (A_l for diffuse, specular kernels, etc.) are
    /// applied at evaluation time in the shader so the same coefficients can
    /// drive both diffuse irradiance and area specular.
    ///
    /// SH basis normalization constants:
    ///   Y00 = 0.282095 (1/(2√π))
    ///   Y1x = 0.488603 (√3/(2√π))
    ///   Y2x varies per sub-band
    fn compute_analytical_sh(
        sun_dir: Vec3,
        sun_color: Vec3,
        ambient_color: Vec3,
    ) -> GpuSHCoefficients {
        // SH basis constants
        let y00: f32 = 0.282095;   // 1/(2√π)
        let y1x: f32 = 0.488603;   // √3/(2√π)
        let y20: f32 = 0.315392;   // √5/(4√π) * (1/2)
        let y21: f32 = 1.092548;   // √15/(2√π)
        let y22: f32 = 0.546274;   // √15/(4√π)

        // Directional light SH projection (raw, no transfer coefficients)
        let d = sun_dir.normalize();
        let mut coeffs = [[0.0f32; 4]; 9];

        // L0 band: ambient + directional
        let l0 = (sun_color + ambient_color) * y00;
        coeffs[0] = [l0.x, l0.y, l0.z, 0.0];

        // L1 band: Y1,-1 = y, Y1,0 = z, Y1,1 = x
        let l1_m1 = sun_color * y1x * d.y;
        let l1_0 = sun_color * y1x * d.z;
        let l1_1 = sun_color * y1x * d.x;
        coeffs[1] = [l1_m1.x, l1_m1.y, l1_m1.z, 0.0];
        coeffs[2] = [l1_0.x, l1_0.y, l1_0.z, 0.0];
        coeffs[3] = [l1_1.x, l1_1.y, l1_1.z, 0.0];

        // L2 band
        let l2_m2 = sun_color * y21 * d.x * d.y;
        let l2_m1 = sun_color * y21 * d.y * d.z;
        let l2_0 = sun_color * y20 * (3.0 * d.z * d.z - 1.0);
        let l2_1 = sun_color * y21 * d.x * d.z;
        let l2_2 = sun_color * y22 * (d.x * d.x - d.y * d.y);
        coeffs[4] = [l2_m2.x, l2_m2.y, l2_m2.z, 0.0];
        coeffs[5] = [l2_m1.x, l2_m1.y, l2_m1.z, 0.0];
        coeffs[6] = [l2_0.x, l2_0.y, l2_0.z, 0.0];
        coeffs[7] = [l2_1.x, l2_1.y, l2_1.z, 0.0];
        coeffs[8] = [l2_2.x, l2_2.y, l2_2.z, 0.0];

        GpuSHCoefficients { coefficients: coeffs }
    }
}

impl RenderPass for EnvProbePass {
    fn prepare(&mut self, ctx: &FrameContext) {
        let shared = ctx.shared;
        let game = ctx.game;

        // Extract sun parameters from lights
        let (sun_direction, sun_color, ambient_color) = {
            let mut sd = Vec3::new(0.0, 0.0, 1.0);
            let mut sc = Vec3::ZERO;
            let mut sa = Vec3::ZERO;
            for (_idx, light) in game.lights.iter() {
                if !light.hidden && light.light_type == LightType::Directional {
                    sd = -light.direction.normalize(); // direction TO the sun
                    sc = light.diffuse_color;
                    sa = light.ambient_color;
                    break;
                }
            }
            (sd, sc, sa)
        };

        let camera_position = game.camera.position;

        // Scale env probe contribution by actual sun presence
        let sun_lum = sun_color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        if sun_lum > 0.001 {
            self.probe_data.env_intensity = 1.0;
            self.probe_data.env_specular_contribution = 0.5;
            self.probe_data.env_diffuse_intensity = 0.3;
        } else {
            self.probe_data.env_intensity = 0.0;
            self.probe_data.env_specular_contribution = 0.0;
            self.probe_data.env_diffuse_intensity = 0.0;
        }

        // Upload probe data
        self.probe_data.probe_position = camera_position.into();
        shared.queue.write_buffer(
            &self.probe_buffer,
            0,
            bytemuck::bytes_of(&self.probe_data),
        );

        // Upload SH coefficients
        let sh = Self::compute_analytical_sh(sun_direction, sun_color, ambient_color);
        shared.queue.write_buffer(
            &self.sh_buffer,
            0,
            bytemuck::bytes_of(&sh),
        );

        self.frame_counter += 1;

        // Upload face cameras if we're updating this frame
        if self.should_update() && !game.debug_cubemap_colors {
            let probe_pos = camera_position;
            let mut proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
            proj.y_axis.y *= -1.0;

            for (face, (dir, up)) in FACE_DIRECTIONS.iter().enumerate() {
                let view = Mat4::look_at_rh(probe_pos, probe_pos + *dir, *up);
                let cam = CameraUniforms {
                    view: view.to_cols_array_2d(),
                    projection: proj.to_cols_array_2d(),
                };
                let offset = (face * self.camera_stride) as u64;
                shared
                    .queue
                    .write_buffer(&self.camera_buffer, offset, bytemuck::bytes_of(&cam));
            }

            for face in 0..6usize {
                let (dir, up) = FACE_DIRECTIONS[face];
                let view = Mat4::look_at_rh(probe_pos, probe_pos + dir, up);
                let inverse_vp = (proj * view).inverse();
                let sky_params = GpuSkyParams {
                    inverse_view_projection: inverse_vp.to_cols_array_2d(),
                    camera_position: probe_pos.into(),
                    _pad: 0.0,
                };
                shared.queue.write_buffer(
                    &self.sky_params_buffers[face],
                    0,
                    bytemuck::bytes_of(&sky_params),
                );
            }
        }
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        if !self.should_update() {
            return;
        }

        let shared = ctx.shared;
        let models = ctx.models;
        let render_list = ctx.render_list;
        let game = ctx.game;

        // Debug: solid color per face to verify cubemap orientation
        const DEBUG_COLORS: [wgpu::Color; 6] = [
            wgpu::Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, // +X = red
            wgpu::Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 }, // -X = cyan
            wgpu::Color { r: 0.0, g: 0.5, b: 0.0, a: 1.0 }, // +Y = dark green
            wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }, // -Y = magenta
            wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }, // +Z = blue
            wgpu::Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 }, // -Z = yellow
        ];

        if game.debug_cubemap_colors {
            for face in 0..6usize {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("env_probe_debug_color"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.face_mip_views[face][0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(DEBUG_COLORS[face]),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }
        } else {
            // Render 6 faces at mip 0: sky background then geometry on top
            for face in 0..6usize {
                // Sky pass: clear to black then draw fullscreen sky quad
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("env_probe_sky"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.face_mip_views[face][0],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    rpass.set_pipeline(&self.sky_pipeline);
                    rpass.set_bind_group(0, &self.sky_bind_groups[face], &[]);
                    rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
                    rpass.draw(0..6, 0..1);
                }

                // Geometry pass: load sky background, clear depth
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("env_probe_face"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.face_mip_views[face][0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                let camera_dynamic_offset = (face * self.camera_stride) as u32;
                rpass.set_bind_group(0, &self.camera_bind_group, &[camera_dynamic_offset]);

                // Draw models
                for (obj_slot, &(_obj_idx, model_idx)) in render_list.iter().enumerate() {
                    let model_dynamic_offset = (obj_slot * shared.model_stride) as u32;
                    let nm_dynamic_offset = (obj_slot * shared.node_matrices_stride) as u32;
                    let gpu_model = &models[model_idx];

                    for mesh in &gpu_model.meshes {
                        match mesh.vertex_type {
                            VertexType::Rigid => {
                                rpass.set_pipeline(&self.forward_pipeline);
                                rpass.set_bind_group(
                                    1,
                                    &shared.model_bind_group,
                                    &[model_dynamic_offset],
                                );
                            }
                            VertexType::Skinned => {
                                rpass.set_pipeline(&self.forward_skinned_pipeline);
                                rpass.set_bind_group(
                                    1,
                                    &shared.model_bind_group,
                                    &[model_dynamic_offset],
                                );
                                rpass.set_bind_group(
                                    3,
                                    &shared.node_matrices_bind_group,
                                    &[nm_dynamic_offset],
                                );
                            }
                        }

                        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        rpass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );

                        for part in &mesh.parts {
                            if let Some(material) = gpu_model.materials.get(part.material_index) {
                                rpass.set_bind_group(2, &material.bind_group, &[]);
                            }
                            rpass.draw_indexed(
                                part.index_start..part.index_start + part.index_count,
                                0,
                                0..1,
                            );
                        }
                    }
                }
            }
        }

        // Mip downsample chain
        for face in 0..6usize {
            for mip in 1..ENV_PROBE_MIP_COUNT as usize {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("env_downsample"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.face_mip_views[face][mip],
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

                rpass.set_pipeline(&self.downsample_pipeline);
                rpass.set_bind_group(0, &self.downsample_bind_groups[face][mip - 1], &[]);
                rpass.set_vertex_buffer(0, shared.quad_vertex_buffer.slice(..));
                rpass.draw(0..6, 0..1);
            }
        }
    }
}

