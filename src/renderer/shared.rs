use crate::{
    animation::MAXIMUM_NUMBER_OF_MODEL_NODES,
    dds::create_fallback_texture,
    gpu_types::*,
    lights::GpuLightingUniforms,
    materials::GpuMaterialProps,
    renderer::{MAX_OBJECTS, helpers::{
        create_1x1_texture, create_screen_texture, sampler_entry, tex_entry, uniform_entry,
    }},
};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// G-Buffer + intermediate render targets
// ---------------------------------------------------------------------------

pub struct GBuffer {
    // NOTE: position_depth MRT was removed — world position is now reconstructed from
    // the depth buffer + inverse_view_projection in shaders that need it. This frees
    // one MRT slot (was Rgba16Float, now reconstructed from depth) that can
    // be used for a velocity buffer or other data in the future.
    pub normal_view: wgpu::TextureView,
    pub albedo_specular_view: wgpu::TextureView,
    pub material_view: wgpu::TextureView,
    pub depth_view: wgpu::TextureView,
}

pub struct IntermediateTargets {
    pub ssao_view: wgpu::TextureView,
    pub ssao_blur_view: wgpu::TextureView,
    pub lighting_base_view: wgpu::TextureView,
    pub god_rays_view: wgpu::TextureView,
    pub bloom_mip_views: Vec<wgpu::TextureView>,
    pub bloom_mip_sizes: Vec<(u32, u32)>,
    pub post_composite_view: wgpu::TextureView,
}

pub struct FallbackTextures {
    pub white_view: wgpu::TextureView,
    pub default_normal_view: wgpu::TextureView,
    pub black_view: wgpu::TextureView,
}

pub fn create_gbuffer(device: &wgpu::Device, w: u32, h: u32) -> GBuffer {
    let normal = create_screen_texture(device, w, h, wgpu::TextureFormat::Rg16Float, "gb_normal");

    let albedo_specular = create_screen_texture(
        device,
        w,
        h,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        "gb_albedo_spec",
    );

    let material = create_screen_texture(device, w, h, wgpu::TextureFormat::Rgba16Float, "gb_material");

    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gb_depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let dv = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());

    GBuffer {
        normal_view: dv(&normal),
        albedo_specular_view: dv(&albedo_specular),
        material_view: dv(&material),
        depth_view: dv(&depth),
    }
}

pub fn create_intermediates(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    surface_format: wgpu::TextureFormat,
) -> IntermediateTargets {
    let ssao = create_screen_texture(
        device,
        (w / 2).max(1),
        (h / 2).max(1),
        wgpu::TextureFormat::Rgba16Float,
        "ssao",
    );

    let ssao_blur = create_screen_texture(
        device,
        (w / 2).max(1),
        (h / 2).max(1),
        wgpu::TextureFormat::Rgba16Float,
        "ssao_blur",
    );

    let lighting_base = create_screen_texture(
        device,
        w,
        h,
        wgpu::TextureFormat::Rg11b10Ufloat,
        "lighting_base",
    );

    let god_rays = create_screen_texture(
        device,
        (w / 2).max(1),
        (h / 2).max(1),
        wgpu::TextureFormat::Rg11b10Ufloat,
        "god_rays",
    );

    let post_composite = create_screen_texture(
        device,
        w,
        h,
        surface_format,
        "post_composite",
    );

    let mut bloom_mip_views = Vec::with_capacity(BLOOM_MIP_COUNT);
    let mut bloom_mip_sizes = Vec::with_capacity(BLOOM_MIP_COUNT);
    let mut mip_w = (w / 2).max(1);
    let mut mip_h = (h / 2).max(1);

    for i in 0..BLOOM_MIP_COUNT {
        let tex = create_screen_texture(
            device,
            mip_w,
            mip_h,
            wgpu::TextureFormat::Rgba16Float,
            &format!("bloom_mip_{i}"),
        );

        bloom_mip_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        bloom_mip_sizes.push((mip_w, mip_h));

        mip_w = (mip_w / 2).max(1);
        mip_h = (mip_h / 2).max(1);
    }

    let dv = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());

    IntermediateTargets {
        ssao_view: dv(&ssao),
        ssao_blur_view: dv(&ssao_blur),
        lighting_base_view: dv(&lighting_base),
        god_rays_view: dv(&god_rays),
        bloom_mip_views,
        bloom_mip_sizes,
        post_composite_view: dv(&post_composite),
    }
}

// ---------------------------------------------------------------------------
// SharedResources
// ---------------------------------------------------------------------------

pub struct SharedResources {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    // Samplers
    pub filtering_sampler: wgpu::Sampler,
    pub bloom_sampler: wgpu::Sampler,
    pub nearest_sampler: wgpu::Sampler,
    pub shadow_comparison_sampler: wgpu::Sampler,

    // Fullscreen quad
    pub quad_vertex_buffer: wgpu::Buffer,

    // Camera
    pub camera_bgl: wgpu::BindGroupLayout,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,

    // Model transforms
    pub model_bgl: wgpu::BindGroupLayout,
    pub model_buffer: wgpu::Buffer,
    pub model_stride: usize,
    pub model_bind_group: wgpu::BindGroup,

    // Node/bone matrices
    pub node_matrices_bgl: wgpu::BindGroupLayout,
    pub node_matrices_buffer: wgpu::Buffer,
    pub node_matrices_stride: usize,
    pub node_matrices_bind_group: wgpu::BindGroup,

    // Lighting
    pub lighting_buffer: wgpu::Buffer,
    pub atmosphere_buffer: wgpu::Buffer,
    pub sky_params_buffer: wgpu::Buffer,
    pub lighting_uniforms_bgl: wgpu::BindGroupLayout,
    pub lighting_uniforms_bind_group: wgpu::BindGroup,

    // Material BGL
    pub material_bgl: wgpu::BindGroupLayout,

    // Fallback textures
    pub fallback_textures: FallbackTextures,
}

impl SharedResources {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
    ) -> Self {
        // --- Samplers ---
        let filtering_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("filtering_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let bloom_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let shadow_comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_comparison_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Fallback textures ---
        let white_tex = create_fallback_texture(&device, &queue);
        let white_view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let default_normal_tex = create_1x1_texture(
            &device,
            &queue,
            [128, 128, 255, 255],
            "default_normal",
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let default_normal_view =
            default_normal_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let black_tex = create_1x1_texture(
            &device,
            &queue,
            [0, 0, 0, 255],
            "black",
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let black_view = black_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let fallback_textures = FallbackTextures {
            white_view,
            default_normal_view,
            black_view,
        };

        // --- Fullscreen quad vertex buffer ---
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad_vb"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // --- Bind group layouts ---
        let camera_bgl = create_camera_bgl(&device);
        let model_bgl = create_model_bgl(&device);
        let material_bgl = create_material_bgl(&device);
        let node_matrices_bgl = create_node_matrices_bgl(&device);
        let lighting_uniforms_bgl = create_lighting_uniforms_bgl(&device);

        // --- Uniform buffers ---
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment as usize;
        let min_storage_alignment = device.limits().min_storage_buffer_offset_alignment as usize;
        let model_size = size_of::<ModelUniforms>();
        let model_stride = ((model_size + min_alignment - 1) / min_alignment) * min_alignment;

        let node_matrix_raw_size = MAXIMUM_NUMBER_OF_MODEL_NODES * size_of::<[[f32; 4]; 4]>();
        let node_matrices_stride = ((node_matrix_raw_size + min_storage_alignment - 1)
            / min_storage_alignment)
            * min_storage_alignment;

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buffer"),
            size: size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model_buffer"),
            size: (MAX_OBJECTS * model_stride) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let node_matrices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node_matrices_buffer"),
            size: (MAX_OBJECTS * node_matrices_stride) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lighting_buffer"),
            size: size_of::<GpuLightingUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atmosphere_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmosphere_buffer"),
            size: size_of::<GpuAtmosphereData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sky_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_params_buffer"),
            size: size_of::<GpuSkyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind groups ---
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model_bg"),
            layout: &model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &model_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(model_size as u64),
                }),
            }],
        });
        let node_matrices_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("node_matrices_bg"),
            layout: &node_matrices_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &node_matrices_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(node_matrix_raw_size as u64),
                }),
            }],
        });
        let lighting_uniforms_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_uniforms_bg"),
            layout: &lighting_uniforms_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lighting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: atmosphere_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sky_params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            config,
            filtering_sampler,
            bloom_sampler,
            nearest_sampler,
            shadow_comparison_sampler,
            quad_vertex_buffer,
            camera_bgl,
            camera_buffer,
            camera_bind_group,
            model_bgl,
            model_buffer,
            model_stride,
            model_bind_group,
            node_matrices_bgl,
            node_matrices_buffer,
            node_matrices_stride,
            node_matrices_bind_group,
            lighting_buffer,
            atmosphere_buffer,
            sky_params_buffer,
            lighting_uniforms_bgl,
            lighting_uniforms_bind_group,
            material_bgl,
            fallback_textures,
        }
    }
}

// ---------------------------------------------------------------------------
// Bind group layout creation functions
// ---------------------------------------------------------------------------

pub fn create_camera_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera_bgl"),
        entries: &[uniform_entry(
            0,
            size_of::<CameraUniforms>() as u64,
            wgpu::ShaderStages::VERTEX,
            false,
        )],
    })
}

pub fn create_model_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("model_bgl"),
        entries: &[uniform_entry(
            0,
            size_of::<ModelUniforms>() as u64,
            wgpu::ShaderStages::VERTEX,
            true,
        )],
    })
}

pub fn create_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let filterable = wgpu::TextureSampleType::Float {
        filterable: true,
    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material_bgl"),
        entries: &[
            tex_entry(0, filterable),
            tex_entry(1, filterable),
            tex_entry(2, filterable),
            tex_entry(3, filterable),
            tex_entry(4, filterable),
            sampler_entry(5, wgpu::SamplerBindingType::Filtering),
            uniform_entry(
                6,
                size_of::<GpuMaterialProps>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
        ],
    })
}

pub fn create_node_matrices_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("node_matrices_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(
                    (MAXIMUM_NUMBER_OF_MODEL_NODES * size_of::<[[f32; 4]; 4]>()) as u64,
                ),
            },
            count: None,
        }],
    })
}

pub fn create_lighting_uniforms_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting_uniforms_bgl"),
        entries: &[
            uniform_entry(
                0,
                size_of::<GpuLightingUniforms>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
            uniform_entry(
                1,
                size_of::<GpuAtmosphereData>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
            uniform_entry(
                2,
                size_of::<GpuSkyParams>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
        ],
    })
}
