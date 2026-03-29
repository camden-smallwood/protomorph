use crate::renderer::shared::SharedResources;

const NOISE_SIZE: u32 = 128;
const WEATHER_SIZE: u32 = 512;

const DETAIL_SIZE: u32 = 32;

pub struct CloudTextures {
    pub noise_texture: wgpu::Texture,     // 128^3 Perlin-Worley base shape
    pub detail_texture: wgpu::Texture,    // 32^3 high-freq Worley for edge erosion
    pub weather_texture: wgpu::Texture,   // 512x512 coverage/type map
    pub blue_noise_texture: wgpu::Texture,
}

pub fn generate(shared: &SharedResources) -> CloudTextures {
    let device = &shared.device;
    let queue = &shared.queue;

    // --- 3D noise texture (128^3) via compute shader ---
    let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cloud_noise_3d"),
        size: wgpu::Extent3d {
            width: NOISE_SIZE,
            height: NOISE_SIZE,
            depth_or_array_layers: NOISE_SIZE,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let noise_view = noise_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = device.create_shader_module(wgpu::include_wgsl!("../../assets/shaders/cloud_noise.wgsl"));

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cloud_noise_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cloud_noise_pipeline_layout"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cloud_noise_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cloud_noise_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&noise_view),
        }],
    });

    // --- 32^3 detail noise texture ---
    let detail_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cloud_detail_3d"),
        size: wgpu::Extent3d {
            width: DETAIL_SIZE, height: DETAIL_SIZE, depth_or_array_layers: DETAIL_SIZE,
        },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let detail_view = detail_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Detail BGL + bind group (bindings 0 = base output, 1 = detail output)
    let detail_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cloud_detail_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        }],
    });

    let detail_shader = device.create_shader_module(wgpu::include_wgsl!(
        "../../assets/shaders/cloud_noise.wgsl"
    ));

    let detail_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cloud_detail_pipeline_layout"),
        bind_group_layouts: &[&detail_bgl],
        immediate_size: 0,
    });

    let detail_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cloud_detail_pipeline"),
        layout: Some(&detail_pipeline_layout),
        module: &detail_shader,
        entry_point: Some("cs_detail"),
        compilation_options: Default::default(),
        cache: None,
    });

    let detail_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cloud_detail_bg"),
        layout: &detail_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&detail_view),
        }],
    });

    // --- Dispatch both compute passes ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cloud_noise_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloud_noise_compute"),
            ..Default::default()
        });
        // Base noise: 128^3, workgroup 4^3
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(NOISE_SIZE / 4, NOISE_SIZE / 4, NOISE_SIZE / 4);

        // Detail noise: 32^3, workgroup 4^3
        cpass.set_pipeline(&detail_pipeline);
        cpass.set_bind_group(0, &detail_bind_group, &[]);
        cpass.dispatch_workgroups(DETAIL_SIZE / 4, DETAIL_SIZE / 4, DETAIL_SIZE / 4);
    }

    queue.submit(std::iter::once(encoder.finish()));

    // --- 2D weather map (512^2) via CPU FBM ---
    let weather_texture = generate_weather_map(device, queue);
    let blue_noise_texture = generate_blue_noise(device, queue);

    CloudTextures {
        noise_texture,
        detail_texture,
        weather_texture,
        blue_noise_texture,
    }
}

// --- CPU-side weather map generation ---

fn generate_weather_map(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let size = WEATHER_SIZE as usize;
    let mut data = vec![0u8; size * size * 4];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            // Multi-octave FBM for coverage with large-scale variation
            let large_scale = fbm2(u, v, 3, 2.0, 0.6);
            let detail = fbm2(u, v, 6, 5.0, 0.5);
            // Blend: large patches with detail breakup
            let coverage = (large_scale * 0.6 + detail * 0.4).clamp(0.0, 1.0);

            // Cloud type: smooth variation
            let cloud_type = fbm2(u + 73.7, v + 91.3, 3, 2.0, 0.5);

            let idx = (y * size + x) * 4;
            data[idx] = (coverage * 255.0) as u8;
            data[idx + 1] = (cloud_type.clamp(0.0, 1.0) * 255.0) as u8;
            data[idx + 2] = 0;
            data[idx + 3] = 255;
        }
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cloud_weather_map"),
        size: wgpu::Extent3d {
            width: WEATHER_SIZE,
            height: WEATHER_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(WEATHER_SIZE * 4),
            rows_per_image: Some(WEATHER_SIZE),
        },
        wgpu::Extent3d {
            width: WEATHER_SIZE,
            height: WEATHER_SIZE,
            depth_or_array_layers: 1,
        },
    );

    texture
}

// Simple 2D value noise + FBM for CPU weather map generation

fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn hash2d(ix: i32, iy: i32) -> f32 {
    let h = pcg_hash((ix as u32).wrapping_add((iy as u32).wrapping_mul(1013)));
    h as f32 / u32::MAX as f32
}

fn value_noise_2d(x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();

    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);

    let a = hash2d(ix, iy);
    let b = hash2d(ix + 1, iy);
    let c = hash2d(ix, iy + 1);
    let d = hash2d(ix + 1, iy + 1);

    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uy
}

fn fbm2(x: f32, y: f32, octaves: u32, base_freq: f32, persistence: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = base_freq;
    let mut max_val = 0.0f32;

    for _ in 0..octaves {
        value += value_noise_2d(x * frequency, y * frequency) * amplitude;
        max_val += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    value / max_val
}

// --- Blue noise (white noise with R2 sequence for good distribution) ---

fn generate_blue_noise(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let size = 256usize;
    let mut data = vec![0u8; size * size];

    // Use interleaved gradient noise pattern (Jorge Jimenez)
    // Better than pure white noise for dithering
    for y in 0..size {
        for x in 0..size {
            let val = interleaved_gradient_noise(x as f32, y as f32);
            data[y * size + x] = (val * 255.0) as u8;
        }
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cloud_blue_noise"),
        size: wgpu::Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size as u32),
            rows_per_image: Some(size as u32),
        },
        wgpu::Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: 1,
        },
    );

    texture
}

fn interleaved_gradient_noise(x: f32, y: f32) -> f32 {
    let val = 52.9829189 * ((0.06711056 * x + 0.00583715 * y).fract());
    val.fract()
}
