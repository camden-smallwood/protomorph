use crate::{
    lights::LightType,
    renderer::{
        cloud_noise::CloudTextures,
        shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
    },
};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CloudParams {
    // --- Per-frame (offsets 0-159) ---
    inverse_view_projection: [[f32; 4]; 4], // 0
    camera_position: [f32; 3],               // 64
    time: f32,                               // 76
    sun_direction: [f32; 3],                 // 80 (vec3 align 16)
    sun_intensity: f32,                      // 92
    sun_color: [f32; 3],                     // 96 (vec3 align 16)
    cloud_bottom: f32,                       // 108
    cloud_top: f32,                          // 112
    coverage: f32,                           // 116
    wind_x: f32,                             // 120
    wind_z: f32,                             // 124
    wind_speed: f32,                         // 128
    base_noise_scale: f32,                   // 132
    extinction_coeff: f32,                   // 136
    frame_index: u32,                        // 140
    quarter_w: u32,                          // 144
    quarter_h: u32,                          // 148
    _pad: [u32; 2],                          // 152

    // --- Atmosphere (offsets 160-191) ---
    rayleigh_coefficients: [f32; 3],         // 160 (vec3 align 16)
    mie_coefficient: f32,                    // 172
    mie_g: f32,                              // 176
    inscatter_scale: f32,                    // 180
    reference_height: f32,                   // 184
    sun_air_mass_scale: f32,                 // 188

    // --- Cloud lighting scalars (offsets 192-223) ---
    rayleigh_height_scale: f32,              // 192
    mie_height_scale: f32,                   // 196
    cloud_phase_g_forward: f32,              // 200
    cloud_phase_g_back: f32,                 // 204
    cloud_phase_blend: f32,                  // 208
    cloud_optical_depth_scale: f32,          // 212
    cloud_light_sample_dist: f32,            // 216
    _pad3: f32,                              // 220

    // --- Cloud colors (offsets 224-335, vec3s at 16-byte aligned offsets) ---
    cloud_albedo: [f32; 3],                  // 224 (vec3 align 16)
    _pad4: f32,                              // 236
    cloud_sky_ambient_day: [f32; 3],         // 240 (vec3 align 16)
    _pad5: f32,                              // 252
    cloud_sky_ambient_sunset: [f32; 3],      // 256 (vec3 align 16)
    _pad6: f32,                              // 268
    cloud_ground_ambient_day: [f32; 3],      // 272 (vec3 align 16)
    _pad7: f32,                              // 284
    cloud_ground_ambient_sunset: [f32; 3],   // 288 (vec3 align 16)
    _pad8: f32,                              // 300
    cloud_bg_day: [f32; 3],                  // 304 (vec3 align 16)
    _pad9: f32,                              // 316
    cloud_bg_sunset: [f32; 3],               // 320 (vec3 align 16)
    _pad10: f32,                             // 332
    // Total: 336 bytes (336%16=0)
}

pub struct CloudRaymarchPass {
    compute_pipeline: wgpu::ComputePipeline,
    input_bgl: wgpu::BindGroupLayout,
    output_bgl: wgpu::BindGroupLayout,
    input_bind_group: wgpu::BindGroup,
    output_bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    // Keep noise textures alive + for resize rebuild
    noise_texture: wgpu::Texture,
    detail_texture: wgpu::Texture,
    weather_texture: wgpu::Texture,
    blue_noise_texture: wgpu::Texture,
}

impl CloudRaymarchPass {
    pub fn new(
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
        cloud_textures: CloudTextures,
    ) -> Self {
        let device = &shared.device;

        // Input bind group layout: uniforms + textures
        let input_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud_raymarch_input_bgl"),
            entries: &[
                // 0: CloudParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<CloudParams>() as u64,
                        ),
                    },
                    count: None,
                },
                // 1: 3D noise texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: weather map
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: blue noise
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: filtering sampler (repeat)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 5: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 6: nearest sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 7: 3D detail noise texture (32^3)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Output bind group layout: storage texture
        let output_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud_raymarch_output_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        // Zeroed initial buffer — prepare() fills everything before first dispatch
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud_params"),
            contents: &vec![0u8; size_of::<CloudParams>()],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let CloudTextures {
            noise_texture,
            detail_texture,
            weather_texture,
            blue_noise_texture,
        } = cloud_textures;

        let noise_view = noise_texture.create_view(&Default::default());
        let detail_view = detail_texture.create_view(&Default::default());
        let weather_view = weather_texture.create_view(&Default::default());
        let blue_noise_view = blue_noise_texture.create_view(&Default::default());

        let input_bind_group = Self::create_input_bind_group(
            device,
            &input_bgl,
            &params_buffer,
            &noise_view,
            &detail_view,
            &weather_view,
            &blue_noise_view,
            &gbuffer.depth_view,
            &shared.filtering_sampler,
            &shared.nearest_sampler,
        );

        let output_bind_group = Self::create_output_bind_group(
            device,
            &output_bgl,
            &intermediates.cloud_raymarch_view,
        );

        let shader = device
            .create_shader_module(wgpu::include_wgsl!("../../assets/shaders/cloud_raymarch.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud_raymarch_pipeline_layout"),
            bind_group_layouts: &[&input_bgl, &output_bgl],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cloud_raymarch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            compute_pipeline,
            input_bgl,
            output_bgl,
            input_bind_group,
            output_bind_group,
            params_buffer,
            noise_texture,
            detail_texture,
            weather_texture,
            blue_noise_texture,
        }
    }

    fn create_input_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params_buffer: &wgpu::Buffer,
        noise_view: &wgpu::TextureView,
        detail_view: &wgpu::TextureView,
        weather_view: &wgpu::TextureView,
        blue_noise_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        filtering_sampler: &wgpu::Sampler,
        nearest_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud_raymarch_input_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(noise_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(weather_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(blue_noise_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(filtering_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(nearest_sampler) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(detail_view) },
            ],
        })
    }

    fn create_output_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud_raymarch_output_bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output_view),
            }],
        })
    }
}

impl RenderPass for CloudRaymarchPass {
    fn prepare(&mut self, ctx: &FrameContext) {
        let game = ctx.game;
        let vp = game.camera.projection * game.camera.view;
        let inverse_vp = vp.inverse();

        // Find directional light (sun)
        let (sun_dir, sun_color) = {
            let mut sd = Vec3::new(0.0, 0.0, 1.0);
            let mut sc = Vec3::ONE;
            for (_idx, light) in game.lights.iter() {
                if !light.hidden && light.light_type == LightType::Directional {
                    sd = -light.direction.normalize();
                    sc = light.diffuse_color;
                    break;
                }
            }
            (sd, sc)
        };

        let (qw, qh) = ctx.intermediates.cloud_quarter_size;

        let params = CloudParams {
            inverse_view_projection: inverse_vp.to_cols_array_2d(),
            camera_position: game.camera.position.into(),
            time: game.total_time,
            sun_direction: sun_dir.into(),
            sun_intensity: game.sky.cloud_sun_intensity,
            sun_color: sun_color.into(),
            cloud_bottom: game.sky.cloud_bottom,
            cloud_top: game.sky.cloud_top,
            coverage: game.sky.cloud_coverage,
            wind_x: game.sky.cloud_wind_x,
            wind_z: game.sky.cloud_wind_z,
            wind_speed: game.sky.cloud_wind_speed,
            base_noise_scale: game.sky.cloud_noise_scale,
            extinction_coeff: game.sky.cloud_extinction,
            frame_index: ctx.frame_index,
            quarter_w: qw,
            quarter_h: qh,
            _pad: [0; 2],
            rayleigh_coefficients: game.sky.rayleigh_coefficients,
            mie_coefficient: game.sky.mie_coefficient,
            mie_g: game.sky.mie_g,
            inscatter_scale: game.sky.inscatter_scale,
            reference_height: game.sky.reference_height,
            sun_air_mass_scale: game.sky.sun_air_mass_scale,
            rayleigh_height_scale: game.sky.rayleigh_height_scale,
            mie_height_scale: game.sky.mie_height_scale,
            cloud_phase_g_forward: game.sky.cloud_phase_g_forward,
            cloud_phase_g_back: game.sky.cloud_phase_g_back,
            cloud_phase_blend: game.sky.cloud_phase_blend,
            cloud_optical_depth_scale: game.sky.cloud_optical_depth_scale,
            cloud_light_sample_dist: game.sky.cloud_light_sample_dist,
            _pad3: 0.0,
            cloud_albedo: game.sky.cloud_albedo,
            _pad4: 0.0,
            cloud_sky_ambient_day: game.sky.cloud_sky_ambient_day,
            _pad5: 0.0,
            cloud_sky_ambient_sunset: game.sky.cloud_sky_ambient_sunset,
            _pad6: 0.0,
            cloud_ground_ambient_day: game.sky.cloud_ground_ambient_day,
            _pad7: 0.0,
            cloud_ground_ambient_sunset: game.sky.cloud_ground_ambient_sunset,
            _pad8: 0.0,
            cloud_bg_day: game.sky.cloud_bg_day,
            _pad9: 0.0,
            cloud_bg_sunset: game.sky.cloud_bg_sunset,
            _pad10: 0.0,
        };

        ctx.shared
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let (qw, qh) = ctx.intermediates.cloud_quarter_size;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloud_raymarch"),
            ..Default::default()
        });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.input_bind_group, &[]);
        cpass.set_bind_group(1, &self.output_bind_group, &[]);
        cpass.dispatch_workgroups(
            (qw + 7) / 8,
            (qh + 7) / 8,
            1,
        );
    }

    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        intermediates: &IntermediateTargets,
    ) {
        let noise_view = self.noise_texture.create_view(&Default::default());
        let detail_view = self.detail_texture.create_view(&Default::default());
        let weather_view = self.weather_texture.create_view(&Default::default());
        let blue_noise_view = self.blue_noise_texture.create_view(&Default::default());

        self.input_bind_group = Self::create_input_bind_group(
            &shared.device,
            &self.input_bgl,
            &self.params_buffer,
            &noise_view,
            &detail_view,
            &weather_view,
            &blue_noise_view,
            &gbuffer.depth_view,
            &shared.filtering_sampler,
            &shared.nearest_sampler,
        );

        self.output_bind_group = Self::create_output_bind_group(
            &shared.device,
            &self.output_bgl,
            &intermediates.cloud_raymarch_view,
        );
    }
}
