use crate::renderer::{
    create_fullscreen_pipeline, depth_tex_entry, sampler_entry,
    shared::{FrameContext, GBuffer, IntermediateTargets, RenderPass, SharedResources},
    tex_entry, uniform_entry,
};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rand::Rng;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SsaoParams {
    pub kernel_samples: [[f32; 4]; 32],
    pub strength: f32,
    pub falloff: f32,
    pub radius: f32,
    pub noise_scale_x: f32,
    pub noise_scale_y: f32,
    pub _pad: [f32; 3],
}

pub struct SsaoPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    noise_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl SsaoPass {
    pub fn new(shared: &SharedResources, gbuffer: &GBuffer) -> Self {
        let unfilterable = wgpu::TextureSampleType::Float {
            filterable: false,
        };

        let bgl = shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao_bgl"),
            entries: &[
                tex_entry(0, unfilterable),
                depth_tex_entry(1),
                tex_entry(2, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(3, wgpu::SamplerBindingType::NonFiltering),
                uniform_entry(4, size_of::<SsaoParams>() as u64, wgpu::ShaderStages::FRAGMENT, false),
            ],
        });

        let pipeline = create_fullscreen_pipeline(
            &shared.device,
            wgpu::include_wgsl!("../../assets/shaders/ssao.wgsl"),
            &[&bgl],
            &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            "ssao_pipeline",
        );

        let mut ssao_params = Self::generate_params();
        let ssao_half_w = (shared.config.width / 2).max(1) as f32;
        let ssao_half_h = (shared.config.height / 2).max(1) as f32;
        ssao_params.noise_scale_x = ssao_half_w / 4.0;
        ssao_params.noise_scale_y = ssao_half_h / 4.0;

        let params_buffer = shared.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao_params"),
            contents: bytemuck::bytes_of(&ssao_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let noise_view = Self::create_noise_texture(&shared.device, &shared.queue);

        let bind_group = Self::create_bind_group(
            &shared.device, &bgl, gbuffer, &noise_view, &shared.nearest_sampler, &params_buffer,
        );

        Self { pipeline, bgl, params_buffer, noise_view, bind_group }
    }

    fn generate_params() -> SsaoParams {
        let mut rng = rand::rng();
        let mut kernel = [[0.0f32; 4]; 32];

        for i in 0..32 {
            let mut sample = Vec3::new(
                rng.random::<f32>() * 2.0 - 1.0,
                rng.random::<f32>() * 2.0 - 1.0,
                rng.random::<f32>(),
            )
            .normalize();

            let scale_factor = i as f32 / 32.0;
            let scale = 0.1 + scale_factor * scale_factor * 0.9;
            sample *= scale;

            kernel[i] = [sample.x, sample.y, sample.z, 0.0];
        }

        SsaoParams {
            kernel_samples: kernel,
            strength: 0.025,
            falloff: 0.00005,
            radius: 0.1,
            noise_scale_x: 0.0,
            noise_scale_y: 0.0,
            _pad: [0.0; 3],
        }
    }

    fn create_noise_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
        let mut rng = rand::rng();
        let mut noise_data = [0u8; 4 * 4 * 4];

        for i in 0..16 {
            let x = rng.random::<f32>() * 2.0 - 1.0;
            let y = rng.random::<f32>() * 2.0 - 1.0;
            noise_data[i * 4] = ((x * 0.5 + 0.5) * 255.0) as u8;
            noise_data[i * 4 + 1] = ((y * 0.5 + 0.5) * 255.0) as u8;
            noise_data[i * 4 + 2] = 128;
            noise_data[i * 4 + 3] = 255;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao_noise"),
            size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
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
            &noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_bind_group(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer, noise_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler, params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&gbuffer.depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(noise_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        })
    }
}

impl RenderPass for SsaoPass {
    fn resize(
        &mut self,
        shared: &SharedResources,
        gbuffer: &GBuffer,
        _intermediates: &IntermediateTargets,
    ) {
        let ssao_half_w = (shared.config.width / 2).max(1) as f32;
        let ssao_half_h = (shared.config.height / 2).max(1) as f32;
        let noise_scale_data = [ssao_half_w / 4.0, ssao_half_h / 4.0];
        let noise_scale_offset = std::mem::offset_of!(SsaoParams, noise_scale_x) as u64;
        shared.queue.write_buffer(&self.params_buffer, noise_scale_offset, bytemuck::cast_slice(&noise_scale_data));

        self.bind_group = SsaoPass::create_bind_group(
            &shared.device, &self.bgl, gbuffer, &self.noise_view, &shared.nearest_sampler, &self.params_buffer,
        );
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ssao_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.intermediates.ssao_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, ctx.shared.quad_vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
}

