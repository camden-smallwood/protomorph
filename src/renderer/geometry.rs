use crate::{
    models::ModelVertex,
    renderer::{
        color_clear_attach, draw_models,
        shared::{FrameContext, RenderPass, SharedResources},
    },
};

// ===========================================================================
// G-Buffer Pass — writes normals, albedo, material + depth in one pass
// No separate depth prepass — depth is written here with Greater comparison.
// ===========================================================================

pub struct GBufferPass {
    pipeline: wgpu::RenderPipeline,
}

impl GBufferPass {
    pub fn new(shared: &SharedResources) -> Self {
        let shader = shared.device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/geometry.wgsl"
        ));

        let layout = shared.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("geometry_layout"),
            bind_group_layouts: &[&shared.camera_bgl, &shared.model_bgl, &shared.material_bgl, &shared.node_matrices_bgl],
            immediate_size: 0,
        });

        let pipeline = shared.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("geometry_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ModelVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rg16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8UnormSrgb, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rg11b10Ufloat, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
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
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }
}

impl RenderPass for GBufferPass {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let gbuffer = ctx.gbuffer;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("geometry_pass"),
            color_attachments: &[
                Some(color_clear_attach(&gbuffer.normal_view)),
                Some(color_clear_attach(&gbuffer.albedo_specular_view)),
                Some(color_clear_attach(&gbuffer.material_view)),
                Some(color_clear_attach(&gbuffer.emissive_view)),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &ctx.shared.camera_bind_group, &[]);
        draw_models(&mut rpass, ctx.shared, ctx, true);
    }
}
