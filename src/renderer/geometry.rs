use crate::{
    models::{VertexRigid, VertexSkinned, VertexType},
    renderer::{
        color_clear_attach,
        shared::{FrameContext, RenderPass, SharedResources},
    },
};

// ===========================================================================
// Shared draw loop — all three passes iterate models the same way
// ===========================================================================

fn draw_models<'a>(
    rpass: &mut wgpu::RenderPass<'a>,
    shared: &'a SharedResources,
    rigid_pipeline: &'a wgpu::RenderPipeline,
    skinned_pipeline: &'a wgpu::RenderPipeline,
    ctx: &'a FrameContext<'a>,
) {
    rpass.set_bind_group(0, &shared.camera_bind_group, &[]);

    for (obj_slot, &(_obj_idx, model_idx)) in ctx.render_list.iter().enumerate() {
        let model_dynamic_offset = (obj_slot * shared.model_stride) as u32;
        let nm_dynamic_offset = (obj_slot * shared.node_matrices_stride) as u32;

        let gpu_model = &ctx.models[model_idx];

        for mesh in &gpu_model.meshes {
            match mesh.vertex_type {
                VertexType::Rigid => {
                    rpass.set_pipeline(rigid_pipeline);
                    rpass.set_bind_group(1, &shared.model_bind_group, &[model_dynamic_offset]);
                }
                VertexType::Skinned => {
                    rpass.set_pipeline(skinned_pipeline);
                    rpass.set_bind_group(1, &shared.model_bind_group, &[model_dynamic_offset]);
                    rpass.set_bind_group(
                        3,
                        &shared.node_matrices_bind_group,
                        &[nm_dynamic_offset],
                    );
                }
            }

            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

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

// ===========================================================================
// Depth Pre-pass
// ===========================================================================

pub struct DepthPrepass {
    pipeline: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,
}

impl DepthPrepass {
    pub fn new(shared: &SharedResources) -> Self {
        Self {
            pipeline: Self::create_pipeline(
                &shared.device,
                &shared.camera_bgl,
                &shared.model_bgl,
                &shared.material_bgl,
            ),
            skinned_pipeline: Self::create_skinned_pipeline(
                &shared.device,
                &shared.camera_bgl,
                &shared.model_bgl,
                &shared.material_bgl,
                &shared.node_matrices_bgl,
            ),
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/depth_prepass.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth_prepass_layout"),
            bind_group_layouts: &[camera_bgl, model_bgl, material_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth_prepass_pipeline"),
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
                targets: &[],
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
        })
    }

    fn create_skinned_pipeline(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
        node_matrices_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/depth_prepass_skinned.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth_prepass_skinned_layout"),
            bind_group_layouts: &[camera_bgl, model_bgl, material_bgl, node_matrices_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth_prepass_skinned_pipeline"),
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
                targets: &[],
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
        })
    }
}

impl RenderPass for DepthPrepass {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth_prepass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        draw_models(
            &mut rpass,
            ctx.shared,
            &self.pipeline,
            &self.skinned_pipeline,
            ctx,
        );
    }
}

// ===========================================================================
// G-Buffer Pass
// ===========================================================================

pub struct GBufferPass {
    pipeline: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,
}

impl GBufferPass {
    pub fn new(shared: &SharedResources) -> Self {
        Self {
            pipeline: Self::create_pipeline(
                &shared.device,
                &shared.camera_bgl,
                &shared.model_bgl,
                &shared.material_bgl,
            ),
            skinned_pipeline: Self::create_skinned_pipeline(
                &shared.device,
                &shared.camera_bgl,
                &shared.model_bgl,
                &shared.material_bgl,
                &shared.node_matrices_bgl,
            ),
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/geometry.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("geometry_layout"),
            bind_group_layouts: &[camera_bgl, model_bgl, material_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("geometry_pipeline"),
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
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
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
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Equal,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_skinned_pipeline(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
        node_matrices_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/geometry_skinned.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("geometry_skinned_layout"),
            bind_group_layouts: &[camera_bgl, model_bgl, material_bgl, node_matrices_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("geometry_skinned_pipeline"),
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
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
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
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Equal,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
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
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        draw_models(
            &mut rpass,
            ctx.shared,
            &self.pipeline,
            &self.skinned_pipeline,
            ctx,
        );
    }
}

// ===========================================================================
// Emissive Forward Pass
// ===========================================================================

pub struct EmissiveForwardPass {
    pipeline: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,
}

const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

impl EmissiveForwardPass {
    pub fn new(shared: &SharedResources) -> Self {
        Self {
            pipeline: Self::create_pipeline(
                &shared.device,
                &shared.camera_bgl,
                &shared.model_bgl,
                &shared.material_bgl,
            ),
            skinned_pipeline: Self::create_skinned_pipeline(
                &shared.device,
                &shared.camera_bgl,
                &shared.model_bgl,
                &shared.material_bgl,
                &shared.node_matrices_bgl,
            ),
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/emissive_forward.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("emissive_layout"),
            bind_group_layouts: &[camera_bgl, model_bgl, material_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("emissive_pipeline"),
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
                    format: wgpu::TextureFormat::Rg11b10Ufloat,
                    blend: Some(ADDITIVE_BLEND),
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
                depth_compare: wgpu::CompareFunction::Equal,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_skinned_pipeline(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
        node_matrices_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../assets/shaders/emissive_forward_skinned.wgsl"
        ));

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("emissive_skinned_layout"),
            bind_group_layouts: &[camera_bgl, model_bgl, material_bgl, node_matrices_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("emissive_skinned_pipeline"),
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
                    format: wgpu::TextureFormat::Rg11b10Ufloat,
                    blend: Some(ADDITIVE_BLEND),
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
                depth_compare: wgpu::CompareFunction::Equal,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    }
}

impl RenderPass for EmissiveForwardPass {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &FrameContext) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("emissive_forward_pass"),
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
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        draw_models(
            &mut rpass,
            ctx.shared,
            &self.pipeline,
            &self.skinned_pipeline,
            ctx,
        );
    }
}
