use crate::{
    models::{VertexRigid, VertexSkinned, VertexType},
    renderer::{
        GpuModel, color_clear_attach,
        shared::{GBuffer, SharedResources},
    },
};

pub struct GeometryPass {
    // Depth pre-pass pipelines (depth-only, alpha-test)
    depth_prepass_pipeline: wgpu::RenderPipeline,
    depth_prepass_skinned_pipeline: wgpu::RenderPipeline,

    // G-buffer pipelines (color output, depth Equal/no-write)
    pipeline: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,

    // Forward emissive pipelines (additive blend onto lighting buffer, depth Equal/no-write)
    emissive_pipeline: wgpu::RenderPipeline,
    emissive_skinned_pipeline: wgpu::RenderPipeline,
}

impl GeometryPass {
    pub fn new(shared: &SharedResources) -> Self {
        let depth_prepass_pipeline = create_depth_prepass_pipeline(
            &shared.device,
            &shared.camera_bgl,
            &shared.model_bgl,
            &shared.material_bgl,
        );

        let depth_prepass_skinned_pipeline = create_depth_prepass_skinned_pipeline(
            &shared.device,
            &shared.camera_bgl,
            &shared.model_bgl,
            &shared.material_bgl,
            &shared.node_matrices_bgl,
        );

        let pipeline = create_geometry_pipeline(
            &shared.device,
            &shared.camera_bgl,
            &shared.model_bgl,
            &shared.material_bgl,
        );

        let skinned_pipeline = create_geometry_skinned_pipeline(
            &shared.device,
            &shared.camera_bgl,
            &shared.model_bgl,
            &shared.material_bgl,
            &shared.node_matrices_bgl,
        );

        let emissive_pipeline = create_emissive_pipeline(
            &shared.device,
            &shared.camera_bgl,
            &shared.model_bgl,
            &shared.material_bgl,
        );

        let emissive_skinned_pipeline = create_emissive_skinned_pipeline(
            &shared.device,
            &shared.camera_bgl,
            &shared.model_bgl,
            &shared.material_bgl,
            &shared.node_matrices_bgl,
        );

        Self {
            depth_prepass_pipeline,
            depth_prepass_skinned_pipeline,
            pipeline,
            skinned_pipeline,
            emissive_pipeline,
            emissive_skinned_pipeline,
        }
    }

    /// Depth pre-pass: writes depth buffer only (with alpha-test).
    /// Must be called before `record()` to populate the depth buffer.
    pub fn record_depth_prepass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        shared: &'a SharedResources,
        gbuffer: &'a GBuffer,
        models: &'a [GpuModel],
        render_list: &[(crate::objects::ObjectIndex, usize)],
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth_prepass"),
            color_attachments: &[],
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

        rpass.set_bind_group(0, &shared.camera_bind_group, &[]);

        for (obj_slot, &(_obj_idx, model_idx)) in render_list.iter().enumerate() {
            let model_dynamic_offset = (obj_slot * shared.model_stride) as u32;
            let nm_dynamic_offset = (obj_slot * shared.node_matrices_stride) as u32;

            let gpu_model = &models[model_idx];

            for mesh in &gpu_model.meshes {
                match mesh.vertex_type {
                    VertexType::Rigid => {
                        rpass.set_pipeline(&self.depth_prepass_pipeline);
                        rpass.set_bind_group(1, &shared.model_bind_group, &[model_dynamic_offset]);
                    }

                    VertexType::Skinned => {
                        rpass.set_pipeline(&self.depth_prepass_skinned_pipeline);
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

    /// G-buffer pass: outputs normals, albedo, material properties.
    /// Depth compare is Equal (pre-pass already wrote depth), so zero overdraw.
    pub fn record<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        shared: &'a SharedResources,
        gbuffer: &'a GBuffer,
        models: &'a [GpuModel],
        render_list: &[(crate::objects::ObjectIndex, usize)],
    ) {
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

        rpass.set_bind_group(0, &shared.camera_bind_group, &[]);

        for (obj_slot, &(_obj_idx, model_idx)) in render_list.iter().enumerate() {
            let model_dynamic_offset = (obj_slot * shared.model_stride) as u32;
            let nm_dynamic_offset = (obj_slot * shared.node_matrices_stride) as u32;

            let gpu_model = &models[model_idx];

            for mesh in &gpu_model.meshes {
                match mesh.vertex_type {
                    VertexType::Rigid => {
                        rpass.set_pipeline(&self.pipeline);
                        rpass.set_bind_group(1, &shared.model_bind_group, &[model_dynamic_offset]);
                    }

                    VertexType::Skinned => {
                        rpass.set_pipeline(&self.skinned_pipeline);
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

    /// Forward emissive pass: additively blends emissive contribution into the lighting buffer.
    /// Runs after deferred lighting. Uses Equal depth compare (reuses pre-pass depth).
    pub fn record_emissive<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        shared: &'a SharedResources,
        gbuffer: &'a GBuffer,
        lighting_view: &'a wgpu::TextureView,
        models: &'a [GpuModel],
        render_list: &[(crate::objects::ObjectIndex, usize)],
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("emissive_forward_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: lighting_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
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

        rpass.set_bind_group(0, &shared.camera_bind_group, &[]);

        for (obj_slot, &(_obj_idx, model_idx)) in render_list.iter().enumerate() {
            let model_dynamic_offset = (obj_slot * shared.model_stride) as u32;
            let nm_dynamic_offset = (obj_slot * shared.node_matrices_stride) as u32;

            let gpu_model = &models[model_idx];

            for mesh in &gpu_model.meshes {
                match mesh.vertex_type {
                    VertexType::Rigid => {
                        rpass.set_pipeline(&self.emissive_pipeline);
                        rpass.set_bind_group(1, &shared.model_bind_group, &[model_dynamic_offset]);
                    }

                    VertexType::Skinned => {
                        rpass.set_pipeline(&self.emissive_skinned_pipeline);
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
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

fn create_depth_prepass_pipeline(
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

fn create_depth_prepass_skinned_pipeline(
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

fn create_geometry_pipeline(
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

fn create_geometry_skinned_pipeline(
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

fn create_emissive_pipeline(
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

fn create_emissive_skinned_pipeline(
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
