use crate::{
    gpu_types::{
        CameraUniforms, GpuShadowData, CSM_CASCADE_COUNT, CSM_MAP_SIZE,
        MAX_POINT_SHADOW_CASTERS, MAX_SPOT_SHADOW_CASTERS,
        SHADOW_CAMERA_SLOTS, SHADOW_MAP_SIZE, SPOT_SHADOW_MAP_SIZE,
    },
    lights::LightType,
    model::{VertexRigid, VertexSkinned, VertexType},
    renderer::{
        GpuModel,
        helpers::{sampler_entry, uniform_entry},
        shared::SharedResources,
    },
};
use bytemuck::Zeroable;
use glam::{Mat4, Vec3, Vec4};

#[allow(dead_code)]
pub struct ShadowPass {
    pipeline: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,
    cascade_pipeline: wgpu::RenderPipeline,
    cascade_skinned_pipeline: wgpu::RenderPipeline,

    camera_bgl: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    camera_stride: usize,
    camera_bind_group: wgpu::BindGroup,

    uniform_buffer: wgpu::Buffer,

    pub bgl: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,

    point_cubemaps: Vec<wgpu::Texture>,
    point_cubemap_views: Vec<wgpu::TextureView>,
    point_face_views: Vec<[wgpu::TextureView; 6]>,
    spot_maps: Vec<wgpu::Texture>,
    spot_views: Vec<wgpu::TextureView>,

    cascade_texture: wgpu::Texture,
    cascade_array_view: wgpu::TextureView,
    cascade_layer_views: [wgpu::TextureView; CSM_CASCADE_COUNT],
}

impl ShadowPass {
    pub fn new(shared: &SharedResources) -> Self {
        let camera_bgl = create_shadow_camera_bgl(&shared.device);
        let bgl = create_shadow_bgl(&shared.device);

        let point_spot_bias = wgpu::DepthBiasState {
            constant: 2,
            slope_scale: 0.0, // slope bias moved to shader (per-pixel smooth, not per-triangle)
            clamp: 0.0,
        };
        let cascade_bias = wgpu::DepthBiasState {
            constant: 3,
            slope_scale: 3.0,
            clamp: 0.0,
        };

        let pipeline = create_shadow_pipeline(&shared.device, &camera_bgl, &shared.model_bgl, point_spot_bias);
        let skinned_pipeline = create_shadow_skinned_pipeline(
            &shared.device,
            &camera_bgl,
            &shared.model_bgl,
            &shared.node_matrices_bgl,
            point_spot_bias,
        );
        let cascade_pipeline = create_shadow_pipeline(&shared.device, &camera_bgl, &shared.model_bgl, cascade_bias);
        let cascade_skinned_pipeline = create_shadow_skinned_pipeline(
            &shared.device,
            &camera_bgl,
            &shared.model_bgl,
            &shared.node_matrices_bgl,
            cascade_bias,
        );

        let min_alignment = shared.device.limits().min_uniform_buffer_offset_alignment as usize;
        let camera_size = size_of::<CameraUniforms>();
        let camera_stride = ((camera_size + min_alignment - 1) / min_alignment) * min_alignment;
        let camera_buffer = shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_camera_buffer"),
            size: (SHADOW_CAMERA_SLOTS * camera_stride) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform_buffer"),
            size: size_of::<GpuShadowData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_camera_bg"),
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

        // Shadow textures
        let mut point_cubemaps = Vec::with_capacity(MAX_POINT_SHADOW_CASTERS);
        let mut point_cubemap_views = Vec::with_capacity(MAX_POINT_SHADOW_CASTERS);
        let mut point_face_views = Vec::with_capacity(MAX_POINT_SHADOW_CASTERS);

        for i in 0..MAX_POINT_SHADOW_CASTERS {
            let tex = shared.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("shadow_point_cubemap_{i}")),
                size: wgpu::Extent3d {
                    width: SHADOW_MAP_SIZE,
                    height: SHADOW_MAP_SIZE,
                    depth_or_array_layers: 6,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let cube_view = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });

            let face_views: [wgpu::TextureView; 6] = std::array::from_fn(|face| {
                tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("shadow_point_{i}_face_{face}")),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: face as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            });

            point_cubemaps.push(tex);
            point_cubemap_views.push(cube_view);
            point_face_views.push(face_views);
        }

        let mut spot_maps = Vec::with_capacity(MAX_SPOT_SHADOW_CASTERS);
        let mut spot_views = Vec::with_capacity(MAX_SPOT_SHADOW_CASTERS);

        for i in 0..MAX_SPOT_SHADOW_CASTERS {
            let tex = shared.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("shadow_spot_map_{i}")),
                size: wgpu::Extent3d {
                    width: SPOT_SHADOW_MAP_SIZE,
                    height: SPOT_SHADOW_MAP_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

            spot_maps.push(tex);
            spot_views.push(view);
        }

        // Cascade shadow map texture (2D array, one layer per cascade)
        let cascade_texture = shared.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_cascade"),
            size: wgpu::Extent3d {
                width: CSM_MAP_SIZE,
                height: CSM_MAP_SIZE,
                depth_or_array_layers: CSM_CASCADE_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cascade_array_view = cascade_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let cascade_layer_views: [wgpu::TextureView; CSM_CASCADE_COUNT] =
            std::array::from_fn(|i| {
                cascade_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("shadow_cascade_layer_{i}")),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            });

        let bind_group = create_shadow_bind_group(
            &shared.device,
            &bgl,
            &point_cubemap_views,
            &spot_views,
            &shared.shadow_comparison_sampler,
            &uniform_buffer,
            &cascade_array_view,
        );

        Self {
            pipeline,
            skinned_pipeline,
            cascade_pipeline,
            cascade_skinned_pipeline,
            camera_bgl,
            camera_buffer,
            camera_stride,
            camera_bind_group,
            uniform_buffer,
            bgl,
            bind_group,
            point_cubemaps,
            point_cubemap_views,
            point_face_views,
            spot_maps,
            spot_views,
            cascade_texture,
            cascade_array_view,
            cascade_layer_views,
        }
    }

    /// Upload shadow camera matrices and shadow uniform data.
    /// Returns (point_shadow_casters, spot_shadow_casters, shadow_assignments).
    pub fn prepare(
        &self,
        shared: &SharedResources,
        game_lights: &crate::lights::LightStore,
        camera_view: Mat4,
        camera_proj: Mat4,
        point_shadow_casters: &mut Vec<(usize, usize)>,
        spot_shadow_casters: &mut Vec<(usize, usize)>,
        shadow_assignments: &mut Vec<(usize, i32)>,
    ) {
        point_shadow_casters.clear();
        spot_shadow_casters.clear();
        shadow_assignments.clear();

        {
            let mut gpu_slot = 0usize;
            for (light_idx, light) in game_lights.iter() {
                if light.hidden {
                    continue;
                }

                if gpu_slot >= crate::lights::MAX_LIGHTS {
                    break;
                }

                if light.casts_shadow {
                    match light.light_type {
                        LightType::Point if point_shadow_casters.len() < MAX_POINT_SHADOW_CASTERS => {
                            let si = point_shadow_casters.len() as i32;
                            shadow_assignments.push((gpu_slot, si));
                            point_shadow_casters.push((gpu_slot, light_idx.0));
                        }

                        LightType::Spot if spot_shadow_casters.len() < MAX_SPOT_SHADOW_CASTERS => {
                            let si = spot_shadow_casters.len() as i32;
                            shadow_assignments.push((gpu_slot, si));
                            spot_shadow_casters.push((gpu_slot, light_idx.0));
                        }

                        LightType::Directional => {
                            shadow_assignments.push((gpu_slot, 0));
                        }

                        _ => {}
                    }
                }

                gpu_slot += 1;
            }
        }

        let shadow_near = 0.5f32;
        let shadow_far = 50.0f32;
        let shadow_bias = 0.005f32;
        let mut shadow_data = GpuShadowData::zeroed();

        let face_directions: [(Vec3, Vec3); 6] = [
            (Vec3::X, Vec3::new(0.0, -1.0, 0.0)),
            (-Vec3::X, Vec3::new(0.0, -1.0, 0.0)),
            (Vec3::Y, Vec3::new(0.0, 0.0, 1.0)),
            (-Vec3::Y, Vec3::new(0.0, 0.0, -1.0)),
            (Vec3::Z, Vec3::new(0.0, -1.0, 0.0)),
            (-Vec3::Z, Vec3::new(0.0, -1.0, 0.0)),
        ];

        let mut point_proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, shadow_near, shadow_far);
        point_proj.y_axis.y = -point_proj.y_axis.y;

        for (pi, &(_gpu_slot, store_idx)) in point_shadow_casters.iter().enumerate() {
            let light = game_lights.get(crate::lights::LightIndex(store_idx));
            let pos = light.position;

            shadow_data.point_params[pi] = [shadow_near, shadow_far, shadow_bias, 0.0];

            for (face, (dir, up)) in face_directions.iter().enumerate() {
                let view = Mat4::look_at_rh(pos, pos + *dir, *up);

                let cam = CameraUniforms {
                    view: view.to_cols_array_2d(),
                    projection: point_proj.to_cols_array_2d(),
                };

                let slot = pi * 6 + face;
                let offset = (slot * self.camera_stride) as u64;

                shared.queue.write_buffer(&self.camera_buffer, offset, bytemuck::bytes_of(&cam));
            }
        }

        for (si, &(_gpu_slot, store_idx)) in spot_shadow_casters.iter().enumerate() {
            let light = game_lights.get(crate::lights::LightIndex(store_idx));
            let pos = light.position;
            let dir = light.direction.normalize();

            // Convert half-angle to full cone + small margin; old * 2.4 wasted resolution
            let fov = (light.outer_cutoff.to_radians() * 2.0 + 0.05).max(0.1);
            // Tighter near/far than point lights — improves depth precision at distance
            let spot_near = 1.0f32;
            let spot_far = 30.0f32;
            let proj = Mat4::perspective_rh(fov, 1.0, spot_near, spot_far);

            let up = if dir.z.abs() > 0.99 { Vec3::Y } else { Vec3::Z };
            let view = Mat4::look_at_rh(pos, pos + dir, up);
            let view_proj = proj * view;

            shadow_data.spot_view_proj[si] = view_proj.to_cols_array_2d();
            shadow_data.spot_params[si] = [
                shadow_bias,
                1.0 / SPOT_SHADOW_MAP_SIZE as f32, // texel_size for PCF + normal offset
                0.0,
                0.0,
            ];

            let cam = CameraUniforms {
                view: view.to_cols_array_2d(),
                projection: proj.to_cols_array_2d(),
            };

            let slot = MAX_POINT_SHADOW_CASTERS * 6 + si;
            let offset = (slot * self.camera_stride) as u64;

            shared.queue.write_buffer(&self.camera_buffer, offset, bytemuck::bytes_of(&cam));
        }

        // Cascade shadow maps for the first directional light with shadows
        const CASCADE_SPLITS: [f32; 4] = [0.05, 8.0, 25.0, 80.0];

        let mut found_directional = false;
        for (_light_idx, light) in game_lights.iter() {
            if light.hidden || !light.casts_shadow || light.light_type != LightType::Directional {
                continue;
            }
            if found_directional {
                break;
            }
            found_directional = true;

            let sun_dir = light.direction.normalize();
            let inv_vp = (camera_proj * camera_view).inverse();

            for cascade in 0..CSM_CASCADE_COUNT {
                let near_split = CASCADE_SPLITS[cascade];
                let far_split = CASCADE_SPLITS[cascade + 1];

                // Convert view-space depth splits to NDC z values (wgpu [0,1] range)
                // For perspective_rh: z_ndc = proj[2][2] * z_view + proj[3][2]) / (-z_view)
                // z_view is negative in RH, so we pass -near_split, -far_split
                let near_ndc = (camera_proj.z_axis.z * (-near_split) + camera_proj.w_axis.z)
                    / near_split;
                let far_ndc = (camera_proj.z_axis.z * (-far_split) + camera_proj.w_axis.z)
                    / far_split;

                // 8 frustum corners in NDC
                let ndc_corners = [
                    Vec4::new(-1.0, -1.0, near_ndc, 1.0),
                    Vec4::new(1.0, -1.0, near_ndc, 1.0),
                    Vec4::new(-1.0, 1.0, near_ndc, 1.0),
                    Vec4::new(1.0, 1.0, near_ndc, 1.0),
                    Vec4::new(-1.0, -1.0, far_ndc, 1.0),
                    Vec4::new(1.0, -1.0, far_ndc, 1.0),
                    Vec4::new(-1.0, 1.0, far_ndc, 1.0),
                    Vec4::new(1.0, 1.0, far_ndc, 1.0),
                ];

                // Unproject to world space
                let mut world_corners = [Vec3::ZERO; 8];
                let mut center = Vec3::ZERO;
                for (i, ndc) in ndc_corners.iter().enumerate() {
                    let world = inv_vp * *ndc;
                    world_corners[i] = world.truncate() / world.w;
                    center += world_corners[i];
                }
                center /= 8.0;

                // Build light view matrix
                let up = if sun_dir.z.abs() > 0.99 {
                    Vec3::Y
                } else {
                    Vec3::Z
                };
                let light_view = Mat4::look_at_rh(center - sun_dir * 50.0, center, up);

                // Bounding sphere of frustum slice — use sphere radius for X/Y
                // so off-screen shadow casters near the frustum are still captured.
                let mut sphere_radius = 0.0f32;
                for corner in &world_corners {
                    sphere_radius = sphere_radius.max((*corner - center).length());
                }

                // Transform corners to light space for Z range only
                let mut min_z = f32::MAX;
                let mut max_z = f32::MIN;
                for corner in &world_corners {
                    let ls = (light_view * Vec4::new(corner.x, corner.y, corner.z, 1.0)).truncate();
                    min_z = min_z.min(ls.z);
                    max_z = max_z.max(ls.z);
                }

                // Extend Z range in both directions:
                // toward the light (max_z) to capture casters between light and frustum
                // (e.g. objects behind the camera), and away (min_z) for depth margin.
                max_z += 100.0;
                min_z -= 100.0;

                // Build orthographic projection:
                // X/Y: sphere-based (captures off-screen shadow casters)
                // Z: tight AABB + extension
                let ortho_proj = Mat4::orthographic_rh(
                    -sphere_radius, sphere_radius,
                    -sphere_radius, sphere_radius,
                    -max_z, -min_z,
                );

                let cascade_vp = ortho_proj * light_view;
                shadow_data.cascade_view_proj[cascade] = cascade_vp.to_cols_array_2d();

                shadow_data.cascade_texel_sizes[cascade] =
                    (2.0 * sphere_radius) / CSM_MAP_SIZE as f32;

                // Upload camera uniform for this cascade's render pass
                let cam = CameraUniforms {
                    view: light_view.to_cols_array_2d(),
                    projection: ortho_proj.to_cols_array_2d(),
                };
                let slot = MAX_POINT_SHADOW_CASTERS * 6 + MAX_SPOT_SHADOW_CASTERS + cascade;
                let offset = (slot * self.camera_stride) as u64;
                shared
                    .queue
                    .write_buffer(&self.camera_buffer, offset, bytemuck::bytes_of(&cam));
            }

            shadow_data.cascade_splits = [
                CASCADE_SPLITS[1],
                CASCADE_SPLITS[2],
                CASCADE_SPLITS[3],
                0.0,
            ];

        }

        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&shadow_data));
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: &SharedResources,
        models: &[GpuModel],
        render_list: &[(crate::objects::ObjectIndex, usize)],
        point_shadow_casters: &[(usize, usize)],
        spot_shadow_casters: &[(usize, usize)],
        has_directional_shadow: bool,
    ) {
        // Point light shadow passes (6 faces per caster)
        for (pi, &(_gpu_slot, _store_idx)) in point_shadow_casters.iter().enumerate() {
            for face in 0..6 {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_point_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.point_face_views[pi][face],
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                let camera_slot = pi * 6 + face;
                let camera_dynamic_offset = (camera_slot * self.camera_stride) as u32;
                rpass.set_bind_group(0, &self.camera_bind_group, &[camera_dynamic_offset]);

                Self::draw_models(&mut rpass, shared, models, render_list, &self.pipeline, &self.skinned_pipeline);
            }
        }

        // Spot light shadow passes (1 per caster)
        for (si, &(_gpu_slot, _store_idx)) in spot_shadow_casters.iter().enumerate() {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_spot_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.spot_views[si],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            let camera_slot = MAX_POINT_SHADOW_CASTERS * 6 + si;
            let camera_dynamic_offset = (camera_slot * self.camera_stride) as u32;
            rpass.set_bind_group(0, &self.camera_bind_group, &[camera_dynamic_offset]);

            Self::draw_models(&mut rpass, shared, models, render_list, &self.pipeline, &self.skinned_pipeline);
        }

        // Cascade shadow passes (one per cascade layer)
        if has_directional_shadow {
            for cascade in 0..CSM_CASCADE_COUNT {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow_cascade_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.cascade_layer_views[cascade],
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                let camera_slot =
                    MAX_POINT_SHADOW_CASTERS * 6 + MAX_SPOT_SHADOW_CASTERS + cascade;
                let camera_dynamic_offset = (camera_slot * self.camera_stride) as u32;
                rpass.set_bind_group(0, &self.camera_bind_group, &[camera_dynamic_offset]);

                Self::draw_models(&mut rpass, shared, models, render_list, &self.cascade_pipeline, &self.cascade_skinned_pipeline);
            }
        }
    }

    fn draw_models<'a>(
        rpass: &mut wgpu::RenderPass<'a>,
        shared: &'a SharedResources,
        models: &'a [GpuModel],
        render_list: &[(crate::objects::ObjectIndex, usize)],
        rigid_pipeline: &'a wgpu::RenderPipeline,
        skinned_pipeline: &'a wgpu::RenderPipeline,
    ) {
        for (obj_slot, &(_obj_idx, model_idx)) in render_list.iter().enumerate() {
            let model_dynamic_offset = (obj_slot * shared.model_stride) as u32;
            let nm_dynamic_offset = (obj_slot * shared.node_matrices_stride) as u32;
            let gpu_model = &models[model_idx];

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
                            2,
                            &shared.node_matrices_bind_group,
                            &[nm_dynamic_offset],
                        );
                    }
                }

                rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                for part in &mesh.parts {
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
// Bind group layouts + creation
// ---------------------------------------------------------------------------

fn create_shadow_camera_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_camera_bgl"),
        entries: &[uniform_entry(
            0,
            size_of::<CameraUniforms>() as u64,
            wgpu::ShaderStages::VERTEX,
            true,
        )],
    })
}

fn create_shadow_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            sampler_entry(4, wgpu::SamplerBindingType::Comparison),
            uniform_entry(
                5,
                size_of::<GpuShadowData>() as u64,
                wgpu::ShaderStages::FRAGMENT,
                false,
            ),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn create_shadow_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    point_cubemap_views: &[wgpu::TextureView],
    spot_views: &[wgpu::TextureView],
    comparison_sampler: &wgpu::Sampler,
    uniform_buffer: &wgpu::Buffer,
    cascade_array_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&point_cubemap_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&point_cubemap_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&spot_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&spot_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(comparison_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(cascade_array_view),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

fn create_shadow_pipeline(
    device: &wgpu::Device,
    shadow_camera_bgl: &wgpu::BindGroupLayout,
    model_bgl: &wgpu::BindGroupLayout,
    bias: wgpu::DepthBiasState,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!(
        "../../assets/shaders/shadow.wgsl"
    ));

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow_layout"),
        bind_group_layouts: &[shadow_camera_bgl, model_bgl],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: size_of::<VertexRigid>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            }],
            compilation_options: Default::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias,
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_shadow_skinned_pipeline(
    device: &wgpu::Device,
    shadow_camera_bgl: &wgpu::BindGroupLayout,
    model_bgl: &wgpu::BindGroupLayout,
    node_matrices_bgl: &wgpu::BindGroupLayout,
    bias: wgpu::DepthBiasState,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!(
        "../../assets/shaders/shadow_skinned.wgsl"
    ));
    
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow_skinned_layout"),
        bind_group_layouts: &[shadow_camera_bgl, model_bgl, node_matrices_bgl],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow_skinned_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: size_of::<VertexSkinned>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32x4,
                        offset: 56,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 72,
                        shader_location: 2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias,
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}
