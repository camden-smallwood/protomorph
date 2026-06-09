//! S1 — engine-faithful per-object/per-light shadow generate + apply.
//!
//! Replaces the CSM/cubemap-array architecture in `shadow_pass.rs`
//! (which is wrong per the plan: Halo does NOT use cascade shadow
//! mapping). The engine flow per the plan's Phase S spec:
//!
//! 1. Per-caster (object or expensive dynamic light): bind
//!    `_surface_shadow_1` as depth-stencil, render shadow-casting
//!    geometry through a per-caster ortho/perspective camera into the
//!    depth target. Uses `assets/shaders/shadow_generate.wgsl`.
//! 2. Apply pass: a fullscreen quad samples scene depth, reconstructs
//!    world position, transforms to light-space, PCF-samples the shadow
//!    target, and alpha-blends a darkening stamp into the LDR target.
//!    Uses `assets/shaders/shadow_apply.wgsl`.
//! 3. Repeat for each caster against the same shared 512² target.
//!
//! S1 lands the infrastructure (target, shaders, pass skeleton).
//! S2 will tighten the PCF kernel + bias model.
//! S4 will plumb the per-light scheduler + Shadow_Projection cbuffer.
//!
//! Per-object OBB iteration (the actual list of casters) is not yet
//! wired — this pass is a no-op until a caster list is populated.
//! Engine anchors:
//!   - `c_lightmap_shadows_view::submit_visibility_and_render @ 0x1806BB7A0`
//!   - `c_rasterizer::setup_targets_shadow_generate @ 0x180671080`

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::halo::render::shared::SharedResources;

/// One shadow-cast event the apply pass should stamp. Populated each
/// frame by the per-object/per-light scheduler (TODO — see plan S1/S4).
/// The matrix fields mirror engine `Shadow_Projection` (cbuffer slot
/// `0x2B0000`, 4 vec4) + `CAMERA_TO_SHADOW_*` (`0x2E0000..2`).
#[derive(Clone, Copy)]
pub struct ShadowCaster {
    /// World-to-light-clip matrix used both at generate-time (to write
    /// the depth target) and at apply-time (to project scene fragments
    /// into shadow UV space).
    pub world_to_shadow: glam::Mat4,
    /// 0..1 fade — multiplied into the stamp alpha at apply-time. Engine
    /// uses this for distance fade, transition into ambient-only zones,
    /// etc. Default 1.0 = full shadow.
    pub opacity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct ShadowApplyUniforms {
    world_to_shadow: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    shadow_pixel_size: [f32; 4],
    caster_opacity: [f32; 4],
    /// `(sub_res/512, sub_res/512, 0, 0)` — UV scale into the populated
    /// sub-rect of `_surface_shadow_1`. See shader doc.
    shadow_uv_scale: [f32; 4],
    /// World-space light direction (toward the light). Engine
    /// `SHADOW_DIRECTION_WORLDSPACE = p_lighting_constant_9.xyz`.
    light_dir_world: [f32; 4],
    /// `(quality_mode, _, _, _)` — PCF kernel selector. 0=default 3×3,
    /// 1=fancy 5×5 predicated, 2=faster 2×2 bilinear. Read from
    /// `PROTOMORPH_SHADOW_PCF` env var via `pcf_quality_from_env()`.
    pcf_quality: [f32; 4],
}

/// Read `PROTOMORPH_SHADOW_PCF` env var → kernel index.
/// Cached lazily so `getenv` runs once per process.
pub fn pcf_quality_from_env() -> u32 {
    use std::sync::OnceLock;
    static QUALITY: OnceLock<u32> = OnceLock::new();
    *QUALITY.get_or_init(|| match std::env::var("PROTOMORPH_SHADOW_PCF").ok().as_deref() {
        Some("fancy") => 1,
        Some("faster") => 2,
        _ => 0,
    })
}

/// `PROTOMORPH_SHADOW_DEBUG=N.NN` — flat-stamp every shadow-apply
/// fragment with alpha=N.NN regardless of depth/PCF math. Lets us
/// confirm whether the stamp is reaching lighting_base. 0 = off
/// (normal math). Values in [0.05, 0.9] are visually useful.
pub fn shadow_debug_stamp_alpha() -> f32 {
    use std::sync::OnceLock;
    static ALPHA: OnceLock<f32> = OnceLock::new();
    *ALPHA.get_or_init(|| {
        std::env::var("PROTOMORPH_SHADOW_DEBUG")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    })
}

/// `PROTOMORPH_SHADOW_NO_COSINE=1` forces cosine_falloff = 1.0 in the
/// shadow apply PS. Diagnostic: if shadows appear with this but not
/// without, the receiver normal MRT is the bug (cleared normals or
/// decode mismatch).
pub fn shadow_no_cosine_from_env() -> f32 {
    use std::sync::OnceLock;
    static ON: OnceLock<f32> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("PROTOMORPH_SHADOW_NO_COSINE").ok().as_deref() {
        Some("1") | Some("true") => 1.0,
        _ => 0.0,
    })
}

/// `PROTOMORPH_SHADOW_FRUSTUM_ONLY=1` makes the apply PS stamp a flat
/// 0.4-alpha darkening for every fragment whose world-position lands
/// INSIDE the caster's shadow frustum, bypassing PCF / depth compare.
/// Diagnostic: any caster whose matrix is correct will show a darker
/// silhouette/rectangle under it. Casters whose matrix lands no
/// receivers in the frustum will be invisible.
pub fn shadow_frustum_only_from_env() -> f32 {
    use std::sync::OnceLock;
    static ON: OnceLock<f32> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("PROTOMORPH_SHADOW_FRUSTUM_ONLY").ok().as_deref() {
        Some("1") | Some("true") => 1.0,
        _ => 0.0,
    })
}

/// Per-caster uniform slot stride. 256 is the spec maximum for
/// `min_uniform_buffer_offset_alignment`, so dynamic offsets of
/// `caster_index × STRIDE` are valid on every device. Each slot holds
/// one `ShadowGenerateUniforms` (128 B) or `ShadowApplyUniforms` (208 B).
const SHADOW_CASTER_STRIDE: u64 = 256;

/// Max shadow casters drawn per frame. The generate/apply uniform
/// buffers are sized for this; casters past it are dropped (logged once).
/// Object shadows only — far above the engine's throttled
/// `shadow_generate_count`, but protomorph has no LOD throttle yet.
pub const MAX_SHADOW_CASTERS: u32 = 512;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct ShadowGenerateUniforms {
    world_to_shadow: [[f32; 4]; 4],
    /// Object-to-world for THIS caster. Folded in here (vs the shared
    /// `model_bind_group` slot) so shadows aren't tied to the on-screen
    /// `render_list` — any shadow-casting object can be drawn, including
    /// ones culled from view whose shadow still lands on-screen.
    model: [[f32; 4]; 4],
}

/// State for the shadow-generate + apply pipelines.
///
/// `submit_caster` is the per-caster orchestration: open a depth-only
/// generate rpass (caller fills in the geometry draws) → close the
/// rpass → run a screen-space apply pass that stamps darkening onto
/// the LDR target. Called once per shadow-cast object/light per frame.
pub struct ShadowApplyPass {
    apply_pipeline: wgpu::RenderPipeline,
    apply_bgl: wgpu::BindGroupLayout,
    apply_uniform: wgpu::Buffer,
    depth_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    normal_sampler: wgpu::Sampler,
    /// Reuse-cache for the per-caster apply bind group. Key folds in
    /// the varying inputs (scene_depth, shadow_depth, receiver_normal
    /// views). Self-owned uniform + samplers are baked into every
    /// entry. Cleared by the caller on resize via `invalidate_cache()`.
    apply_bg_cache: std::cell::RefCell<crate::halo::render::bind_group_cache::BindGroupCache>,

    generate_pipeline: wgpu::RenderPipeline,
    generate_uniform: wgpu::Buffer,
    generate_uniform_bg: wgpu::BindGroup,
}

impl ShadowApplyPass {
    pub fn new(shared: &SharedResources) -> Self {
        // Bind group layout — matches `shadow_apply.wgsl` @group(0).
        let apply_bgl = shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_apply_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // Per-caster slot addressed by dynamic offset.
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = shared.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_apply_pipeline_layout"),
            bind_group_layouts: &[&apply_bgl],
            immediate_size: 0,
        });

        let shader = shared.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_apply_wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/shadow_apply.wgsl").into(),
            ),
        });

        let apply_pipeline = shared.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_apply_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // Engine `_alpha_blend_alpha_blend` per
                // `set_alpha_blend_mode_no_cache @ 0x18066E930` packed value
                // 0x3C98A4C5 → SRC_ALPHA / INV_SRC_ALPHA / ADD for both
                // color and alpha. Dual-target so the bloom-extract feed
                // (`hdr_dark`) sees the same darkening the LDR target gets
                // — auto-exposure would otherwise undershoot ~0.6 stops
                // from sampling the un-shadowed brighter scene. Same blend
                // on both RTs since DARK_COLOR_MULTIPLIER = 1.0 on PC.
                targets: &{
                    let factors = wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    };
                    let blend = wgpu::BlendState { color: factors, alpha: factors };
                    [
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ]
                },
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // One slot per caster (dynamic offset). Sized for MAX_SHADOW_CASTERS.
        let apply_uniform = shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_apply_uniform"),
            size: MAX_SHADOW_CASTERS as u64 * SHADOW_CASTER_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_sampler = shared.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_apply_depth_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let shadow_sampler = shared.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_apply_compare_sampler"),
            // POINT comparison — engine `shadow_apply_hlsl.hlsl` samples
            // the shadow depth with `tex2D_offset_point` and does the
            // bilinear smoothing via MANUAL blend weights in the kernel.
            // A Linear comparison sampler would add a hardware 2×2 PCF to
            // every tap, turning the 3×3 kernel into a ~4×4 blur — the
            // main remaining over-softening vs the engine.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let normal_sampler = shared.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_apply_normal_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ----- Shadow generate (depth-only) pipeline -----
        // @group(0) = ShadowGenerateUniforms {world_to_shadow, model},
        // one slot per caster addressed by dynamic offset. The model is
        // folded in (no shared model_bind_group), so shadows are not tied
        // to the on-screen render_list.
        let generate_bgl = shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_generate_uniforms_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let generate_uniform = shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_generate_uniform"),
            size: MAX_SHADOW_CASTERS as u64 * SHADOW_CASTER_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let generate_uniform_bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_generate_uniforms_bg"),
            layout: &generate_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // One slot window; the dynamic offset selects the caster.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &generate_uniform,
                    offset: 0,
                    size: std::num::NonZeroU64::new(SHADOW_CASTER_STRIDE),
                }),
            }],
        });
        let generate_layout = shared.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_generate_pipeline_layout"),
            bind_group_layouts: &[&generate_bgl],
            immediate_size: 0,
        });
        let generate_shader = shared.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_generate_wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/shaders/shadow_generate.wgsl").into(),
            ),
        });
        let generate_pipeline = shared.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_generate_pipeline"),
            layout: Some(&generate_layout),
            vertex: wgpu::VertexState {
                module: &generate_shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::halo::geometry::ModelVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &generate_shader,
                entry_point: Some("fs_main"),
                // Engine: PS=NULL → no color attachments.
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Front-face cull would shadow-acne the back side of
                // objects; back-face cull matches engine `_z_buffer_
                // mode_shadow_generate` defaults.
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                // Shadow target is cleared to 1.0 (engine default), so
                // Less wins on closer fragments.
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    // Engine `_z_buffer_mode_shadow_generate` uses a
                    // small constant + slope bias to push the recorded
                    // depth slightly farther from the receiver, killing
                    // self-shadow acne. Picked conservative values; S2
                    // will refine via `terrain_fx.hlsl:1476-1485`.
                    constant: 4,
                    slope_scale: 1.5,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            apply_pipeline,
            apply_bgl,
            apply_uniform,
            depth_sampler,
            shadow_sampler,
            normal_sampler,
            apply_bg_cache: std::cell::RefCell::new(
                crate::halo::render::bind_group_cache::BindGroupCache::new(),
            ),
            generate_pipeline,
            generate_uniform,
            generate_uniform_bg,
        }
    }

    /// Drop every cached apply bind group. Caller invokes on resize
    /// after intermediate views are rebuilt.
    pub fn invalidate_cache(&self) {
        self.apply_bg_cache.borrow_mut().clear();
    }

    /// Open a depth-only render pass into `_surface_shadow_1` for one
    /// caster `caster_index`. The caller draws shadow-casting geometry
    /// through the already-bound shadow_generate pipeline (set_vertex_buffer,
    /// set_index_buffer, draw_indexed — the per-caster `world_to_shadow`
    /// + `model` are already in the bound uniform slot). When the returned
    /// rpass is dropped, depth has been written.
    ///
    /// Engine equivalent: `setup_targets_shadow_generate(clear=1, ...)
    /// → object_renderer::render_shadows_generate(...) →
    /// structure_renderer::render_shadows_generate(...)`.
    ///
    /// Each caster writes a DISTINCT slot (`caster_index × STRIDE`) of the
    /// generate uniform buffer, so a single per-frame submit sees every
    /// caster's own matrix — no last-write-wins clobber. Pairs with
    /// `apply_caster`, which must use the SAME `caster_index`.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_generate_pass<'a>(
        &'a self,
        shared: &SharedResources,
        encoder: &'a mut wgpu::CommandEncoder,
        shadow_depth_view: &'a wgpu::TextureView,
        world_to_shadow: glam::Mat4,
        model_matrix: glam::Mat4,
        caster_index: u32,
        sub_res: u32,
    ) -> wgpu::RenderPass<'a> {
        let uniforms = ShadowGenerateUniforms {
            world_to_shadow: world_to_shadow.to_cols_array_2d(),
            model: model_matrix.to_cols_array_2d(),
        };
        let slot = caster_index as u64 * SHADOW_CASTER_STRIDE;
        shared.queue.write_buffer(&self.generate_uniform, slot, bytemuck::bytes_of(&uniforms));

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_generate_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: shadow_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        // Engine: per-object resolution clamp. Render into a
        // `sub_res × sub_res` sub-rect at the top-left of the 512²
        // target; the apply pass scales UVs to match. The remaining
        // texels keep the LoadOp::Clear=1.0 value so off-rect samples
        // read "no occluder" → unshadowed.
        let sub = sub_res.max(8) as f32;
        rpass.set_viewport(0.0, 0.0, sub, sub, 0.0, 1.0);
        rpass.set_pipeline(&self.generate_pipeline);
        rpass.set_bind_group(0, &self.generate_uniform_bg, &[slot as u32]);
        rpass
    }

    /// Run the screen-space stamp for one caster. The shadow depth
    /// target must have been populated by a prior generate pass with
    /// the SAME `world_to_shadow` matrix. Outputs an alpha-blend
    /// darkening into `lighting_base_view`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_caster(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        lighting_base_view: &wgpu::TextureView,
        hdr_dark_view: &wgpu::TextureView,
        scene_depth_view: &wgpu::TextureView,
        shadow_depth_view: &wgpu::TextureView,
        receiver_normal_view: &wgpu::TextureView,
        camera_view: glam::Mat4,
        camera_projection: glam::Mat4,
        world_to_shadow: glam::Mat4,
        light_dir_world: glam::Vec3,
        opacity: f32,
        caster_index: u32,
        sub_res: u32,
    ) {
        let inv_vp = (camera_projection * camera_view).inverse();
        let res_max = crate::halo::render::shared::SHADOW_DEPTH_RESOLUTION as f32;
        let pixel_size = std::f32::consts::SQRT_2 / res_max;
        let uv_scale = (sub_res.max(8) as f32) / res_max;
        let ld = light_dir_world.normalize_or_zero();
        let uniforms = ShadowApplyUniforms {
            world_to_shadow: world_to_shadow.to_cols_array_2d(),
            inverse_view_projection: inv_vp.to_cols_array_2d(),
            shadow_pixel_size: [pixel_size, 0.0, 0.0, 0.0],
            caster_opacity: [opacity.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
            shadow_uv_scale: [uv_scale, uv_scale, 0.0, 0.0],
            light_dir_world: [ld.x, ld.y, ld.z, 0.0],
            pcf_quality: [
                pcf_quality_from_env() as f32,
                shadow_debug_stamp_alpha(),
                shadow_no_cosine_from_env(),
                shadow_frustum_only_from_env(),
            ],
        };
        // Distinct slot per caster — no clobber across the single submit.
        let slot = caster_index as u64 * SHADOW_CASTER_STRIDE;
        shared.queue.write_buffer(&self.apply_uniform, slot, bytemuck::bytes_of(&uniforms));

        // Cache key folds in the varying-per-caster views; the uniform
        // buffer + samplers are stable. Each caster's bind group is
        // built once and reused for every subsequent frame as long as
        // its shadow_depth_view stays alive.
        use crate::halo::render::bind_group_cache::{key_from_ptrs, ptr_addr};
        let cache_key = key_from_ptrs(&[
            ptr_addr(scene_depth_view),
            ptr_addr(shadow_depth_view),
            ptr_addr(receiver_normal_view),
        ]);
        let bind_group = self.apply_bg_cache.borrow_mut().get_or_create(
            cache_key,
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow_apply_bind_group"),
                layout: &self.apply_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        // One slot window; the dynamic offset selects the
                        // caster at bind time.
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.apply_uniform,
                            offset: 0,
                            size: std::num::NonZeroU64::new(SHADOW_CASTER_STRIDE),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(scene_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(shadow_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(receiver_normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&self.normal_sampler),
                    },
                ],
            }),
        );

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_apply_pass"),
            color_attachments: &[
                // RT0 — `lighting_base` (LDR composite read by FinalCompositePass).
                Some(wgpu::RenderPassColorAttachment {
                    view: lighting_base_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                // RT1 — `hdr_dark` (engine `_surface_screenshot_composite_cubemap`,
                // the bloom-extract / auto-exposure feed). Same darkening
                // as RT0 since DARK_COLOR_MULTIPLIER = 1.0 on PC.
                Some(wgpu::RenderPassColorAttachment {
                    view: hdr_dark_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_pipeline(&self.apply_pipeline);
        rpass.set_bind_group(0, bind_group.as_ref(), &[slot as u32]);
        rpass.draw(0..3, 0..1);
    }
}

/// Compute the per-caster shadow target sub-resolution from the
/// projected sphere diameter on screen, clamped to `[16..512]`.
///
/// Engine `c_lightmap_shadows_view::render` per-caster:
///   r = (proj_sphere_diameter_pixels × shadow_quality_lod) / k_shadow_resolution
///   gen_res = clamp(r × k_shadow_resolution, 16, 512) = clamp(diameter × lod, 16, 512)
///
/// `shadow_quality_lod` lives at `s_performance_throttles+52`
/// (`c_performance_throttles::m_current_throttles @ 0x182582948`); we
/// don't have a perf-throttles port yet so we default to 1.0.
///
/// `proj_sphere_diameter_pixels` for a perspective camera is
/// `2 × radius × screen_height / (2 × dist × tan(fov_y/2))`.
pub fn shadow_resolution_for_caster(
    caster_center: glam::Vec3,
    caster_radius: f32,
    camera_position: glam::Vec3,
    camera_fov_y_radians: f32,
    screen_height: u32,
) -> u32 {
    let dist = (caster_center - camera_position).length().max(1e-3);
    let half_fov = (camera_fov_y_radians * 0.5).max(1e-4);
    let diameter_pixels =
        (caster_radius * (screen_height as f32)) / (dist * half_fov.tan());
    // shadow_quality_lod default 1.0 (perf throttle not ported).
    let res = diameter_pixels.clamp(16.0, 512.0);
    res.round() as u32
}

/// Build a basic ortho world-to-shadow matrix for an OBB-bounded
/// caster. `light_dir` points TOWARD the light; the matrix projects
/// world points into [-1, 1] light-clip space along that axis.
///
/// Engine equivalent: `c_lightmap_shadows_view::setup_camera @
/// 0x1806BB...` builds an ortho camera through the caster's
/// `projection_bounds` (a 3-basis OBB) toward the dominant light.
/// This sphere-bound version is preserved as a fallback for callers
/// without an AABB; new code should prefer [`build_caster_obb_ortho`]
/// which projects the 8 AABB corners into light-space for tight
/// non-spherical bounds.
pub fn build_caster_ortho(
    caster_center: glam::Vec3,
    caster_extents: glam::Vec3,
    light_dir: glam::Vec3,
) -> glam::Mat4 {
    // Build a light-space basis. Choose `up` perpendicular to light_dir.
    let forward = -light_dir.normalize_or_zero();
    let up_world = if forward.z.abs() < 0.99 {
        glam::Vec3::Z
    } else {
        glam::Vec3::Y
    };
    let right = forward.cross(up_world).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();

    // Half-size of the ortho box. The loop passes `splat(shadow_radius)`
    // (= 3× the bounding-sphere radius, per `object_shadow_get_potential_
    // bounds @ 0x1806BAAD0`). Take the max extent → a cube of that
    // half-size (~6r across). Using `.length()` instead circumscribed the
    // sphere with a cube of half-size `radius·√3 ≈ 5.2r` — ~1.7× too loose
    // on every axis, so the caster's silhouette covered ~3× fewer shadow-
    // map texels → blurry. `max_element` packs the caster into far more
    // depth texels (sharper) while still covering the 3× ground bound.
    let radius = caster_extents.max_element();
    let eye = caster_center - forward * radius;
    let view = glam::Mat4::look_at_rh(eye, caster_center, up);
    let proj = glam::Mat4::orthographic_rh(
        -radius, radius, -radius, radius, 0.01, radius * 2.5,
    );
    proj * view
}

/// Build a tight light-aligned ortho world-to-shadow matrix for a
/// caster whose model-space AABB is `(aabb_min, aabb_max)` transformed
/// to world space by `model_matrix`. Engine-faithful equivalent of
/// `c_lightmap_shadows_view::setup_camera`'s `m_projection_bounds`
/// path: project the 8 AABB corners into a light-aligned basis, take
/// the per-axis min/max, and build the camera frustum from those tight
/// extents instead of an inflated bounding-sphere cube.
///
/// Visible win over [`build_caster_ortho`]: for elongated objects
/// (crates, weapon racks, etc.) the perpendicular axes shrink, raising
/// effective shadow resolution along the long axis without changing
/// the output target size.
pub fn build_caster_obb_ortho(
    aabb_min: glam::Vec3,
    aabb_max: glam::Vec3,
    model_matrix: glam::Mat4,
    light_dir: glam::Vec3,
) -> glam::Mat4 {
    // Light-space basis. `forward` points AWAY from the light (camera
    // looks toward the receiver along -light_dir).
    let forward = -light_dir.normalize_or_zero();
    let up_world = if forward.z.abs() < 0.99 {
        glam::Vec3::Z
    } else {
        glam::Vec3::Y
    };
    let right = forward.cross(up_world).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();

    // Project the 8 AABB corners through model_matrix into world, then
    // onto the (right, up, forward) basis. Track per-axis min/max.
    let corners = [
        glam::Vec3::new(aabb_min.x, aabb_min.y, aabb_min.z),
        glam::Vec3::new(aabb_max.x, aabb_min.y, aabb_min.z),
        glam::Vec3::new(aabb_min.x, aabb_max.y, aabb_min.z),
        glam::Vec3::new(aabb_max.x, aabb_max.y, aabb_min.z),
        glam::Vec3::new(aabb_min.x, aabb_min.y, aabb_max.z),
        glam::Vec3::new(aabb_max.x, aabb_min.y, aabb_max.z),
        glam::Vec3::new(aabb_min.x, aabb_max.y, aabb_max.z),
        glam::Vec3::new(aabb_max.x, aabb_max.y, aabb_max.z),
    ];
    let mut min_l = glam::Vec3::splat(f32::INFINITY);
    let mut max_l = glam::Vec3::splat(f32::NEG_INFINITY);
    for c in corners {
        let world = model_matrix.transform_point3(c);
        let lx = world.dot(right);
        let ly = world.dot(up);
        let lz = world.dot(forward);
        min_l = min_l.min(glam::Vec3::new(lx, ly, lz));
        max_l = max_l.max(glam::Vec3::new(lx, ly, lz));
    }

    // Camera position: behind the OBB along -forward. Engine pattern
    // (`c_lightmap_shadows_view::setup_camera`): eye is `-2 * extents.x *
    // forward + center` and `z_near = extents.x`, `z_far = 3 * extents.x`.
    // We additionally extend `z_far` past the OBB's back face by
    // `RECEIVER_BUFFER × half_l.z` to capture shadow receivers on the
    // ground / surfaces below the caster — without this, the OBB
    // tightly bounds the caster geometry and ground fragments fall
    // OUTSIDE the frustum (early-return → no shadow). Engine relies on
    // attachment-tree expansion of `m_projection_bounds` for the same
    // effect; we get there with an explicit padding factor.
    let center_l = (min_l + max_l) * 0.5;
    // Floor each half-extent so a near-flat object (a decal-thin AABB, a
    // single-plane mesh) can't collapse the ortho to a zero-width frustum
    // → NaN projection. 5 cm is well below any real caster's dimensions.
    let half_l = ((max_l - min_l) * 0.5).max(glam::Vec3::splat(0.05));
    // Engine `object_shadow_visible`/`setup_camera`: the along-light volume
    // depth (`extents[0]`) is sized from the object's PERPENDICULAR
    // (projected) radius (`extents[0] ≈ 1.5 × projected_radius`), then the
    // ortho spans `[extents, 3·extents]` from an eye `2·extents` behind the
    // OBB center. NOT a fixed buffer off the along-light extent.
    //
    // Why this is the right shape: the screenspace apply darkens EVERY
    // receiver in the light column behind the occluder, so the volume's far
    // plane is what stops a caster from stamping a phantom second shadow on
    // geometry further down-light (a weapon on a ledge → the wall below).
    // A small/flat caster has a tiny perpendicular radius → short volume →
    // far plane just past its footprint, so the wall below is OUTSIDE the
    // frustum (culled). A tall caster is large perpendicular to a steep
    // sun → deep volume that reaches its ground footprint. (`shadow_falloff`
    // in the apply softens the tail; the far-plane cull does the heavy work.)
    // Size the depth from the object's BOUNDING-SPHERE radius (≈ the AABB's
    // enclosing sphere), matching the engine's `projected_radius` (taken
    // from the object's bounding sphere in `render_lightmap_shadow_calculate_
    // radius`). NOT the perpendicular AABB width: that's tiny for a thin TALL
    // caster (the antenna mast/ladder), which gave it a too-short volume and
    // clipped its footprint ("barely visible"). The sphere radius is large
    // for tall objects (reaches the footprint) and small for compact ones
    // (far plane culls the geometry below a ledge → no phantom).
    let radius = half_l.length();
    let extents = radius * 2.0;
    let center_world = right * center_l.x + up * center_l.y + forward * center_l.z;
    let eye = center_world - forward * (2.0 * extents);
    let view = glam::Mat4::look_at_rh(eye, center_world, up);
    let z_far = 3.0 * extents;
    // Ortho frustum: ±half on the perpendicular axes, [near, z_far] along
    // forward. Lateral padding so silhouettes that bleed past the AABB
    // (alpha-tested foliage etc.) aren't clipped at the frustum sides.
    const LATERAL_BUFFER: f32 = 1.5;
    let proj = glam::Mat4::orthographic_rh(
        -half_l.x * LATERAL_BUFFER, half_l.x * LATERAL_BUFFER,
        -half_l.y * LATERAL_BUFFER, half_l.y * LATERAL_BUFFER,
        0.01, z_far,
    );
    proj * view
}
