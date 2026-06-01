//! Mirror of `Ares/source/render/render_sky.h` (35 lines).
//!
//! Sky scenery rendering. Halo's sky system has two pieces:
//!   1. Per-cluster sky tag — the `_object_source_sky` objects
//!      placed at the BSP cluster level (sky boxes, distant
//!      mountains). These render via
//!      `c_object_renderer::submit_and_render_sky` with a special
//!      offset-by-camera node-matrix transform so they appear
//!      infinitely far.
//!   2. Procedural atmospheric extinction + inscatter — the
//!      sky_atm_parameters tag (already loaded in protomorph) that
//!      modulates everything else's `compute_scattering()`.
//!
//! `c_sky_renderer::setup_sky_ravi_constants` packs the sky's
//! lightprobe SH coefficients into a shader cbuffer slot so any
//! material can sample them as the ambient SH backdrop.

use crate::halo::structures::clusters::ClusterReference;
use crate::halo::geometry::{ModelUniforms, ModelVertex, SkyVertColorVertex};
use crate::halo::render::shared::SharedResources;
use glam::{Mat4, Vec3};
use std::mem::size_of;
use wgpu::util::DeviceExt;

/// Per-frame GPU state for the sky pass: dedicated single-slot
/// model_u + node_matrices buffers + matching bind groups (using
/// the same `model_bgl` / `node_matrices_bgl` as the placement pool
/// so the sky pipeline layout matches BSP shaders bit-for-bit).
///
/// Sky stays out of `shared.model_buffer` / `shared.node_matrices_buffer`
/// (the placement pool) so the game-side `model_data[]` index stays
/// in sync with the renderer's placement-driven model uploads.
pub struct SkyGpu {
    pub model_buffer: wgpu::Buffer,
    pub model_bind_group: wgpu::BindGroup,
    pub node_matrices_buffer: wgpu::Buffer,
    pub node_matrices_bind_group: wgpu::BindGroup,
    /// Dedicated pipeline for the engine sky-shader entry
    /// (`sky_dome_simple_hlsl.hlsl`'s `default_vs` / `default_ps` —
    /// engine `_entry_point_vertex_color_lighting`, idx=14). Built
    /// once in `SkyGpu::new`; consumed by `submit_and_render_sky(1, ...)`
    /// so sky vert-color meshes render through
    /// `vert_color × g_exposure + sun_glare` instead of the generic
    /// `static_sh` material pipeline (which would sample the lightmap
    /// atlas at `uv=(0,0)` for sky verts and return garbage SH).
    /// Variant for opaque meshes (`blend: None`).
    pub pipeline: wgpu::RenderPipeline,
    /// Same shader, alpha-blend variant
    /// (SrcAlpha/OneMinusSrcAlpha, Add). Used by transparent sky
    /// meshes whose authored material is `blend_mode = alpha_blend`
    /// (clouds_cirrus, dust_clouds on snowbound; comparable on shrine,
    /// guardian, etc.). Same vertex layout + bind groups as the
    /// opaque variant — only the color-target blend state differs.
    pub pipeline_alpha_blend: wgpu::RenderPipeline,
    /// Additive variant (One/One, Add) for `blend_mode = additive`
    /// sky materials (aurora-class).
    pub pipeline_additive: wgpu::RenderPipeline,
    /// Multiply variant (Dst/Zero, Add) for `blend_mode = multiply`
    /// (rare; some atmospheric-effect sky materials).
    pub pipeline_multiply: wgpu::RenderPipeline,
    /// Pre-multiplied alpha variant (One/OneMinusSrcAlpha, Add).
    pub pipeline_pre_multiplied_alpha: wgpu::RenderPipeline,
    /// Double-multiply variant (DstColor/SrcColor, Add).
    pub pipeline_double_multiply: wgpu::RenderPipeline,
    /// Empty bind group bound at slot 2 before each sky draw — replaces
    /// whatever per-material BG the previous static-lighting draw left
    /// in slot 2. wgpu validates at draw time that every bound BG's
    /// layout matches the active pipeline layout's corresponding entry,
    /// so we explicitly overwrite the slot to satisfy the check.
    pub empty_bind_group: wgpu::BindGroup,
}

impl SkyGpu {
    /// Pick the sky pipeline matching the material's `blend_mode`
    /// choice. Verbatim engine names from the rmsh rmdf's `blend_mode`
    /// category; the variant call site dispatches on the same string
    /// used by `pipeline_cache::blend_state_for_mode`.
    pub fn pipeline_for_blend_mode(&self, blend_mode: &str) -> &wgpu::RenderPipeline {
        match blend_mode {
            "opaque" => &self.pipeline,
            "alpha_blend" => &self.pipeline_alpha_blend,
            "additive" => &self.pipeline_additive,
            "multiply" => &self.pipeline_multiply,
            "pre_multiplied_alpha" => &self.pipeline_pre_multiplied_alpha,
            "double_multiply" => &self.pipeline_double_multiply,
            n => panic!(
                "[protomorph] sky pipeline missing for blend_mode '{n}' — \
                 add a variant to `SkyGpu` + dispatch arm here. \
                 (per feedback_wgsl_must_mirror_hlsl.md)"
            ),
        }
    }
}

impl SkyGpu {
    pub fn new(shared: &SharedResources) -> Self {
        let device = &shared.device;
        // Model buffer — 1 slot of `ModelUniforms` (model_stride
        // bytes, std140-aligned). Initial content is identity until
        // the per-frame update writes the camera-position translation.
        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky_model_buffer"),
            contents: bytemuck::bytes_of(&ModelUniforms {
                model: Mat4::IDENTITY.to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_model_bg"),
            layout: &shared.model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &model_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<ModelUniforms>() as u64),
                }),
            }],
        });

        // Node matrices buffer — sky models always have 1 node (the
        // root); even multi-node skies (rare) get one identity bone
        // matrix that doesn't move with the camera, so the static
        // identity content is correct for the entire pass lifetime.
        // Allocate node_matrices_stride bytes so the bind group's
        // dynamic_offset=true expectation is satisfied.
        let nm_zeros: Vec<u8> = vec![0u8; shared.node_matrices_stride];
        let mut nm_init = nm_zeros;
        // Identity 4x4 at slot 0.
        let identity = Mat4::IDENTITY.to_cols_array();
        let identity_bytes = bytemuck::cast_slice::<f32, u8>(&identity);
        nm_init[..identity_bytes.len()].copy_from_slice(identity_bytes);
        let node_matrices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky_node_matrices_buffer"),
            contents: &nm_init,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let node_matrices_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_node_matrices_bg"),
            layout: &shared.node_matrices_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &node_matrices_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(shared.node_matrices_stride as u64),
                }),
            }],
        });
        // Dedicated sky pipeline — verbatim port of engine
        // `sky_dome_simple_hlsl.hlsl::default_vs/default_ps`. Bind groups:
        //   - group 0: camera_bgl (only `camera@0` and `engine_exposure@9`
        //     are read; the rest of the BGL goes unused — wgpu permits
        //     pipeline shaders to declare a strict subset of the BGL).
        //   - group 1: model_bgl (`model_u@0`).
        //   - group 2: empty placeholder (sky shader is parameter-less,
        //     but wgpu requires an entry at this index so node_matrices
        //     stays at @group(3)).
        //   - group 3: node_matrices_bgl (`node_matrices@0`).
        // Vertex layout: slot 0 = ModelVertex, slot 1 = SkyVertColorVertex
        // (engine `_vertex_buffer_usage_vert_color`).
        // RT shape mirrors the SL pass: 2 × Rgba16Float. Depth =
        // reverse-Z `GreaterEqual` with `depth_write_enabled: false`
        // matching engine `c_object_renderer::render_static_lighting @
        // 0x1806E4A80:5` z_buffer_mode_read.
        let sky_empty_bgl =
            shared.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sky_empty_group2_bgl"),
                entries: &[],
            });
        let empty_bind_group = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_empty_group2_bg"),
            layout: &sky_empty_bgl,
            entries: &[],
        });
        let sky_shader = shared.device.create_shader_module(wgpu::include_wgsl!(
            "../../../assets/halo_shaders/entry_sky_dome_simple.wgsl"
        ));
        let pipeline_layout =
            shared.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sky_pipeline_layout"),
                bind_group_layouts: &[
                    &shared.camera_bgl,
                    &shared.model_bgl,
                    &sky_empty_bgl,
                    &shared.node_matrices_bgl,
                ],
                immediate_size: 0,
            });
        // One pipeline-construction closure parameterized by blend
        // state — every sky variant uses the same shader, vertex layout,
        // bind groups, primitive state, and depth state. Only the
        // color-target blend differs. Mirrors engine: for the
        // `_entry_point_vertex_color_lighting` (idx=14) entry, the
        // shader is identical for every material; the rmsh
        // `blend_mode` option just toggles the device-state blend.
        let make_pipeline =
            |label: &str, blend: Option<wgpu::BlendState>| -> wgpu::RenderPipeline {
                shared.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &sky_shader,
                        entry_point: Some("default_vs"),
                        buffers: &[ModelVertex::layout(), SkyVertColorVertex::layout()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &sky_shader,
                        entry_point: Some("default_ps"),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend,
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend,
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
                        depth_compare: wgpu::CompareFunction::GreaterEqual,
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };

        let pipeline = make_pipeline("sky_pipeline", None);
        let pipeline_alpha_blend = make_pipeline(
            "sky_pipeline_alpha_blend",
            Some(crate::halo::render_methods::pipeline_cache::blend_state_for_mode("alpha_blend")),
        );
        let pipeline_additive = make_pipeline(
            "sky_pipeline_additive",
            Some(crate::halo::render_methods::pipeline_cache::blend_state_for_mode("additive")),
        );
        let pipeline_multiply = make_pipeline(
            "sky_pipeline_multiply",
            Some(crate::halo::render_methods::pipeline_cache::blend_state_for_mode("multiply")),
        );
        let pipeline_pre_multiplied_alpha = make_pipeline(
            "sky_pipeline_pre_multiplied_alpha",
            Some(crate::halo::render_methods::pipeline_cache::blend_state_for_mode("pre_multiplied_alpha")),
        );
        let pipeline_double_multiply = make_pipeline(
            "sky_pipeline_double_multiply",
            Some(crate::halo::render_methods::pipeline_cache::blend_state_for_mode("double_multiply")),
        );

        Self {
            model_buffer,
            model_bind_group,
            node_matrices_buffer,
            node_matrices_bind_group,
            pipeline,
            pipeline_alpha_blend,
            pipeline_additive,
            pipeline_multiply,
            pipeline_pre_multiplied_alpha,
            pipeline_double_multiply,
            empty_bind_group,
        }
    }

    /// Sky model matrix is identity. Engine-faithful for MCC: dllcache
    /// strips `render_sky_modify_node_matrices @ 0x1806e5b50` to `return 1;`
    /// and `render_object_adjust_skinning_for_sky @ 0x180696d40` has zero
    /// callers. Sky meshes render at authored world coords; the per-frame
    /// camera-translation parallax-lock that Reach's `c_object_renderer
    /// ::adjust_sky_object_position @ 0x82741570` did is dead code in MCC.
    ///
    /// Earlier protomorph commits attempted parallax-lock to fix
    /// "trees moving weirdly when camera moves"; the actual cause was a
    /// duplicate-render bug (sky drawn both via `render_list` and via
    /// `submit_and_render_sky`). With the duplicate gone, identity is
    /// engine-correct — sky stays put in world space as the camera moves.
    pub fn update_model_matrix(&self, _queue: &wgpu::Queue, _camera_position: Vec3) {
        // Identity is initialized in `new`; no per-frame upload needed.
    }
}

/// `c_sky_renderer` (h:14-19, 1B). All-static namespace.
#[derive(Debug, Default)]
pub struct SkyRenderer;

impl SkyRenderer {
    /// `c_sky_renderer::setup_sky_ravi_constants @ h:17`.
    /// Uploads the active sky's SH probe to the ravi cbuffer slot
    /// so all materials can read it. v1 stub.
    pub fn setup_sky_ravi_constants() {
        // todo: pull from `g_lighting_interface.sh_probe_*` and
        // upload via render_method_data set_*_sh_constants.
    }
}

/// `render_sky_cluster_is_visible_sky_cluster @ h:23`. Returns
/// whether the given cluster is part of the sky group (the BSP
/// `cluster_is_sky_bit` flag). v1 stub returns false until
/// cluster sky-flags are tracked.
pub fn render_sky_cluster_is_visible_sky_cluster(_cluster_reference: ClusterReference) -> bool {
    false
}

/// `render_sky_modify_node_matrices @ h:24`. Halo applies an
/// `offset` translation (the camera position) to every sky node
/// matrix so the sky appears infinitely far — when the player
/// moves, the sky stays parallax-anchored to them.
///
/// Returns true on success.
pub fn render_sky_modify_node_matrices(
    offset: Vec3,
    node_matrices: &[Mat4],
    destination_matrices: &mut [Mat4],
) -> bool {
    if node_matrices.len() != destination_matrices.len() {
        return false;
    }
    let translate = Mat4::from_translation(offset);
    for (src, dst) in node_matrices.iter().zip(destination_matrices.iter_mut()) {
        *dst = translate * *src;
    }
    true
}

/// `render_sky_prepare_for_window @ h:25`. Per-window sky setup
/// (e.g. picks the right sky tag for the player window's BSP).
pub fn render_sky_prepare_for_window() {}

/// `render_sky_get_scale @ h:26`. The scale factor applied to sky
/// scenery (sky boxes are typically rendered at 0.1× world scale
/// to fit infinite-distance illusion + reduce z-fighting).
pub fn render_sky_get_scale() -> f32 {
    1.0
}
