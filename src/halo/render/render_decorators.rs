//! `c_decorator_renderer` — GPU-instanced foliage. Renders every
//! `ScenarioDecoratorPlacement` painted across the active level via a
//! single instanced draw call per `.decorator_set`.
//!
//! Engine equivalent: `c_structure_renderer::render_decorators @
//! 0x18068F1B0` (called from `render_albedo`). The engine has 6
//! dedicated decorator shader variants (Wind / DynamicLights / Static /
//! DominantLightOnly / WavyDynamicLights / ShadedDynamicLights — see
//! `blam_tags::decorator_set::RenderShader`); we ship a single v1
//! fallback shader (`entry_decorator.wgsl`) that renders all of them
//! identically with alpha-test cutout. Per-variant ports come later.
//!
//! Data path:
//! - `LoadedScenario::scenario.decorators[block].sets[set].placements[]`
//!   — authoring positions/rotations/scales/tints.
//! - `LoadedScenario::decorator_sets[block][set]` — the .decorator_set
//!   tag (texture path, render_model_path, fade distances, etc).
//! - `LoadedScenario::decorator_render_models[block][set]` — the
//!   .render_model carrying the actual mesh.
//!
//! At scenario load: walk every painted placement, build a per-set
//! instance buffer (matrix + tint), upload the set's mesh + texture
//! to GPU. At render time: one `draw_indexed` per set with
//! `instance_count = placements.len()`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::halo::bitmaps::atlas_decode::DecodedAtlas;
use crate::halo::bitmaps::bitmap_group_2d_get_real_pixel;
use crate::halo::geometry::{
    oct_encode_snorm8, pack_tangent_with_sign, pack_weights_unorm8, ModelVertex,
};
use crate::halo::render::shared::{FrameContext, SharedResources};
use crate::halo::scenario::loader::LoadedScenario;
use blam_tags::decorator_set::RenderShader;
use blam_tags::math::{RealPoint2d, RealQuaternion};

/// Number of `s_decorator_set::e_render_shader` variants — see
/// `blam_tags::decorator_set::RenderShader`. Used to size the per-variant
/// pipeline lookup tables.
const N_DECORATOR_VARIANTS: usize = 6;

/// Scale applied to per-set start/end fade distances. Engine equivalent:
/// `decorator_fade_distance_scale / fov_scale` — `decorator_fade_distance_scale`
/// is read from `s_performance_throttles` at runtime (see
/// `c_structure_renderer::render_decorators @ 0x1806901a0` line 234)
/// and `fov_scale` from `rasterizer_camera_get_fov_scale`. Until those
/// globals plumb in, hardcode 5× so authored tag values like thistle's
/// (7, 12) match typical Halo gameplay foliage ranges of ~35-60 wu.
const DECORATOR_FADE_DISTANCE_SCALE: f32 = 5.0;

fn render_shader_index(s: RenderShader) -> usize {
    match s {
        RenderShader::WindDynamicLights => 0,
        RenderShader::DynamicLights => 1,
        RenderShader::Static => 2,
        RenderShader::DominantLightOnly => 3,
        RenderShader::WavyDynamicLights => 4,
        RenderShader::ShadedDynamicLights => 5,
    }
}

fn render_shader_label(s: RenderShader) -> &'static str {
    match s {
        RenderShader::WindDynamicLights => "WindDynamicLights",
        RenderShader::DynamicLights => "DynamicLights",
        RenderShader::Static => "Static",
        RenderShader::DominantLightOnly => "DominantLightOnly",
        RenderShader::WavyDynamicLights => "WavyDynamicLights",
        RenderShader::ShadedDynamicLights => "ShadedDynamicLights",
    }
}

/// Per-frame wind globals. Drives wind / wavy variants. Engine
/// equivalent: `setup_wind_constants(get_wind_definition(camera_pos),
/// &g_wind_values)` once per frame at the top of `render_decorators`.
///
/// Layout matches the engine's `g_wind_values.{wind_data, wind_data2}`
/// pair — see `s_wind_definition::update @ 0x180907F80` (the per-frame
/// integrator that produces them) and `wind.fx::sample_wind` (the
/// VS-side sampler that consumes them):
///
/// ```text
/// wind_data.xy = (wind_x, wind_y) ÷ 2²⁰   // ring-buffered position [0, 1)
/// wind_data.z  = 1 ÷ decorator_gust_size
/// wind_data.w  = 0
/// wind_data2.xy = bend × (sin(direction), cos(direction))
/// wind_data2.z  = oscillation
/// wind_data2.w  = global time (ticks + frac)
/// ```
///
/// `sample_wind(world_xy)` then computes
/// `texc = world_xy × wind_data.z + wind_data.xy`, samples the noise
/// bitmap (a separate vertex sampler — currently faked with a
/// procedural noise function until the bitmap loader lands), and
/// returns `noise.xy × wind_data2.z + wind_data2.xy`.
///
/// `wave_flow` is a separate per-set tag knob (the WAVY variant's
/// (sin_period_xy, time_coef, vertex_z_coef)) — bound via
/// `DecoratorSetUniforms.wave_flow` (per-set CBV @ slot 0x160005),
/// NOT here in the global per-frame block.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct WindGlobals {
    pub wind_data: [f32; 4],
    pub wind_data2: [f32; 4],
}

/// Per-set uniform — encodes the variant index + tag-driven knobs.
/// Bound at `material_bgl @ 3` (per-set). Read by VS to branch the
/// animation/shading path.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct DecoratorSetUniforms {
    /// `(variant_idx, translucency, contrast_x, contrast_y)`.
    /// variant_idx packed as f32 (0..5 — see `render_shader_index`).
    pub knobs: [f32; 4],
    /// `(start_fade_distance, end_fade_distance, _pad, _pad)` — VS
    /// computes `alpha = saturate((end − dist) / (end − start))` per
    /// placement, fading foliage to transparent (PS discards on
    /// alpha-test) past the cull range. Engine equivalent: per-cluster
    /// `LOD_constants` × camera distance; we apply per-placement so we
    /// don't need cluster grouping.
    pub fade: [f32; 4],
    /// `wave_flow` — per-set WAVY-variant phase parameters. Engine
    /// binds at slot `0x160005` from the decorator_set tag's
    /// `(wavelength X, wavelength Y, wave speed, wave frequency)`
    /// quartet. Used by `decorators.hlsl:237`:
    ///   phase = vertex.z × wave_flow.w
    ///         + wind_data2.w × wave_flow.z
    ///         + dot(inst.xy, wave_flow.xy)
    /// Components:
    ///   .x/.y = world-space spatial period along X/Y (per-instance phase)
    ///   .z    = temporal frequency (waves/sec)
    ///   .w    = vertex.z multiplier (top of decorator sways more)
    /// All zero for non-WAVY sets — the WAVY branch in the shader
    /// is gated on `variant == VARIANT_WAVY_DYNAMIC` so unused.
    pub wave_flow: [f32; 4],
}
use half::f16;

/// Per-instance vertex stream entry. 96 bytes — 4×4 model matrix + RGBA
/// tint + lightmap UV / raycast valid / motion scale. 26K instances →
/// ~2.4 MB. Matches `entry_decorator.wgsl`'s `inst_m0..m3 + inst_tint
/// + inst_lightmap_uv` attribute group at locations 7-12.
///
/// `lightmap_uv` is unused since Phase 9.0 cutover — `placement.tint_color`
/// carries the engine-faithful bake from `light_placement.rs` (which
/// internally calls `c_geometry_sampler::sample` for the 10-sample weighted
/// SH accumulation). Stays in the vertex struct for GPU buffer layout
/// stability; renderer always writes `(0, 0, 0)` and the shader ignores it.
///
/// `lightmap_uv[3]` doubles as the per-placement `motion_scale` (0-1)
/// from `placement.motion_scale: i8` (engine reads as u8 / 255). Used
/// by wind/wavy variants to scale per-vertex displacement.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct DecoratorInstance {
    /// Column-major 4×4 model matrix = `translate(position) * rotate(rotation) * uniform_scale(scale)`.
    pub model: [[f32; 4]; 4],
    /// Per-instance tint color (RGB) + 1.0 alpha. Painted per-placement
    /// in the editor; survives in `ScenarioDecoratorPlacement::tint_color`.
    pub tint: [f32; 4],
    /// `[u, v, raycast_valid_flag, motion_scale]`. raycast_valid = 1.0
    /// when the per-placement vertical raycast hit a BSP triangle (use
    /// atlas SH); 0.0 when missed (shader falls back to global SH).
    /// motion_scale = `placement.motion_scale as u8 / 255.0` — drives
    /// wind/wavy displacement amplitude.
    pub lightmap_uv: [f32; 4],
}

impl DecoratorInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0,  shader_location: 7 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 8 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 9 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 10 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 11 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 80, shader_location: 12 },
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// One (mesh × instance-batch) draw within a decorator set. A set with
/// multiple `decorator_types[k].mesh` values produces one subgroup per
/// distinct mesh — placements are bucketed by their resolved mesh
/// index. Engine equivalent: `c_structure_renderer::render_decorators`
/// dispatches once per (set, mesh) pair via the per-type mesh selector.
/// Lightweight view into a subpart of a decorator render_model — the
/// shared vertex pool + this subpart's triangle-list. Lets the
/// per-subgroup build loop work uniformly whether the source data is
/// the legacy `RenderMesh` (de-stripped whole-mesh) or the per-subpart
/// slice (`DecoratorSubpartGeometry::subpart_indices[k]`).
struct SubpartMeshView<'a> {
    vertices: &'a [blam_tags::render_model::RenderVertex],
    indices: &'a [u32],
}

pub struct DecoratorSubgroup {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
    /// Source render_model mesh index — kept for diagnostics.
    pub mesh_index: u32,
    /// Sorted instance-buffer ranges grouped by their containing
    /// BSP cluster. Each run draws independently with that cluster's
    /// `simple_lights_offset` bound — engine-faithful: a placement
    /// gets the SAME stli lights its enclosing cluster gets, no per-
    /// frame popping. Empty `cluster_runs` is treated as the entire
    /// instance buffer at the default empty slot.
    pub cluster_runs: Vec<DecoratorClusterRun>,
}

/// One per-cluster instance span within a `DecoratorSubgroup`.
pub struct DecoratorClusterRun {
    /// BSP index into `LoadedScenario::active_bsps`. Looked up at
    /// render time via `ctx.structure_renderer.bsps[bsp_idx]`.
    pub bsp_idx: u32,
    /// Cluster index into that BSP's
    /// `BspGpu::cluster_simple_lights_offsets`.
    pub cluster_idx: u32,
    pub instance_offset: u32,
    pub instance_count: u32,
    /// World-space bounding sphere over all placements in this run.
    /// Used at draw time for camera-frustum + far-fade culling — engine
    /// stores the equivalent (sphere_center, sphere_radius) packed into
    /// `decorator_group.compression_offset.xyz` + `.w` per
    /// `c_structure_renderer::render_decorators @ 0x1806901A0` (line
    /// 207). Tested via `visibility_region_test_sphere`.
    pub bounding_center: [f32; 3],
    pub bounding_radius: f32,
}

/// Per-set GPU resources. One of these per
/// `LoadedScenario::decorator_sets[block][set]` that successfully
/// loaded a .decorator_set + .render_model + .bitmap.
pub struct LoadedDecoratorSet {
    /// Per-set bind group: texture + sampler + wind globals + per-set
    /// uniforms. Group 1 of the decorator pipeline.
    pub bind_group: wgpu::BindGroup,
    /// Per-set uniform buffer (variant_idx + translucency + contrast).
    /// Held so the bind group keeps it alive.
    pub set_uniform_buffer: wgpu::Buffer,
    /// Two-sided foliage (no back-face culling). True for cards/leaves;
    /// false for solid props like rocks.
    pub two_sided: bool,
    /// `s_decorator_set::render_shader` — selects the variant pipeline
    /// at draw. Phase 5 ports wind/wavy/dominant/shaded into a single
    /// VS that branches on this index via the per-set uniform buffer.
    pub render_shader: RenderShader,
    /// `s_decorator_set::start_fade_distance` (default 7 m). Distance
    /// at which placements start vertex-collapsing per the engine VS
    /// LOD-fade. Used by `render_into_pass`'s far-distance cull.
    pub start_fade_distance: f32,
    /// `s_decorator_set::end_fade_distance` (default 12 m). Distance
    /// past which placements are fully invisible. Engine cull uses
    /// `(end_fade + sphere_radius) × (1 - early_cull)`.
    pub end_fade_distance: f32,
    /// `s_decorator_set::early_cull` ∈ [0,1] — conservative shrink on
    /// the far cull boundary. Higher values cull groups slightly past
    /// the fade boundary even sooner.
    pub early_cull: f32,
    /// One subgroup per distinct mesh referenced by this set's
    /// placements (via `decorator_set.decorator_types[type_index].mesh`).
    pub subgroups: Vec<DecoratorSubgroup>,
    /// Diagnostic label — the .decorator_set tag path. Not used at
    /// draw; present for debug logs.
    pub label: String,
}

pub struct DecoratorRenderer {
    /// Per-variant × per-cull-mode pipeline cache. Indexed by
    /// `[variant][two_sided as usize]`. Engine renders decorators
    /// ONCE in the albedo pass; pipeline targets `(_surface_accum_HDR,
    /// normal G-buffer)` and the lit color flows through to the SL
    /// target via the albedo→SL copy-blit.
    pub pipelines: [[wgpu::RenderPipeline; 2]; N_DECORATOR_VARIANTS],
    pub material_bgl: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub sets: Vec<LoadedDecoratorSet>,
    /// Per-frame wind globals — uploaded each frame in `record()` from
    /// the scenario clock. Bound at `material_bgl @ 2`.
    pub wind_buffer: wgpu::Buffer,
    /// Engine-faithful wind ring-buffer state — `s_wind_definition::
    /// update @ 0x180907F80` integrates `(freq × sin/cos(direction) ×
    /// dt) × 2²⁰` into a u20 ring buffer per frame. We hold them as
    /// i32 (engine: `int`); `& 0xFFFFF` after each integration step.
    /// Maps to `wind_data.xy = wind_state.xy / 2²⁰` at upload time.
    pub wind_state_x: i32,
    pub wind_state_y: i32,
    /// Engine clock — `s_wind_values.{wind_time_ticks, fractional_wind_time}`.
    /// `t = wind_time_ticks + fractional_wind_time / 65536` in seconds.
    /// Drives the input to direction/speed `TagFunction::evaluate` calls.
    pub wind_time_ticks: i32,
    pub wind_fractional_time: i32,
    /// Cached `total_time` from the prior frame — used to recover dt
    /// from monotonic clock when `g_main_render_timing_data->game_dt`
    /// isn't available (engine path). First-frame dt = 0 → no
    /// integration step taken.
    pub wind_last_total_time: f32,
    /// Resolved wind tag for THIS frame. Set by the per-frame
    /// orchestrator before `update_wind_state` is called. `None` when
    /// neither the active BSP nor the globals fallback authored a wind
    /// tag — wind freezes (engine: same behavior).
    pub current_wind: Option<blam_tags::wind::Wind>,
    /// Wind noise bitmap (`random4_warp.bitmap` by default — BC5/DXN
    /// 128×128 signed 2-channel noise). Sampled in the VS by
    /// `sample_wind`. Engine binds to vertex sampler 0 via
    /// `setup_wind_constants @ 0x180908290`. Replaced at scenario load
    /// when the wind tag's `gust_noise_bitmap` resolves; falls back to
    /// the 1×1 black placeholder (uniform-zero noise → all wind motion
    /// driven by the constant `bend × dir` term in `wind_data2.xy`).
    pub wind_noise_view: wgpu::TextureView,
    pub wind_noise_sampler: wgpu::Sampler,
    /// Held to keep the GPU texture alive for as long as the view it
    /// produced. Replaced atomically with the view in `set_wind_noise`.
    pub wind_noise_texture: wgpu::Texture,
}

impl DecoratorRenderer {
    pub fn new(shared: &SharedResources) -> Self {
        let device = &shared.device;

        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decorator_material_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Wind globals — per-frame uniform. VS-only.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            size_of::<WindGlobals>() as u64
                        ),
                    },
                    count: None,
                },
                // Per-set uniforms (variant_idx + translucency + contrast).
                // VS reads variant for branching; PS reads contrast for
                // shaded-light cosine modulation.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            size_of::<DecoratorSetUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // Wind noise bitmap — VS-visible. Engine: vertex sampler
                // 0, BC5/DXN signed 2-channel `random4_warp.bitmap`.
                // Wrap-addressed so the ring-buffer wrap is silent.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let wind_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decorator_wind_globals"),
            size: size_of::<WindGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("decorator_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Wind noise sampler — wrap+linear, no mips. Engine:
        // `set_vertex_sampler_address_mode(0, _wrap, _wrap, _wrap)` +
        // `set_vertex_sampler_filter_mode(0, _bilinear)`.
        let wind_noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("decorator_wind_noise_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        // Placeholder 1×1 black texture — replaced by `set_wind_noise`
        // when the scenario's wind tag resolves. Format chosen to match
        // the BC-decoded noise bitmap (signed [-1, 1]); 1×1 black gives
        // `sample_wind` a uniform-zero noise field, so the wind motion
        // collapses to the constant `bend × dir` term — sane fallback.
        let wind_noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("decorator_wind_noise_placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Snorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        shared.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &wind_noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8, 0u8],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(2), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let wind_noise_view = wind_noise_texture.create_view(&Default::default());

        let shader = device.create_shader_module(wgpu::include_wgsl!(
            "../../../assets/halo_shaders/entry_decorator.wgsl"
        ));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decorator_pipeline_layout"),
            bind_group_layouts: &[&shared.camera_bgl, &material_bgl],
            immediate_size: 0,
        });

        // Engine-faithful: decorators render ONLY in the albedo pass
        // (`c_structure_renderer::render_decorators @ 0x1806901A0`,
        // called from `c_player_view::render_albedo`). The unified
        // `default_ps` returns a 2-MRT `albedo_pixel` with the LIT
        // color in slot 0 (`_surface_accum_HDR`) and encoded normal
        // + albedo.w in slot 1. The lit color flows through to the
        // SL target via the albedo→SL copy-blit (see
        // `views/player_view.rs::render_static_lighting`).
        let albedo_targets: [Option<wgpu::ColorTargetState>; 2] = [
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];

        let make_pipeline = |label: &str, cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("default_vs"),
                    buffers: &[ModelVertex::layout(), DecoratorInstance::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("default_ps"),
                    targets: &albedo_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    // GreaterEqual (reverse-Z) — `c_rasterizer::set_z_buffer_mode(_z_buffer_mode_write)`
                    // in the engine `render_decorators` head.
                    depth_compare: wgpu::CompareFunction::GreaterEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // Per-variant × per-cull pipeline matrix.
        let pipelines = std::array::from_fn(|variant_idx| {
            let label_one = format!("decorator_pipeline_v{}_one_sided", variant_idx);
            let label_two = format!("decorator_pipeline_v{}_two_sided", variant_idx);
            [
                make_pipeline(&label_one, Some(wgpu::Face::Back)),
                make_pipeline(&label_two, None),
            ]
        });

        Self {
            pipelines,
            material_bgl,
            sampler,
            sets: Vec::new(),
            wind_buffer,
            wind_state_x: 0,
            wind_state_y: 0,
            wind_time_ticks: 0,
            wind_fractional_time: 0,
            wind_last_total_time: 0.0,
            current_wind: None,
            wind_noise_view,
            wind_noise_sampler,
            wind_noise_texture,
        }
    }

    /// Walk the loaded scenario and build per-set GPU resources for every
    /// (decorator_set + render_model + bitmap) chain that resolved.
    /// Sets where any link in the chain is missing get skipped silently
    /// (already logged at scenario load).
    pub fn load_scenario(&mut self, shared: &SharedResources, scenario: &LoadedScenario) {
        self.sets.clear();
        let device = &shared.device;
        let queue = &shared.queue;

        // Phase 9.0 cutover: Raycaster construction dropped. The new bake
        // (`light_placement`) calls `c_geometry_sampler::sample` internally
        // which handles all ray casting through `collision_test_vector` +
        // `geometry_test_collision_result`. No XY-grid BVH anymore.
        //
        // Vertical-UV fallback also dropped — the shader's atlas-sample-at-
        // world_up path was a workaround for missing bake values. With the
        // engine-faithful bake, `placement.tint_color` carries the correct
        // RGB and the renderer encodes it via `hdr_encode` directly. UV
        // stays at (0, 0) and `raycast_valid = 0`.

        // Decode the per-BSP lightprobe atlas + compression scalars. The
        // new bake still needs these — `c_geometry_sampler::sample`'s
        // per-pixel branch reads atlas pixels at the hit's lightmap UV.
        // One-shot at scenario load.
        let bake_resources = decode_bake_resources(scenario);
        if bake_resources.is_empty() {
            eprintln!("[decorators] no atlas+compress decoded; placements use unbaked tint");
        }

        // Build the per-material postprocess sampler lookup that
        // `c_geometry_sampler::sample → sample_diffuse_textures` reads
        // for the bounce-light albedo. Engine equivalent: the global
        // tag-heap registry of `c_render_method` + its postprocess. The
        // guard installs the lookup on the thread-local; dropping it on
        // function exit frees the bitmap data.
        let bake_material_lookup =
            crate::halo::decorators::bake_material_lookup::BakeMaterialLookup::build(
                scenario,
                &scenario.tags_root,
            );
        let _bake_lookup_guard =
            crate::halo::decorators::bake_material_lookup::BakeLookupGuard::install(
                bake_material_lookup,
            );

        // PROTOMORPH_DIAG_GROUND_AT="x,y,z" — one-shot diag: cast a ray
        // straight DOWN from (x, y, z + 1.0) to find the ground hit, then
        // dump the per-pixel atlas SH at that hit. This is what the
        // terrain shader IMPLICITLY reads when rendering a ground pixel
        // at that world location. Used to compare against decorator
        // sample-diag's per-ray SH (same atlas, different hit UVs) to
        // diagnose dim-decorator-on-bright-ground complaints.
        if let Ok(s) = std::env::var("PROTOMORPH_DIAG_GROUND_AT") {
            let parts: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            if parts.len() == 3 {
                eprintln!("[ground-diag] target world XYZ=({:.3},{:.3},{:.3})", parts[0], parts[1], parts[2]);
                // Dump the per-BSP compression vectors that gate every SH
                // coef. Atlas SH formula = `(p_a + p_b - 1) × 2 × cv[2k].x`
                // — if cv is small, all SH values come out small regardless
                // of underlying atlas pixel content. Riverworld's atlas
                // appears uniformly dim per the diags; cv being undersized
                // would explain it.
                if let Some(bsp) = scenario.active_bsps.first() {
                    if let Some(lbsp) = &bsp.lightmap {
                        let cvs = &lbsp.compression_vectors;
                        eprintln!(
                            "[ground-diag] compression_vectors len={} (need ≥9 for full SH chain)",
                            cvs.len(),
                        );
                        for (i, v) in cvs.iter().take(20).enumerate() {
                            eprintln!(
                                "  cv[{i}] = ({:.4}, {:.4}, {:.4})",
                                v.i, v.j, v.k,
                            );
                        }
                    }
                }
                if let Some(ba) = bake_resources.first() {
                    use crate::halo::geometry::geometry_sampling::{
                        geometry_test_vector, sample as c_geometry_sampler_sample,
                        GeometrySample, GeometrySamplerOutcome, GeometryTestResult,
                    };
                    use blam_tags::math::{RealPoint3d, RealVector3d};

                    // Cast geometry_test_vector ourselves to grab lightmap_uv
                    // for the hit, so we can dump raw atlas pixel data
                    // separately from the SH evaluation pipeline.
                    let ray_start_probe = RealPoint3d { x: parts[0], y: parts[1], z: parts[2] + 1.0 };
                    let ray_dir_probe = RealVector3d { i: 0.0, j: 0.0, k: -10.0 };
                    let mut probe_test = GeometryTestResult::default();
                    let _ = geometry_test_vector(
                        scenario, ray_start_probe, ray_dir_probe,
                        true, -1, false, false, true, &mut probe_test,
                    );
                    let lm_uv = probe_test.lightmap_uv;
                    eprintln!(
                        "[ground-diag] lightmap_uv at hit: ({:.4}, {:.4})",
                        lm_uv.x, lm_uv.y,
                    );
                    let raw_uv = RealPoint2d { x: lm_uv.x, y: lm_uv.y };
                    // SH coef index 0 = atlas image_index 0 + 1.
                    let pa = bitmap_group_2d_get_real_pixel(
                        &ba.sh, 0, &raw_uv, 0, true,
                    );
                    let pb = bitmap_group_2d_get_real_pixel(
                        &ba.sh, 1, &raw_uv, 0, true,
                    );
                    eprintln!(
                        "[ground-diag] raw atlas pixel_a img0: R={:.4} G={:.4} B={:.4} A={:.4}",
                        pa.red, pa.green, pa.blue, pa.alpha,
                    );
                    eprintln!(
                        "[ground-diag] raw atlas pixel_b img1: R={:.4} G={:.4} B={:.4} A={:.4}",
                        pb.red, pb.green, pb.blue, pb.alpha,
                    );
                    let cv0 = scenario.active_bsps.first()
                        .and_then(|b| b.lightmap.as_ref())
                        .and_then(|l| l.compression_vectors.first())
                        .map(|v| v.i).unwrap_or(0.0);
                    let alpha_sq = pa.alpha * pb.alpha;
                    let signed_rgb = (pa.red + pb.red - 1.0,
                                      pa.green + pb.green - 1.0,
                                      pa.blue + pb.blue - 1.0);
                    eprintln!(
                        "[ground-diag] decode: alpha² = {:.4}, signed_value_RGB = ({:.4}, {:.4}, {:.4}), cv[0] = {:.4}",
                        alpha_sq, signed_rgb.0, signed_rgb.1, signed_rgb.2, cv0,
                    );
                    eprintln!(
                        "[ground-diag] sh_r[0] (alpha-gated) = signed × 2 × alpha² × cv = {:.4}",
                        signed_rgb.0 * 2.0 * alpha_sq * cv0,
                    );
                    eprintln!(
                        "[ground-diag] sh_r[0] (alpha-IGNORED) = signed × 2 × cv = {:.4}  ← what we'd get without alpha gating",
                        signed_rgb.0 * 2.0 * cv0,
                    );

                    let ray_start = RealPoint3d { x: parts[0], y: parts[1], z: parts[2] + 1.0 };
                    let ray_dir = RealVector3d { i: 0.0, j: 0.0, k: -10.0 };
                    let mut sample = GeometrySample::default();
                    let outcome = c_geometry_sampler_sample(
                        scenario,
                        Some(&ba.sh),
                        ba.intensity.as_ref(),
                        /*lightprobe_pixels_per_probe*/ 8,
                        /*dominant_pixels_per_probe*/ 2,
                        ray_start, ray_dir,
                        /*ignore_all_objects*/ true,
                        /*ignore_object_index*/ -1,
                        /*light_probe*/ true,
                        /*diffuse*/ true,
                        &mut sample,
                    );
                    eprintln!(
                        "[ground-diag] outcome={outcome:?} hit=({:.3},{:.3},{:.3}) normal=({:.3},{:.3},{:.3})",
                        sample.sample_point.x, sample.sample_point.y, sample.sample_point.z,
                        sample.normal.i, sample.normal.j, sample.normal.k,
                    );
                    eprintln!(
                        "[ground-diag] sh_r[0..4]=({:.4},{:.4},{:.4},{:.4})",
                        sample.light_probe_r[0], sample.light_probe_r[1],
                        sample.light_probe_r[2], sample.light_probe_r[3],
                    );
                    eprintln!(
                        "[ground-diag] sh_g[0..4]=({:.4},{:.4},{:.4},{:.4})",
                        sample.light_probe_g[0], sample.light_probe_g[1],
                        sample.light_probe_g[2], sample.light_probe_g[3],
                    );
                    eprintln!(
                        "[ground-diag] sh_b[0..4]=({:.4},{:.4},{:.4},{:.4})",
                        sample.light_probe_b[0], sample.light_probe_b[1],
                        sample.light_probe_b[2], sample.light_probe_b[3],
                    );
                    eprintln!(
                        "[ground-diag] diffuse=({:.4},{:.4},{:.4})",
                        sample.diffuse.i, sample.diffuse.j, sample.diffuse.k,
                    );
                    // Terrain shader ultimately evaluates ravi(N, SH) at the
                    // pixel's interpolated normal for the ambient term. Dump
                    // that for an up-normal (the closest reference to "ground
                    // shaded at a horizontal surface") so the user can compare
                    // a single scalar against the decorator's bake bright.
                    let n_up = glam::Vec3::Z;
                    let sh_r9: &[f32; 9] = (&sample.light_probe_r[..9]).try_into().unwrap();
                    let sh_g9: &[f32; 9] = (&sample.light_probe_g[..9]).try_into().unwrap();
                    let sh_b9: &[f32; 9] = (&sample.light_probe_b[..9]).try_into().unwrap();
                    let ambient_r = crate::halo::decorators::light_placement::ravi_order_3(n_up, sh_r9).max(0.0);
                    let ambient_g = crate::halo::decorators::light_placement::ravi_order_3(n_up, sh_g9).max(0.0);
                    let ambient_b = crate::halo::decorators::light_placement::ravi_order_3(n_up, sh_b9).max(0.0);
                    eprintln!(
                        "[ground-diag] ravi(+Z, SH) ambient_only=({:.4},{:.4},{:.4})  — what the bake's plain ravi computes",
                        ambient_r, ambient_g, ambient_b,
                    );
                    eprintln!(
                        "[ground-diag] dom_dir=({:.3},{:.3},{:.3}) dom_intensity=({:.4},{:.4},{:.4})",
                        sample.dominant_light_dir.i, sample.dominant_light_dir.j, sample.dominant_light_dir.k,
                        sample.dominant_light_intensity.red,
                        sample.dominant_light_intensity.green,
                        sample.dominant_light_intensity.blue,
                    );
                    // Mirror of `ravi_order_2_with_dominant_light` (terrain
                    // shader's actual diffuse evaluator). Subtract the dom
                    // contribution from SH L1, ravi the remainder, then add
                    // back analytical Lambertian `max(0, N·L) × intensity × 0.281`.
                    // Done at the diag layer only (no dependency on the
                    // shader's matrix-packed cbuffer layout).
                    let dom_dir = glam::Vec3::new(
                        sample.dominant_light_dir.i,
                        sample.dominant_light_dir.j,
                        sample.dominant_light_dir.k,
                    );
                    let dom_int = glam::Vec3::new(
                        sample.dominant_light_intensity.red,
                        sample.dominant_light_intensity.green,
                        sample.dominant_light_intensity.blue,
                    );
                    let dom_norm = dom_dir.length();
                    let n_dot_l_pos =
                        if dom_norm > 1e-5 { (dom_dir / dom_norm).dot(n_up).max(0.0) } else { 0.0 };
                    let analytical_lambert = dom_int * n_dot_l_pos * 0.281;
                    eprintln!(
                        "[ground-diag] analytical sun lambert (+Z)=({:.4},{:.4},{:.4})  — the ~0.281×N·L term the bake MISSES",
                        analytical_lambert.x, analytical_lambert.y, analytical_lambert.z,
                    );
                    eprintln!(
                        "[ground-diag] TOTAL terrain diffuse ≈ ambient + sun = ({:.4},{:.4},{:.4})",
                        ambient_r + analytical_lambert.x,
                        ambient_g + analytical_lambert.y,
                        ambient_b + analytical_lambert.z,
                    );
                    let _ = outcome;
                }
            } else {
                eprintln!("[ground-diag] PROTOMORPH_DIAG_GROUND_AT needs 3 floats: x,y,z");
            }
        }

        let mut total_placements = 0usize;
        let mut total_hits = 0usize;
        let mut total_baked = 0usize;

        for (block_idx, block) in scenario.scenario.decorators.iter().enumerate() {
            // `decorators[block_idx]` parallels `scenario.structure_bsps[block_idx]`.
            // Find which active BSP that's mapped to (zone-set pruning
            // may have dropped some). When found, snapshot each
            // cluster's AABB so each placement can be assigned to a
            // containing cluster — the per-cluster `simple_lights_offset`
            // is then bound at draw time. When the BSP isn't active
            // (off-zone), placements still render but with the empty
            // simple_lights slot.
            let active_bsp_pos = scenario
                .active_bsps
                .iter()
                .position(|b| b.scenario_bsp_index == block_idx);
            #[derive(Clone, Copy)]
            struct ClusterAabb { min: glam::Vec3, max: glam::Vec3 }
            let cluster_aabbs: Vec<ClusterAabb> = match active_bsp_pos {
                Some(i) => scenario.active_bsps[i]
                    .sbsp
                    .clusters
                    .iter()
                    .map(|c| ClusterAabb {
                        min: glam::Vec3::new(c.bounds_x.lower, c.bounds_y.lower, c.bounds_z.lower),
                        max: glam::Vec3::new(c.bounds_x.upper, c.bounds_y.upper, c.bounds_z.upper),
                    })
                    .collect(),
                None => Vec::new(),
            };
            let bsp_idx_for_runtime = active_bsp_pos.map(|i| i as u32);

            // Returns the cluster index whose AABB contains `pos`, or
            // the nearest cluster's center as a fallback.
            let find_cluster = |pos: glam::Vec3| -> Option<u32> {
                if cluster_aabbs.is_empty() {
                    return None;
                }
                for (i, aabb) in cluster_aabbs.iter().enumerate() {
                    if pos.cmpge(aabb.min).all() && pos.cmple(aabb.max).all() {
                        return Some(i as u32);
                    }
                }
                let mut best = (f32::INFINITY, 0u32);
                for (i, aabb) in cluster_aabbs.iter().enumerate() {
                    let center = (aabb.min + aabb.max) * 0.5;
                    let d2 = (pos - center).length_squared();
                    if d2 < best.0 {
                        best = (d2, i as u32);
                    }
                }
                Some(best.1)
            };

            for (set_idx, set_entry) in block.sets.iter().enumerate() {
                if set_entry.placements.is_empty() {
                    continue;
                }
                let Some(loaded_set) = scenario
                    .decorator_sets
                    .get(block_idx)
                    .and_then(|sets| sets.get(set_idx))
                    .and_then(|t| t.as_ref())
                else {
                    continue;
                };
                let Some(rm) = scenario
                    .decorator_render_models
                    .get(block_idx)
                    .and_then(|rms| rms.get(set_idx))
                    .and_then(|t| t.as_ref())
                else {
                    continue;
                };
                let Some(subpart_geom) = scenario
                    .decorator_subpart_geometry
                    .get(block_idx)
                    .and_then(|g| g.get(set_idx))
                    .and_then(|t| t.as_ref())
                else {
                    continue;
                };
                if subpart_geom.vertices.is_empty()
                    || subpart_geom.subpart_indices.is_empty()
                {
                    continue;
                }

                // Resolve per-decorator-type → render_model subpart
                // index by string-id chain:
                //   decorator_types[T].mesh
                //     → decorator_set.render_model_instance_names[mesh]
                //     → string_id
                //     → render_model.instance_placements.iter()
                //          .position(|ip| ip.name == string_id)
                //
                // The result aligns with `subpart_geom.subpart_indices`
                // (which is parallel to render_model.instance_placements
                // by construction in `extract_decorator_subparts`).
                // Engine: this is the same resolution chain
                // `c_structure_renderer::render_decorators @ 0x1806901A0`
                // does CPU-side per draw.
                let type_to_subpart: Vec<Option<usize>> = loaded_set
                    .decorator_types
                    .iter()
                    .map(|dt| {
                        let mesh_idx = dt.mesh;
                        if mesh_idx < 0 { return None; }
                        let mesh_idx = mesh_idx as usize;
                        let name = loaded_set
                            .render_model_instance_names
                            .get(mesh_idx)?;
                        rm.instance_placements
                            .iter()
                            .position(|ip| ip.name == *name)
                    })
                    .collect();

                // Bucket placements by their resolved subpart index.
                // Falls back to subpart 0 when the type chain doesn't
                // resolve, but counts those — high fallback rates point
                // to a string-id mismatch.
                let mut placements_by_subpart: std::collections::BTreeMap<
                    u32,
                    Vec<usize>,
                > = std::collections::BTreeMap::new();
                let mut fallback_count = 0usize;
                for (i, p) in set_entry.placements.iter().enumerate() {
                    let raw_t = p.type_index as i32;
                    let resolved = if raw_t >= 0
                        && (raw_t as usize) < type_to_subpart.len()
                    {
                        type_to_subpart[raw_t as usize]
                    } else {
                        None
                    };
                    let subpart_idx = match resolved {
                        Some(sp) if sp < subpart_geom.subpart_indices.len() => sp as u32,
                        _ => {
                            fallback_count += 1;
                            0
                        }
                    };
                    placements_by_subpart
                        .entry(subpart_idx)
                        .or_default()
                        .push(i);
                }
                if placements_by_subpart.is_empty() {
                    continue;
                }
                if fallback_count > 0 {
                    eprintln!(
                        "[decorators]   {} type_index→subpart fallbacks for set '{}' (string-id mismatch?)",
                        fallback_count, loaded_set.render_model_path,
                    );
                }

                let mut subgroups: Vec<DecoratorSubgroup> = Vec::new();
                for (subpart_idx, placements) in &placements_by_subpart {
                    // Each subgroup carries the SHARED vertex pool +
                    // the per-subpart triangle-list. Same vertex_buffer
                    // contents across subgroups in a set; per-subpart
                    // uniqueness comes from the index buffer.
                    let Some(subpart_tri_list) =
                        subpart_geom.subpart_indices.get(*subpart_idx as usize)
                    else {
                        continue;
                    };
                    if subpart_tri_list.is_empty() {
                        continue;
                    }

                    // Per-subpart pose (`render_model.instance_placements[i]`):
                    // each subpart has its own (`forward`, `left`, `up`)
                    // basis + `position` + `scale` that lifts the shared
                    // mesh's vertices into the subpart's authored frame.
                    // The engine cache-builder bakes this into the
                    // per-instance position+quaternion at compile time;
                    // the runtime VS only does
                    // `world = R(inst_quat)*local + inst_pos` (no per-
                    // subpart logic). We compose the two transforms here
                    // CPU-side per `c_decorator_renderer::initialize`.
                    // Without this, subparts authored with non-identity
                    // anchor pose (e.g. a fern subpart whose local origin
                    // sits at its bounding-box center) sink into the
                    // ground because the placement.position lands the
                    // mesh's local origin (= geometric center) on the
                    // surface instead of the subpart's authored anchor
                    // (= bottom).
                    let subpart_pose: Mat4 = match rm
                        .instance_placements
                        .get(*subpart_idx as usize)
                    {
                        Some(ip) => {
                            let basis = glam::Mat3::from_cols(
                                Vec3::new(ip.forward.i, ip.forward.j, ip.forward.k),
                                Vec3::new(ip.left.i, ip.left.j, ip.left.k),
                                Vec3::new(ip.up.i, ip.up.j, ip.up.k),
                            );
                            let scale = if ip.scale > 0.0 { ip.scale } else { 1.0 };
                            let translation = Vec3::new(
                                ip.position.x,
                                ip.position.y,
                                ip.position.z,
                            );
                            Mat4::from_translation(translation)
                                * Mat4::from_mat3(basis * scale)
                        }
                        None => Mat4::IDENTITY,
                    };
                    // Stand-in for the rest of the loop's mesh references.
                    // The loop body below originally consumed `mesh.vertices`
                    // + `mesh.indices`; we redirect those to the subpart
                    // geometry without changing the rest of the per-subgroup
                    // build (vertex buffer build, instance buffer build, etc).
                    let mesh = SubpartMeshView {
                        vertices: &subpart_geom.vertices,
                        indices: subpart_tri_list,
                    };
                    let mesh = &mesh;
                    let mesh_idx = subpart_idx;

                    // Engine `light_placement @ tool.exe sub_140C4AF00:151`
                    // reads `render_model.render_geometry.compression_info[0]`
                    // (hardcoded element 0) for the bake's mesh extent.
                    // ALL placements of a decorator_set use the SAME bounds
                    // — the overall render_model's, NOT per-subpart vertex
                    // AABBs. For multi-subpart decorator sets (small / med /
                    // large fern variants) this means every variant gets the
                    // same `radius` / `r_threshold` / sample positions,
                    // matching what tool.exe writes into the cached bake.
                    //
                    // Earlier per-subpart vertex-AABB approach produced
                    // variable ray lengths per subpart → small subparts'
                    // shorter rays missed the floor on rotated placements
                    // → drop / unlit fallback → "all over the place" pattern
                    // on riverworld. Verified divergence 2026-05-26.
                    //
                    // TEST REVERT 2026-05-26 (later same day): the
                    // compression_info[0] path made tall multi-subpart fern
                    // sets bake dim & sky-biased (bright ≈ 0.07-0.12 vs
                    // thistle/clover ≈ 0.50). Going back to per-subpart
                    // vertex AABB to verify ferns light correctly when each
                    // variant samples its own true extent.
                    let subpart_aabb: crate::halo::decorators::light_placement::MeshBounds = {
                        let _ = &rm.compression_info_0; // unused on this path
                        let mut mn = Vec3::splat(f32::INFINITY);
                        let mut mx = Vec3::splat(f32::NEG_INFINITY);
                        for &i in subpart_tri_list.iter() {
                            if let Some(v) = subpart_geom.vertices.get(i as usize) {
                                let local = Vec3::new(v.position.x, v.position.y, v.position.z);
                                let lifted = subpart_pose.transform_point3(local);
                                mn = mn.min(lifted);
                                mx = mx.max(lifted);
                            }
                        }
                        if mn.x.is_finite() {
                            crate::halo::decorators::light_placement::MeshBounds { min: mn, max: mx }
                        } else {
                            crate::halo::decorators::light_placement::MeshBounds {
                                min: Vec3::splat(-0.5),
                                max: Vec3::splat(0.5),
                            }
                        }
                    };

                    let verts: Vec<ModelVertex> =
                        mesh.vertices.iter().map(convert_decorator_vertex).collect();
                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("decorator_vb"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("decorator_ib"),
                            contents: bytemuck::cast_slice(&mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                    // Sort placements by their containing cluster so the
                    // emitted instance buffer has contiguous per-cluster
                    // ranges. Without `bsp_idx_for_runtime` (off-zone
                    // BSP), every placement falls into a single
                    // "no-cluster" bucket that uses the empty slot.
                    let mut placements_with_cluster: Vec<(usize, u32)> = placements
                        .iter()
                        .map(|&p_ref| {
                            let p = &set_entry.placements[p_ref];
                            let pos = glam::Vec3::new(p.position.x, p.position.y, p.position.z);
                            // Sentinel `u32::MAX` = no-cluster; sorts
                            // last so it forms a single trailing run.
                            let c = find_cluster(pos).unwrap_or(u32::MAX);
                            (p_ref, c)
                        })
                        .collect();
                    placements_with_cluster.sort_by_key(|(_, c)| *c);

                    let mut instances: Vec<DecoratorInstance> =
                        Vec::with_capacity(placements_with_cluster.len());
                    let mut cluster_runs: Vec<DecoratorClusterRun> = Vec::new();
                    let mut current_cluster: Option<u32> = None;
                    let mut current_run_start: u32 = 0;
                    // Track per-run bounding box; flush converts to a
                    // bounding sphere stored on the run for camera-frustum +
                    // far-fade culling at draw time.
                    let mut run_aabb_min = glam::Vec3::splat(f32::INFINITY);
                    let mut run_aabb_max = glam::Vec3::splat(f32::NEG_INFINITY);
                    let mut sum_decoded = glam::Vec3::ZERO;
                    let mut decoded_count = 0usize;
                    let mut sum_fallback = glam::Vec3::ZERO;
                    let mut fallback_count = 0usize;
                    // PROTOMORPH_DIAG_BAKE_HIST=1 — per-set alpha + black-output
                    // histogram (2026-05-26 chain-audit follow-up). Confirms
                    // whether the stubbed-m_diffuse hypothesis explains the
                    // dark ferns: high-alpha placements blend toward
                    // m_diffuse=0 → near-black final_rgb.
                    let hist_diag = std::env::var("PROTOMORPH_DIAG_BAKE_HIST")
                        .ok()
                        .as_deref()
                        == Some("1");
                    let mut hist_alpha_zero = 0usize;
                    let mut hist_alpha_low = 0usize;
                    let mut hist_alpha_high = 0usize;
                    let mut hist_alpha_one = 0usize;
                    let mut hist_black_decoded = 0usize;
                    let mut hist_dropped = 0usize;
                    let mut hist_sum_bright = glam::Vec3::ZERO;
                    let mut hist_sum_albedo = glam::Vec3::ZERO;
                    let mut hist_sum_hits = 0u64;
                    let mut hist_min_hits = 11u8;
                    let mut hist_bright_count = 0usize;
                    let hist_branch_snapshot_before = if hist_diag {
                        crate::halo::geometry::geometry_sampling::DIAG_LIGHTPROBE_BRANCH_COUNTS
                            .with(|c| *c.borrow())
                    } else { [0u64; 4] };
                    // Engine `cluster_run.bounding_sphere_radius` is pre-padded
                    // at publish time to enclose blade GEOMETRY around placement
                    // points. Our spheres come from placement-position AABBs
                    // only; under camera-frustum cull (the walker isn't
                    // producing a usable region yet) a tight sphere clips
                    // off-axis blades. ×4 pad empirically covers riverworld
                    // foliage. REMOVE once the projections walker reliably
                    // populates `region.clusters[]` for every visible cluster.
                    const DECORATOR_RADIUS_PAD: f32 = 4.0;
                    let make_sphere = |min: glam::Vec3, max: glam::Vec3| -> ([f32; 3], f32) {
                        let center = (min + max) * 0.5;
                        let radius = (max - center).length().max(0.25) * DECORATOR_RADIUS_PAD;
                        ([center.x, center.y, center.z], radius)
                    };
                    for &(p_ref, cluster_idx) in &placements_with_cluster {
                        // Emit a run when the cluster_idx changes.
                        match current_cluster {
                            None => {
                                current_cluster = Some(cluster_idx);
                                current_run_start = instances.len() as u32;
                                run_aabb_min = glam::Vec3::splat(f32::INFINITY);
                                run_aabb_max = glam::Vec3::splat(f32::NEG_INFINITY);
                            }
                            Some(prev) if prev != cluster_idx => {
                                let count = instances.len() as u32 - current_run_start;
                                if count > 0 {
                                    if let Some(bsp_idx) = bsp_idx_for_runtime {
                                        // Skip the sentinel "no-cluster"
                                        // bucket — it gets emitted at
                                        // the empty slot via a final
                                        // run with cluster_idx = MAX.
                                        let (center, radius) =
                                            make_sphere(run_aabb_min, run_aabb_max);
                                        cluster_runs.push(DecoratorClusterRun {
                                            bsp_idx,
                                            cluster_idx: prev,
                                            instance_offset: current_run_start,
                                            instance_count: count,
                                            bounding_center: center,
                                            bounding_radius: radius,
                                        });
                                    }
                                }
                                current_cluster = Some(cluster_idx);
                                current_run_start = instances.len() as u32;
                                run_aabb_min = glam::Vec3::splat(f32::INFINITY);
                                run_aabb_max = glam::Vec3::splat(f32::NEG_INFINITY);
                            }
                            _ => {}
                        }
                        let p = &set_entry.placements[p_ref];
                        total_placements += 1;
                        let world_pos = glam::Vec3::new(p.position.x, p.position.y, p.position.z);
                        run_aabb_min = run_aabb_min.min(world_pos);
                        run_aabb_max = run_aabb_max.max(world_pos);

                        // Phase 9.0: vertical-UV fallback dropped along
                        // with the Raycaster. UV stays at (0, 0) and
                        // `raycast_valid = 0` — the new engine-faithful
                        // bake makes the shader's atlas-sample-at-world_up
                        // workaround unnecessary.
                        let uv = [0.0_f32, 0.0_f32];
                        let raycast_valid = 0.0_f32;

                        // Engine-faithful bake: `light_placement` @ tool.exe
                        // `sub_140C4AF00`. Casts 10 sample rays via
                        // `c_geometry_sampler::sample` over the placement's
                        // sample-pattern table (Ground/Hanging) + weighted
                        // SH accumulation + RGBE encode. Returns None when
                        // all 10 rays miss (engine drops the placement).
                        let baked_color = if let Some(ba) = bake_resources.first() {
                            // Use first active BSP's index for the bake.
                            // Multi-BSP scenarios will need per-block→BSP
                            // mapping; Phase 9.1 follow-up.
                            let bake_bsp_idx = bsp_idx_for_runtime.unwrap_or(0) as i32;
                            crate::halo::decorators::light_placement::light_placement(
                                p,
                                loaded_set,
                                subpart_aabb,
                                bake_bsp_idx,
                                scenario,
                                Some(&ba.sh),
                                ba.intensity.as_ref(),
                                /*lightprobe_pixels_per_probe*/ 8,
                                /*dominant_pixels_per_probe*/ 2,
                            )
                        } else {
                            None
                        };
                        let (encoded, valid) = match baked_color {
                            Some(bc) => (bc.hdr_encoded, true),
                            None => (encode_tint_as_unbaked(&p.tint_color), false),
                        };
                        let decoded = {
                            // Inline RGBE decode for stats (mirror of the
                            // shader's `tint.rgb × exp2(tint.a × 63.75 − 31.75)`).
                            let r = encoded[0] as f32 / 255.0;
                            let g = encoded[1] as f32 / 255.0;
                            let b = encoded[2] as f32 / 255.0;
                            let a = encoded[3] as f32 / 255.0;
                            let exp = (a * 63.75 - 31.75).exp2();
                            glam::Vec3::new(r, g, b) * exp
                        };
                        if valid {
                            total_baked += 1;
                            sum_decoded += decoded;
                            decoded_count += 1;
                        } else {
                            sum_fallback += decoded;
                            fallback_count += 1;
                        }
                        if hist_diag {
                            // PROTOMORPH_DIAG_DUMP_POSITIONS=<substring>
                            // dumps each placement's world position +
                            // brightness for matching decorator sets.
                            // Used 2026-05-26 to find adjacent fern/grass
                            // pairs for direct same-position comparison.
                            if let Ok(name_filter) = std::env::var("PROTOMORPH_DIAG_DUMP_POSITIONS") {
                                if set_entry.decorator_set.contains(&name_filter) {
                                    let b = baked_color
                                        .map(|bc| (bc.diag_brightness.x, bc.diag_brightness.y, bc.diag_brightness.z, bc.diag_sample_hits))
                                        .unwrap_or((-1.0, -1.0, -1.0, 0));
                                    eprintln!(
                                        "[pos-dump] {} placement={} pos=({:.2},{:.2},{:.2}) bright=({:.4},{:.4},{:.4}) hits={}",
                                        set_entry.decorator_set,
                                        p_ref,
                                        p.position.x, p.position.y, p.position.z,
                                        b.0, b.1, b.2, b.3,
                                    );
                                }
                            }
                            let alpha = (p.ground_tint as i32 & 0xFF) as f32 / 255.0;
                            if alpha <= 0.0 {
                                hist_alpha_zero += 1;
                            } else if alpha < 0.5 {
                                hist_alpha_low += 1;
                            } else if alpha < 0.999 {
                                hist_alpha_high += 1;
                            } else {
                                hist_alpha_one += 1;
                            }
                            if !valid {
                                hist_dropped += 1;
                            } else if decoded.length_squared() < 1e-4 {
                                hist_black_decoded += 1;
                            }
                            if let Some(bc) = baked_color {
                                hist_sum_bright += bc.diag_brightness;
                                hist_sum_albedo += bc.diag_albedo;
                                hist_sum_hits += bc.diag_sample_hits as u64;
                                if bc.diag_sample_hits < hist_min_hits {
                                    hist_min_hits = bc.diag_sample_hits;
                                }
                                hist_bright_count += 1;
                            }
                        }
                        instances.push(build_instance(
                            p,
                            uv,
                            raycast_valid,
                            encoded,
                            None, // motion_scale_override deferred (was an L8 DominantLightOnly tweak)
                            subpart_pose,
                        ));
                    }
                    // Flush the final cluster run.
                    if let (Some(prev), Some(bsp_idx)) = (current_cluster, bsp_idx_for_runtime) {
                        if prev != u32::MAX {
                            let count = instances.len() as u32 - current_run_start;
                            if count > 0 {
                                let (center, radius) =
                                    make_sphere(run_aabb_min, run_aabb_max);
                                cluster_runs.push(DecoratorClusterRun {
                                    bsp_idx,
                                    cluster_idx: prev,
                                    instance_offset: current_run_start,
                                    instance_count: count,
                                    bounding_center: center,
                                    bounding_radius: radius,
                                });
                            }
                        }
                    }
                    // Per-set summary — useful for verifying the bake
                    // is producing engine-faithful values when porting
                    // new maps. Comment out if log volume gets noisy.
                    let avg_baked = if decoded_count > 0 {
                        sum_decoded / decoded_count as f32
                    } else { glam::Vec3::ZERO };
                    let avg_fb = if fallback_count > 0 {
                        sum_fallback / fallback_count as f32
                    } else { glam::Vec3::ZERO };
                    eprintln!(
                        "[decorators]   {} avg ambient: baked=({:.4},{:.4},{:.4}) over {}, fallback=({:.4},{:.4},{:.4}) over {}",
                        set_entry.decorator_set,
                        avg_baked.x, avg_baked.y, avg_baked.z, decoded_count,
                        avg_fb.x, avg_fb.y, avg_fb.z, fallback_count,
                    );
                    if hist_diag {
                        let inv_n = if hist_bright_count > 0 {
                            1.0 / hist_bright_count as f32
                        } else { 0.0 };
                        let avg_b = hist_sum_bright * inv_n;
                        let avg_a = hist_sum_albedo * inv_n;
                        let avg_hits = if hist_bright_count > 0 {
                            hist_sum_hits as f32 / hist_bright_count as f32
                        } else { 0.0 };
                        let branch_after = crate::halo::geometry::geometry_sampling::DIAG_LIGHTPROBE_BRANCH_COUNTS
                            .with(|c| *c.borrow());
                        let pv = branch_after[0] - hist_branch_snapshot_before[0];
                        let pp = branch_after[1] - hist_branch_snapshot_before[1];
                        let sg = branch_after[2] - hist_branch_snapshot_before[2];
                        let nl = branch_after[3] - hist_branch_snapshot_before[3];
                        eprintln!(
                            "[bake-hist] block={} set={} ({}) \
                             n={} alpha[=0={} 0<a<.5={} .5<=a<1={} =1={}] \
                             black_decoded={} dropped={} \
                             avg_bright=({:.4},{:.4},{:.4}) avg_albedo=({:.4},{:.4},{:.4}) \
                             avg_hits={:.2}/10 min_hits={} \
                             render_flags=0x{:02X} vis_gate={} sample_pat={:?} \
                             variant={} \
                             branches[pv={} pp={} single={} none={}]",
                            block_idx,
                            set_idx,
                            set_entry.decorator_set,
                            decoded_count + fallback_count,
                            hist_alpha_zero,
                            hist_alpha_low,
                            hist_alpha_high,
                            hist_alpha_one,
                            hist_black_decoded,
                            hist_dropped,
                            avg_b.x, avg_b.y, avg_b.z,
                            avg_a.x, avg_a.y, avg_a.z,
                            avg_hits,
                            if hist_min_hits == 11 { 0 } else { hist_min_hits },
                            loaded_set.render_flags,
                            loaded_set.dont_sample_lighting_through_geometry(),
                            loaded_set.lighting_sample_pattern,
                            render_shader_label(loaded_set.render_shader),
                            pv, pp, sg, nl,
                        );
                    }
                    let instance_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("decorator_instances"),
                            contents: bytemuck::cast_slice(&instances),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    subgroups.push(DecoratorSubgroup {
                        vertex_buffer,
                        index_buffer,
                        index_count: mesh.indices.len() as u32,
                        instance_buffer,
                        instance_count: instances.len() as u32,
                        mesh_index: *mesh_idx,
                        cluster_runs,
                    });
                }

                if subgroups.is_empty() {
                    continue;
                }

                // Texture upload — same path as BSP / object materials.
                let texture_view = if loaded_set.texture_path.is_empty() {
                    None
                } else {
                    let tex_path = blam_tags::paths::resolve_tag_path(
                        &scenario.tags_root,
                        &loaded_set.texture_path,
                        "bitmap",
                    );
                    crate::halo::bitmaps::load(&tex_path).map(|inline_tex| {
                        crate::halo::structures::bsp_gpu::upload_inline_texture(
                            shared,
                            &inline_tex,
                            /* linear = */ false,
                        )
                    })
                };
                // Skip any set whose texture failed to load — drawing a
                // 1×1 magenta texture would just paint the screen with
                // 26K hot-pink instances.
                let Some(texture_view) = texture_view else {
                    eprintln!(
                        "[decorators] skip {}: bitmap '{}' didn't load",
                        loaded_set.render_model_path, loaded_set.texture_path,
                    );
                    continue;
                };

                // Per-set uniforms — variant + tag-driven shading knobs.
                // `contrast` IS a tag field on `decorator_set` after all
                // (`shaded dark` / `shaded bright`); engine reads it via
                // `c_rasterizer::set_shader_constant(0x190000u, 1, &set->contrast)`.
                // Only ShadedDynamicLights (render_shader == 5) consumes
                // it (`s_decorator_set::decorator_uses_contrast_ratio`
                // returns `render_shader == 5`); for other variants the
                // PS modulation is a no-op via the variant gate.
                //
                // Riverworld values: thistle (Shaded) has (0.9, 0.5) →
                // dark side 90% brightness, sun-facing 140%. Earlier
                // (1, 0) default rendered all Shaded variants flat —
                // identity-ish but not engine-faithful.
                let set_uniforms = DecoratorSetUniforms {
                    knobs: [
                        render_shader_index(loaded_set.render_shader) as f32,
                        loaded_set.translucency,
                        loaded_set.shaded_dark,
                        loaded_set.shaded_bright,
                    ],
                    fade: [
                        loaded_set.start_fade_distance.max(0.1) * DECORATOR_FADE_DISTANCE_SCALE,
                        loaded_set.end_fade_distance
                            .max(loaded_set.start_fade_distance + 0.1)
                            * DECORATOR_FADE_DISTANCE_SCALE,
                        0.0,
                        0.0,
                    ],
                    // Per-set WAVY phase params from `decorator_set` tag.
                    // Engine binds at slot `0x160005` via `setup_wave_flow`.
                    // Components: (wavelength_x, wavelength_y, wave_speed,
                    // wave_frequency). Ignored by non-WAVY variants.
                    wave_flow: [
                        loaded_set.wavelength_x,
                        loaded_set.wavelength_y,
                        loaded_set.wave_speed,
                        loaded_set.wave_frequency,
                    ],
                };
                let set_uniform_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("decorator_set_uniforms"),
                        contents: bytemuck::bytes_of(&set_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("decorator_bg"),
                    layout: &self.material_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.wind_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: set_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.wind_noise_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&self.wind_noise_sampler),
                        },
                    ],
                });

                // Use a stable label for runtime diagnostics.
                let label = if !set_entry.decorator_set.is_empty() {
                    set_entry.decorator_set.clone()
                } else {
                    format!("decorator_set[{}][{}]", block_idx, set_idx)
                };

                let _ = queue;
                let render_shader = loaded_set.render_shader;
                self.sets.push(LoadedDecoratorSet {
                    bind_group,
                    set_uniform_buffer,
                    two_sided: loaded_set.is_two_sided(),
                    render_shader,
                    start_fade_distance: loaded_set.start_fade_distance,
                    end_fade_distance: loaded_set.end_fade_distance,
                    early_cull: loaded_set.early_cull,
                    subgroups,
                    label,
                });
            }
        }

        // Per-set variant breakdown — useful when verifying that the
        // .decorator_set tag's `render_shader` enum is being read
        // correctly and to spot which variants Phase 5 needs to port.
        let mut variant_counts = [0usize; N_DECORATOR_VARIANTS];
        for set in &self.sets {
            variant_counts[render_shader_index(set.render_shader)] += 1;
        }
        for (idx, &count) in variant_counts.iter().enumerate() {
            if count > 0 {
                let label = match idx {
                    0 => "WindDynamicLights",
                    1 => "DynamicLights",
                    2 => "Static",
                    3 => "DominantLightOnly",
                    4 => "WavyDynamicLights",
                    5 => "ShadedDynamicLights",
                    _ => "Unknown",
                };
                eprintln!("[decorators]   variant {}: {} sets", label, count);
            }
        }

        let total_instances: usize = self
            .sets
            .iter()
            .flat_map(|s| s.subgroups.iter().map(|g| g.instance_count as usize))
            .sum();
        let total_subgroups: usize = self.sets.iter().map(|s| s.subgroups.len()).sum();
        eprintln!(
            "[decorators] loaded {} sets, {} subgroups, {} total instances",
            self.sets.len(),
            total_subgroups,
            total_instances,
        );
        if total_placements > 0 {
            eprintln!(
                "[decorators] vertical raycast hit {}/{} placements ({:.1}%); 10-sample bake {} valid ({:.1}%)",
                total_hits,
                total_placements,
                100.0 * total_hits as f32 / total_placements as f32,
                total_baked,
                100.0 * total_baked as f32 / total_placements as f32,
            );
        }
    }

    /// Update per-frame wind/wave globals. Must run before the
    /// render-albedo rpass opens (queue.write_buffer requires the GPU
    /// to be outside a pass). Engine equivalent: `s_wind_definition::
    /// update @ 0x180907F80` followed by `setup_wind_constants @
    /// 0x180908290`, both invoked from `c_structure_renderer::
    /// render_decorators @ 0x1806901A0`'s prologue.
    ///
    /// Engine integration math (verbatim from research):
    /// ```text
    /// fractional_wind_time += dt × 65536
    /// wind_time_ticks      += fractional_wind_time / 65536
    /// fractional_wind_time %= 65536
    /// t = wind_time_ticks + fractional_wind_time / 65536
    /// direction_radians = function_eval(direction, t) × π/180
    /// (sin_dir, cos_dir) = (sin(d), cos(d))
    /// speed = function_eval(speed, t) × 0.14666666 × 0.005
    /// bend  = function_eval(bend, t)
    /// osc   = function_eval(oscillation, t)
    /// freq  = function_eval(frequency, t)
    /// wind_x = ((freq × sin_dir × dt) × 2²⁰ + wind_x) mod 2²⁰
    /// wind_y = ((freq × cos_dir × dt) × 2²⁰ + wind_y) mod 2²⁰
    /// ```
    /// Upload an `InlineTexture` (from `bitmaps::load`) as the wind
    /// noise bitmap and replace the placeholder. Engine equivalent:
    /// `setup_wind_constants @ 0x180908290` looks up the bitmap via
    /// `bitmap_group_get_bitmap_hardware_format` and binds to vertex
    /// sampler 0. Bind groups built BEFORE this call still reference
    /// the old (placeholder) view; new bind groups pick up the
    /// replacement. Call this BEFORE `load_scenario` so the per-set
    /// bind groups bind the real noise texture.
    pub fn set_wind_noise(
        &mut self,
        shared: &SharedResources,
        tex: &crate::halo::render_methods::materials::InlineTexture,
    ) {
        let device = &shared.device;
        let queue = &shared.queue;
        let wgpu_format = crate::halo::render::inline_to_wgpu_format(tex.format, true);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("decorator_wind_noise"),
            size: wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: tex.layers,
            },
            mip_level_count: tex.mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut offset = 0usize;
        for layer in 0..tex.layers {
            for mip in 0..tex.mip_count {
                let mw = (tex.width >> mip).max(1);
                let mh = (tex.height >> mip).max(1);
                let level_bytes = tex.format.level_bytes(mw, mh);
                let (extent_w, extent_h, rows, bytes_per_row);
                if tex.format.is_compressed() {
                    let bw = ((mw + 3) / 4).max(1);
                    let bh = ((mh + 3) / 4).max(1);
                    extent_w = bw * 4;
                    extent_h = bh * 4;
                    rows = bh;
                    bytes_per_row = bw * tex.format.block_bytes();
                } else {
                    extent_w = mw;
                    extent_h = mh;
                    rows = mh;
                    bytes_per_row = mw * tex.format.bytes_per_pixel();
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.data[offset..offset + level_bytes],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rows),
                    },
                    wgpu::Extent3d {
                        width: extent_w,
                        height: extent_h,
                        depth_or_array_layers: 1,
                    },
                );
                offset += level_bytes;
            }
        }
        let view = texture.create_view(&Default::default());
        self.wind_noise_view = view;
        self.wind_noise_texture = texture;
    }

    /// Verbatim port of `s_wind_definition::update @ 0x180907F80`. Reads
    /// the resolved wind tag (set by the per-frame orchestrator into
    /// `self.current_wind`), evaluates the 5 function blocks at
    /// game-clock time `t`, integrates the 20-bit ring buffer, and
    /// uploads `g_wind_values.{wind_data, wind_data2}` to the GPU.
    ///
    /// When `current_wind` is None (no per-bsp wind, no globals
    /// fallback), uploads zero wind state — matches engine behavior
    /// when `get_wind_definition` returns null.
    pub fn update_per_frame_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        total_time: f32,
    ) {
        if self.sets.is_empty() {
            return;
        }

        // dt from the prior call. Engine source = `g_main_render_timing_data->game_dt`
        // — frame delta in seconds, zero on pause. We approximate via
        // wall-clock total_time delta. First-frame dt = 0 (engine
        // accumulator is zero-initialized).
        let mut dt = (total_time - self.wind_last_total_time).max(0.0);
        if self.wind_last_total_time == 0.0 {
            dt = 0.0;
        }
        self.wind_last_total_time = total_time;

        // No wind tag → freeze (engine: get_wind_definition returns null,
        // setup_wind_constants binds default white sampler + last-frame
        // values). Upload neutral data and bail.
        let Some(wind) = self.current_wind.as_ref() else {
            let neutral = WindGlobals {
                wind_data: [0.0, 0.0, 1.0, 0.0],
                wind_data2: [0.0, 0.0, 0.0, total_time],
            };
            queue.write_buffer(&self.wind_buffer, 0, bytemuck::bytes_of(&neutral));
            return;
        };

        // Engine ring-buffer time accumulator (16-bit fractional ticks/sec):
        //   fractional += dt × 65536
        //   if fractional > 65536: ticks += fractional / 65536; fractional %= 65536
        //   t = ticks + fractional / 65536    (seconds)
        let v7 = (dt * 65536.0) as i32 + self.wind_fractional_time;
        self.wind_fractional_time = v7;
        if v7 > 0x10000 {
            self.wind_time_ticks = self.wind_time_ticks.wrapping_add(v7 / 0x10000);
            self.wind_fractional_time = v7 % 0x10000;
        }
        let t = self.wind_fractional_time as f32 * (1.0 / 65536.0)
            + self.wind_time_ticks as f32;

        // Evaluate the 5 function blocks. direction/speed take game-clock
        // `t` as input; bend/osc/freq take normalized speed.
        let dir_deg = wind.direction.evaluate(t, 0.5);
        let dir_rad = dir_deg * std::f32::consts::PI / 180.0;
        let sin_dir = dir_rad.sin();
        let cos_dir = dir_rad.cos();

        let speed_mph_raw = wind.speed.evaluate(t, 0.5);
        // Optional debug amplification — riverworld authors a 7.5 mph
        // breeze (speed_norm ≈ 0.005) which feeds bend/osc/freq curves
        // far below their `speed_norm = 1` headline values. Set
        // `PROTOMORPH_WIND_AMPLIFY=20` to scale 7.5 mph → 150 mph
        // (speed_norm ≈ 0.11) for visibility testing. Off by default
        // (engine value).
        let amp = {
            use std::sync::OnceLock;
            static AMP: OnceLock<f32> = OnceLock::new();
            *AMP.get_or_init(|| {
                std::env::var("PROTOMORPH_WIND_AMPLIFY")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0)
                    .max(0.0)
            })
        };
        let speed_mph = speed_mph_raw * amp;
        // Engine: speed_norm = speed * 0.14666666 * 0.005 = speed / (6.818 * 200).
        // Reach makes it explicit as `(speed/6.818)/200`.
        let speed_norm = speed_mph * 0.14666666 * 0.005;

        let bend = wind.bend.evaluate(speed_norm, 0.5);
        let osc = wind.oscillation.evaluate(speed_norm, 0.5);
        let freq = wind.frequency.evaluate(speed_norm, 0.5);

        // 20-bit fixed-point ring buffer drift. Engine uses signed `int`
        // and wraps via `& 0xFFFFF`. MCC adds (drift IN direction);
        // Reach subtracts. Mirror MCC.
        const TWO_POW_20: f32 = 1_048_576.0;
        const RING_MASK: i32 = 0xFFFFF;
        let dx = (freq * sin_dir * dt * TWO_POW_20) as i32;
        let dy = (freq * cos_dir * dt * TWO_POW_20) as i32;
        self.wind_state_x = self.wind_state_x.wrapping_add(dx) & RING_MASK;
        self.wind_state_y = self.wind_state_y.wrapping_add(dy) & RING_MASK;

        let gust_size = if wind.gust_size > 0.001 { wind.gust_size } else { 1.0 };
        let wind_data = [
            self.wind_state_x as f32 / TWO_POW_20,
            self.wind_state_y as f32 / TWO_POW_20,
            1.0 / gust_size,
            0.0,
        ];
        let wind_data2 = [
            bend * sin_dir,
            bend * cos_dir,
            osc,
            t,
        ];

        let wind_globals = WindGlobals {
            wind_data,
            wind_data2,
        };
        queue.write_buffer(&self.wind_buffer, 0, bytemuck::bytes_of(&wind_globals));
    }

    /// `c_structure_renderer::render_decorators @ 0x1806901A0`.
    /// Dispatched INSIDE `c_player_view::render_albedo` — the engine
    /// renders decorators ONCE there. `default_ps` does sample +
    /// light + alpha-test + writes the LIT color to the
    /// (`_surface_accum_HDR`, normal G-buffer) MRT pair. The lit
    /// color flows through to the SL target via the albedo→SL
    /// copy-blit (`setup_targets_static_lighting`).
    pub fn render_into_pass<'rp>(
        &'rp self,
        rpass: &mut wgpu::RenderPass<'rp>,
        ctx: &'rp FrameContext<'_>,
    ) {
        if self.sets.is_empty() {
            return;
        }

        // Per-group cull mirrors `c_structure_renderer::render_decorators
        // @ 0x1806901A0` lines 207-226:
        //   far_test = sqrt(d2) > (1 - early_cull) × (end_fade + sphere_radius)
        //   cull     = !visibility_region_test_sphere(...) || far_test
        // Engine uses PVS-driven `visibility_region_test_sphere`; until
        // the projections walker stops false-negative-ing visible
        // neighbour clusters, fall back to a 5-plane Gribb-Hartmann
        // camera frustum (skip far — perspective_infinite_reverse_rh
        // puts it at infinity). `decorator_fade_distance_scale` and
        // `rasterizer_camera_get_fov_scale()` aren't ported (perf
        // throttle) so we treat them as 1.0.
        let cam_pos: glam::Vec3 = ctx.game.camera.position.into();
        let frustum_planes: [glam::Vec4; 5] = {
            let vp = ctx.game.camera.projection * ctx.game.camera.view;
            let row = |n: usize| {
                glam::Vec4::new(vp.x_axis[n], vp.y_axis[n], vp.z_axis[n], vp.w_axis[n])
            };
            let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
            let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2];
            raw.map(|p| {
                let n = p.truncate().length().max(1e-6);
                p / n
            })
        };
        let sphere_in_frustum = |center: glam::Vec3, radius: f32| -> bool {
            for plane in &frustum_planes {
                let signed =
                    plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
                if signed < -radius {
                    return false;
                }
            }
            true
        };

        // Albedo pass: bind the placeholder camera group (G-buffer
        // views are placeholder 1×1 textures while the real ones
        // are bound as color targets).
        rpass.set_bind_group(
            0,
            &ctx.shared.camera_bind_group,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13 shares the ravi cursor.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );

        let mut runs_total: u32 = 0;
        let mut runs_drawn: u32 = 0;
        let mut culled_far: u32 = 0;
        let mut culled_frustum: u32 = 0;
        for set in &self.sets {
            let variant_idx = render_shader_index(set.render_shader);
            let cull_idx = if set.two_sided { 1 } else { 0 };
            rpass.set_pipeline(&self.pipelines[variant_idx][cull_idx]);
            rpass.set_bind_group(1, &set.bind_group, &[]);
            // Per-set far-cull boundary. Engine: `(1 - early_cull) ×
            // (end_fade + sphere_radius)`. We pre-compute the
            // shared `(1 - early_cull) × end_fade` factor; sphere_radius
            // is added per-run.
            let cull_factor = (1.0 - set.early_cull).max(0.0);
            let end_fade = set.end_fade_distance.max(0.0);
            for sub in &set.subgroups {
                rpass.set_vertex_buffer(0, sub.vertex_buffer.slice(..));
                rpass.set_vertex_buffer(1, sub.instance_buffer.slice(..));
                rpass.set_index_buffer(sub.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                if sub.cluster_runs.is_empty() {
                    // No per-run grouping (off-cluster placements) —
                    // fall back to drawing the whole instance buffer.
                    runs_total += 1;
                    runs_drawn += 1;
                    rpass.draw_indexed(0..sub.index_count, 0, 0..sub.instance_count);
                } else {
                    // Engine-faithful per-cluster simple_lights binding —
                    // each cluster gets its own simple_lights slot via
                    // dynamic offset.
                    for run in &sub.cluster_runs {
                        runs_total += 1;
                        let center = glam::Vec3::from_array(run.bounding_center);
                        let radius = run.bounding_radius;
                        let dist = (center - cam_pos).length();
                        if dist > cull_factor * (end_fade + radius) {
                            culled_far += 1;
                            continue;
                        }
                        if !sphere_in_frustum(center, radius) {
                            culled_frustum += 1;
                            continue;
                        }
                        let simple_lights_offset = ctx
                            .structure_renderer
                            .bsps
                            .get(run.bsp_idx as usize)
                            .and_then(|b| {
                                b.cluster_simple_lights_offsets
                                    .get(run.cluster_idx as usize)
                                    .copied()
                            })
                            .unwrap_or(crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET);
                        rpass.set_bind_group(
                            0,
                            &ctx.shared.camera_bind_group,
                            &[
                                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                                simple_lights_offset,
                                // dominant_light @ binding 13.
                                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                            ],
                        );
                        rpass.draw_indexed(
                            0..sub.index_count,
                            0,
                            run.instance_offset..run.instance_offset + run.instance_count,
                        );
                        runs_drawn += 1;
                    }
                }
            }
        }

        // One-shot cull diagnostic — print once on the first frame
        // after a scenario load so the user can see the per-frame
        // run-drop rate. Comparable to the shadow `[shadow] cull:`
        // line emitted by views/player_view.rs.
        static CULL_DIAG_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !CULL_DIAG_LOGGED.load(std::sync::atomic::Ordering::Relaxed)
            && !CULL_DIAG_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "[decorators] cull: runs_total={} drawn={} far={} frustum={}",
                runs_total, runs_drawn, culled_far, culled_frustum,
            );
        }
    }
}


/// Convert a `blam_tags::render_model::RenderVertex` to the GPU
/// [`ModelVertex`] format used by every other forward pipeline. Same
/// shape as `loader::convert_vertex` — kept inline here so the
/// decorator path doesn't depend on the object-loader internals.
fn convert_decorator_vertex(v: &blam_tags::render_model::RenderVertex) -> ModelVertex {
    let normal = Vec3::new(v.normal.i, v.normal.j, v.normal.k);
    let texcoord_u = f16::from_f32(v.texcoord.x);
    let texcoord_v = f16::from_f32(v.texcoord.y);
    let raw_tangent = Vec3::new(v.tangent.i, v.tangent.j, v.tangent.k);
    let raw_binormal = Vec3::new(v.binormal.i, v.binormal.j, v.binormal.k);
    let (tangent_dir, binormal_dir) = if raw_tangent.length_squared() > 0.0001 {
        (raw_tangent.normalize(), raw_binormal.normalize_or_zero())
    } else {
        let stub = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let stub = stub.cross(normal).normalize_or_zero();
        (stub, normal.cross(stub))
    };
    ModelVertex {
        position: [v.position.x, v.position.y, v.position.z],
        texcoord: [texcoord_u.to_bits(), texcoord_v.to_bits()],
        normal: oct_encode_snorm8(normal),
        _pad: [0, 0],
        tangent: pack_tangent_with_sign(tangent_dir, normal, binormal_dir),
        node_indices: v.node_indices,
        node_weights: pack_weights_unorm8(v.node_weights),
        // Decorators don't use lightmap UVs.
        lightmap_texcoord: [0, 0],
    }
}

fn build_instance(
    p: &blam_tags::scenario::ScenarioDecoratorPlacement,
    lightmap_uv: [f32; 2],
    raycast_valid: f32,
    baked_hdr: [u8; 4],
    motion_scale_override: Option<u8>,
    subpart_pose: Mat4,
) -> DecoratorInstance {
    let pos = Vec3::new(p.position.x, p.position.y, p.position.z);
    let rot = quat_from_real(&p.rotation);
    let scale = if p.scale > 0.0 { p.scale } else { 1.0 };
    // Engine cache-build composition (see the subpart-pose lookup site
    // for the engine reference): the per-subpart `instance_placement`
    // pose lifts the shared mesh's vertex into the subpart's authored
    // anchor frame; the per-placement transform then places that frame
    // in the world. M = T(p.pos) × R(p.rot) × S(p.scale) × subpart_pose.
    let placement_transform =
        Mat4::from_scale_rotation_translation(Vec3::splat(scale), rot, pos);
    let model = placement_transform * subpart_pose;
    // motion_scale source is dual: scenario tag's `placement.motion_scale`
    // for Wind/Wavy/Static/Shaded variants, OR the bake's L8 anti-double-
    // count factor for `DominantLightOnly` (tool.exe `sub_140C4AF00:324`).
    // `motion_scale_override` is `Some(byte)` only when the bake produced
    // it (DominantLightOnly + dom_color was supplied). All other variants
    // fall through to the scenario byte. Engine `setup_default_lighting`
    // matches: the runtime VS reads the byte as "sun intensity" for the
    // dominant-light variant, "wind sway intensity" for wind/wavy.
    let motion_scale_byte = motion_scale_override
        .unwrap_or(p.motion_scale as u8);
    let motion_scale = motion_scale_byte as f32 / 255.0;
    // `tint` carries the engine's `instance_color` — HDR-encoded
    // `(R, G, B, exp_byte)` per `decorators.hlsl:282`. Shader decodes
    // as `tint.rgb × exp2(tint.a × 63.75 − 31.75)`. When the bake fails
    // we encode the painted tint at exp=127 (= 2^0 = 1.0 multiplier).
    let inv = 1.0 / 255.0;
    DecoratorInstance {
        model: model.to_cols_array_2d(),
        tint: [
            baked_hdr[0] as f32 * inv,
            baked_hdr[1] as f32 * inv,
            baked_hdr[2] as f32 * inv,
            baked_hdr[3] as f32 * inv,
        ],
        lightmap_uv: [lightmap_uv[0], lightmap_uv[1], raycast_valid, motion_scale],
    }
}

/// Fallback when no atlas+compress is available: encode the painted
/// tint at exp=0 so the shader's HDR decode (`tint × exp2(0)`)
/// recovers the painted color directly.
fn encode_tint_as_unbaked(tint: &blam_tags::math::RealPoint3d) -> [u8; 4] {
    let r = (tint.x.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (tint.y.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (tint.z.clamp(0.0, 1.0) * 255.0).round() as u8;
    // exp_byte = 127 → encoded.a = 127/255 ≈ 0.498 → exp2(0.498*63.75 - 31.75) = exp2(~0) = 1.0.
    [r, g, b, 127]
}

/// Decode the per-BSP lightprobe atlas + compression scalars CPU-side
/// for the bake. Walks `scenario.active_bsps[].lightmap` and loads the
/// `.bitmap` data from disk via the existing bitmap loader. Returns
/// one [`BakeAtlas`] per BSP that has a lightmap.
///
/// Includes both SH atlas (8 BC3 layers, 4 SH coefs) AND the
/// dominant-light-intensity atlas (2 BC3 layers, 1 intensity Vec3) —
/// engine `c_geometry_sampler::sample`'s per-pixel branch decodes both
/// then runs `reconstruct_quadratic_lightprobe_from_linear_and_intensity`
/// to fill L2 SH bands. `intensity` is `None` when the BSP's lightmap
/// tag has no intensity bitmap reference or the load fails — callers
/// fall back to leaving L2=0 (Phase A behavior).
pub(crate) fn decode_bake_resources(
    scenario: &LoadedScenario,
) -> Vec<BakeAtlas> {
    let mut out = Vec::new();
    for bsp in &scenario.active_bsps {
        let Some(lbsp) = &bsp.lightmap else { continue };
        if lbsp.lightprobe_texture.is_empty() {
            continue;
        }
        let load_atlas = |tag_path: &str| -> Option<DecodedAtlas> {
            if tag_path.is_empty() {
                return None;
            }
            let path = blam_tags::paths::resolve_tag_path(&scenario.tags_root, tag_path, "bitmap");
            let tag = blam_tags::file::TagFile::read(&path).ok()?;
            let bitmap = blam_tags::bitmap::Bitmap::new(&tag).ok()?;
            let img = bitmap.image(0)?;
            DecodedAtlas::from_bitmap(&img)
        };
        let Some(sh_atlas) = load_atlas(&lbsp.lightprobe_texture) else { continue };
        let intensity_atlas = load_atlas(&lbsp.dominant_light_intensity_texture);

        // Mirror of `upload_lightmap_atlases` compress packing
        // (render::mod.rs:684-688) — 9 ranges from `compression_vectors[2k].x`
        // packed 3-per-vec4. Atlas decoder reads `[0][x/y/z]` and `[1][x]`
        // for SH coefs; `[1][y]` for intensity.
        let r = |i: usize| lbsp.compression_vectors.get(i).map(|v| v.i).unwrap_or(0.0);
        let compress: [[f32; 4]; 3] = [
            [r(0), r(2), r(4), 0.0],
            [r(6), r(8), r(10), 0.0],
            [r(12), r(14), r(16), 0.0],
        ];
        out.push(BakeAtlas { sh: sh_atlas, intensity: intensity_atlas, compress });
    }
    out
}

/// Per-BSP CPU-side lightmap atlas pair + compression vectors. Used by
/// the decorator bake AND the geometry sampler to reproduce engine's
/// per-pixel lightprobe decode + L2 reconstruct.
pub(crate) struct BakeAtlas {
    pub sh: DecodedAtlas,
    /// `None` when the BSP's lightmap has no intensity bitmap or the
    /// load failed — callers should skip the `reconstruct` step (L2 = 0,
    /// Phase A behavior).
    pub intensity: Option<DecodedAtlas>,
    /// Three vec4s packing 9 SH compression ranges + the dominant
    /// intensity range. Layout matches dllcache's `submit_extern_texture`
    /// case 11 + HLSL's `lightmap_compress` cbuffer.
    pub compress: [[f32; 4]; 3],
}

fn quat_from_real(q: &RealQuaternion) -> Quat {
    // glam Quat is (x, y, z, w); RealQuaternion is (i, j, k, w).
    let v = Quat::from_xyzw(q.i, q.j, q.k, q.w);
    if v.length_squared() > 0.0001 {
        v.normalize()
    } else {
        Quat::IDENTITY
    }
}
