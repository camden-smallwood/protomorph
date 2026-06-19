//! GPU particle grid core (engine `particle_block.cpp`): the persistent
//! `RawParticleState` grid, the spawn scatter-copy, the per-emitter update,
//! the billboard render, and the screen-space distortion warp.

use crate::halo::effects::particle_states::{ParticleState, RawParticleState};
use crate::halo::effects::{
    ParticleAlbedo, MAX_SPAWN_PER_FRAME, PARTICLE_GRID_SIZE, PARTICLE_ROW_COUNT,
};
use blam_tags::render_method::AlphaBlendMode;
use crate::halo::render::shared::SharedResources;

use super::common_gpu::{blend_state, MAX_SPRITE_FRAMES};

/// Bytes per `RawParticleState` (20 × u32) — exact DX11 layout.
const RAW_PARTICLE_BYTES: u64 = 80;

/// Max render batches the per-region physics table holds.
const MAX_BATCHES: u64 = 256;

/// Distortion accumulation + scene-copy format. Matches `lighting_base`
/// (Rgba16Float) so the warp target/copy are 1:1 and signed displacement
/// deltas accumulate without clamping.
const DISTORTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Allocate the distortion targets for a full-res viewport of `(w, h)`. The
/// accumulation buffer is HALF-res (engine `window_pixel_bounds / 2` in
/// `submit_distortions`) — the displacement is authored in resolution-
/// independent UV units, so the apply just linear-upsamples it (a slightly
/// softer warp, matching the engine). The scene copy stays full-res (it's the
/// warp source, sampled at full-res UVs). Returns `(accum_view, scene_copy_tex,
/// scene_copy_view)`.
fn create_distortion_targets(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::Texture, wgpu::TextureView) {
    let mk = |label: &str, tw: u32, th: u32| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: tw.max(1), height: th.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DISTORTION_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let accum = mk("particle_distortion_accum", w, h);
    let scene_copy = mk("particle_distortion_scene_copy", w, h);
    let accum_view = accum.create_view(&wgpu::TextureViewDescriptor::default());
    let scene_copy_view = scene_copy.create_view(&wgpu::TextureViewDescriptor::default());
    (accum_view, scene_copy, scene_copy_view)
}

/// A read-only storage bind-group-layout entry for the compute update.
fn update_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Update-kernel params — `dt` + the region routing the per-emitter
/// physics table is indexed by (a particle's batch = `slot / region_len`,
/// clamped to `batch_count`). Mirrors the engine binding that emitter's
/// `set_shader_update_state` per the per-emitter `frame_advance`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateParams {
    dt: f32,
    region_len: u32,
    batch_count: u32,
    _pad: u32,
}

pub struct ParticleGpu {
    /// Persistent grid — `PARTICLE_GRID_SIZE` × 64B `RawParticleState`.
    grid: wgpu::Buffer,
    /// This frame's packed staging states (storage, read by `cs_spawn`).
    spawn_buf: wgpu::Buffer,
    /// This frame's per-spawn grid addresses (storage, read by `cs_spawn`).
    address_buf: wgpu::Buffer,
    /// `vec4<u32>` — `.x` = spawn count this frame.
    params: wgpu::Buffer,
    spawn_bind_group: wgpu::BindGroup,
    spawn_pipeline: wgpu::ComputePipeline,
    /// [`UpdateParams`] — dt + region routing.
    update_params: wgpu::Buffer,
    /// Per-batch physics (`vec4<f32>` = gravity·grav_mod, air, rot, 0),
    /// indexed by region. `MAX_BATCHES` entries.
    physics_table: wgpu::Buffer,
    /// Per-batch curve-eval tables (update re-evaluates props from these).
    eval_properties: wgpu::Buffer, // MAX_BATCHES × 13 × EvalProperty
    eval_functions: wgpu::Buffer,  // MAX_BATCHES × 32 × EvalFunction
    eval_colors: wgpu::Buffer,     // MAX_BATCHES × 8 × vec4
    /// Per-batch per-frame state uniforms (system_age, ...), 28 × vec4.
    eval_state: wgpu::Buffer,
    /// Per-batch WORLD self-acceleration (`vec4` = xyz, 0), uploaded each
    /// frame. The update adds `self_accel·dt` to velocity.
    self_accel_table: wgpu::Buffer,
    /// Per-row owning batch (engine row allocation), 448 × u32. Uploaded
    /// each frame from the CPU [`crate::halo::effects::RowPool`]; the update
    /// kernel routes each slot's physics/curves by `row_batch[slot / 16]`.
    row_batch_buf: wgpu::Buffer,
    /// CPU copy of the row→batch table (for the per-row render draws).
    row_batch: Vec<u32>,
    update_bind_group: wgpu::BindGroup,
    update_pipeline: wgpu::ComputePipeline,
    batch_count: u32,
    // --- billboard render (per render-batch) ---
    render_shader: wgpu::ShaderModule,
    /// group0 = camera + grid + frame params (shared across batches).
    group0_layout: wgpu::BindGroupLayout,
    // group0 is rebuilt each frame in `render` because it binds the
    // gbuffer depth view (recreated on resize) for the soft depth fade.
    /// group1 = per-batch material (base/alpha/palette + sampler + mat).
    group1_layout: wgpu::BindGroupLayout,
    render_pl: wgpu::PipelineLayout,
    /// `vec4<f32>` — view exposure, size scale, near_clip, depth_fade_range.
    frame_params: wgpu::Buffer,
    /// group0 bind group for the color render, rebuilt each frame by
    /// [`Self::prepare_color_render`] (binds the gbuffer depth view, which
    /// is recreated on resize). `draw_emitter` binds it inside the shared
    /// transparents rpass. `None` until the first prepare.
    render_group0: Option<wgpu::BindGroup>,
    /// SEPARATE params buffer for the distortion pass. Must NOT share
    /// `frame_params` with the color render: `queue.write_buffer` writes to one
    /// buffer are all applied before the command buffer executes with the LAST
    /// write winning, so a shared buffer would feed the distortion pass's
    /// exposure=0 back into the (earlier) color pass → black particles.
    distortion_params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// 1×1 white stand-in for unbound material slots (e.g. no palette).
    fallback_tex: wgpu::Texture,
    /// blend mode → render pipeline (lazily built, shared).
    pipelines: std::collections::HashMap<AlphaBlendMode, wgpu::RenderPipeline>,
    // --- distortion (specialized_rendering particles → screen warp) ---
    /// Distortion particles accumulate a screen-space displacement here
    /// (cleared to 0.5, additive). Engine `_surface_aux_mini2`.
    distortion_accum_view: wgpu::TextureView,
    /// Scene-color copy used as the warp source (ping-pong: read here,
    /// write the warped result back into the scene target).
    scene_copy: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,
    /// Current size of the distortion targets (resized with the viewport).
    distortion_size: (u32, u32),
    /// Distortion accumulation pipeline (shares the render bind layout; FS
    /// writes displacement, additive blend, target = the accum format).
    distortion_pipeline: wgpu::RenderPipeline,
    /// Full-screen warp that applies the accumulated displacement.
    apply_pipeline: wgpu::RenderPipeline,
    apply_layout: wgpu::BindGroupLayout,
    apply_sampler: wgpu::Sampler,
    /// One per loaded particle system; owns a grid region + material.
    batches: Vec<BatchGpu>,
    size_scale: f32,
    /// Map-readable staging copy of the grid (debug verification only).
    readback: wgpu::Buffer,
    frame: u64,
    pub total_spawned: u64,
}

/// A registered render batch — its material bind group + blend pipeline.
/// Its grid slots are no longer a contiguous region; they live in whatever
/// rows the engine allocator handed this batch's emitters (see
/// [`ParticleGpu::row_batch`]). The batch's index in `batches` IS its id.
struct BatchGpu {
    bind_group: wgpu::BindGroup,
    blend_mode: AlphaBlendMode,
    /// Distortion variant — skipped at render until distortion lands.
    distortion: bool,
    /// Engine `DontRenderSystem` inverted: `false` → spawned + updated but
    /// never drawn (skipped in both the color and distortion passes).
    renders: bool,
    // Keep the per-batch GPU resources alive.
    _views: [wgpu::TextureView; 3],
    _mat_buf: wgpu::Buffer,
}

/// Per-batch material handed to [`ParticleGpu::register_batches`]. The
/// caller (scenario load) resolves the bitmaps to views.
pub struct BatchMaterial {
    pub albedo: ParticleAlbedo,
    pub blend_mode: AlphaBlendMode,
    pub black_point: bool,
    pub base_view: wgpu::TextureView,
    pub alpha_view: wgpu::TextureView,
    /// Palette gradient view; pass the fallback for non-palettized.
    pub palette_view: Option<wgpu::TextureView>,
    /// `[gravity·grav_mod, air_friction, rot_friction, 0]` — this
    /// system's physics, routed to its grid region by the update kernel.
    pub physics: [f32; 4],
    /// Compiled curve-evaluation tables — the update kernel re-evaluates
    /// color/size/alpha/etc. from these each frame.
    pub eval: crate::halo::effects::particle_properties::EmitterEvalTables,
    /// Engine `m_initial_color` — rgb tint + HDR exponent in `.w`. White
    /// `[1,1,1,0]` unless the emitter's appearance flags request lightprobe
    /// tinting (bit3/bit4). Resolved per-emitter at register time from the
    /// scene lightprobe; folded into the particle color in the render VS.
    pub initial_color: [f32; 4],
    /// Baked sprite corner `(xy = bottom-left, zw = extent)` — the billboard
    /// vertex is `shift·corner.zw + corner.xy` (engine
    /// `c_particle_definition::postprocess_frame_animation`). Default square
    /// sprite = `(-1,-1,2,2)`.
    pub corner: [f32; 4],
    /// Extra in-plane sprite rotation in turns (the
    /// bitmap-authored-vertically +0.25, resolved CPU-side from the typed
    /// appearance flag).
    pub rotation_offset: f32,
    /// Whether the shader's `depth_fade` category is on — gates the soft
    /// depth fade (off for water_spray; the engine doesn't fade it).
    pub depth_fade: bool,
    /// Whether the shader's `fog` category is on — gates the per-vertex
    /// analytic atmosphere fog (engine particle_render_hlsl.hlsl:673:
    /// `m_color ×= extinction; m_color_add += inscatter·v_exposure`).
    pub fog: bool,
    /// Distortion variant — skipped at render (distortion unimplemented;
    /// drawing it shows the warp bitmap as rainbow garbage).
    pub distortion: bool,
    /// Engine `DontRenderSystem` inverted — `false` → spawned + updated but
    /// never drawn. Default `true`.
    pub renders: bool,
    /// Billboard basis selector — raw engine `particle_billboard_type_enum`
    /// (0..9). Drives `billboard_basis` in the VS.
    pub billboard: u32,
    /// Motion-blur aspect-stretch scale (prt3 `motion_blur_aspect_scale`
    /// when the `motion blur` appearance bit is set, else 0). The VS adds
    /// `|velocity|·scale/(30·size)` to the sprite aspect → streaks fast
    /// particles along their motion (engine particle_render_hlsl.hlsl:588).
    pub motion_blur_aspect_scale: f32,
    /// Engine always-on near-camera fade (set_shader_render_state
    /// @0x1806a67c0): `m_near_range = 1/near_fade_range` (0 = disabled, no
    /// fade) — the VS does `alpha *= saturate(near_range·(depth−near_cutoff))`.
    pub near_range: f32,
    /// `m_near_cutoff` = `near_fade_cutoff` (or `near_fade_override` when the
    /// system's `override near fade` flag, bit 11, is set).
    pub near_cutoff: f32,
    /// Edge fade (appearance bit 7 `fade when viewed edge-on`): `edge_range =
    /// 1/radians(angle_fade_range)` (0 = disabled), `edge_cutoff =
    /// radians(angle_fade_cutoff)`. Carried in `MatParams.edge.xy`; the VS
    /// does `alpha *= saturate(edge_range·(billboard_angle − edge_cutoff))`
    /// (engine `set_shader_render_state @0x1806a67c0` / `particle_render.hlsl:714`).
    pub edge_range: f32,
    pub edge_cutoff: f32,
    /// Appearance `intensity affects alpha` (bit 6, by-name resolved): the
    /// VS multiplies output alpha by `m_intensity` (engine
    /// `particle_render_hlsl.hlsl:701-703`). Carried to the shader in
    /// `MatParams.fade.z`.
    pub intensity_affects_alpha: bool,
    /// Appearance `random u/v mirror` (bits 0/1, by-name): per-particle
    /// sprite texcoord flip (engine `particle_render_hlsl.hlsl:623-630`).
    /// Packed into `MatParams.fade.w` (bit0 = u, bit1 = v).
    pub flip_u: bool,
    pub flip_v: bool,
    /// Baked sprite-sheet frame UV rects in engine `frame_texcoord` form
    /// `(left, top, right-left, bottom-top)`. Empty (or ≤1 entry) → plain
    /// bitmap, the render samples the full `[0,1]` quad (no regression for
    /// non-animated particles). Clamped to [`MAX_SPRITE_FRAMES`].
    pub frames: Vec<[f32; 4]>,
    /// `frame_blend` shader category — cross-fade consecutive frames.
    pub frame_blend: bool,
    /// Sprite-sheet animation playback flags (`compute_variant`).
    pub anim_one_shot: bool,
    pub anim_backwards: bool,
}

impl ParticleGpu {
    pub fn new(shared: &SharedResources) -> Self {
        let device = &shared.device;
        let grid_bytes = PARTICLE_GRID_SIZE as u64 * RAW_PARTICLE_BYTES;
        // Staging holds packed RawParticleState (80B); the CPU builds the
        // full state and the spawn kernel scatter-copies it.
        let spawn_bytes = MAX_SPAWN_PER_FRAME as u64 * RAW_PARTICLE_BYTES;
        let address_bytes = MAX_SPAWN_PER_FRAME as u64 * 4;

        let grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_grid"),
            size: grid_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spawn_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_spawn_staging"),
            size: spawn_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let address_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_spawn_addresses"),
            size: address_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_spawn_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_grid_readback"),
            size: grid_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_spawn"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/halo_shaders/particle_spawn.wgsl").into(),
            ),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_spawn_bgl"),
            entries: &[
                // grid (read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // staging states (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // addresses (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let spawn_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_spawn_bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: spawn_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: address_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_spawn_pl"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let spawn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_spawn_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_spawn"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- update pipeline (cs_update over the whole grid) ---
        let update_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_update_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let physics_table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_physics_table"),
            size: MAX_BATCHES * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let self_accel_table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_self_accel_table"),
            size: MAX_BATCHES * 32, // 2 world interpolant vectors / batch
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let row_batch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_row_batch"),
            size: PARTICLE_ROW_COUNT as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Engine-exact transition/periodic sub-function LUTs (blam-tags
        // FUNCTION_TABLES: transition rows 1..=7 then periodic rows 1..=11,
        // each 1024 bytes). Uploaded once as f32 (byte/255) and sampled by
        // the eval engine (particle_eval.wgsl `lut_transition`/`lut_periodic`),
        // matching `transition_function_evaluate`/`periodic_function_evaluate`.
        let func_table_data: Vec<f32> = blam_tags::tag_function::FUNCTION_TABLES
            .iter()
            .map(|&b| b as f32 * (1.0 / 255.0))
            .collect();
        let func_table_bytes: &[u8] = bytemuck::cast_slice(&func_table_data);
        let func_tables = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_func_tables"),
            size: func_table_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        func_tables
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(func_table_bytes);
        func_tables.unmap();
        use crate::halo::effects::particle_properties::{
            EvalFunction, EvalProperty, EVAL_COLORS, EVAL_FUNCS, EVAL_PROPS, EVAL_STATE_SLOTS,
        };
        let storage_buf = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let prop_sz = std::mem::size_of::<EvalProperty>() as u64;
        let fn_sz = std::mem::size_of::<EvalFunction>() as u64;
        let eval_properties = storage_buf(
            "particle_eval_properties",
            MAX_BATCHES * EVAL_PROPS as u64 * prop_sz,
        );
        let eval_functions = storage_buf(
            "particle_eval_functions",
            MAX_BATCHES * EVAL_FUNCS as u64 * fn_sz,
        );
        let eval_colors = storage_buf(
            "particle_eval_colors",
            MAX_BATCHES * EVAL_COLORS as u64 * 16,
        );
        let eval_state = storage_buf(
            "particle_eval_state",
            MAX_BATCHES * EVAL_STATE_SLOTS as u64 * 16,
        );
        // The update kernel = the eval engine (particle_eval.wgsl) +
        // the integrate/property-write body (particle_update.wgsl).
        let update_src = format!(
            "{}\n{}",
            include_str!("../../../assets/halo_shaders/particle_eval.wgsl"),
            include_str!("../../../assets/halo_shaders/particle_update.wgsl"),
        );
        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_update"),
            source: wgpu::ShaderSource::Wgsl(update_src.into()),
        });
        let update_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_update_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // per-batch physics table + the curve-eval tables (3-6),
                // all read-only storage.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                update_storage_entry(3),
                update_storage_entry(4),
                update_storage_entry(5),
                update_storage_entry(6),
                update_storage_entry(7),
                update_storage_entry(8), // row_batch
                update_storage_entry(9), // transition/periodic LUTs
            ],
        });
        let update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_update_bg"),
            layout: &update_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: update_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: physics_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: eval_properties.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: eval_functions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: eval_colors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: eval_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self_accel_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: row_batch_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: func_tables.as_entire_binding() },
            ],
        });
        let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_update_pl"),
            bind_group_layouts: &[&update_layout],
            immediate_size: 0,
        });
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_update_pipeline"),
            layout: Some(&update_pl),
            module: &update_shader,
            entry_point: Some("cs_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- billboard render (per-batch) ---
        let frame_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_frame_params"),
            size: 32, // two vec4: [exposure,size,near,fade] + [distortion_strength,..]
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let distortion_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_distortion_params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("particle_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // 1×1 white fallback for unbound slots (e.g. no palette).
        let fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particle_fallback_white"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        shared.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_render"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/halo_shaders/particle_render.wgsl").into(),
            ),
        });
        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_render_group0"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX), // camera
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // grid
                uniform_entry(2, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT), // frame params
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }, // scene depth (sampled for the soft depth fade)
                // Frame-default atmosphere setting (the cluster-weighted
                // accumulator entry of the shared atmosphere table) — the
                // VS analytic-fog `compute_scattering` source.
                uniform_entry(4, wgpu::ShaderStages::VERTEX),
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_render_group1"),
            entries: &[
                tex_entry(0), // base
                tex_entry(1), // alpha
                tex_entry(2), // palette
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // mat — read in the VS (sprite corner) + FS (albedo/blend).
                uniform_entry(4, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_render_pl"),
            bind_group_layouts: &[&group0_layout, &group1_layout],
            immediate_size: 0,
        });

        // --- distortion: accumulate displacement, then warp the scene ---
        let (dw, dh) = (shared.config.width.max(1), shared.config.height.max(1));
        let (distortion_accum_view, scene_copy, scene_copy_view) =
            create_distortion_targets(device, dw, dh);
        // Accumulation pipeline: same bind layout as the color render (camera/
        // grid/frame/depth + material), but the FS writes displacement and the
        // blend is additive so overlapping particles sum their warp.
        let distortion_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_distortion_pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_particle"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_distortion"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DISTORTION_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        // Full-screen warp pass.
        let apply_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_distortion_apply"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../assets/halo_shaders/particle_distortion_apply.wgsl").into(),
            ),
        });
        let apply_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_distortion_apply_layout"),
            entries: &[
                tex_entry(0), // scene copy
                tex_entry(1), // displacement accum
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let apply_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_distortion_apply_pl"),
            bind_group_layouts: &[&apply_layout],
            immediate_size: 0,
        });
        let apply_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_distortion_apply_pipeline"),
            layout: Some(&apply_pl),
            vertex: wgpu::VertexState {
                module: &apply_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &apply_shader,
                entry_point: Some("fs_apply"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DISTORTION_FORMAT, // = lighting_base (Rgba16Float)
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let apply_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("particle_distortion_apply_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            grid,
            spawn_buf,
            address_buf,
            params,
            spawn_bind_group,
            spawn_pipeline,
            update_params,
            physics_table,
            eval_properties,
            eval_functions,
            eval_colors,
            eval_state,
            self_accel_table,
            row_batch_buf,
            row_batch: vec![crate::halo::effects::ROW_BATCH_FREE; PARTICLE_ROW_COUNT as usize],
            update_bind_group,
            update_pipeline,
            batch_count: 0,
            render_shader,
            group0_layout,
            render_group0: None,
            group1_layout,
            render_pl,
            frame_params,
            distortion_params,
            sampler,
            fallback_tex,
            pipelines: std::collections::HashMap::new(),
            distortion_accum_view,
            scene_copy,
            scene_copy_view,
            distortion_size: (dw, dh),
            distortion_pipeline,
            apply_pipeline,
            apply_layout,
            apply_sampler,
            batches: Vec::new(),
            size_scale: 1.0,
            readback,
            frame: 0,
            total_spawned: 0,
        }
    }

    /// Scatter this frame's particle births into the grid. The CPU packs
    /// each [`ParticleState`] into the 80-byte grid format and assigns a
    /// grid address (ring-allocated within the routed batch's region);
    /// `cs_spawn` then copies staging→grid. No-op with no births/batches.
    pub fn dispatch_spawn(
        &mut self,
        shared: &SharedResources,
        births: &[ParticleState],
        slots: &[u32],
    ) {
        self.frame += 1;
        if births.is_empty() || self.batches.is_empty() {
            return;
        }
        let count = (births.len() as u32).min(MAX_SPAWN_PER_FRAME) as usize;

        // Pack states; each birth already carries the grid slot the engine
        // row allocator assigned it CPU-side (engine `m_gpu_address`). The
        // scatter shader copies staging[i] → grid[slot].
        let mut staged: Vec<RawParticleState> = Vec::with_capacity(count);
        let mut addresses: Vec<u32> = Vec::with_capacity(count);
        for (i, b) in births[..count].iter().enumerate() {
            staged.push(b.pack());
            addresses.push(slots.get(i).copied().unwrap_or(0));
        }
        let count = count as u32;
        self.total_spawned += count as u64;

        shared
            .queue
            .write_buffer(&self.spawn_buf, 0, bytemuck::cast_slice(&staged));
        shared
            .queue
            .write_buffer(&self.address_buf, 0, bytemuck::cast_slice(&addresses));
        shared
            .queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&[count, 0u32, 0u32, 0u32]));

        let mut encoder = shared
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("particle_spawn_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_spawn_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spawn_pipeline);
            pass.set_bind_group(0, &self.spawn_bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        shared.queue.submit(Some(encoder.finish()));

        if std::env::var("PROTOMORPH_DIAG_PARTICLES").is_ok() && self.frame <= 6 {
            eprintln!(
                "[particles] frame {} — spawned {} (batches={}, total={})",
                self.frame, count, self.batches.len(), self.total_spawned,
            );
            self.debug_readback(shared, count.min(8));
        }
    }

    /// Integrate the whole grid one step (age + per-region physics).
    /// Engine `frame_advance_all @0x1806A4F30` dispatches the update
    /// per-emitter with that emitter's `set_shader_update_state`; we
    /// achieve the same by routing each particle to its batch's physics
    /// (batch = `slot / region_len`).
    pub fn dispatch_update(
        &mut self,
        shared: &SharedResources,
        dt: f32,
        self_accel: &[[f32; 4]],
        state: &[[f32; 4]],
        row_batch: &[u32],
    ) {
        if dt <= 0.0 || self.batch_count == 0 {
            return;
        }
        // Per-frame state uniforms (system_age/game_time/LOD/seeds) feeding the
        // curve eval. Per-particle slots are overridden in-shader from the
        // packed particle; these are the system/location/game-level inputs.
        if !state.is_empty() {
            // Clamp to the destination buffer capacity — the caller sizes these
            // by the (unclamped) emitter count, which register_batches caps at
            // MAX_BATCHES. Writing the full slice would overflow the buffer.
            let cap = (self.eval_state.size() as usize) / 16;
            let state = &state[..state.len().min(cap)];
            shared
                .queue
                .write_buffer(&self.eval_state, 0, bytemuck::cast_slice(state));
        }
        let p = UpdateParams {
            dt,
            region_len: 0,
            batch_count: self.batch_count,
            _pad: 0,
        };
        shared
            .queue
            .write_buffer(&self.update_params, 0, bytemuck::bytes_of(&p));
        if !self_accel.is_empty() {
            let cap = (self.self_accel_table.size() as usize) / 16;
            let self_accel = &self_accel[..self_accel.len().min(cap)];
            shared
                .queue
                .write_buffer(&self.self_accel_table, 0, bytemuck::cast_slice(self_accel));
        }
        // Upload the per-row batch table (engine row allocation routing) and
        // keep a CPU copy for the per-row render draws.
        if !row_batch.is_empty() {
            let n = row_batch.len().min(self.row_batch.len());
            self.row_batch[..n].copy_from_slice(&row_batch[..n]);
            shared
                .queue
                .write_buffer(&self.row_batch_buf, 0, bytemuck::cast_slice(&self.row_batch));
        }

        let mut encoder = shared
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("particle_update_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.update_bind_group, &[]);
            pass.dispatch_workgroups(PARTICLE_GRID_SIZE.div_ceil(64), 1, 1);
        }
        shared.queue.submit(Some(encoder.finish()));
    }

    /// Register one render batch per loaded particle system: split the
    /// grid into equal regions, build each batch's material bind group,
    /// and ensure its blend pipeline exists. Replaces any prior batches.
    /// `size_scale` multiplies authored particle sizes on screen.
    pub fn register_batches(
        &mut self,
        shared: &SharedResources,
        mut materials: Vec<BatchMaterial>,
        size_scale: f32,
    ) {
        self.size_scale = size_scale;
        self.batches.clear();
        if materials.is_empty() {
            return;
        }
        // Every per-batch GPU buffer (physics_table, eval_*, eval_state,
        // self_accel_table) is sized MAX_BATCHES. With more emitter batches
        // than that, the write_buffer calls below would write past the buffer
        // end → wgpu validation panic. Clamp + warn instead of crashing.
        if materials.len() > MAX_BATCHES as usize {
            eprintln!(
                "[gpu_particle] {} emitter batches exceeds MAX_BATCHES={}; dropping {} extra batch(es)",
                materials.len(),
                MAX_BATCHES,
                materials.len() - MAX_BATCHES as usize,
            );
            materials.truncate(MAX_BATCHES as usize);
        }
        let n = materials.len() as u32;
        self.batch_count = n;
        // Upload the per-region physics table.
        let mut table = vec![[0.0f32; 4]; n as usize];
        for (i, m) in materials.iter().enumerate() {
            if i < table.len() {
                table[i] = m.physics;
            }
        }
        shared
            .queue
            .write_buffer(&self.physics_table, 0, bytemuck::cast_slice(&table));

        // Upload the per-batch curve-eval tables (flat, batch-indexed).
        use crate::halo::effects::particle_properties::{
            EVAL_COLORS, EVAL_FUNCS, EVAL_PROPS, EVAL_STATE_SLOTS,
        };
        let mut props = Vec::with_capacity(n as usize * EVAL_PROPS);
        let mut funcs = Vec::with_capacity(n as usize * EVAL_FUNCS);
        let mut cols = Vec::with_capacity(n as usize * EVAL_COLORS);
        for m in materials.iter() {
            props.extend_from_slice(&m.eval.properties);
            funcs.extend_from_slice(&m.eval.functions);
            cols.extend_from_slice(&m.eval.colors);
        }
        shared.queue.write_buffer(&self.eval_properties, 0, bytemuck::cast_slice(&props));
        shared.queue.write_buffer(&self.eval_functions, 0, bytemuck::cast_slice(&funcs));
        shared.queue.write_buffer(&self.eval_colors, 0, bytemuck::cast_slice(&cols));
        // Zero the per-frame state uniforms (system_age etc.); the
        // visible props are particle-age-keyed. (Per-frame system_age
        // wiring lands with P4's pulse.)
        let zeros = vec![[0.0f32; 4]; n as usize * EVAL_STATE_SLOTS];
        shared.queue.write_buffer(&self.eval_state, 0, bytemuck::cast_slice(&zeros));

        let device = &shared.device;
        for m in materials.into_iter() {
            self.ensure_pipeline(shared, m.blend_mode);
            let mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particle_mat_params"),
                // v (vec4<u32>) + corner (vec4) + extra (vec4) +
                // initial_color (vec4) + frames_meta (vec4<u32>) +
                // fade (vec4) + edge (vec4) + frame_uv[MAX_SPRITE_FRAMES].
                size: 112 + (MAX_SPRITE_FRAMES as u64) * 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            shared.queue.write_buffer(
                &mat_buf,
                0,
                bytemuck::cast_slice(&[
                    m.albedo as u32,
                    m.black_point as u32,
                    m.blend_mode as u32,
                    m.billboard, // v.w = billboard basis selector
                ]),
            );
            shared
                .queue
                .write_buffer(&mat_buf, 16, bytemuck::cast_slice(&m.corner));
            // extra = (rotation_offset, depth_fade_on, motion_blur_aspect_scale, 0)
            let depth_fade = if m.depth_fade { 1.0f32 } else { 0.0 };
            shared.queue.write_buffer(
                &mat_buf,
                32,
                bytemuck::cast_slice(&[
                    m.rotation_offset,
                    depth_fade,
                    m.motion_blur_aspect_scale,
                    0.0,
                ]),
            );
            // initial_color (vec4) = engine m_initial_color (rgb tint + HDR
            // exponent .w). White [1,1,1,0] for untinted emitters.
            shared
                .queue
                .write_buffer(&mat_buf, 48, bytemuck::cast_slice(&m.initial_color));
            // frames_meta = (frame_count, anim_flags, frame_blend, fog_on).
            // frame_count ≤ 1 → render samples the full quad (plain bitmap).
            let frame_count = m.frames.len().min(MAX_SPRITE_FRAMES) as u32;
            let anim_flags =
                (m.anim_one_shot as u32) | ((m.anim_backwards as u32) << 1);
            // Diagnostic: PROTOMORPH_NO_PARTICLE_FOG=1 disables the analytic
            // particle fog to isolate whether it causes the s3d_turf red tint.
            let fog_on = m.fog && std::env::var("PROTOMORPH_NO_PARTICLE_FOG").is_err();
            shared.queue.write_buffer(
                &mat_buf,
                64,
                bytemuck::cast_slice(&[
                    frame_count,
                    anim_flags,
                    m.frame_blend as u32,
                    fog_on as u32,
                ]),
            );
            // fade = (near_range, near_cutoff, intensity_affects_alpha, 0).
            // x/y = engine always-on near-camera fade (set_shader_render_state
            // @0x1806a67c0): alpha *= saturate(near_range·(view_depth −
            // near_cutoff)), near_range = 1/near_fade_range (0 ⇒ disabled).
            // z = appearance bit 6 (`m_color.w *= m_intensity`); w = random
            // UV-mirror flags (bit0 flip-u, bit1 flip-v, appearance bits 0/1).
            let flip_mask = (m.flip_u as u32) | ((m.flip_v as u32) << 1u32);
            shared.queue.write_buffer(
                &mat_buf,
                80,
                bytemuck::cast_slice(&[
                    m.near_range,
                    m.near_cutoff,
                    if m.intensity_affects_alpha { 1.0f32 } else { 0.0 },
                    flip_mask as f32,
                ]),
            );
            // edge = (edge_range, edge_cutoff, 0, 0) — appearance bit 7 fade
            // when viewed edge-on (engine set_shader_render_state
            // @0x1806a67c0). edge_range = 1/radians(angle_fade_range)
            // (0 ⇒ disabled), edge_cutoff = radians(angle_fade_cutoff).
            shared.queue.write_buffer(
                &mat_buf,
                96,
                bytemuck::cast_slice(&[m.edge_range, m.edge_cutoff, 0.0f32, 0.0f32]),
            );
            // frame_uv[MAX_SPRITE_FRAMES] (zero-padded).
            let mut frame_uv = [[0.0f32; 4]; MAX_SPRITE_FRAMES];
            for (dst, src) in frame_uv.iter_mut().zip(m.frames.iter()) {
                *dst = *src;
            }
            shared
                .queue
                .write_buffer(&mat_buf, 112, bytemuck::cast_slice(&frame_uv));
            let palette = m
                .palette_view
                .unwrap_or_else(|| self.fallback_tex.create_view(&Default::default()));
            let views = [m.base_view, m.alpha_view, palette];
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_render_group1_bg"),
                layout: &self.group1_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&views[0]) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&views[1]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&views[2]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 4, resource: mat_buf.as_entire_binding() },
                ],
            });
            self.batches.push(BatchGpu {
                bind_group,
                blend_mode: m.blend_mode,
                distortion: m.distortion,
                renders: m.renders,
                _views: views,
                _mat_buf: mat_buf,
            });
        }
        eprintln!(
            "[particles] registered {} render batches (engine row allocation)",
            self.batches.len(),
        );
    }

    /// A 1×1 white fallback view (for the palette slot of non-palettized
    /// materials, or a failed bitmap load).
    pub fn fallback_view(&self) -> wgpu::TextureView {
        self.fallback_tex.create_view(&Default::default())
    }

    /// Lazily build the render pipeline for a blend mode.
    fn ensure_pipeline(&mut self, shared: &SharedResources, blend_mode: AlphaBlendMode) {
        if self.pipelines.contains_key(&blend_mode) {
            return;
        }
        let pipeline = shared.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_render_pipeline"),
            layout: Some(&self.render_pl),
            vertex: wgpu::VertexState {
                module: &self.render_shader,
                entry_point: Some("vs_particle"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // Particles now draw INSIDE the shared transparents rpass
            // (F-full) so they sort against glass/water, which binds the
            // scene depth as a READ-ONLY attachment (Depth32Float). The
            // pipeline must declare a matching depth-stencil state, but
            // keeps depth WRITE OFF and compare ALWAYS: occlusion + the
            // soft intersection still come entirely from the in-PS depth
            // fade (which samples the same depth via the DepthOnly aspect
            // view — legal because the attachment is read-only). This
            // preserves the standalone pass's exact behaviour while making
            // the texture safe to both attach and sample.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &self.render_shader,
                entry_point: Some("fs_particle"),
                compilation_options: Default::default(),
                // Two color targets to match the transparents rpass
                // (RT0 lighting_base + RT1 hdr_dark). On PC DARK_MULT=1.0
                // so RT1 == RT0; the FS writes the same color to both.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: blend_state(blend_mode),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: blend_state(blend_mode),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.insert(blend_mode, pipeline);
    }

    /// Prepare the per-frame color-render state (F-full): write
    /// `frame_params` and rebuild the group0 bind group (camera + grid +
    /// frame params + scene depth). Must run BEFORE the shared transparents
    /// rpass opens — the bind group is then bound per-batch by
    /// [`Self::draw_emitter`] from inside that rpass. `depth_view` is the
    /// scene depth sample view (DepthOnly aspect), legal to sample while
    /// the same texture is attached read-only. `exposure` is the engine
    /// view exposure; `near_clip` linearizes reverse-Z; `fade_range` is
    /// the soft-fade distance. Engine `compute_depth_fade`.
    pub fn prepare_color_render(
        &mut self,
        shared: &SharedResources,
        depth_view: &wgpu::TextureView,
        exposure: f32,
        near_clip: f32,
        fade_range: f32,
        // V_ILLUM_EXPOSURE (= render_exposure · illum_render_scale) — the
        // engine self-illum exposure used by additive / add_src×srcalpha
        // particles (frame.v2.x). Distinct from `exposure` (render_exposure,
        // used by alpha_blend), so additive embers aren't over-brightened in
        // dark scenes where auto-exposure climbs but self-illum stays fixed.
        v_illum_exposure: f32,
    ) {
        // Diagnostic overrides for the soft depth-fade (frame.v.w = range):
        //   PROTOMORPH_NO_SOFT_Z=1  -> range≈0 (sharp cutoff, no soft fade;
        //                              hard occlusion still applies)
        //   PROTOMORPH_SOFT_Z_RANGE=<wu> -> force the fade distance
        // Used to isolate whether the waterfall-foam "vanishes when looking
        // down" symptom is the soft fade (foam seen against the near water
        // pool, which is in scene_depth) vs hard occlusion / sort.
        let fade_range = if std::env::var("PROTOMORPH_NO_SOFT_Z").is_ok() {
            0.0
        } else if let Some(r) = std::env::var("PROTOMORPH_SOFT_Z_RANGE")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
        {
            r
        } else {
            fade_range
        };
        // frame.v2.y: PROTOMORPH_NO_PARTICLE_DEPTH=1 disables the particle
        // depth-occlusion block entirely (soft + hard) — isolates whether a
        // vanishing layer is occluded by scene_depth (opaque + water).
        let no_depth = if std::env::var("PROTOMORPH_NO_PARTICLE_DEPTH").is_ok() { 1.0 } else { 0.0 };
        shared.queue.write_buffer(
            &self.frame_params,
            0,
            bytemuck::cast_slice(&[
                exposure, self.size_scale, near_clip, fade_range,
                v_illum_exposure, no_depth, 0.0, 0.0,
            ]),
        );
        self.render_group0 = Some(shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_render_group0_bg"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shared.camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.frame_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                // Frame-default atmosphere (table entry 0, the cluster-
                // weighted accumulator) — fogged particles' analytic
                // scattering source. Engine: `v_atmosphere_constant_*`
                // set by the `render_method_submit` default path.
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &shared.atmosphere_buffer,
                        offset: crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET as u64,
                        size: wgpu::BufferSize::new(176), // GpuAtmosphereData: 11 vec4
                    }),
                },
            ],
        }));
    }

    /// True for a registered batch that the color pass should draw (valid
    /// index, has a built blend pipeline, NOT a distortion batch — those
    /// draw in the separate displacement pass).
    pub fn is_color_batch(&self, batch_index: usize) -> bool {
        self.batches.get(batch_index).is_some_and(|b| {
            b.renders && !b.distortion && self.pipelines.contains_key(&b.blend_mode)
        })
    }

    /// Number of registered render batches.
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Draw exactly ONE emitter-instance's particles (engine
    /// `c_particle_emitter_gpu::render`): bind the owning batch's pipeline +
    /// color group0 + material, then one instanced draw per `(base, count)`
    /// span (a grid row's occupied slots, `base = 16*(row+1)-used`). Used by
    /// `TransparentDispatch::EmitterInstance` so each emitter sorts at its
    /// own world depth. No-op if group0 isn't prepared or the batch isn't a
    /// drawable color batch. Caller re-binds its own group0 after (the
    /// particle group0 layout differs from the structure layout).
    pub fn draw_emitter<'rp>(
        &'rp self,
        rpass: &mut wgpu::RenderPass<'rp>,
        batch_index: usize,
        spans: &[(u32, u32)],
    ) {
        let Some(group0) = self.render_group0.as_ref() else { return };
        let Some(batch) = self.batches.get(batch_index) else { return };
        if batch.distortion || !batch.renders {
            return;
        }
        let Some(pipeline) = self.pipelines.get(&batch.blend_mode) else { return };
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, group0, &[]);
        rpass.set_bind_group(1, &batch.bind_group, &[]);
        for &(base, count) in spans {
            if count == 0 {
                continue;
            }
            rpass.draw(0..6, base..base + count);
        }
    }

    /// Reallocate the distortion targets when the viewport changes. Called
    /// from the renderer's `resize`.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 || self.distortion_size == (width, height) {
            return;
        }
        let (accum_view, scene_copy, scene_copy_view) =
            create_distortion_targets(device, width, height);
        self.distortion_accum_view = accum_view;
        self.scene_copy = scene_copy;
        self.scene_copy_view = scene_copy_view;
        self.distortion_size = (width, height);
    }

    /// Apply distortion particles to the scene. Three steps (engine
    /// `submit_distortions` → `apply_distortions`):
    ///   1. Draw distortion batches into the accumulation buffer (cleared to
    ///      0.5, additive) — each writes a screen-space displacement delta.
    ///   2. Copy the current scene color into the ping-pong source.
    ///   3. Full-screen warp: sample the scene copy at `uv + (accum-0.5)`,
    ///      write the warped result back into the scene target.
    /// No-op when no distortion batches are loaded. `strength` scales the
    /// displacement (UV fraction); `near_clip`/`fade_range` feed the same
    /// soft depth fade as the color pass.
    pub fn render_distortion(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        scene_texture: &wgpu::Texture,
        scene_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        near_clip: f32,
        fade_range: f32,
        strength: f32,
    ) {
        if !self.batches.iter().any(|b| b.distortion) {
            return;
        }
        shared.queue.write_buffer(
            &self.distortion_params,
            0,
            bytemuck::cast_slice(&[0.0f32, self.size_scale, near_clip, fade_range, strength, 0.0, 0.0, 0.0]),
        );
        let group0 = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_distortion_group0_bg"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shared.camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.distortion_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                // Layout requires the atmosphere slot; the distortion FS
                // ignores color_add, so the binding is inert here.
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &shared.atmosphere_buffer,
                        offset: crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET as u64,
                        size: wgpu::BufferSize::new(176),
                    }),
                },
            ],
        });
        // 1. Accumulate displacement (clear to 0.5 = zero displacement).
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particle_distortion_accumulate"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.distortion_accum_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.5, g: 0.5, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &group0, &[]);
            pass.set_pipeline(&self.distortion_pipeline);
            for (bi, batch) in self.batches.iter().enumerate() {
                if !batch.distortion || !batch.renders {
                    continue;
                }
                let bi = bi as u32;
                pass.set_bind_group(1, &batch.bind_group, &[]);
                let mut row = 0u32;
                let rows = PARTICLE_ROW_COUNT;
                while row < rows {
                    if self.row_batch[row as usize] != bi {
                        row += 1;
                        continue;
                    }
                    let start = row;
                    while row < rows && self.row_batch[row as usize] == bi {
                        row += 1;
                    }
                    pass.draw(0..6, start * 16..row * 16);
                }
            }
        }
        // 2. Copy the scene color into the ping-pong source.
        encoder.copy_texture_to_texture(
            scene_texture.as_image_copy(),
            self.scene_copy.as_image_copy(),
            wgpu::Extent3d {
                width: self.distortion_size.0,
                height: self.distortion_size.1,
                depth_or_array_layers: 1,
            },
        );
        // 3. Warp the scene by the accumulated displacement.
        let apply_bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_distortion_apply_bg"),
            layout: &self.apply_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.scene_copy_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.distortion_accum_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.apply_sampler) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particle_distortion_apply"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.apply_pipeline);
        pass.set_bind_group(0, &apply_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Map the grid back and log the first `n` slots' unpacked state —
    /// pos/vel/age — the spawn (Phase 2) + integrate (Phase 3)
    /// verification. Synchronous (blocks on poll); debug-only.
    fn debug_readback(&self, shared: &SharedResources, n: u32) {
        let bytes = PARTICLE_GRID_SIZE as u64 * RAW_PARTICLE_BYTES;
        let mut encoder = shared
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("particle_readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.grid, 0, &self.readback, 0, bytes);
        shared.queue.submit(Some(encoder.finish()));

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = shared
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        // Block until the map callback fires (debug-only path).
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        for slot in 0..n as usize {
            let base = slot * 20;
            let px = f32::from_bits(words[base]);
            let py = f32::from_bits(words[base + 1]);
            let pz = f32::from_bits(words[base + 2]);
            let size = f32::from_bits(words[base + 3]);
            let vel_lo = words[base + 4];
            let vel_hi = words[base + 5];
            let vx = f16_to_f32((vel_lo & 0xFFFF) as u16);
            let vy = f16_to_f32((vel_lo >> 16) as u16);
            let vz = f16_to_f32((vel_hi & 0xFFFF) as u16);
            let speed = (vx * vx + vy * vy + vz * vz).sqrt();
            // time words: [12]=(birth,invlife), [13]=(age,intensity)
            let invlife = f16_to_f32((words[base + 12] >> 16) as u16);
            let age = f16_to_f32((words[base + 13] & 0xFFFF) as u16);
            eprintln!(
                "[particles]   slot {slot}: pos=({px:.2}, {py:.2}, {pz:.2}) size={size:.3} \
                 vel=({vx:.3}, {vy:.3}, {vz:.3}) |v|={speed:.3} age={age:.3} invlife={invlife:.3}",
            );
        }
        drop(data);
        self.readback.unmap();
    }
}

/// Decode an IEEE half (u16) to f32. (wgpu readback gives us raw bytes;
/// the `half` crate is already a dependency but a local decoder keeps
/// this debug helper self-contained.)
fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

#[cfg(test)]
mod eval_tests {
    use blam_tags::tag_function::{
        periodic_function_evaluate, transition_function_evaluate, FUNCTION_TABLES,
    };

    fn tbyte(base: usize, k: usize) -> f32 {
        FUNCTION_TABLES[base + k] as f32 / 255.0
    }

    /// The combined eval-engine + update kernel (with the new LUT binding 9
    /// and the rewritten transition/periodic/spline2 paths) must be valid
    /// WGSL. naga parses + validates exactly as wgpu does at runtime.
    #[test]
    fn wgsl_update_shader_validates() {
        let src = format!(
            "{}\n{}",
            include_str!("../../../assets/halo_shaders/particle_eval.wgsl"),
            include_str!("../../../assets/halo_shaders/particle_update.wgsl"),
        );
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("eval+update WGSL parse failed: {e:?}"));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("eval+update WGSL validation failed: {e:?}"));
    }

    /// The particle render shader (with the new `MatParams.edge` field +
    /// the edge-fade VS term) must be valid WGSL.
    #[test]
    fn wgsl_render_shader_validates() {
        let src = include_str!("../../../assets/halo_shaders/particle_render.wgsl");
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("particle_render WGSL parse failed: {e:?}"));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("particle_render WGSL validation failed: {e:?}"));
    }

    #[test]
    fn function_tables_present() {
        assert_eq!(FUNCTION_TABLES.len(), 18432); // 7*1024 transition + 11*1024 periodic
    }

    /// Endpoints are exact grid hits (no interpolation): transition(t, 0)=row[0],
    /// transition(t, 1)=row[1023]. Verified against the dllcache table bytes.
    #[test]
    fn transition_endpoints_exact() {
        for ftype in 1u8..=7 {
            let base = (ftype as usize - 1) * 1024;
            let lo = transition_function_evaluate(ftype, 0.0);
            let hi = transition_function_evaluate(ftype, 1.0);
            assert!((lo - tbyte(base, 0)).abs() < 1e-4, "type {ftype} x=0");
            assert!((hi - tbyte(base, 1023)).abs() < 1e-4, "type {ftype} x=1");
        }
        // type 0 = identity
        assert!((transition_function_evaluate(0, 0.37) - 0.37).abs() < 1e-6);
    }

    /// Engine semantics: type 6 = "one" (≈1 everywhere), 7 = "zero" (≈0),
    /// 5 = cosine ease (≈0.5 mid), and "early" front-loads vs "late".
    #[test]
    fn transition_shapes() {
        for &x in &[0.1f32, 0.3, 0.5, 0.7, 0.9] {
            assert!((transition_function_evaluate(6, x) - 1.0).abs() < 1e-3, "one@{x}");
            assert!(transition_function_evaluate(7, x).abs() < 1e-3, "zero@{x}");
            // early (1) front-loads → above late (3) on the interior.
            assert!(
                transition_function_evaluate(1, x) > transition_function_evaluate(3, x),
                "early>late @{x}"
            );
        }
        let mid = transition_function_evaluate(5, 0.49);
        assert!((mid - 0.486).abs() < 0.02, "cosine ease mid = {mid}");
    }

    /// Each interior sample must lie within the bracket of the two table
    /// entries the engine index straddles (robust to ±1 fp index drift).
    #[test]
    fn transition_interior_bracketed() {
        for ftype in 1u8..=5 {
            let base = (ftype as usize - 1) * 1024;
            for n in 1..=19 {
                let x = n as f32 / 20.0;
                let got = transition_function_evaluate(ftype, x);
                let k = (x * 1023.0) as usize;
                let lo = tbyte(base, k.saturating_sub(1));
                let hi = tbyte(base, (k + 2).min(1023));
                let (a, b) = (lo.min(hi), lo.max(hi));
                assert!(
                    got >= a - 1e-3 && got <= b + 1e-3,
                    "type {ftype} x {x}: {got} not in [{a},{b}]"
                );
            }
        }
    }

    /// periodic(0)≡1, periodic(1)≡0, and cosine (type 2) starts at the table
    /// top (byte 0xff → 1.0) at time 0.
    #[test]
    fn periodic_constants_and_cosine_phase() {
        for &t in &[0.0f32, 0.21, 0.5, 0.83] {
            assert!((periodic_function_evaluate(0, t) - 1.0).abs() < 1e-6, "one@{t}");
            assert!(periodic_function_evaluate(1, t).abs() < 1e-3, "zero@{t}");
        }
        // type 2 cosine: row byte[0] = 0xff → ~1.0 at time 0.
        let c0 = periodic_function_evaluate(2, 0.0);
        assert!((c0 - 1.0).abs() < 1e-3, "cosine@0 = {c0}");
    }
}
