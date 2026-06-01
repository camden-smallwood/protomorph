//! Mirror of `Ares/source/render/screen_postprocess.{h,cpp}`.
//! Halo's `c_screen_postprocess` is all statics; we mirror the
//! public surface as free functions in the `c_screen_postprocess`
//! module that take the owning `ScreenPostprocess` (which holds
//! the wgpu pipelines + scratch surfaces + the three settings
//! statics) by `&mut`.
//!
//! Bodies sourced from dllcache (per-method addresses noted on
//! each fn).

use crate::halo::render::shared::SharedResources;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use wgpu::util::DeviceExt;

/// Exposure-readback slot state. Stored in `AtomicU8`.
const EXPOSURE_SLOT_IDLE: u8 = 0;
const EXPOSURE_SLOT_IN_FLIGHT: u8 = 1;
const EXPOSURE_SLOT_READY: u8 = 2;

/// Mirror of `c_screen_postprocess::s_settings` (Ares 56 bytes).
#[derive(Debug, Clone, Copy)]
pub struct ScreenPostprocessSettings {
    pub postprocess: bool,                 // 0x0
    pub display_alpha: i32,                // 0x4
    pub accum: i32,                        // 0x8
    pub bloom_source: i32,                 // 0xC
    pub accum_filter: i32,                 // 0x10
    pub bloom: i32,                        // 0x14
    pub bling: i32,                        // 0x18
    pub persist: bool,                     // 0x1C
    pub downsample: i32,                   // 0x20
    pub postprocess_exposure: f32,         // 0x24
    pub allow_debug_display: bool,         // 0x28
    pub tone_curve: bool,                  // 0x29
    pub tone_curve_white_point: f32,       // 0x2C
    pub auto_exposure_lock: bool,          // 0x30
    pub debug_mode: i32,                   // 0x34
    /// Engine `s_bUseLightShafts` (`@ 0x18118aef5`, default `0x01`).
    /// Toggled at runtime by `lightshafts_enable @ 0x1806B6E40`. Engine
    /// gates `render_lightshafts` on `(cfxs.flags & 2) != 0 && this`.
    ///
    /// **Protomorph default: false** — the engine-faithful 5-pass
    /// chain is implemented, but on s3d_* MP maps the authored cfxs
    /// (`intensity=4`, `falloff_radius=0.4`) cascades against our
    /// dim-scene rendering (auto-exposure pegs at ~+2.4 stops, then
    /// bloom + lightshafts tip mid-tones past the lum threshold) into
    /// a screen-wide white-out. Revisit once the underlying lighting
    /// rendering produces brighter raw values (see
    /// `project_lighting_exposure`). Set
    /// `PROTOMORPH_ENABLE_LIGHTSHAFTS=1` to force-enable for testing.
    pub s_b_use_light_shafts: bool,
}

/// Halo's `c_screen_postprocess::accept_edited_settings @ 0x18034d890`
/// applies the user's editable settings on game-tick. Defaults sourced
/// verbatim from engine static initializer at
/// `.data:000000018118AEB0 c_screen_postprocess::x_settings_internal`:
///
/// ```
/// c_screen_postprocess::s_settings <1, 0, 1, 2, 1, 1, 0, 0, 0, 1.0, 1, 1, 1.0, 0, 0>
/// ```
///
/// Field order from the IDA struct dump: `postprocess, display_alpha,
/// accum, bloom_source, accum_filter, bloom, bling, persist, downsample,
/// postprocess_exposure, allow_debug_display, tone_curve,
/// tone_curve_white_point, auto_exposure_lock, debug_mode`.
const DEFAULT_SETTINGS: ScreenPostprocessSettings = ScreenPostprocessSettings {
    postprocess: true,
    display_alpha: 0,
    accum: 1,
    bloom_source: 2,
    accum_filter: 1,
    bloom: 1,
    bling: 0,
    persist: false,
    downsample: 0,
    postprocess_exposure: 1.0,
    allow_debug_display: true,
    tone_curve: true,
    // Engine ships this as 1.0 (NOT 1.4938016). The 1.4938016 constant
    // appears in the tone-curve coefficient computation as a hardcoded
    // scaler (`(1.4938016/WHITE) * 1.0041494` etc.). For
    // WHITE=1.4938016 the math reduces to: max=1.4938, linear=1.0041,
    // quadratic=0, cubic=-0.15 — saturates at c=1.4938. This is the
    // softer rolloff the protomorph-debug-branch hardcoded as constants
    // before the cbuffer plumbing landed. Engine's
    // `c_screen_postprocess::x_settings_internal @ .data:18118AEB0`
    // ships WHITE=1.0 (which produces a steeper `(1.0, 1.5, 0, -0.5)`
    // curve), but we use the softer rolloff because the engine curve
    // pegs midtones higher and combined with our bloom-pyramid
    // shift-by-one-level (more spread, less concentrated) ends up
    // visibly darker than engine reference. Until a full
    // auto-exposure+bloom calibration pass, this knob gives the
    // closer-to-engine LOOK at the cost of literal cbuffer fidelity.
    // See `final_composite_pass.rs::compute_tone_curve_constants` and
    // `/tmp/divergence_tracker.md` D-415.
    tone_curve_white_point: 1.4938016,
    auto_exposure_lock: false,
    debug_mode: 0,
    // Engine ships this as `0x01`; we default to false because of the
    // bloom × lightshafts cascade against dim-scene auto-exposure. See
    // the field doc for the full story.
    s_b_use_light_shafts: false,
};

/// Subset of `c_camera_fx_values` that the bloom + bling chain
/// reads each frame. Halo updates this from the active
/// `c_camera_fx_settings` (cfxs tag) via interpolation; we snapshot
/// the tag's authored values directly until the per-frame update
/// path is wired.
#[derive(Debug, Clone, Copy)]
pub struct CameraFxBloomDefaults {
    pub bloom_point: f32,
    pub bloom_inherent: f32,
    pub bloom_intensity: f32,
    pub exposure_anti_bloom: f32,
    pub bloom_large_color: [f32; 3],
    pub bloom_medium_color: [f32; 3],
    pub bloom_small_color: [f32; 3],
    pub bling_intensity: f32,
    pub bling_size: f32,
    pub bling_angle: f32,
    pub bling_count: u16,
    /// Engine `c_camera_fx_values::m_exposure.m_exposure` — the
    /// auto-exposure result in stops. Feeds the bloom extract's `v21`:
    /// `v21 = scripted_exposure_apply(m_exposure) + g_exposure_stops + m_exposure_boost`.
    /// With scripted/stops/boost all zero, this is the dominant term.
    pub exposure_stops: f32,
    /// Engine `c_camera_fx_values::m_exposure_boost`. Additive stops
    /// offset feeding `v21`. Default 0.
    pub exposure_boost: f32,
}

impl CameraFxBloomDefaults {
    pub const DEFAULT: Self = Self {
        bloom_point: 1.0,
        bloom_inherent: 0.05,
        bloom_intensity: 0.66943294,
        exposure_anti_bloom: 0.0,
        bloom_large_color: [1.0, 1.0, 1.0],
        bloom_medium_color: [1.0, 1.0, 1.0],
        bloom_small_color: [1.0, 1.0, 1.0],
        bling_intensity: 0.0,
        bling_size: 1.0,
        bling_angle: 0.0,
        bling_count: 0,
        exposure_stops: 0.0,
        exposure_boost: 0.0,
    };

    /// Snapshot the bloom + bling fields from a parsed
    /// `.camera_fx_settings` (cfxs) tag. `bloom_intensity` is normalized
    /// against Halo's canonical `0.66943294` ratio so the existing
    /// `v23` extract-scale formula matches; the tag's bloom_intensity
    /// is then a pure (0..1) modulator on top.
    pub fn from_tag(cfx: &blam_tags::camera_fx_settings::CameraFxSettings) -> Self {
        // Each parameter struct carries flags + value (+ blend tuning
        // that the engine consumes per-frame in
        // c_camera_fx_values::update). For the snapshot defaults we
        // only need the static values; per-frame interpolation lands
        // separately when `update` is fully ported.
        let large = cfx.bloom_large_color.effective_color();
        let medium = cfx.bloom_medium_color.effective_color();
        let small = cfx.bloom_small_color.effective_color();
        Self {
            bloom_point: cfx.bloom_point.value,
            bloom_inherent: cfx.bloom_inherent.value,
            bloom_intensity: cfx.bloom_intensity.value,
            exposure_anti_bloom: cfx.auto_exposure_anti_bloom.value,
            bloom_large_color: [large.red, large.green, large.blue],
            bloom_medium_color: [medium.red, medium.green, medium.blue],
            bloom_small_color: [small.red, small.green, small.blue],
            bling_intensity: cfx.bling_intensity.value,
            bling_size: cfx.bling_size.value,
            bling_angle: cfx.bling_angle_deg.value,
            bling_count: cfx.bling_count.value,
            // No m_exposure at tag-snapshot time — runtime path populates
            // these via `from_camera_fx_values`. Default 0 (identity).
            exposure_stops: 0.0,
            exposure_boost: 0.0,
        }
    }

    /// Refresh the snapshot from per-frame-blended `c_camera_fx_values`.
    /// Called by `Renderer::setup_camera_fx_parameters` after the L3
    /// master `update` runs — keeps the postprocess pass reading the
    /// blended values until those readers move to direct `CameraFxValues`
    /// access (L7).
    pub fn from_camera_fx_values(
        values: &crate::halo::render::camera_fx_settings::CameraFxValues,
    ) -> Self {
        Self {
            bloom_point: values.bloom_point,
            bloom_inherent: values.bloom_inherent,
            bloom_intensity: values.bloom_intensity,
            exposure_anti_bloom: values.exposure_anti_bloom,
            bloom_large_color: [
                values.bloom_large_color.x,
                values.bloom_large_color.y,
                values.bloom_large_color.z,
            ],
            bloom_medium_color: [
                values.bloom_medium_color.x,
                values.bloom_medium_color.y,
                values.bloom_medium_color.z,
            ],
            bloom_small_color: [
                values.bloom_small_color.x,
                values.bloom_small_color.y,
                values.bloom_small_color.z,
            ],
            bling_intensity: values.bling_intensity,
            bling_size: values.bling_size,
            bling_angle: values.bling_angle,
            bling_count: values.bling_count,
            // Engine `c_camera_fx_values::m_exposure.m_exposure` +
            // `m_exposure_boost`. Used in postprocess_player_view to
            // compute `v21 = scripted_apply(m_exposure) + g_stops + boost`.
            exposure_stops: values.exposure.exposure,
            exposure_boost: values.exposure_boost,
        }
    }
}

/// Common postprocess uniforms — `ps_postprocess_pixel_size`,
/// `ps_postprocess_scale`, `intensity_vector`,
/// `DARK_COLOR_MULTIPLIER`. The `kernel_5` shader uses an extended
/// variant; see [`Kernel5Uniforms`] below.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostprocessUniforms {
    pixel_size: [f32; 4],
    scale: [f32; 4],
    intensity_vector: [f32; 4],
    dark_color_multiplier: [f32; 4],
}

/// `kernel_5` shader's full uniform block — base postprocess fields
/// plus the 5 kernel entries (offset_x, offset_y, weight, _).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Kernel5Uniforms {
    pixel_size: [f32; 4],
    scale: [f32; 4],
    kernel: [[f32; 4]; 5],
}

/// `spike_blur_*.hlsl` cbuffer (4 vec4s). Slots map to
/// `c_screen_postprocess` register 0x4D0000..0x4D0003.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpikeBlurUniforms {
    /// `source_pixel_size` — `(1/w, 1/h, w, h)` of the source surface.
    source_pixel_size: [f32; 4],
    /// `(start_offset.xy, step_delta.zw)`. Halo uses `start = (0,0)`.
    offset_delta: [f32; 4],
    /// `(R, G, B, _)` starting weight — Halo uses `(1, 1, 1)`.
    initial_color: [f32; 4],
    /// `(R, G, B, _)` per-step multiplicative falloff (chromatic).
    delta_color: [f32; 4],
}

/// `ssao.wgsl`'s uniform layout. 7 vec4s = 112 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniforms {
    texcoord_scale: [f32; 4],
    frustum_scale: [f32; 4],
    ssao_params: [f32; 4],
    mv_row_0: [f32; 4],
    mv_row_1: [f32; 4],
    mv_row_2: [f32; 4],
    depth_constants: [f32; 4],
}

/// `radial_blur.wgsl`'s `BlurWeights` cbuffer — 16 sample weights
/// packed as 4 vec4s. Engine slot `0x430002` written 4 vec4s at a time.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RadialBlurWeights {
    w0: [f32; 4],
    w1: [f32; 4],
    w2: [f32; 4],
    w3: [f32; 4],
}

/// All 9 ported postprocess pipelines. `apply_binary_op_ex` /
/// `copy` / `gaussian_blur_fixed` dispatch by shader index — the
/// indices come from `rasterizer_globals.explicit_shaders[]`:
///   10 = add, 11 = blur_11_horizontal, 12 = blur_11_vertical,
///   34 = kernel_5, 48 = downsample_4x4_block_bloom,
///   66 = bloom_add_alpha1, 67 = downsample_4x4_block_bloom_ldr.
struct Pipelines {
    apply_bloom_curve: wgpu::RenderPipeline,
    horizontal_gaussian_blur: wgpu::RenderPipeline,
    vertical_gaussian_blur: wgpu::RenderPipeline,
    downsample_4x4_block_bloom: wgpu::RenderPipeline,
    /// Engine shader 67 — `downsample_4x4_block_bloom_ldr.hlsl`. Runs
    /// after shader 48 to produce the 1/4-resolution `aux_exposure_5`
    /// (the actual pyramid input). Re-extracts bloom from the scene at
    /// half the resolution + MAX-combines with a point-sample of the
    /// half-res `aux_bloom`. Without this step the bloom pyramid
    /// operates one level higher than engine, producing visibly
    /// sharper/brighter highlights.
    downsample_4x4_block_bloom_ldr: wgpu::RenderPipeline,
    blur_11_horizontal: wgpu::RenderPipeline,
    blur_11_vertical: wgpu::RenderPipeline,
    add: wgpu::RenderPipeline,
    bloom_add_alpha1: wgpu::RenderPipeline,
    kernel_5: wgpu::RenderPipeline,
    /// Single-tap copy. Used to ferry the bloom_add_alpha1 result
    /// out of `aux_exposure_7` back into `aux_bloom` so the blit
    /// bind group reads from a stable texture handle.
    postprocess_copy: wgpu::RenderPipeline,
    /// `spike_blur_horizontal.hlsl` — linear-Y point-X 8-tap.
    /// Used by `bling_generate` for vertical-ish spike directions.
    /// Default-blend variant (overwrites destination).
    spike_blur_h: wgpu::RenderPipeline,
    /// Additive-blend variant of `spike_blur_h` for accumulating
    /// successive spikes into `aux_star`.
    spike_blur_h_add: wgpu::RenderPipeline,
    /// `spike_blur_vertical.hlsl` — linear-X point-Y 8-tap.
    /// Used by `bling_generate` for horizontal-ish spike directions.
    spike_blur_v: wgpu::RenderPipeline,
    /// Additive-blend variant of `spike_blur_v`.
    spike_blur_v_add: wgpu::RenderPipeline,
    /// `bloom_star_combine.wgsl` — final bloom + bling combine with
    /// independent per-component intensities. Mirrors the bloom+star
    /// contribution computed inside Halo's `copy_accumulation_target`.
    bloom_star_combine: wgpu::RenderPipeline,
    /// `downsample_4x4_block.wgsl` — plain 4-tap box filter used to
    /// build the multi-scale bloom pyramid (1/2 → 1/4 → 1/8).
    /// Halo's shader index 49.
    downsample_4x4_block: wgpu::RenderPipeline,
    /// `lightshafts.wgsl` — engine shader index 97. Sun-cone extract
    /// for `c_screen_postprocess::render_lightshafts`. Uses bgl_2tex
    /// (color + depth).
    lightshafts_extract: wgpu::RenderPipeline,
    /// `radial_blur.wgsl` — engine shader index 98. 16-tap log-spaced
    /// radial blur from the sun screen position. Uses a custom
    /// bind group layout (`bgl_radial_blur`) — 1 texture + 2 uniforms
    /// (PostprocessParams + BlurWeights).
    radial_blur: wgpu::RenderPipeline,
    /// Additive-blend variant of `postprocess_copy` — used for the
    /// final lightshafts blit into the scene HDR (engine `copy(51)` +
    /// `set_alpha_blend_mode(additive)`).
    lightshafts_additive_blit: wgpu::RenderPipeline,
    /// `ssao.wgsl` — engine shader 95. SSAO compute. Custom layout
    /// (`bgl_ssao`).
    ssao: wgpu::RenderPipeline,
    /// `ssao_blur.wgsl` — engine shader 96 (first pass). Diagonal-cross
    /// blur with depth discontinuity rejection. Uses bgl_2tex (depth +
    /// ssao views).
    ssao_blur: wgpu::RenderPipeline,
    /// Same shader as `ssao_blur` but with `blend_mode=multiply` and
    /// targets the scene HDR (Rg11b10Ufloat) — engine's V-blur pass
    /// that ALSO multiplies AO into accum_HDR in one dispatch.
    ssao_blur_multiply: wgpu::RenderPipeline,
    /// `exposure_downsample.wgsl` — engine shader index 35. Reads
    /// 1/16-res HDR (intensity in `.a`) + `auto_exposure_weight`
    /// (slot 1) and outputs a single half-float average luminance
    /// in log2-stops to a 1×1 R16Float ring slot. Driven by
    /// `c_screen_postprocess::downsample_generate` step 3.
    exposure_downsample: wgpu::RenderPipeline,
}

/// Half-resolution bloom scratch surface. Each maps to a Halo
/// `e_surface` enum value the bloom chain references; we own them
/// dedicated to ScreenPostprocess until SurfaceTable allocation
/// (canonical g_surface_group_descriptions[] walker) lands.
struct BloomSurface {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// Owning struct for the wgpu pipelines + scratch surfaces backing
/// `c_screen_postprocess`'s static methods.
pub struct ScreenPostprocess {
    /// Per-pass `wgpu::BindGroup` reuse cache. Keys fold in the
    /// addresses of input views/samplers; values are `Arc<BindGroup>`.
    /// Wrapped in `RefCell` because most pass methods take `&self`.
    /// Cleared on resize when intermediate views are rebuilt.
    bg_cache: std::cell::RefCell<crate::halo::render::bind_group_cache::BindGroupCache>,
    /// `c_screen_postprocess::x_editable_settings` (Ares static).
    pub editable_settings: ScreenPostprocessSettings,
    /// `c_screen_postprocess::x_settings_internal` (Ares static).
    pub settings_internal: ScreenPostprocessSettings,
    /// Snapshot of the active per-scenario `c_camera_fx_values`. Halo
    /// reads these every frame from the camera; we keep a default
    /// here until the scenario camera_fx pipe is wired.
    pub camera_fx: CameraFxBloomDefaults,

    pipelines: Pipelines,

    bgl_1tex: wgpu::BindGroupLayout,
    bgl_2tex: wgpu::BindGroupLayout,
    /// Depth (slot 0, sample_type=Depth) + color (slot 2, filterable
    /// float) + samplers + uniforms. Used by the SSAO blur passes
    /// which can't share `bgl_2tex` because the depth view's sample
    /// type is Depth, not Float.
    bgl_ssao_blur: wgpu::BindGroupLayout,
    /// Color at slot 0, depth at slot 2 — engine convention for
    /// `c_screen_postprocess::render_lightshafts`. Mirror of
    /// `bgl_ssao_blur` with the depth/color slots swapped.
    bgl_color_depth: wgpu::BindGroupLayout,
    bgl_kernel5: wgpu::BindGroupLayout,
    /// 1 tex + 2 uniforms — for radial_blur (color sampler + standard
    /// PostprocessParams + per-pass BlurWeights).
    bgl_radial_blur: wgpu::BindGroupLayout,

    sampler_bilinear: wgpu::Sampler,
    sampler_point: wgpu::Sampler,

    uniform_buffer: wgpu::Buffer,
    kernel5_uniform_buffer: wgpu::Buffer,
    spike_uniform_buffer: wgpu::Buffer,
    /// 64-byte uniform buffer holding the 16 radial-blur sample weights
    /// (4 vec4s). Updated each radial_blur dispatch.
    radial_blur_weights_buffer: wgpu::Buffer,
    /// 4×4 RGBA8 random-rotation noise table for SSAO (engine
    /// `g_ssao_noise_texture`, built once at init via `InitTextureSSAONoise`).
    ssao_noise_texture: wgpu::Texture,
    ssao_noise_view: wgpu::TextureView,
    /// Wrap+point sampler for the noise texture (engine sets these
    /// modes on sampler 2 inside `render_ssao`).
    ssao_noise_sampler: wgpu::Sampler,
    /// Full-res AO output (engine: `_surface_albedo` repurposed during
    /// `render_ssao`). RGBA8 single-channel content (only .r matters).
    ssao_raw: BloomSurface,
    /// Full-res blurred AO. Engine writes to `_surface_post_LDR`
    /// after the H-blur pass, then re-reads as input for V-blur+multiply.
    ssao_blurred: BloomSurface,
    /// `bgl_ssao` — depth + normal + noise + uniforms (3 textures, 3
    /// samplers, 1 uniform = 7 bindings).
    bgl_ssao: wgpu::BindGroupLayout,
    /// SSAO uniforms (TEXCOORD_SCALE, FRUSTUM_SCALE, SSAO_PARAMS, MV
    /// rows, depth_constants).
    ssao_uniform_buffer: wgpu::Buffer,

    /// `_surface_aux_bloom` (= 25, 1/2 res). Receives the extract
    /// output, then overwritten with the final multi-scale bloom
    /// result. Read by the final composite blit.
    aux_bloom: BloomSurface,
    /// E1 — `_surface_aux_small2` (engine, 1/2 res). Half-res copy of
    /// the HDR composite that gets a 2-pass binomial gaussian blur
    /// (existing `horizontal_gaussian_blur` + `vertical_gaussian_blur`
    /// pipelines). Read by `final_composite.wgsl` as the "blurry"
    /// source for DoF blending.
    dof_small: BloomSurface,
    /// E1 — `_surface_aux_tiny2` (engine, 1/2 res). Ping-pong target
    /// between the H-pass and V-pass blur reads.
    dof_temp: BloomSurface,
    /// `_surface_aux_star` (= 27, 1/2 res). Bling spike output. Halo
    /// overloads this with the 1/8 res "tiny" downsample inside
    /// postprocess_bloom_buffer; we keep aux_star at 1/2 res for
    /// bling and use a separate aux_tiny for the downsample chain.
    aux_star: BloomSurface,
    /// 1/2-res Rgba16Float bloom-pipeline scratch surface. The name
    /// is **misleading** — this is NOT engine surface 35 (which is
    /// `_surface_aux_exposure_7`, the last slot of the 1×1 R16Float
    /// auto-exposure ring; see `aux_exposure` below). Real role:
    /// gaussian_blur H-pass intermediate, multi-scale combine
    /// destination before final copy back to aux_bloom, spike_blur
    /// scratch. Rename to `bloom_half_temp` or similar in a future
    /// cleanup; left alone for now to keep bloom paths identical.
    aux_exposure_7: BloomSurface,
    /// `c_exposure::m_render_target_queue`-eligible surfaces 28..35,
    /// engine name `_surface_aux_exposure_0`..`_surface_aux_exposure_7`.
    /// Each slot is 1×1 `R16Float` and stores a single half-float
    /// average-luminance sample written by shader 35
    /// (`downsample_generate` step 3). `m_next_render_target` selects
    /// which slot the next frame writes into; `update_sampled_exposure`
    /// pushes the chosen slot's surface index into the 3-deep
    /// `m_render_target_queue` so the read 3 frames later picks up the
    /// correct sample. Allocated once — 1×1 doesn't depend on display
    /// size, so `resize` ignores them.
    aux_exposure: [BloomSurface; 8],
    // `c_exposure::m_next_render_target` — class-level static in engine
    // (`downsample_generate @ 0x1806B5C00:51-55`). Single counter shared
    // across all `c_exposure` instances (i.e., all player_windows).
    // Hoisted to module-scope `EXPOSURE_M_NEXT_RENDER_TARGET` in
    // `camera_fx_settings.rs`; accessed via `Exposure::next_render_target()`.
    /// `auto_exposure_weight.bitmap` — engine `default_textures[7]`,
    /// the 18×10 monochrome center-weighted metering pattern bound at
    /// sampler slot 1 in shader 35 (`exposure_downsample`). Baked
    /// constant in the binary; allocated once at init.
    auto_exposure_weight_view: wgpu::TextureView,
    auto_exposure_weight_sampler: wgpu::Sampler,
    /// 32-byte uniform for the exposure_downsample dispatch
    /// (`pixel_size`, `scale`).
    exposure_downsample_uniform: wgpu::Buffer,
    /// 8 GPU→CPU staging buffers (one per ring slot), 4 bytes each
    /// (1× R16Float = 2 bytes, padded to 4 to satisfy
    /// `COPY_BUFFER_ALIGNMENT`). The engine reads ring slots directly
    /// via `lock_read` on a 360 EDRAM-resident surface; on PC + wgpu
    /// we must stage through a `MAP_READ`-capable buffer.
    /// `schedule_exposure_readback` schedules `copy_texture_to_buffer(
    /// aux_exposure[N] → staging[N])`; `begin_exposure_map(N)` is
    /// called post-submit to start an async map; the 3-frame-old
    /// slot is read non-blocking via `try_read_exposure_sample`.
    exposure_staging_buffers: [wgpu::Buffer; 8],
    /// Per-slot map state: 0=Idle (not mapped, not in flight), 1=InFlight
    /// (map_async issued, awaiting GPU), 2=Ready (callback fired,
    /// `get_mapped_range` valid). Engine equivalent: D3D11_MAP_FLAG_DO_NOT_WAIT
    /// returning nullptr if not ready. `wgpu::Device::poll(Wait)` would
    /// drain the entire device queue (gfx-rs/wgpu#4589), so we use
    /// non-blocking `Poll` + this state machine.
    exposure_slot_state: [Arc<AtomicU8>; 8],
    /// `_surface_aux_exposure_5` (= 36) — 1/4 res. Engine's actual
    /// pyramid input: filled by shader 67 after shader 48's
    /// half-res `aux_bloom` extract. `postprocess_bloom_buffer` starts
    /// its downsample chain from here, not from `aux_bloom`, which is
    /// why the pyramid operates at 1/4 → 1/8 → 1/16 instead of
    /// 1/2 → 1/4 → 1/8.
    aux_exposure_5: BloomSurface,
    /// 1/8 res — engine's `_surface_aux_chud_overlay` reused as the
    /// medium-scale source in postprocess_bloom_buffer (engine
    /// `downsample_generate` second arg `dest_buffer_tiny`).
    aux_mini: BloomSurface,
    /// 1/16 res — engine's `_surface_aux_star` reused as the
    /// large-scale source in postprocess_bloom_buffer. Distinct from
    /// the protomorph `aux_star` field (which stays at 1/2 res for
    /// bling output).
    aux_tiny: BloomSurface,
    /// 1/8 res blur temp + medium+large combine destination.
    aux_temp_quarter: BloomSurface,
    /// 1/16 res blur temp.
    aux_temp_eighth: BloomSurface,

    /// Cached display dims so resize re-allocates with the right size.
    display_width: u32,
    display_height: u32,

    /// `g_pColorGradingTextures[0/1]` — the two 16³ RGBA8 LUT textures
    /// the engine ping-pongs between for crossfading. `update_color_grading`
    /// bakes into the inactive slot then swaps; the shader samples both
    /// and lerps by `color_grading_blend_factor`.
    color_grading_textures: [wgpu::Texture; 2],
    /// Cached views (one per texture) so callers don't re-create per
    /// frame.
    color_grading_views: [wgpu::TextureView; 2],
    /// `g_fColorGradingBlendFactor` — crossfade weight in [0, 1]. 0 =
    /// 100% texture[1] (new), 1 = 100% texture[0] (old). Decays toward
    /// 0 over `cfxs.blend_time` seconds. Animation pipe lands later;
    /// for now we snap to 0 on every update.
    color_grading_blend_factor: f32,
    /// Active/back index. The engine swaps these every time the
    /// settings change; the BACK index is where the new bake lands.
    color_grading_active: usize,
}

impl ScreenPostprocess {
    /// Cached `create_bind_group` lookup. `key` should hash the
    /// addresses of every bound resource used by the descriptor.
    /// Cleared on resize.
    #[inline]
    fn cached_bg(
        &self,
        key: u64,
        desc: impl FnOnce() -> wgpu::BindGroup,
    ) -> Arc<wgpu::BindGroup> {
        self.bg_cache.borrow_mut().get_or_create(key, desc)
    }

    pub fn new(shared: &SharedResources) -> Self {
        let device = &shared.device;

        // Bind group layouts. Halo's HLSL declares samplers at
        // sampler-register slots; our wgpu mapping has tex+sampler
        // pair at bindings (0,1), (2,3), and uniforms at the next
        // free binding.
        let bgl_1tex = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl_1tex"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl_2tex = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl_2tex"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl_kernel5 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl_kernel5"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 1 tex (color) + 2 uniforms (PostprocessParams + BlurWeights) for radial_blur.
        let bgl_radial_blur = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl_radial_blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_radial_blur = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess_pl_radial_blur"),
            bind_group_layouts: &[&bgl_radial_blur],
            immediate_size: 0,
        });

        // SSAO bgl: 3 textures + 3 samplers + 1 uniform = 7 bindings.
        let tex_entry = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let smp_entry = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        // Depth-bound slot: SSAO samples the scene depth surface, which
        // we allocate as Depth32Float. wgpu's view sample type is
        // therefore `Depth`, not `Float`. Filterable-float at slot 0
        // mismatches at bind-group create (validation: "expects sample
        // type Float, but was given a view with sample type Depth").
        let depth_tex_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl_ssao = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl_ssao"),
            entries: &[
                depth_tex_entry, smp_entry(1),  // depth (sample type = Depth)
                tex_entry(2), smp_entry(3),  // normal
                tex_entry(4), smp_entry(5),  // noise
                wgpu::BindGroupLayoutEntry {
                    // VS reads `frustum_scale` + `texcoord_scale` from the
                    // same uniform that the FS reads `ssao_params` etc. from.
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_ssao = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess_pl_ssao"),
            bind_group_layouts: &[&bgl_ssao],
            immediate_size: 0,
        });

        let layout_1tex = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess_pl_1tex"),
            bind_group_layouts: &[&bgl_1tex],
            immediate_size: 0,
        });
        let layout_2tex = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess_pl_2tex"),
            bind_group_layouts: &[&bgl_2tex],
            immediate_size: 0,
        });
        // SSAO blur passes sample the scene depth (Depth32Float) at
        // slot 0 — its sample type is `Depth`, not Float, so it can't
        // share `bgl_2tex`. Same shape as bgl_2tex otherwise.
        let bgl_ssao_blur = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl_ssao_blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_ssao_blur = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess_pl_ssao_blur"),
            bind_group_layouts: &[&bgl_ssao_blur],
            immediate_size: 0,
        });
        // Lightshafts samples color at slot 0, depth at slot 2 (engine
        // pixel-shader register convention) — needs its own layout
        // because bgl_2tex has both slots as filterable Float.
        let bgl_color_depth = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl_color_depth"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_color_depth = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess_pl_color_depth"),
            bind_group_layouts: &[&bgl_color_depth],
            immediate_size: 0,
        });
        let layout_kernel5 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postprocess_pl_kernel5"),
            bind_group_layouts: &[&bgl_kernel5],
            immediate_size: 0,
        });

        // Helper closure: build a fullscreen-pass pipeline.
        let make_pipeline_blend = |label: &str,
                                   source: &str,
                                   format: wgpu::TextureFormat,
                                   layout: &wgpu::PipelineLayout,
                                   blend: Option<wgpu::BlendState>|
         -> wgpu::RenderPipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module, entry_point: Some("vs_main"),
                    buffers: &[], compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module, entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format, blend, write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let make_pipeline = |label: &str,
                             source: &str,
                             format: wgpu::TextureFormat,
                             layout: &wgpu::PipelineLayout|
         -> wgpu::RenderPipeline {
            make_pipeline_blend(label, source, format, layout, None)
        };

        // Engine `set_alpha_blend_mode_no_cache @ 0x18066E930` packs every
        // mode with the alpha channel using the SAME factors+op as RGB.
        // We previously had `alpha: BlendComponent::REPLACE` which writes
        // src.a verbatim — wrong against engine's additive (ONE/ONE/ADD)
        // and multiply (DEST_ALPHA/ZERO/ADD).

        // Additive blend: dest = src + dest. Engine `_alpha_blend_additive`,
        // packed value 0x3C884442. Used for the multi-spike accumulator in
        // `bling_generate`.
        let additive_color = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let additive_blend = wgpu::BlendState {
            color: additive_color,
            alpha: additive_color,
        };
        // Multiply blend: dest = src * dest. Engine `_alpha_blend_multiply`,
        // packed value 0x3C84E429 → color DEST_COLOR/ZERO/ADD, alpha
        // DEST_ALPHA/ZERO/ADD. Used by the SSAO V-pass to darken
        // `accum_HDR` by AO factor.
        let multiply_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::DstAlpha,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let f = wgpu::TextureFormat::Rgba16Float;
        let pipelines = Pipelines {
            apply_bloom_curve: make_pipeline(
                "apply_bloom_curve",
                include_str!("../../../assets/shaders/apply_bloom_curve.wgsl"),
                f, &layout_1tex,
            ),
            horizontal_gaussian_blur: make_pipeline(
                "horizontal_gaussian_blur",
                include_str!("../../../assets/shaders/horizontal_gaussian_blur.wgsl"),
                f, &layout_1tex,
            ),
            vertical_gaussian_blur: make_pipeline(
                "vertical_gaussian_blur",
                include_str!("../../../assets/shaders/vertical_gaussian_blur.wgsl"),
                f, &layout_1tex,
            ),
            downsample_4x4_block_bloom_ldr: make_pipeline(
                "downsample_4x4_block_bloom_ldr",
                include_str!("../../../assets/shaders/downsample_4x4_block_bloom_ldr.wgsl"),
                f, &layout_2tex,
            ),
            downsample_4x4_block_bloom: make_pipeline(
                "downsample_4x4_block_bloom",
                include_str!("../../../assets/shaders/downsample_4x4_block_bloom.wgsl"),
                f, &layout_1tex,
            ),
            blur_11_horizontal: make_pipeline(
                "blur_11_horizontal",
                include_str!("../../../assets/shaders/blur_11_horizontal.wgsl"),
                f, &layout_1tex,
            ),
            blur_11_vertical: make_pipeline(
                "blur_11_vertical",
                include_str!("../../../assets/shaders/blur_11_vertical.wgsl"),
                f, &layout_1tex,
            ),
            add: make_pipeline(
                "add",
                include_str!("../../../assets/shaders/add.wgsl"),
                f, &layout_2tex,
            ),
            bloom_add_alpha1: make_pipeline(
                "bloom_add_alpha1",
                include_str!("../../../assets/shaders/bloom_add_alpha1.wgsl"),
                f, &layout_2tex,
            ),
            kernel_5: make_pipeline(
                "kernel_5",
                include_str!("../../../assets/shaders/kernel_5.wgsl"),
                f, &layout_kernel5,
            ),
            postprocess_copy: make_pipeline(
                "postprocess_copy",
                include_str!("../../../assets/shaders/postprocess_copy.wgsl"),
                f, &layout_1tex,
            ),
            spike_blur_h: make_pipeline(
                "spike_blur_h",
                include_str!("../../../assets/shaders/spike_blur_horizontal.wgsl"),
                f, &layout_1tex,
            ),
            spike_blur_h_add: make_pipeline_blend(
                "spike_blur_h_add",
                include_str!("../../../assets/shaders/spike_blur_horizontal.wgsl"),
                f, &layout_1tex, Some(additive_blend),
            ),
            spike_blur_v: make_pipeline(
                "spike_blur_v",
                include_str!("../../../assets/shaders/spike_blur_vertical.wgsl"),
                f, &layout_1tex,
            ),
            spike_blur_v_add: make_pipeline_blend(
                "spike_blur_v_add",
                include_str!("../../../assets/shaders/spike_blur_vertical.wgsl"),
                f, &layout_1tex, Some(additive_blend),
            ),
            bloom_star_combine: make_pipeline(
                "bloom_star_combine",
                include_str!("../../../assets/shaders/bloom_star_combine.wgsl"),
                f, &layout_2tex,
            ),
            downsample_4x4_block: make_pipeline(
                "downsample_4x4_block",
                include_str!("../../../assets/shaders/downsample_4x4_block.wgsl"),
                f, &layout_1tex,
            ),
            lightshafts_extract: make_pipeline(
                "lightshafts_extract",
                include_str!("../../../assets/shaders/lightshafts.wgsl"),
                // Engine `c_screen_postprocess::copy(97, source, _surface_albedo)`
                // writes the extracted bright-cone result at FULL-RES into
                // `_surface_albedo` (= our `albedo_view`, Rg11b10Ufloat).
                // Shader 49 then 4-tap box-averages that into 1/2-res
                // aux_bloom — and the per-pixel nonlinearity of
                // `saturate((lum-0.92)*10)` makes "mask-each-then-average"
                // produce different results from "bilinear-sample-then-mask".
                wgpu::TextureFormat::Rgba16Float, &layout_color_depth,
            ),
            radial_blur: make_pipeline(
                "radial_blur",
                include_str!("../../../assets/shaders/radial_blur.wgsl"),
                f, &layout_radial_blur,
            ),
            lightshafts_additive_blit: make_pipeline_blend(
                "lightshafts_additive_blit",
                include_str!("../../../assets/shaders/postprocess_copy.wgsl"),
                // Additive blit targets the HDR scene color
                // (`dest_view`) which protomorph allocates as
                // Rg11b10Ufloat — match the pipeline format to the
                // pass attachment. Engine `_surface_accum_HDR` is the
                // same surface ssao_blur_multiply writes to.
                wgpu::TextureFormat::Rgba16Float, &layout_1tex, Some(additive_blend),
            ),
            ssao: make_pipeline(
                "ssao",
                include_str!("../../../assets/shaders/ssao.wgsl"),
                wgpu::TextureFormat::Rgba8Unorm, &layout_ssao,
            ),
            ssao_blur: make_pipeline(
                "ssao_blur",
                include_str!("../../../assets/shaders/ssao_blur.wgsl"),
                wgpu::TextureFormat::Rgba8Unorm, &layout_ssao_blur,
            ),
            ssao_blur_multiply: make_pipeline_blend(
                "ssao_blur_multiply",
                include_str!("../../../assets/shaders/ssao_blur.wgsl"),
                wgpu::TextureFormat::Rgba16Float, &layout_ssao_blur, Some(multiply_blend),
            ),
            exposure_downsample: make_pipeline(
                "exposure_downsample",
                include_str!("../../../assets/shaders/exposure_downsample.wgsl"),
                wgpu::TextureFormat::R16Float, &layout_2tex,
            ),
        };

        let sampler_bilinear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("postprocess_bilinear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let sampler_point = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("postprocess_point"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("postprocess_uniforms"),
            contents: bytemuck::bytes_of(&PostprocessUniforms {
                pixel_size: [0.0; 4], scale: [0.0; 4],
                intensity_vector: [0.299, 0.587, 0.114, 0.0],
                dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let kernel5_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kernel5_uniforms"),
            contents: bytemuck::bytes_of(&Kernel5Uniforms {
                pixel_size: [0.0; 4], scale: [1.0; 4], kernel: [[0.0; 4]; 5],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let spike_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spike_blur_uniforms"),
            contents: bytemuck::bytes_of(&SpikeBlurUniforms {
                source_pixel_size: [0.0; 4],
                offset_delta: [0.0; 4],
                initial_color: [1.0, 1.0, 1.0, 0.0],
                delta_color: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let radial_blur_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radial_blur_weights"),
            contents: bytemuck::bytes_of(&RadialBlurWeights::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // `InitTextureSSAONoise @ 0x1806b6c10` — verbatim port. 4×4
        // RGBA8 random-rotation table. The 16 angle indices are a
        // pre-shuffled permutation of [0..15] decoded from the engine's
        // hardcoded `rndTab` constants (with the implicit overflow
        // into the adjacent `data` slot the IDA decompile shows).
        let ssao_noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao_noise"),
            size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssao_noise_data = build_ssao_noise_texels();
        shared.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ssao_noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &ssao_noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * 4),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        );
        let ssao_noise_view = ssao_noise_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let ssao_noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao_noise_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let ssao_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao_uniforms"),
            contents: bytemuck::bytes_of(&SsaoUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // `auto_exposure_weight.bitmap` — engine `default_textures[7]`,
        // the 18×10 monochrome center-weighted metering pattern bound at
        // sampler slot 1 in shader 35. Baked into the binary; uploaded
        // once at init.
        let auto_exposure_weight_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("auto_exposure_weight"),
            size: wgpu::Extent3d {
                width: AUTO_EXPOSURE_WEIGHT_WIDTH,
                height: AUTO_EXPOSURE_WEIGHT_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let weight_rgba = build_auto_exposure_weight_rgba();
        shared.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &auto_exposure_weight_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &weight_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * AUTO_EXPOSURE_WEIGHT_WIDTH),
                rows_per_image: Some(AUTO_EXPOSURE_WEIGHT_HEIGHT),
            },
            wgpu::Extent3d {
                width: AUTO_EXPOSURE_WEIGHT_WIDTH,
                height: AUTO_EXPOSURE_WEIGHT_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let auto_exposure_weight_view = auto_exposure_weight_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Engine binds sampler slot 1 with bilinear + clamp inside
        // `downsample_generate` immediately before the shader-35 copy.
        let auto_exposure_weight_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("auto_exposure_weight_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let exposure_downsample_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("exposure_downsample_uniforms"),
            contents: bytemuck::bytes_of(&PostprocessUniforms {
                pixel_size: [0.0; 4],
                scale: [0.0; 4],
                intensity_vector: [0.0; 4],
                dark_color_multiplier: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Phase C — staging buffers for the 3-frame deferred readback
        // path. wgpu's `copy_texture_to_buffer` requires
        // `bytes_per_row` to be a multiple of
        // `COPY_BYTES_PER_ROW_ALIGNMENT` (= 256), so each staging
        // buffer is sized at one full row of padding even though
        // only the first two bytes (one R16Float texel) are
        // meaningful.
        let exposure_staging_buffers: [wgpu::Buffer; 8] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "exposure_staging_0", 1 => "exposure_staging_1",
                    2 => "exposure_staging_2", 3 => "exposure_staging_3",
                    4 => "exposure_staging_4", 5 => "exposure_staging_5",
                    6 => "exposure_staging_6", _ => "exposure_staging_7",
                }),
                size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let exposure_slot_state: [Arc<AtomicU8>; 8] =
            std::array::from_fn(|_| Arc::new(AtomicU8::new(EXPOSURE_SLOT_IDLE)));

        let display_width = shared.config.width;
        let display_height = shared.config.height;
        let (w2, h2) = (display_width / 2, display_height / 2);
        let (w4, h4) = (display_width / 4, display_height / 4);
        let (w8, h8) = (display_width / 8, display_height / 8);
        let (w16, h16) = (display_width / 16, display_height / 16);
        let aux_bloom = make_bloom_surface(device, "aux_bloom", w2, h2);
        let aux_star = make_bloom_surface(device, "aux_star", w2, h2);
        let aux_exposure_7 = make_bloom_surface(device, "aux_exposure_7", w2, h2);
        let dof_small = make_bloom_surface(device, "dof_small", w2, h2);
        let dof_temp = make_bloom_surface(device, "dof_temp", w2, h2);
        let aux_exposure: [BloomSurface; 8] = std::array::from_fn(|i| {
            make_exposure_ring_surface(device, i)
        });
        // Engine pyramid: aux_exposure_5 (1/4) → aux_mini (1/8) → aux_tiny (1/16).
        // Shifted DOWN one level vs the previous protomorph layout — engine's
        // `postprocess_bloom_buffer` starts at aux_exposure_5 (1/4 res), produced
        // by shader 67 from aux_bloom (1/2 res) + composite_HDR (full res).
        let aux_exposure_5 = make_bloom_surface(device, "aux_exposure_5", w4, h4);
        let aux_mini = make_bloom_surface(device, "aux_mini", w8, h8);
        let aux_tiny = make_bloom_surface(device, "aux_tiny", w16, h16);
        let aux_temp_quarter = make_bloom_surface(device, "aux_temp_quarter", w8, h8);
        let aux_temp_eighth = make_bloom_surface(device, "aux_temp_eighth", w16, h16);
        let ssao_raw = make_ssao_surface(device, "ssao_raw", display_width, display_height);
        let ssao_blurred = make_ssao_surface(device, "ssao_blurred", display_width, display_height);

        // `c_screen_postprocess::color_grading_init` — allocates the
        // ping-pong LUT pair. Both start at identity.
        let color_grading_textures = [
            crate::halo::render::color_grading::allocate_lut_texture(device, "color_grading_lut_0"),
            crate::halo::render::color_grading::allocate_lut_texture(device, "color_grading_lut_1"),
        ];
        let color_grading_views = [
            color_grading_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            color_grading_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let identity = crate::halo::render::color_grading::bake_lut_to_vec(None);
        crate::halo::render::color_grading::upload_lut(&shared.queue, &color_grading_textures[0], &identity);
        crate::halo::render::color_grading::upload_lut(&shared.queue, &color_grading_textures[1], &identity);

        Self {
            bg_cache: std::cell::RefCell::new(
                crate::halo::render::bind_group_cache::BindGroupCache::new(),
            ),
            editable_settings: DEFAULT_SETTINGS,
            settings_internal: DEFAULT_SETTINGS,
            camera_fx: CameraFxBloomDefaults::DEFAULT,
            pipelines,
            bgl_1tex,
            bgl_2tex,
            bgl_ssao_blur,
            bgl_color_depth,
            bgl_kernel5,
            sampler_bilinear,
            sampler_point,
            uniform_buffer,
            kernel5_uniform_buffer,
            spike_uniform_buffer,
            radial_blur_weights_buffer,
            bgl_radial_blur,
            ssao_noise_texture,
            ssao_noise_view,
            ssao_noise_sampler,
            ssao_raw,
            ssao_blurred,
            bgl_ssao,
            ssao_uniform_buffer,
            aux_bloom,
            aux_star,
            aux_exposure_7,
            dof_small,
            dof_temp,
            aux_exposure,
            auto_exposure_weight_view,
            auto_exposure_weight_sampler,
            exposure_downsample_uniform,
            exposure_staging_buffers,
            exposure_slot_state,
            aux_exposure_5,
            aux_mini,
            aux_tiny,
            aux_temp_quarter,
            aux_temp_eighth,
            display_width,
            display_height,
            color_grading_textures,
            color_grading_views,
            color_grading_blend_factor: 0.0,
            color_grading_active: 0,
        }
    }

    pub fn resize(&mut self, shared: &SharedResources) {
        let device = &shared.device;
        self.display_width = shared.config.width;
        self.display_height = shared.config.height;
        // Drop cached bind groups — every cache entry references at
        // least one view that's about to be rebuilt below.
        self.bg_cache.borrow_mut().clear();
        let (w2, h2) = (self.display_width / 2, self.display_height / 2);
        let (w4, h4) = (self.display_width / 4, self.display_height / 4);
        let (w8, h8) = (self.display_width / 8, self.display_height / 8);
        self.aux_bloom = make_bloom_surface(device, "aux_bloom", w2, h2);
        self.aux_star = make_bloom_surface(device, "aux_star", w2, h2);
        self.aux_exposure_7 = make_bloom_surface(device, "aux_exposure_7", w2, h2);
        self.dof_small = make_bloom_surface(device, "dof_small", w2, h2);
        self.dof_temp = make_bloom_surface(device, "dof_temp", w2, h2);
        let (w16, h16) = (self.display_width / 16, self.display_height / 16);
        self.aux_exposure_5 = make_bloom_surface(device, "aux_exposure_5", w4, h4);
        self.aux_mini = make_bloom_surface(device, "aux_mini", w8, h8);
        self.aux_tiny = make_bloom_surface(device, "aux_tiny", w16, h16);
        self.aux_temp_quarter = make_bloom_surface(device, "aux_temp_quarter", w8, h8);
        self.aux_temp_eighth = make_bloom_surface(device, "aux_temp_eighth", w16, h16);
        self.ssao_raw = make_ssao_surface(device, "ssao_raw", self.display_width, self.display_height);
        self.ssao_blurred = make_ssao_surface(device, "ssao_blurred", self.display_width, self.display_height);
    }

    /// View into the final bloom result — the composite blit reads
    /// this and adds it to the HDR composite before tone curve.
    pub fn bloom_result_view(&self) -> &wgpu::TextureView {
        &self.aux_bloom.view
    }

    /// View into the half-res DoF blur output (E1). The composite blit
    /// reads this as the "blurry" source when DoF is active. Engine:
    /// `_surface_aux_small2` after the 2-pass binomial gaussian.
    pub fn dof_blur_view(&self) -> &wgpu::TextureView {
        &self.dof_small.view
    }
}

/// Full-res SSAO scratch surface (Rgba8Unorm). Engine: `_surface_albedo` /
/// `_surface_post_LDR` repurposed during `render_ssao`.
fn make_ssao_surface(
    device: &wgpu::Device, label: &'static str, w: u32, h: u32,
) -> BloomSurface {
    let w = w.max(1);
    let h = h.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    BloomSurface { tex, view, width: w, height: h }
}

/// `InitTextureSSAONoise @ 0x1806b6c10` — port of the engine's 4×4
/// noise builder.
///
/// Engine flow: writes 4 hardcoded `_DWORD` constants into 16 bytes of
/// stack memory (treating it as the angle-index table rndTab), then
/// loops 4×4 over `rndTab[4*v8 + 8 + v10]`. The IDA decompile's `+ 8`
/// offset overflows past `rndTab[16]` into the adjacent `data` slot —
/// the engine's stack happens to place them contiguously so the read
/// works. We rebuild the same 16-entry permutation table cleanly.
///
/// Each texel encodes a random rotation angle θ as
/// `(R, G, B, A) = ((1-sinθ)/2, (1+sinθ)/2, (1+cosθ)/2, (1+cosθ)/2)` — the
/// SSAO shader dot-products kernel offsets against `rot.xy` and `rot.zw`
/// to scatter the 8-tap kernel directions per pixel.
fn build_ssao_noise_texels() -> [u8; 4 * 4 * 4] {
    // Constants reverse-engineered from the engine's stack writes:
    //   rndTab[8..11]  = LE bytes of 0x09010800 = [0, 8, 1, 9]
    //   rndTab[12..15] =                0x0d050c04 = [4, 12, 5, 13]
    //   data[0..3]     =                0x070f030b = [11, 3, 15, 7]
    //   data[4..7]     =                0x020a0e06 = [6, 14, 10, 2]
    // Concatenated into our 16-entry angle table — clean permutation
    // of [0..15], one entry per texel.
    const ANGLE_TABLE: [u8; 16] = [
        0, 8, 1, 9,
        4, 12, 5, 13,
        11, 3, 15, 7,
        6, 14, 10, 2,
    ];
    let mut data = [0u8; 4 * 4 * 4];
    for row in 0..4_usize {
        for col in 0..4_usize {
            let idx = ANGLE_TABLE[row * 4 + col] as f32;
            // Engine: angle = (idx * 0.0625) * 2π - π
            let angle = (idx * 0.0625) * std::f32::consts::TAU - std::f32::consts::PI;
            let (s, c) = angle.sin_cos();
            let texel_idx = (row * 4 + col) * 4;
            data[texel_idx] = ((1.0 - s) * 0.5 * 255.0) as u8;     // R
            data[texel_idx + 1] = ((1.0 + s) * 0.5 * 255.0) as u8; // G
            data[texel_idx + 2] = ((1.0 + c) * 0.5 * 255.0) as u8; // B
            data[texel_idx + 3] = ((1.0 + c) * 0.5 * 255.0) as u8; // A (= B in engine)
        }
    }
    data
}

fn make_bloom_surface(
    device: &wgpu::Device, label: &'static str, w: u32, h: u32,
) -> BloomSurface {
    let w = w.max(1);
    let h = h.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    BloomSurface { tex, view, width: w, height: h }
}

const AUTO_EXPOSURE_WEIGHT_WIDTH: u32 = 18;
const AUTO_EXPOSURE_WEIGHT_HEIGHT: u32 = 10;

/// 18×10 green-channel weight bitmap, extracted verbatim from
/// `shaders/default_bitmaps/bitmaps/auto_exposure_weight.bitmap`
/// (engine `default_textures[7]`). The engine stores this as
/// "8-bit Monochrome", which the runtime expands so all four
/// channels carry the same value; we bake it directly.
static AUTO_EXPOSURE_WEIGHT_G: [u8; 180] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 3, 6, 13, 25, 40, 58, 73, 82, 82, 73, 58, 40, 25, 13, 6, 2, 1,
    2, 4, 10, 20, 37, 60, 86, 108, 122, 122, 108, 86, 60, 37, 20, 9, 4, 1,
    3, 6, 14, 29, 52, 84, 121, 153, 172, 172, 153, 121, 84, 52, 29, 13, 6, 2,
    4, 8, 17, 37, 67, 108, 155, 196, 221, 221, 196, 155, 108, 67, 37, 17, 7, 3,
    5, 9, 21, 43, 79, 127, 181, 229, 255, 255, 229, 181, 127, 79, 43, 21, 8, 3,
    5, 10, 21, 45, 81, 131, 188, 238, 255, 255, 238, 188, 131, 81, 45, 21, 9, 4,
    5, 8, 19, 38, 70, 113, 161, 204, 229, 229, 204, 161, 113, 70, 38, 19, 8, 3,
    3, 6, 13, 28, 51, 82, 117, 148, 166, 166, 148, 117, 82, 51, 28, 13, 6, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

fn build_auto_exposure_weight_rgba() -> Vec<u8> {
    let mut rgba = Vec::with_capacity(AUTO_EXPOSURE_WEIGHT_G.len() * 4);
    for &g in AUTO_EXPOSURE_WEIGHT_G.iter() {
        rgba.extend_from_slice(&[g, g, g, 0xFF]);
    }
    rgba
}

/// `_surface_aux_exposure_<i>` (engine surfaces 28..35) — 1×1
/// `R16Float` slot in the auto-exposure ring. Each slot stores a
/// single half-float average-luminance sample written by shader 35
/// in `downsample_generate`. `COPY_SRC` is needed so `Phase C`'s
/// `copy_texture_to_buffer` readback can stage the sample into a
/// mappable buffer 3 frames later.
fn make_exposure_ring_surface(device: &wgpu::Device, slot: usize) -> BloomSurface {
    static LABELS: [&str; 8] = [
        "aux_exposure_0", "aux_exposure_1", "aux_exposure_2", "aux_exposure_3",
        "aux_exposure_4", "aux_exposure_5", "aux_exposure_6", "aux_exposure_7",
    ];
    let label = LABELS[slot];
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    BloomSurface { tex, view, width: 1, height: 1 }
}

// Engine `c_screen_postprocess` static methods — collapsed into
// `impl ScreenPostprocess` per project Rust style (drop the `c_`
// namespace, keep engine method names).
impl ScreenPostprocess {
    /// `c_screen_postprocess::accept_edited_settings @ 0x18034d890`.
    pub fn accept_edited_settings(&mut self) {
        self.settings_internal = self.editable_settings;
    }

    /// `c_screen_postprocess::postprocess_final_composite @ 0x1806b49f0`.
    ///
    /// **Screenshot-render-engine path** — the only caller in the
    /// dllcache xref is `c_screenshot_render_engine::composite_single_tile`.
    /// Per-frame rendering does NOT go through this method; it goes
    /// through `postprocess_player_view`'s embedded `copy(51, accum_HDR
    /// → display)` (or `copy(93, ...)` when FXAA is on).
    ///
    /// Engine body (271 lines): looks up bloom-curve / tone-curve /
    /// color grading textures from `rasterizer_globals.default_textures`,
    /// binds them at sampler slots 0/1, the `bloom_surface` argument
    /// at slot 2, plus `depth_stencil` and `aux_small2` at slots 3/4
    /// when DoF is enabled. Writes cbuffer slots:
    ///
    ///   - `0x420000` pixel_size (1/aux_refraction.{w,h}, 1/depth.{w,h} for DoF)
    ///   - `0x1D0000` intensities = `(1, 1, inherent_scale, 0)`
    ///   - `0x1D0001` tone_curve_constants (when `use_tone_curve`):
    ///       `w0 = wp`, `w1 = (1.4938016/wp) * 1.0041494`,
    ///       `w2 = 0`, `w3 = -(1.4938016/wp)^3 * 0.15`
    ///     Without tone curve: `(65535, 1.4938016/wp, 0, 0)`.
    ///   - `0x1D0003` bloom_xform — window_rect normalization.
    ///   - `0x1E0000`/`0x1E0001` depth + DoF blur sigma (DoF only).
    ///
    /// Then `set_scissor_rect(target_rect)` + `draw_fullscreen_quad` +
    /// reset all sampler/scissor state.
    ///
    /// **Protomorph status: STUB** — we don't render screenshots, so
    /// no caller path lights up. The corresponding per-frame logic
    /// lives in `FinalCompositePass`. Adding this method keeps the
    /// engine API surface complete; if/when screenshot rendering
    /// lands, the body fills in.
    #[allow(clippy::too_many_arguments)]
    pub fn postprocess_final_composite(
        &self,
        _explicit_shader_index: i32,
        _display_surface: Option<&wgpu::TextureView>,
        _bloom_surface: Option<&wgpu::TextureView>,
        _inherent_scale: f32,
        _use_tone_curve: bool,
        _true_srgb_output: bool,
        _use_depth_of_field: bool,
        _z_near: f32,
        _z_far: f32,
        _target_rect: Option<[i32; 4]>,
        _window_rect: Option<[f32; 4]>,
    ) -> bool {
        // Engine returns 1 on success, 0 if set_explicit_shaders failed.
        // Stub — see doc comment.
        false
    }

    /// `c_screen_postprocess::render_ssao @ 0x1806b50e0`. Screen-space
    /// ambient occlusion. Engine flow:
    ///
    ///   1. Gate on `m_ssao.m_flags & 2` (bit 1 = enable).
    ///   2. Resolve `radius`, `intensity`, `sample_z_threshold` from
    ///      cfxs (or `dbg_ssao*` debug overrides when set to ≠ -1).
    ///   3. `setup_rasterizer_for_postprocess(true)`.
    ///   4. Write cbuffer slots:
    ///      - `0x4E0000` TEXCOORD_SCALE = `(w/4, h/4, 0, 0)` of `accum_HDR`.
    ///      - `0x4F0000` SSAO_PARAMS = `(radius, 16.0, 1/sample_z_threshold, intensity)`.
    ///      - `0x4F0001` & `0x4E0001` FRUSTUM_SCALE = derived from
    ///        `tan(fov/2) * aspect`, etc.
    ///      - `0x4F0002` world_to_view 3 columns (view-space basis).
    ///   5. Bind `post_HDR` as sampler 1 (point/clamp), `g_ssao_noise_texture`
    ///      as sampler 2 (point/wrap).
    ///   6. `copy(95, depth_stencil → albedo)` — SSAO compute pass.
    ///   7. Bind `albedo` as sampler 1.
    ///   8. `copy(96, depth_stencil → post_LDR)` — H blur.
    ///   9. `set_alpha_blend_mode(multiply)`. Bind `post_LDR` as sampler 1.
    ///  10. `copy(96, depth_stencil → accum_HDR)` — V blur + composite into HDR.
    ///  11. `set_alpha_blend_mode(opaque)`.
    ///
    /// Three pipeline dispatches:
    ///   1. `ssao` (shader 95) → `ssao_raw` (Rgba8Unorm).
    ///   2. `ssao_blur` (shader 96 default) → `ssao_blurred` (Rgba8Unorm).
    ///   3. `ssao_blur_multiply` (shader 96 + multiply blend) → `dest_view`
    ///      (`lighting_base`, Rg11b10Ufloat).
    #[allow(clippy::too_many_arguments)]
    pub fn render_ssao(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        ssao: Option<&blam_tags::camera_fx_settings::SsaoBlock>,
        view: glam::Mat4,
        vertical_fov_rad: f32,
        aspect: f32,
        z_near: f32,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        dest_view: &wgpu::TextureView,
    ) {
        let Some(ssao) = ssao else { return };
        if !ssao.is_enabled() {
            return;
        }

        encoder.push_debug_group("render_ssao");

        // === Engine cbuffer values (lines 49-91 of render_ssao body) ===
        let w = self.display_width as f32;
        let h = self.display_height as f32;

        // 0x4E0000 TEXCOORD_SCALE = (W*0.25, H*0.25, 0, 0) — used by VS to
        // form noise-tiling UVs (4×4 noise repeated across screen).
        let texcoord_scale = [w * 0.25, h * 0.25, 0.0, 0.0];

        // 0x4F0000 SSAO_PARAMS = (radius, 16.0, 1/sample_z_threshold, intensity).
        // FLOAT_16_0 in IDA = 16.0 (unused by our shader but kept faithful).
        let inv_z_threshold = 1.0 / ssao.sample_z_threshold.max(1.0e-6);
        let ssao_params = [ssao.radius, 16.0, inv_z_threshold, ssao.intensity];

        // 0x4F0001/0x4E0001 FRUSTUM_SCALE = (2*tan(fov/2)*aspect, 2*tan(fov/2),
        //                                    1/(2*tan(fov/2)*aspect), 1/(2*tan(fov/2))).
        let tan_half = (vertical_fov_rad * 0.5).tan();
        let f_y = tan_half * 2.0;
        let f_x = f_y * aspect;
        let frustum_scale = [f_x, f_y, 1.0 / f_x, 1.0 / f_y];

        // 0x4F0002 PS_REG_SSAO_MV_*: rows of view matrix. Row 0 NOT negated;
        // rows 1 and 2 negated (engine XORs with sign-bit mask `_xmm`).
        // Halo's view-space convention flips Y/Z relative to its world space.
        let r0 = view.row(0);
        let r1 = view.row(1);
        let r2 = view.row(2);
        let mv_row_0 = [r0.x, r0.y, r0.z, 0.0];
        let mv_row_1 = [-r1.x, -r1.y, -r1.z, 0.0];
        let mv_row_2 = [-r2.x, -r2.y, -r2.z, 0.0];

        // Engine reads depth_constants from a separate global slot we don't
        // model yet. For our reverse-Z infinite-far projection,
        // `lin_depth = 1 / (raw_depth * (1/z_near))`.
        let depth_constants = [1.0 / z_near.max(1.0e-6), 0.0, 0.0, 0.0];

        let uniforms = SsaoUniforms {
            texcoord_scale,
            frustum_scale,
            ssao_params,
            mv_row_0,
            mv_row_1,
            mv_row_2,
            depth_constants,
        };
        shared.queue.write_buffer(&self.ssao_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // The blur passes only consume `pixel_size` — first vec4 of the
        // standard `PostprocessUniforms`. Reuse `uniform_buffer` so we
        // don't tie up a second slot.
        let blur_params = PostprocessUniforms {
            pixel_size: [1.0 / w, 1.0 / h, w, h],
            scale: [0.0; 4],
            intensity_vector: [0.0; 4],
            dark_color_multiplier: [0.0; 4],
        };
        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&blur_params));

        // ============================================================
        // Pass 1 — SSAO compute (shader 95).
        // ============================================================
        // Engine: `copy(95, depth_stencil, _surface_albedo, ...)` with
        //   sampler 1 = post_HDR (clamp/point) — unused by our shader.
        //   sampler 2 = g_ssao_noise_texture (wrap/point).
        use crate::halo::render::bind_group_cache::{key_from_ptrs, ptr_addr};
        let ssao_bg = self.cached_bg(
            key_from_ptrs(&[ptr_addr(depth_view), ptr_addr(normal_view), ptr_addr(&self.bgl_ssao)]),
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao_bg"),
                layout: &self.bgl_ssao,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.ssao_noise_view) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.ssao_noise_sampler) },
                    wgpu::BindGroupEntry { binding: 6, resource: self.ssao_uniform_buffer.as_entire_binding() },
                ],
            }),
        );
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_raw.view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            rp.set_pipeline(&self.pipelines.ssao);
            rp.set_bind_group(0, ssao_bg.as_ref(), &[]);
            rp.draw(0..3, 0..1);
        }

        // ============================================================
        // Pass 2 — Diagonal-cross blur (shader 96, default blend).
        // ============================================================
        // Engine: `copy(96, depth_stencil, _surface_post_LDR, ...)`.
        // We write to ssao_blurred (own surface) instead of overloading
        // the LDR scratch. Inputs: depth (slot 0) + ssao_raw (slot 2).
        let blur_bg_h = self.cached_bg(
            key_from_ptrs(&[ptr_addr(depth_view), ptr_addr(&self.ssao_raw.view), ptr_addr(&self.bgl_ssao_blur)]),
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao_blur_h_bg"),
                layout: &self.bgl_ssao_blur,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.ssao_raw.view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                    wgpu::BindGroupEntry { binding: 4, resource: self.uniform_buffer.as_entire_binding() },
                ],
            }),
        );
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_blur"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_blurred.view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            rp.set_pipeline(&self.pipelines.ssao_blur);
            rp.set_bind_group(0, blur_bg_h.as_ref(), &[]);
            rp.draw(0..3, 0..1);
        }

        // ============================================================
        // Pass 3 — Multiply AO into scene HDR (shader 96 + multiply blend).
        // ============================================================
        // Engine: `set_alpha_blend_mode(_alpha_blend_multiply)` then
        // `copy(96, depth_stencil, _surface_accum_HDR, ...)`. Same shader
        // as pass 2 but the blend factor `(Dst, Zero)` makes the output
        // multiply against the destination — darkening lit pixels by AO.
        let blur_bg_v = self.cached_bg(
            key_from_ptrs(&[ptr_addr(depth_view), ptr_addr(&self.ssao_blurred.view), ptr_addr(dest_view), ptr_addr(&self.bgl_ssao_blur)]),
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao_blur_multiply_bg"),
                layout: &self.bgl_ssao_blur,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.ssao_blurred.view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                    wgpu::BindGroupEntry { binding: 4, resource: self.uniform_buffer.as_entire_binding() },
                ],
            }),
        );
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao_blur_multiply"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dest_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            rp.set_pipeline(&self.pipelines.ssao_blur_multiply);
            rp.set_bind_group(0, blur_bg_v.as_ref(), &[]);
            rp.draw(0..3, 0..1);
        }

        encoder.pop_debug_group();
    }

    /// `c_screen_postprocess::render_lightshafts @ 0x1806b55c0`. Volumetric
    /// god-ray / sun-shaft effect. Faithful port of the dllcache body
    /// — 5 passes, mirroring engine surface routing.
    ///
    ///   1. **Extract** (shader 97 = `lightshafts.wgsl`): source HDR +
    ///      scene depth → `full_res_scratch_view` (= engine
    ///      `_surface_albedo`, FULL res). Per-pixel falloff cone +
    ///      depth-clamp + lum-clamp mask. **Must run at full-res** —
    ///      the `saturate((lum - g_tint.w) * 10)` is a near-step
    ///      function; engine's 4-tap downsample averages 4 INDEPENDENT
    ///      mask evaluations, which differs sharply from masking a
    ///      pre-averaged sample.
    ///   2. **Downsample** (shader 49 = `downsample_4x4_block.wgsl`):
    ///      `full_res_scratch → aux_bloom` (1/2-each-dim). 4-tap box
    ///      average at ±1 source-pixel offsets, divided by 4.
    ///   3. **Radial blur 1** (shader 98 = `radial_blur.wgsl`):
    ///      `aux_bloom → aux_exposure_7`. Quadratic kernel `(i+1)²/sum`,
    ///      tint = `intensity_scaled * tint.rgb`.
    ///   4. **Radial blur 2** (shader 98): `aux_exposure_7 → aux_bloom`,
    ///      uniform `weights[i] = 1/16`, tint = `(1, 1, 1)`. Smooths
    ///      quadratic-kernel striations.
    ///   5. **Additive blit** (shader 51 = `postprocess_copy` with
    ///      additive blend): `aux_bloom → dest_view`.
    ///
    /// `dest_view` should be the scene HDR (caller's `lighting_base`).
    /// `intensity_scaled = intensity * saturate(sun_world_w / z_near)`
    /// — gates out shafts when sun is behind the camera.
    ///
    /// Engine cbuffer slots → our packed `PostprocessParams`:
    ///   - `0x420001 g_tint` ↔ `params.scale`
    ///   - `0x370001 g_sun_pos` ↔ `params.intensity_vector` (repurposed)
    ///   - `0x370002 g_inner_size` ↔ `params.dark_color_multiplier` (repurposed)
    ///   - `0x430001 g_center_scale` ↔ `params.intensity_vector` (radial pass)
    ///   - `0x430002 BLUR_WEIGHTS0..3` ↔ separate `radial_blur_weights_buffer`
    pub fn render_lightshafts(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        lightshafts: Option<&blam_tags::camera_fx_settings::LightshaftsBlock>,
        view: glam::Mat4,
        projection: glam::Mat4,
        vertical_fov_rad: f32,
        aspect: f32,
        z_near: f32,
        source_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        full_res_scratch_view: &wgpu::TextureView,
        dest_view: &wgpu::TextureView,
    ) {
        let Some(ls) = lightshafts else { return };
        if !ls.is_enabled() {
            return;
        }

        // Diagnostic gate — `PROTOMORPH_DISABLE_LIGHTSHAFTS=1` early-returns.
        if std::env::var_os("PROTOMORPH_DISABLE_LIGHTSHAFTS").is_some() {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| eprintln!("[lightshafts] disabled via PROTOMORPH_DISABLE_LIGHTSHAFTS"));
            return;
        }

        // One-shot dump of the cfxs lightshafts block. Helps diagnose
        // brightness/blur bugs by surfacing the per-scenario slider values.
        static DUMPED: std::sync::Once = std::sync::Once::new();
        DUMPED.call_once(|| {
            eprintln!(
                "[lightshafts] flags={:#06x} intensity={} intensity_clamp={} \
                 falloff_radius={} depth_clamp={} blur_radius={} \
                 tint=({:.3}, {:.3}, {:.3}) heading={}° pitch={}°",
                ls.flags,
                ls.intensity,
                ls.intensity_clamp,
                ls.falloff_radius,
                ls.depth_clamp,
                ls.blur_radius,
                ls.tint.red,
                ls.tint.green,
                ls.tint.blue,
                ls.heading,
                ls.pitch,
            );
        });

        encoder.push_debug_group("render_lightshafts");

        // === Sun direction in world space (engine lines 93-104) ===
        let pitch = ls.pitch.to_radians();
        let heading = ls.heading.to_radians();
        let sin_p = pitch.sin();
        let sun_world = glam::Vec4::new(
            heading.cos() * sin_p,
            heading.sin() * sin_p,
            pitch.cos(),
            0.0,
        );

        // === World → clip → screen (engine lines 82-91, 105-110) ===
        // Engine combines projection × view via `combine_projection_and_view_matrix`.
        // glam's `Mat4::mul` is column-vector convention — same as engine
        // after its row-shuffle transpose.
        let proj_view = projection * view;
        let sun_clip = proj_view * sun_world;
        let sun_w = sun_clip.w;
        let inv_w = if sun_w.abs() > 1.0e-6 { 1.0 / sun_w } else { 0.0 };
        let sun_ndc_x = sun_clip.x * inv_w;
        let sun_ndc_y = sun_clip.y * inv_w;
        let sun_screen_x = (sun_ndc_x + 1.0) * 0.5;
        let sun_screen_y = 0.5 - sun_ndc_y * 0.5;

        // === Falloff scales (engine lines 124-130) ===
        let tan_half_fov = (0.5 * vertical_fov_rad).tan();
        let inv_falloff = 1.0 / ls.falloff_radius.max(1.0e-6);
        let outer_x = inv_falloff * tan_half_fov * aspect;
        let outer_y = inv_falloff * tan_half_fov;
        // Engine: 999.99994 = ~1000 (inner cone is 1000× tighter than outer,
        // i.e. effectively only the immediate sun pixels). The literal is
        // the float closest to 1000 representable.
        let inner_x = tan_half_fov * aspect * 999.99994;
        let inner_y = tan_half_fov * 999.99994;

        // === Depth clamp (engine lines 133-134) ===
        let depth_clamp_scale = 2.0 / ls.depth_clamp.max(1.0e-6);
        let depth_clamp_bias = -2.0;

        // === Intensity scaling (engine lines 135-141) ===
        // intensity_scaled = intensity * saturate(sun_w / z_near) — gates
        // out shafts when sun is behind camera.
        let v40 = sun_w / z_near.max(1.0e-6);
        let v39 = v40.clamp(0.0, 1.0);
        let mut intensity_scaled = v39 * ls.intensity;

        // Diagnostic — `PROTOMORPH_LIGHTSHAFTS_INTENSITY_SCALE=0.25`
        // multiplies the final intensity by 0.25 (etc.). Lets us bisect
        // whether engine math differs by a constant scale somewhere.
        if let Ok(scale_str) = std::env::var("PROTOMORPH_LIGHTSHAFTS_INTENSITY_SCALE") {
            if let Ok(scale) = scale_str.parse::<f32>() {
                static SCALE_LOGGED: std::sync::Once = std::sync::Once::new();
                SCALE_LOGGED.call_once(|| {
                    eprintln!("[lightshafts] intensity scaled by env override {}", scale);
                });
                intensity_scaled *= scale;
            }
        }

        let tint_r = intensity_scaled * ls.tint.red;
        let tint_g = intensity_scaled * ls.tint.green;
        let tint_b = intensity_scaled * ls.tint.blue;

        // Runtime-dependent diagnostic — sun position + intensity scaling
        // + cone shape. Throttled to once per ~3s by frame skipping; the
        // values change as the camera moves but for a static scene
        // give us enough to debug.
        static RUNTIME_DUMPED: std::sync::Once = std::sync::Once::new();
        RUNTIME_DUMPED.call_once(|| {
            let tan_half_fov = (0.5 * vertical_fov_rad).tan();
            eprintln!(
                "[lightshafts:runtime] sun_clip=({:.3},{:.3},{:.3},{:.3}) \
                 sun_screen=({:.3},{:.3}) intensity_scaled={:.3} \
                 (sun_w/z_near={:.3}) tan_half_fov={:.3} aspect={:.3} z_near={:.4} \
                 outer=({:.3},{:.3}) tint=({:.3},{:.3},{:.3})",
                sun_clip.x, sun_clip.y, sun_clip.z, sun_clip.w,
                sun_screen_x, sun_screen_y,
                intensity_scaled, v40,
                tan_half_fov, aspect, z_near,
                (1.0 / ls.falloff_radius.max(1.0e-6)) * tan_half_fov * aspect,
                (1.0 / ls.falloff_radius.max(1.0e-6)) * tan_half_fov,
                tint_r, tint_g, tint_b,
            );
        });

        // ============================================================
        // Pass 1 — Extract bright sun-cone pixels.
        // ============================================================
        // Pack PostprocessParams: pixel_size, g_tint, g_sun_pos, g_inner_size.
        let extract_params = PostprocessUniforms {
            pixel_size: [
                1.0 / self.display_width as f32,
                1.0 / self.display_height as f32,
                self.display_width as f32,
                self.display_height as f32,
            ],
            scale: [0.0, 0.0, 0.0, ls.intensity_clamp],          // g_tint (only .w used)
            intensity_vector: [sun_screen_x, sun_screen_y, outer_x, outer_y],  // g_sun_pos
            dark_color_multiplier: [inner_x, inner_y, depth_clamp_scale, depth_clamp_bias],  // g_inner_size
        };
        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&extract_params));

        // Lightshafts uses bgl_color_depth (color at slot 0,
        // Depth32Float at slot 2). bgl_2tex's filterable-float at slot
        // 2 mismatches the depth view's sample type.
        let extract_bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lightshafts_extract"),
            layout: &self.bgl_color_depth,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                wgpu::BindGroupEntry { binding: 4, resource: self.uniform_buffer.as_entire_binding() },
            ],
        });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lightshafts_extract"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // Engine writes shader 97 output to `_surface_albedo`
                    // at FULL res. We use `full_res_scratch_view` (=
                    // `albedo_view`) for the same purpose — it's free
                    // by the time render_lightshafts runs.
                    view: full_res_scratch_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            rp.set_pipeline(&self.pipelines.lightshafts_extract);
            rp.set_bind_group(0, &extract_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // ============================================================
        // Pass 2 — Engine `copy(49, _surface_albedo, _surface_aux_bloom)`:
        // 4-tap box-average downsample from FULL res to 1/2-each-dim.
        // Critical because shader-97 extract's lum-clamp is non-linear:
        // averaging 4 masked values ≠ masking 1 averaged value.
        // ============================================================
        let downsample_params = PostprocessUniforms {
            pixel_size: [
                1.0 / self.display_width as f32,
                1.0 / self.display_height as f32,
                self.display_width as f32,
                self.display_height as f32,
            ],
            scale: [1.0, 1.0, 1.0, 1.0],
            intensity_vector: [0.0; 4],
            dark_color_multiplier: [0.0; 4],
        };
        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&downsample_params));
        let downsample_bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lightshafts_downsample"),
            layout: &self.bgl_1tex,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(full_res_scratch_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buffer.as_entire_binding() },
            ],
        });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lightshafts_downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.aux_bloom.view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            rp.set_pipeline(&self.pipelines.downsample_4x4_block);
            rp.set_bind_group(0, &downsample_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // ============================================================
        // Pass 3 — First radial blur (quadratic weights + tint).
        // ============================================================
        // weights[i] = (i+1)² / sum_of_squares (engine lines 184-216).
        let mut sq_weights = [0.0f32; 16];
        let mut sum = 0.0f32;
        for i in 0..16 {
            let w = ((i + 1) * (i + 1)) as f32;
            sq_weights[i] = w;
            sum += w;
        }
        let inv_sum = 1.0 / sum;
        for w in sq_weights.iter_mut() {
            *w *= inv_sum;
        }
        let mut quad_weights = RadialBlurWeights::default();
        quad_weights.w0 = [sq_weights[0], sq_weights[1], sq_weights[2], sq_weights[3]];
        quad_weights.w1 = [sq_weights[4], sq_weights[5], sq_weights[6], sq_weights[7]];
        quad_weights.w2 = [sq_weights[8], sq_weights[9], sq_weights[10], sq_weights[11]];
        quad_weights.w3 = [sq_weights[12], sq_weights[13], sq_weights[14], sq_weights[15]];
        shared.queue.write_buffer(&self.radial_blur_weights_buffer, 0, bytemuck::bytes_of(&quad_weights));

        let blur1_params = PostprocessUniforms {
            pixel_size: [
                1.0 / self.aux_bloom.width as f32,
                1.0 / self.aux_bloom.height as f32,
                self.aux_bloom.width as f32,
                self.aux_bloom.height as f32,
            ],
            scale: [tint_r, tint_g, tint_b, 1.0],                                  // g_tint
            intensity_vector: [sun_screen_x, sun_screen_y, ls.blur_radius, ls.blur_radius],  // g_center_scale
            dark_color_multiplier: [0.0; 4],
        };
        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&blur1_params));
        // Engine: `copy(98, _surface_aux_bloom, _surface_aux_exposure_5, ...)`.
        self.dispatch_radial_blur(shared, encoder, &self.aux_bloom.view, &self.aux_exposure_7.view, "radial_blur_pass1");

        // ============================================================
        // Pass 4 — Second radial blur (uniform 1/16 weights).
        // ============================================================
        let uniform_weights = RadialBlurWeights {
            w0: [0.0625; 4], w1: [0.0625; 4], w2: [0.0625; 4], w3: [0.0625; 4],
        };
        shared.queue.write_buffer(&self.radial_blur_weights_buffer, 0, bytemuck::bytes_of(&uniform_weights));

        // Engine line 235: pos_scale.x = pow(blur_radius + 1, 0.0625) - 1.
        // Smaller scale = tighter sample spread for smoothing pass.
        let pass2_scale = (ls.blur_radius + 1.0).powf(0.0625) - 1.0;
        let blur2_params = PostprocessUniforms {
            pixel_size: blur1_params.pixel_size,
            scale: [1.0, 1.0, 1.0, 1.0],
            intensity_vector: [sun_screen_x, sun_screen_y, pass2_scale, pass2_scale],
            dark_color_multiplier: [0.0; 4],
        };
        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&blur2_params));
        // Engine: `copy(98, _surface_aux_exposure_5, _surface_aux_bloom, ...)`.
        self.dispatch_radial_blur(shared, encoder, &self.aux_exposure_7.view, &self.aux_bloom.view, "radial_blur_pass2");

        // ============================================================
        // Pass 5 — Additive blit into scene HDR.
        // ============================================================
        // Engine: copy(51, aux_bloom → dest_surface) with set_alpha_blend_mode(additive).
        // Our `lightshafts_additive_blit` pipeline bakes the additive blend.
        let blit_params = PostprocessUniforms {
            pixel_size: blur1_params.pixel_size,
            scale: [1.0, 1.0, 1.0, 1.0],
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&blit_params));
        let blit_bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lightshafts_blit"),
            layout: &self.bgl_1tex,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.aux_bloom.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buffer.as_entire_binding() },
            ],
        });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lightshafts_additive_blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dest_view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            rp.set_pipeline(&self.pipelines.lightshafts_additive_blit);
            rp.set_bind_group(0, &blit_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        encoder.pop_debug_group();
    }

    /// Internal helper — dispatch the radial_blur pipeline with the
    /// already-uploaded uniform buffers.
    fn dispatch_radial_blur(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        dest: &wgpu::TextureView,
        label: &str,
    ) {
        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bgl_radial_blur,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.radial_blur_weights_buffer.as_entire_binding() },
            ],
        });
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        rp.set_pipeline(&self.pipelines.radial_blur);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// `c_screen_postprocess::setup_rasterizer_for_postprocess @ 0x1806b40a0`.
    ///
    /// Engine body sets D3D11 immediate-context state for postprocess
    /// passes:
    ///   - clear all 16 sampler texture bindings
    ///   - if `clear_targets`: detach depth + render targets 1/2/3
    ///   - alpha_blend = opaque, cull = off, scissor = off
    ///   - z_buffer = off, color_write = RGBA
    ///   - PIX debug marker push/pop
    ///
    /// In wgpu, pipeline state is baked into the pipeline at creation
    /// (not set on the encoder), so the alpha_blend / cull / z_buffer /
    /// color_write are properties of the postprocess pipelines we
    /// already create — calling this is a structural marker, not a
    /// state mutation. Stale sampler bindings between passes don't
    /// occur in wgpu either (every render_pass starts with no
    /// bindings), so `clear_sampler_textures` is implicit.
    ///
    /// We push a debug group on the encoder so renderdoc / nsight see
    /// the same scope boundaries the engine emits via PIX. `clear_targets`
    /// is logged but no-op — render targets bind per-pass, not globally.
    pub fn setup_rasterizer_for_postprocess(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        clear_targets: bool,
    ) {
        let label = if clear_targets {
            "setup_postprocess (clear_targets)"
        } else {
            "setup_postprocess"
        };
        encoder.push_debug_group(label);
        encoder.pop_debug_group();
    }

    /// `c_screen_postprocess::color_grading_init @ 0x1806ca070`. Engine
    /// allocates `g_pColorGradingTextures[0/1]` + their staging pair.
    /// We allocate them inline in `new()` (since wgpu's
    /// `queue.write_texture` handles staging implicitly); this method
    /// re-uploads identity LUTs to both slots and resets the swap
    /// state. Called from a future `c_rasterizer::initialize_after_device_creation_or_reset`
    /// port and from any explicit "reset color grading" path.
    pub fn color_grading_init(&mut self, queue: &wgpu::Queue) {
        let identity = crate::halo::render::color_grading::bake_lut_to_vec(None);
        crate::halo::render::color_grading::upload_lut(
            queue,
            &self.color_grading_textures[0],
            &identity,
        );
        crate::halo::render::color_grading::upload_lut(
            queue,
            &self.color_grading_textures[1],
            &identity,
        );
        self.color_grading_blend_factor = 0.0;
        self.color_grading_active = 0;
    }

    /// `c_screen_postprocess::color_grading_release @ 0x1806ca170`.
    /// Engine releases the D3D resource pointers. wgpu's RAII handles
    /// the actual textures via Drop; this method just resets blend
    /// state so a re-init starts clean.
    pub fn color_grading_release(&mut self) {
        self.color_grading_blend_factor = 0.0;
        self.color_grading_active = 0;
    }

    /// `c_screen_postprocess::update_color_grading @ 0x1806ca1d0`. Bake
    /// the authored `s_color_grading_parameter` into the inactive LUT
    /// slot, then swap. Engine: when settings change, swap
    /// `g_pColorGradingTextures[0/1]`, set `g_fColorGradingBlendFactor
    /// = 1 - g_fColorGradingBlendFactor`, then bake into the new
    /// "back" slot (now slot 1 after the swap). Crossfade decays via
    /// the per-frame `update` path (not yet ported).
    ///
    /// `cg = None` (or `cg.flags & 2 == 0`) → identity LUT, matching
    /// the engine's `LABEL_137` fast path.
    pub fn update_color_grading(
        &mut self,
        queue: &wgpu::Queue,
        cg: Option<&blam_tags::camera_fx_settings::ColorGradingBlock>,
        b_update: bool,
    ) {
        // Engine swaps when bUpdate || s_bUseColorGradingOld != s_bUseColorGrading.
        // Without per-frame state animation we just always swap on
        // explicit update so the new bake lands in the freshly-active
        // slot.
        if b_update {
            self.color_grading_active = 1 - self.color_grading_active;
            self.color_grading_blend_factor = 1.0 - self.color_grading_blend_factor;
        }
        let back_slot = 1 - self.color_grading_active;
        let baked = crate::halo::render::color_grading::bake_lut_to_vec(cg);
        crate::halo::render::color_grading::upload_lut(
            queue,
            &self.color_grading_textures[back_slot],
            &baked,
        );
    }

    /// View into the currently-active 16³ color grading LUT. The
    /// final composite shader samples this + the inactive view and
    /// lerps by [`color_grading_blend_factor`].
    pub fn color_grading_active_view(&self) -> &wgpu::TextureView {
        &self.color_grading_views[self.color_grading_active]
    }

    /// View into the inactive (back) LUT — what the engine calls
    /// `g_pColorGradingTextures[1]` when slot 0 is active.
    pub fn color_grading_back_view(&self) -> &wgpu::TextureView {
        &self.color_grading_views[1 - self.color_grading_active]
    }

    /// `g_fColorGradingBlendFactor` — uniform fed to the composite
    /// shader's lerp.
    pub fn color_grading_blend_factor(&self) -> f32 {
        self.color_grading_blend_factor
    }

    /// `c_screen_postprocess::postprocess_player_view @ 0x1806b4160`.
    ///
    /// Engine-faithful port of `postprocess_player_view @ 0x1806b4160`.
    /// Decompile body sequence (229 lines):
    ///
    ///   1. Read `s_screen_effect_settings` saturate/desaturate/hue,
    ///      contrast, color_filter, color_floor → `c_hue_saturation_control`.
    ///   2. Compute gamma = `clamp(gamma_enhance + 1 - gamma_reduce, 0.01, 10)`,
    ///      write to cbuffer slot `0x420005`.
    ///   3. **If postprocess disabled**: `setup_rasterizer_for_postprocess(true)`,
    ///      `copy(7, screenshot_composite_depth → display, scale=(exposure,exposure,exposure,1))`,
    ///      return.
    ///   4. `setup_rasterizer_for_postprocess(true)`.
    ///   5. `v21 = scripted_exposure_apply(exposure) + g_exposure_stops + boost`,
    ///      `v22 = max(1, pow(2, v21 * anti_bloom))`, `v23 = v22 * tone_curve_white_point * 0.66943294`.
    ///   6. Set cbuffer `0x1C0000 = 0` (xmm zero).
    ///   7. `apply_binary_op_ex(48, composite_cubemap → aux_bloom, scale=(v23*bloom_point, bloom_inherent, 1, 1))`.
    ///   8. `apply_binary_op_ex(67, aux_bloom + composite_depth → aux_exposure_5, same scale)`.
    ///   9. `postprocess_bloom_buffer(aux_exposure_5, camera, _res_default)` — multi-scale combine.
    ///  10. **If DoF enabled**: `copy(51, composite_depth → aux_small2)`.
    ///  11. **If postprocess_bloom_buffer != 0** (debug bypass): `copy(7, ... → display)`, return.
    ///  12. `lightshafts_active = (lightshafts.flags & 2) && s_bUseLightShafts`.
    ///  13. **If DoF**: `gaussian_blur(aux_small2, aux_tiny2, sigma)`, `resolve(aux_small2)`.
    ///  14. Pick `dest = (FXAA || lightshafts) ? accum_HDR : display`.
    ///  15. `copy_accumulation_target(dest, g_star_buffer, 1, bloom_intensity, bling_intensity, 0.5, dof, znear, zfar, dof_settings)` — bloom+bling+persist+DoF combine.
    ///  16. **If lightshafts**: `render_lightshafts(accum_HDR → accum_HDR)`. If FXAA off: `copy(51, accum_HDR → display)`, return.
    ///  17. **FXAA path**: write pixel_size to `0x1F0000`, `copy(93, accum_HDR → display)`.
    ///
    /// **Protomorph status: PARTIAL — engine-shape orchestrator.** Runs
    /// the bloom/bling chain via the engine-named helpers. Items listed
    /// below are stubbed since their full ports haven't landed.
    ///
    /// - **Step 1** (`screen_effects` → hue/saturation): no plumbing.
    ///   The `c_hue_saturation_control` cbuffer is uploaded by `FinalCompositePass`.
    /// - **Step 3** (postprocess-disabled fast path): not yet wired —
    ///   our existing `if !postprocess: return` skips bloom but still
    ///   runs `FinalCompositePass`, which is close-enough since both
    ///   paths end in a `copy` to display.
    /// - **Step 8** (depth-mixed extract via shader 67): skipped — needs
    ///   `downsample_4x4_block_bloom_ldr.wgsl` with depth input.
    /// - **Step 10/13** (DoF): deferred.
    /// - **Step 15** (`copy_accumulation_target`): partial — the bloom+bling
    ///   combine matches via `bloom_star_combine`; persist + DoF deferred.
    /// - **Step 16** (`render_lightshafts`): stubbed at signature level.
    /// - **Step 17** (FXAA): deferred.
    #[allow(clippy::too_many_arguments)]
    pub fn postprocess_player_view(
        &mut self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        composite_hdr: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_fx: Option<&blam_tags::camera_fx_settings::CameraFxSettings>,
        view: glam::Mat4,
        projection: glam::Mat4,
        vertical_fov_rad: f32,
        aspect: f32,
        z_near: f32,
        auto_exposure_sensitivity: f32,
        scripted: &crate::halo::render::camera_fx_settings::ScriptedExposure,
        g_exposure_stops: f32,
    ) -> Option<u32> {
        // Step 1 — screen_effects → hue/saturation/gamma. DEFERRED.
        // Step 2 — gamma to cbuffer 0x420005. DEFERRED.

        // Step 3 — postprocess-disabled fast path.
        if !self.settings_internal.postprocess {
            // Engine: copy(7, composite_depth → display, scale=(exp,exp,exp,1)).
            // Our equivalent is letting FinalCompositePass do the blit.
            return None;
        }

        encoder.push_debug_group("postprocess_player_view");

        // Diagnostic — `PROTOMORPH_DISABLE_BLOOM=1` clears aux_bloom and
        // skips the bloom/bling chain entirely. final_composite reads
        // zero bloom, leaving just the raw lighting_base.
        if std::env::var_os("PROTOMORPH_DISABLE_BLOOM").is_some() {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| eprintln!("[bloom] disabled via PROTOMORPH_DISABLE_BLOOM"));
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aux_bloom_clear_for_disable_bloom"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.aux_bloom.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            encoder.pop_debug_group();
            // No bloom pyramid means no `downsample_generate` either —
            // skip the auto-exposure sample this frame. The integrator
            // falls back to `last_actual_brightness` in
            // `setup_camera_fx_parameters` per its no-fresh-sample path.
            return None;
        }

        // Step 4 — setup_rasterizer_for_postprocess(true).
        self.setup_rasterizer_for_postprocess(encoder, true);

        // One-shot bloom-cfxs dump. Enable with `PROTOMORPH_LOG_BLOOM=1`.
        if std::env::var_os("PROTOMORPH_LOG_BLOOM").is_some() {
            static BLOOM_DUMPED: std::sync::Once = std::sync::Once::new();
            BLOOM_DUMPED.call_once(|| {
                eprintln!(
                    "[bloom] cfxs: point={:.4} inherent={:.4} intensity={:.4} \
                     large=({:.3},{:.3},{:.3}) medium=({:.3},{:.3},{:.3}) \
                     small=({:.3},{:.3},{:.3}) anti_bloom={:.4}",
                    self.camera_fx.bloom_point,
                    self.camera_fx.bloom_inherent,
                    self.camera_fx.bloom_intensity,
                    self.camera_fx.bloom_large_color[0],
                    self.camera_fx.bloom_large_color[1],
                    self.camera_fx.bloom_large_color[2],
                    self.camera_fx.bloom_medium_color[0],
                    self.camera_fx.bloom_medium_color[1],
                    self.camera_fx.bloom_medium_color[2],
                    self.camera_fx.bloom_small_color[0],
                    self.camera_fx.bloom_small_color[1],
                    self.camera_fx.bloom_small_color[2],
                    self.camera_fx.exposure_anti_bloom,
                );
                eprintln!(
                    "[bloom] settings: tone_curve_white_point={:.4} bloom={} bling={}",
                    self.settings_internal.tone_curve_white_point,
                    self.settings_internal.bloom,
                    self.settings_internal.bling,
                );
            });
        }

        // Step 5 — exposure-modulated extract scale. Engine
        // `c_screen_postprocess::postprocess_player_view @ 0x1806b4160`
        // lines 28-36:
        //   v21 = scripted_exposure_game_globals.render_apply_scripted_exposure(
        //             camera->m_exposure.m_exposure)
        //       + c_camera_fx_values::g_exposure_stops
        //       + camera->m_exposure_boost;
        //
        // - `render_apply_scripted_exposure` is ported on `ScriptedExposure`.
        //   Engine's singleton `scripted_exposure_game_globals` lives in
        //   game globals and is normally zero-initialized — script
        //   triggers set it to bias / blend exposure. Until the script
        //   plumbing lands we substitute a default-zero instance, which
        //   makes the lerp identity (returns m_exposure unchanged).
        // - `c_camera_fx_values::g_exposure_stops` is a static class
        //   member nudged by `g_render_set_exposure` scripts (also TBD).
        //   Default 0.
        // - `m_exposure_boost` is per-camera; populated by
        //   `c_camera_fx_values::update` from cfxs `exposure_boost`.
        //
        // anti_bloom modulates how much of v21 reaches the exponent
        // in `v22 = max(1, pow(2, v21 * anti_bloom))`. cfxs `anti_bloom = 1`
        // (s3d_edge default) means the bloom threshold rises 1:1 with
        // auto-exposure → post-exposure scene pixels stop passing the
        // bloom curve and the pyramid only catches genuine HDR highlights.
        // Engine: `v21 = render_apply_scripted_exposure(scripted_globals,
        //                   camera->m_exposure.m_exposure)
        //              + c_camera_fx_values::g_exposure_stops
        //              + camera->m_exposure_boost;`
        // Both `scripted_globals` and `g_exposure_stops` are threaded in
        // from the renderer-level state (Phase 2.5 — B3 fix). With
        // default-zero globals this collapses to
        // `m_exposure + 0 + exposure_boost`.
        let v21: f32 = scripted
            .render_apply_scripted_exposure(self.camera_fx.exposure_stops)
            + g_exposure_stops
            + self.camera_fx.exposure_boost;
        let anti_bloom = self.camera_fx.exposure_anti_bloom;
        let v22 = (v21 * anti_bloom).exp2().max(1.0);
        let v23 = v22 * self.settings_internal.tone_curve_white_point * 0.66943294;
        let extract_scale = [
            v23 * self.camera_fx.bloom_point,
            self.camera_fx.bloom_inherent,
            1.0,
            1.0,
        ];
        // Periodic bloom-extract dump for development. Enable with
        // `PROTOMORPH_LOG_BLOOM=1` (off by default — too noisy for
        // normal runs). Logs first 8 frames + every 60 thereafter.
        if std::env::var_os("PROTOMORPH_LOG_BLOOM").is_some() {
            static EXTRACT_FRAME: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            let f = EXTRACT_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if f < 8 || f % 60 == 0 {
                eprintln!(
                    "[bloom@f{:04}] v21={:.3} v22={:.4} v23={:.4} extract_scale=({:.4},{:.4},{:.4},{:.4}) m_exp={:.3} boost={:.3}",
                    f,
                    v21,
                    v22,
                    v23,
                    extract_scale[0],
                    extract_scale[1],
                    extract_scale[2],
                    extract_scale[3],
                    self.camera_fx.exposure_stops,
                    self.camera_fx.exposure_boost,
                );
            }
        }

        // Step 6 — cbuffer 0x1C0000 = zero. (Skipped — slot unused on PC path.)

        // Step 7 — apply_binary_op_ex(48, composite_HDR, _surface_none, aux_bloom).
        // Isolated dispatch: avoids the queue.write_buffer batching
        // race that breaks shared uniform_buffer content when the same
        // buffer is uploaded multiple times in one frame.
        let extract_uniforms_48 = PostprocessUniforms {
            pixel_size: [
                1.0 / self.display_width as f32,
                1.0 / self.display_height as f32,
                self.display_width as f32,
                self.display_height as f32,
            ],
            scale: extract_scale,
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        self.dispatch_1tex_isolated(
            shared, encoder, "shader_48_extract",
            &self.pipelines.downsample_4x4_block_bloom,
            composite_hdr, &self.aux_bloom.view,
            wgpu::Color::TRANSPARENT,
            &extract_uniforms_48,
        );

        // Step 8 — apply_binary_op_ex(67, aux_bloom + composite_HDR
        // → aux_exposure_5). Engine's second extract pass produces the
        // actual 1/4-resolution pyramid input. Shader 67 re-extracts
        // bloom from the scene at quarter-res, then MAX-combines with
        // a point-sample of the half-res `aux_bloom`. Source mapping:
        //   - source_0 (bgl binding 0,1) = aux_bloom (already-extracted bloom)
        //   - source_1 (bgl binding 2,3) = composite_HDR (the scene)
        let extract_uniforms_67 = PostprocessUniforms {
            pixel_size: [
                1.0 / self.aux_bloom.width as f32,
                1.0 / self.aux_bloom.height as f32,
                self.aux_bloom.width as f32,
                self.aux_bloom.height as f32,
            ],
            scale: extract_scale,
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        self.dispatch_2tex_isolated(
            shared, encoder, "shader_67_re_extract",
            &self.pipelines.downsample_4x4_block_bloom_ldr,
            &self.aux_bloom.view,
            composite_hdr,
            &self.aux_exposure_5.view,
            wgpu::Color::TRANSPARENT,
            &extract_uniforms_67,
        );

        // Step 9 — multi-scale bloom pyramid (with the engine
        // downsample_generate inlined at its head, including the
        // exposure_downsample sample taken BEFORE step 2's blur+tint).
        let exposure_slot = self.postprocess_bloom_buffer(
            shared, encoder, auto_exposure_sensitivity,
        );

        // Step 10 — DoF downsample. Engine `copy(51, composite_HDR →
        // aux_small2)` half-res copy of the HDR composite. Plain
        // bilinear copy through `postprocess_copy` — the composite
        // shader gates DoF on `dof_focus.z > 0`, so paying this every
        // frame is cheap relative to the gating cost in the final
        // shader. (Could be skipped when DoF disabled — a perf knob
        // for later.)
        self.copy(
            shared, encoder,
            &self.pipelines.postprocess_copy,
            composite_hdr, [self.display_width, self.display_height],
            &self.dof_small.view,
            wgpu::FilterMode::Linear,
            wgpu::AddressMode::ClampToEdge,
            [1.0, 1.0, 1.0, 1.0],
        );

        // Step 11 — debug bypass (postprocess_bloom_buffer non-zero return). N/A.

        // Step 12 — gate lightshafts.
        let lightshafts = camera_fx.and_then(|cfx| cfx.lightshafts.as_ref());
        let lightshafts_active = lightshafts.is_some_and(|ls| ls.is_enabled());

        // Step 13 — DoF gaussian_blur. Engine `gaussian_blur(aux_small2,
        // aux_tiny2, sigma)` — fixed 11-tap binomial separable. The
        // sigma argument is a no-op in `gaussian_blur` (only
        // `gaussian_blur_fixed` uses a per-call kernel). Two passes
        // ping-pong dof_small ↔ dof_temp, ending with the blurred
        // result back in dof_small (read by final_composite).
        self.copy(
            shared, encoder,
            &self.pipelines.horizontal_gaussian_blur,
            &self.dof_small.view,
            [self.dof_small.width, self.dof_small.height],
            &self.dof_temp.view,
            wgpu::FilterMode::Linear,
            wgpu::AddressMode::ClampToEdge,
            [1.0, 1.0, 1.0, 1.0],
        );
        self.copy(
            shared, encoder,
            &self.pipelines.vertical_gaussian_blur,
            &self.dof_temp.view,
            [self.dof_temp.width, self.dof_temp.height],
            &self.dof_small.view,
            wgpu::FilterMode::Linear,
            wgpu::AddressMode::ClampToEdge,
            [1.0, 1.0, 1.0, 1.0],
        );

        // Step 14-15 — copy_accumulation_target equivalent (bloom + bling combine).
        // Diagnostic: PROTOMORPH_DISABLE_BLING=1 skips bling_generate and
        // zeros bling_scale. Use to bisect: if yellow wash gone with bling
        // disabled, the spike kernel is amplifying scene content too much.
        let bling_disabled = std::env::var_os("PROTOMORPH_DISABLE_BLING").is_some();
        if bling_disabled {
            static LOG: std::sync::Once = std::sync::Once::new();
            LOG.call_once(|| eprintln!("[bling] disabled via PROTOMORPH_DISABLE_BLING"));
        }
        let bling_active = !bling_disabled
            && self.settings_internal.bling != 0
            && self.camera_fx.bling_count > 0
            && self.camera_fx.bling_intensity > 0.0;
        if bling_active {
            self.bling_generate(shared, encoder);
        }
        let bling_scale = if bling_active { self.camera_fx.bling_intensity } else { 0.0 };
        let combine_scale = [1.0, 1.0, 1.0, bling_scale];
        let combine_uniforms = PostprocessUniforms {
            pixel_size: [
                1.0 / self.aux_bloom.width as f32,
                1.0 / self.aux_bloom.height as f32,
                self.aux_bloom.width as f32,
                self.aux_bloom.height as f32,
            ],
            scale: combine_scale,
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        self.dispatch_2tex_isolated(
            shared, encoder, "bloom_star_combine_iso",
            &self.pipelines.bloom_star_combine,
            &self.aux_bloom.view,
            &self.aux_star.view,
            &self.aux_exposure_7.view,
            wgpu::Color::TRANSPARENT,
            &combine_uniforms,
        );
        self.copy_step(shared, encoder, "bloom_finalcopy",
            &self.aux_exposure_7.view, &self.aux_bloom);

        // Step 16 — render_lightshafts. RELOCATED to the caller (in
        // `render_player_view`) so that the engine ordering
        //   copy_accumulation_target → render_lightshafts → display blit
        // is preserved: lightshafts read/write `_surface_accum_HDR`,
        // which `FinalCompositePass::record` writes to *after* this
        // function returns. The `lightshafts_active` gate is still
        // computed here for backwards compat but no longer consumed
        // inside this function.
        let _ = lightshafts_active;
        let _ = (view, projection, vertical_fov_rad, aspect, z_near);

        // Step 17 — FXAA path. DEFERRED — the `copy(51 or 93)` blit
        // lives in `FinalCompositePass::record_blit_to_swapchain`.

        encoder.pop_debug_group();

        exposure_slot
    }

    /// `c_screen_postprocess::postprocess_bloom_buffer @ 0x1806B4710`.
    /// Verbatim from the dllcache decompile. Multi-scale bloom pyramid:
    ///   1. downsample_generate: aux_bloom → aux_mini (1/4) → aux_tiny (1/8)
    ///        +  exposure_downsample(aux_tiny → ring_slot)  ← auto-exposure sample
    ///   2. gaussian_blur(aux_tiny, ..., scale=bi×large_color) — large tint
    ///   3. bloom_add_alpha1(aux_mini, aux_tiny, aux_temp_quarter, bi×medium)
    ///   4. gaussian_blur(aux_temp_quarter, aux_mini, ×1.0) — smooth
    ///   5. bloom_add_alpha1(aux_bloom, aux_temp_quarter, aux_exposure_7, bi×small)
    ///   6. gaussian_blur(aux_exposure_7, aux_bloom, ×1.0) — final smooth
    ///   7. copy aux_exposure_7 → aux_bloom — final result for the blit
    ///
    /// **Step 1's exposure_downsample is critical**: engine
    /// `c_screen_postprocess::downsample_generate @ 0x1806B5C00` samples
    /// the freshly box-downsampled `mini` surface BEFORE step 2's
    /// gaussian_blur_fixed clobbers it with `bi × bloom_large_color`-
    /// tinted blur output. Sampling aux_tiny after the pyramid finishes
    /// reads the bloom-highlight buffer (bloom-curve-thresholded, tinted)
    /// instead of scene radiance, breaking auto-exposure calibration.
    /// Returns the ring slot used, or `None` if the slot's previous
    /// map_async hasn't completed (caller skips the FIFO push for None).
    ///
    /// Halo's actual code inlines bling_generate between steps 4 and 5;
    /// we run it as a peer step in postprocess_player_view since Halo's
    /// surface overloading (aux_star reused for both 1/8 downsample and
    /// bling output) doesn't generalize to our separately-allocated
    /// aux_tiny + aux_star pair.
    fn postprocess_bloom_buffer(
        &mut self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        auto_exposure_sensitivity: f32,
    ) -> Option<u32> {
        encoder.push_debug_group("postprocess_bloom_buffer");

        // Diagnostic: `PROTOMORPH_BLOOM_INTENSITY=N` overrides the per-frame
        // pyramid bloom_intensity. Lets us bisect: 0 = bloom buffer cleared
        // (no contribution), 0.1 = Guardian/Riverworld default, 1.0 = 10x.
        let bi = std::env::var("PROTOMORPH_BLOOM_INTENSITY")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(self.camera_fx.bloom_intensity);
        static BI_LOG: std::sync::Once = std::sync::Once::new();
        BI_LOG.call_once(|| {
            eprintln!(
                "[bloom] pyramid bi={:.3} (cfxs={:.3}, env={})",
                bi,
                self.camera_fx.bloom_intensity,
                std::env::var("PROTOMORPH_BLOOM_INTENSITY")
                    .unwrap_or_else(|_| String::from("unset")),
            );
        });

        // Step 1 — engine `downsample_generate`: two plain box-downsamples
        // then `exposure_downsample` on the result, all before the
        // pyramid blur clobbers aux_tiny.
        let exposure_slot = self.downsample_generate(
            shared, encoder, auto_exposure_sensitivity,
        );

        // Step 2 — gaussian_blur_fixed of aux_tiny with scale =
        // bloom_intensity × bloom_large_color baked into the V-pass.
        // ISOLATED: uses per-call uniform buffers to avoid the
        // queue.write_buffer batching race that affects shared uniform
        // buffers (see `dispatch_1tex_isolated` doc-comment).
        let large_scale = [
            bi * self.camera_fx.bloom_large_color[0],
            bi * self.camera_fx.bloom_large_color[1],
            bi * self.camera_fx.bloom_large_color[2],
            1.0,
        ];
        self.gaussian_blur_fixed_isolated(shared, encoder,
            &self.aux_tiny, &self.aux_temp_eighth, large_scale);

        // Step 3 — bloom_add_alpha1: aux_temp_quarter = bi×medium_color
        // × aux_mini + aux_tiny (upsampled via bilinear sampling).
        let medium_scale = [
            bi * self.camera_fx.bloom_medium_color[0],
            bi * self.camera_fx.bloom_medium_color[1],
            bi * self.camera_fx.bloom_medium_color[2],
            1.0,
        ];
        let medium_uniforms = PostprocessUniforms {
            pixel_size: [
                1.0 / self.aux_temp_quarter.width as f32,
                1.0 / self.aux_temp_quarter.height as f32,
                self.aux_temp_quarter.width as f32,
                self.aux_temp_quarter.height as f32,
            ],
            scale: medium_scale,
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        self.dispatch_2tex_isolated(
            shared, encoder, "combine_medium_into_quarter",
            &self.pipelines.bloom_add_alpha1,
            &self.aux_mini.view,
            &self.aux_tiny.view,
            &self.aux_temp_quarter.view,
            wgpu::Color::TRANSPARENT,
            &medium_uniforms,
        );

        // Step 4 — gaussian_blur_fixed of aux_temp_quarter, using
        // aux_mini as the H-pass temp (clobbering its content; we
        // don't need it anymore).
        self.gaussian_blur_fixed_isolated(shared, encoder,
            &self.aux_temp_quarter, &self.aux_mini, [1.0, 1.0, 1.0, 1.0]);

        // Step 5 — bloom_add_alpha1 / engine shader 10 = `add`:
        //   aux_exposure_7 = bi×small_color × aux_exposure_5 (bilinear upsampled)
        //                  + aux_temp_quarter (bilinear upsampled).
        // Engine line 96-108 of postprocess_bloom_buffer:
        //   apply_binary_op_ex(10, bloom_buffer=aux_exposure_5,
        //                          aux_exposure_6 → aux_bloom,
        //                          scale=(bi*small.x, bi*small.y, bi*small.z, 1));
        let small_scale = [
            bi * self.camera_fx.bloom_small_color[0],
            bi * self.camera_fx.bloom_small_color[1],
            bi * self.camera_fx.bloom_small_color[2],
            1.0,
        ];
        let small_uniforms = PostprocessUniforms {
            pixel_size: [
                1.0 / self.aux_exposure_7.width as f32,
                1.0 / self.aux_exposure_7.height as f32,
                self.aux_exposure_7.width as f32,
                self.aux_exposure_7.height as f32,
            ],
            scale: small_scale,
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        self.dispatch_2tex_isolated(
            shared, encoder, "combine_small_into_half",
            &self.pipelines.bloom_add_alpha1,
            &self.aux_exposure_5.view,
            &self.aux_temp_quarter.view,
            &self.aux_exposure_7.view,
            wgpu::Color::TRANSPARENT,
            &small_uniforms,
        );

        // Step 6 — final gaussian_blur of aux_exposure_7, using
        // aux_bloom as the H-pass temp.
        self.gaussian_blur_fixed_isolated(shared, encoder,
            &self.aux_exposure_7, &self.aux_bloom, [1.0, 1.0, 1.0, 1.0]);

        // Step 7 — copy aux_exposure_7 → aux_bloom so the final
        // result lands in the surface the blit binds.
        self.copy_step(shared, encoder, "multiscale_finalcopy",
            &self.aux_exposure_7.view, &self.aux_bloom);

        encoder.pop_debug_group();

        exposure_slot
    }

    /// `c_screen_postprocess::downsample_generate @ 0x1806B5C00` —
    /// verbatim port of the engine body. Builds the bloom pyramid's
    /// first two downsamples AND runs `exposure_downsample` on the
    /// smaller surface, all in one call. The engine pattern is:
    ///
    /// ```text
    /// copy(49, aux_bloom → tiny);     // plain box-downsample
    /// resolve_surface(tiny);
    /// copy(49, tiny → mini);
    /// resolve_surface(mini);
    /// copy(35, mini → exposure_slot, sensitivity, 1, 1, 1);
    /// update_sampled_exposure(slot);  // pushed by caller in our port
    /// ```
    ///
    /// Returns `Some(slot)` if the readback was scheduled. `None` means
    /// the slot's previous `map_async` hasn't completed yet — caller
    /// should skip the FIFO push and `begin_exposure_map` for this frame.
    fn downsample_generate(
        &mut self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        auto_exposure_sensitivity: f32,
    ) -> Option<u32> {
        // Engine `downsample_generate(bloom_buffer=aux_exposure_5, _,
        // dest_tiny=aux_chud_overlay (1/8), dest_mini=aux_star (1/16))`.
        // Two plain 4-tap box-downsamples: 1/4 → 1/8 → 1/16.
        self.downsample_step(
            shared, encoder, "downsample_to_mini",
            &self.aux_exposure_5, &self.aux_mini,
        );
        self.downsample_step(
            shared, encoder, "downsample_to_tiny",
            &self.aux_mini, &self.aux_tiny,
        );

        // Engine `copy(35, mini, exposure_slot, sensitivity, 1, 1, 1)` —
        // shader 35 (`exposure_downsample`) on the freshly box-downsampled
        // surface. This is BEFORE postprocess_bloom_buffer step 2 blurs
        // and tints aux_tiny.
        let aux_tiny_view = self.aux_tiny_view().clone();
        let aux_tiny_size = self.aux_tiny_size();
        let slot = self.next_exposure_ring_slot() as usize;
        self.exposure_downsample_step(
            shared, encoder,
            &aux_tiny_view, aux_tiny_size, slot, auto_exposure_sensitivity,
        );

        // Engine `update_sampled_exposure(camera->m_exposure, v8)` — the
        // FIFO push uses `&mut Exposure` on the camera which lives on
        // `c_camera_fx_values`. We don't own that here; return the slot
        // and let the caller push when it has both `&mut self` on the
        // renderer's screen_postprocess and `&mut camera_fx_values`.
        let scheduled = self.schedule_exposure_readback(encoder, slot);
        if scheduled { Some(slot as u32) } else { None }
    }

    /// One downsample pass: source → 4×4 box filter at half resolution.
    fn downsample_step(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        source: &BloomSurface,
        dest: &BloomSurface,
    ) {
        let uniforms = PostprocessUniforms {
            pixel_size: [
                1.0 / source.width as f32,
                1.0 / source.height as f32,
                source.width as f32,
                source.height as f32,
            ],
            scale: [0.0; 4],
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        self.dispatch_1tex_isolated(
            shared, encoder, label,
            &self.pipelines.downsample_4x4_block,
            &source.view, &dest.view,
            wgpu::Color::TRANSPARENT,
            &uniforms,
        );
    }

    /// One-shot `run_pipeline_1tex` variant that allocates a FRESH uniform
    /// buffer per call (via `mapped_at_creation`). This is the canonical fix
    /// for wgpu's queue.write_buffer batching behavior: all
    /// `queue.write_buffer` calls submitted in one frame are processed BEFORE
    /// any encoder commands, so the bloom-pyramid's repeated uploads to a
    /// shared `self.uniform_buffer` race — every draw reads the LAST upload.
    /// Allocating a per-call buffer with `mapped_at_creation: true` writes
    /// the data BEFORE the buffer is returned, locking the value at
    /// allocation time and bypassing the batching reorder.
    fn dispatch_1tex_isolated(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::TextureView,
        dest: &wgpu::TextureView,
        clear: wgpu::Color,
        uniforms: &PostprocessUniforms,
    ) {
        use wgpu::util::DeviceExt;
        let local_uniform = shared.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bgl_1tex,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: local_uniform.as_entire_binding() },
            ],
        });
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// 2-texture variant of `dispatch_1tex_isolated`. Same race-avoidance
    /// rationale — per-call uniform buffer via `mapped_at_creation`.
    fn dispatch_2tex_isolated(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        source_a: &wgpu::TextureView,
        source_b: &wgpu::TextureView,
        dest: &wgpu::TextureView,
        clear: wgpu::Color,
        uniforms: &PostprocessUniforms,
    ) {
        use wgpu::util::DeviceExt;
        let local_uniform = shared.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bgl_2tex,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source_a) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(source_b) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 4, resource: local_uniform.as_entire_binding() },
            ],
        });
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Isolated `gaussian_blur_fixed` — both the H-pass (PostprocessUniforms)
    /// and V-pass (Kernel5Uniforms) get fresh per-call uniform buffers,
    /// bypassing the queue.write_buffer batching race that affects shared
    /// `self.uniform_buffer` / `self.kernel5_uniform_buffer`.
    fn gaussian_blur_fixed_isolated(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        target: &BloomSurface,
        temp: &BloomSurface,
        scale: [f32; 4],
    ) {
        use wgpu::util::DeviceExt;
        encoder.push_debug_group("gaussian_blur_fixed_isolated");

        // H-pass: blur_11_horizontal with isolated PostprocessUniforms.
        let h_uniforms = PostprocessUniforms {
            pixel_size: [
                1.0 / target.width as f32,
                1.0 / target.height as f32,
                target.width as f32,
                target.height as f32,
            ],
            scale: [0.0; 4],
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        self.dispatch_1tex_isolated(
            shared, encoder, "blur_11_horizontal_iso",
            &self.pipelines.blur_11_horizontal,
            &target.view, &temp.view,
            wgpu::Color::TRANSPARENT,
            &h_uniforms,
        );

        // V-pass: kernel_5 with isolated Kernel5Uniforms (scale × kernel taps).
        const KV_W0: f32 = 10.0 / 512.0;
        const KV_W1: f32 = 120.0 / 512.0;
        const KV_W2: f32 = 252.0 / 512.0;
        const KV_OFF0: f32 = -4.0 - 1.0 / 10.0;
        const KV_OFF1: f32 = -2.0 - 36.0 / 120.0;
        const KV_OFF2: f32 = 0.0 - 126.0 / 252.0;
        const KV_OFF3: f32 = 2.0 - 84.0 / 120.0;
        const KV_OFF4: f32 = 4.0 - 9.0 / 10.0;
        let v_uniforms = Kernel5Uniforms {
            pixel_size: [
                1.0 / temp.width as f32,
                1.0 / temp.height as f32,
                temp.width as f32,
                temp.height as f32,
            ],
            scale,
            kernel: [
                [0.5, KV_OFF0, KV_W0, 0.0],
                [0.5, KV_OFF1, KV_W1, 0.0],
                [0.5, KV_OFF2, KV_W2, 0.0],
                [0.5, KV_OFF3, KV_W1, 0.0],
                [0.5, KV_OFF4, KV_W0, 0.0],
            ],
        };
        let v_uniform_buf = shared.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kernel_5_v_pass_iso"),
            contents: bytemuck::bytes_of(&v_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kernel_5_v_pass_iso_bg"),
            layout: &self.bgl_kernel5,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&temp.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: v_uniform_buf.as_entire_binding() },
            ],
        });
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("kernel_5_v_pass_iso"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&self.pipelines.kernel_5);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
        drop(rp);

        encoder.pop_debug_group();
    }

    /// `c_screen_postprocess::bling_generate @ 0x1806B5E70`.
    ///
    /// Engine-faithful port of the dllcache body. Per spike:
    ///   1. spike_blur(aux_bloom → aux_exposure_7, angle, falloff, skip=1)
    ///   2. spike_blur(aux_exposure_7 → aux_star, angle, falloff, skip=8)
    ///      — additive blend on pass 2 so subsequent spikes accumulate.
    ///
    /// Halo's per-channel falloff:
    ///   v6 = exp(log(0.002) / max(0.1, bling_size/4))
    ///   falloff_r = v6 * 0.973     // R attenuates fastest
    ///   falloff_g = v6 * 0.98      // G slightly slower
    ///   falloff_b = v6              // B slowest → blue extends furthest
    /// = chromatic dispersion ("prismatic" lens-flare color shift).
    ///
    /// For multi-spike (`bling_count > 1`), the angle increments by
    /// `360/count` per spike. Halo runs each spike's pass-2 into a
    /// scratch surface (`aux_normal`) then `apply_binary_op_ex(10=add)`
    /// to combine into the accumulator (`aux_refraction`). We achieve
    /// the same result with native additive-blend on the pass-2
    /// pipeline → spikes accumulate directly into aux_star without
    /// the explicit add shader pass or extra scratch surfaces.
    pub fn bling_generate(
        &mut self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        encoder.push_debug_group("bling_generate");

        let count = self.camera_fx.bling_count.max(1) as i32;

        // Halo's falloff base — see fn-doc.
        let v4 = (self.camera_fx.bling_size * 0.25).max(0.1);
        let v5 = 0.002_f32.ln() / v4;
        let v6 = v5.exp();
        let falloff_r = v6 * 0.973;
        let falloff_g = v6 * 0.98;
        let falloff_b = v6;

        // Apply the camera's bling_intensity by scaling the initial
        // color tap weight uniformly — Halo's bling_intensity feeds
        // into the per-channel falloff base via a separate path we
        // don't have visibility into; this approximation modulates
        // the overall spike brightness which is the dominant effect.
        let intensity = self.camera_fx.bling_intensity;

        // Clear aux_star once before the spike loop so the additive
        // blend on pass-2 works correctly for the first spike.
        Self::clear_surface(shared, encoder, &self.aux_star.view);

        for spike in 0..count {
            // Halo distributes spikes around the full 360° (not 180°
            // half-circle — the spike is single-sided per pass, so
            // the rotation needs the full circle to avoid duplicates).
            let angle_deg = self.camera_fx.bling_angle
                + spike as f32 * 360.0 / count as f32;

            // Pass 1: tight (skip=1) blur of aux_bloom into aux_exposure_7.
            // bling_intensity is folded into pass 1's initial_color so
            // the final spike amplitude is `intensity × source`.
            self.spike_blur(
                shared, encoder, "spike_pass1",
                &self.aux_bloom, &self.aux_exposure_7,
                angle_deg, falloff_r, falloff_g, falloff_b, 1.0, intensity,
                /*additive=*/ false,
            );

            // Pass 2: wide (skip=8) blur of aux_exposure_7 into aux_star
            // with additive blending. Pass-2 intensity = 1.0 so it just
            // preserves pass-1's amplitude (intensity already applied).
            self.spike_blur(
                shared, encoder, "spike_pass2",
                &self.aux_exposure_7, &self.aux_star,
                angle_deg, falloff_r, falloff_g, falloff_b, 8.0, 1.0,
                /*additive=*/ true,
            );
        }

        encoder.pop_debug_group();
    }

    /// `spike_blur(source, dest, angle_deg, falloff_r/g/b, skip)`
    /// from `screen_postprocess.h:84`. Picks the linear-Y or linear-X
    /// shader based on the angle quadrant per the dllcache logic.
    fn spike_blur(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        source: &BloomSurface,
        dest: &BloomSurface,
        angle_deg: f32,
        falloff_r: f32,
        falloff_g: f32,
        falloff_b: f32,
        skip: f32,
        intensity: f32,
        additive: bool,
    ) {
        // Angle-quadrant selection — verbatim from dllcache pseudocode.
        let v18 = angle_deg + 450.0;
        let v19 = (v18 * 2.0 / 180.0) + 0.5;
        let v20 = (v19 as i32) & 3;
        let v15: f32 = if v20 == 1 || v20 == 0 { 1.0 } else { -1.0 };
        let use_linear_y = v20 == 0 || v20 == 2; // v22==3 in dllcache

        // Per-axis spike step in pixels (v23 = primary, v24 = secondary).
        let (dx_pixels, dy_pixels, step_norm) = if use_linear_y {
            // v22 = 3 branch: dx = tan(v21) * sign * skip, dy = sign * skip
            let v21 = v18.to_radians();
            let v23 = v15 * skip;
            let v24 = v21.tan() * v23;
            let step = (skip / v21.cos()).abs();
            (v24, v23, step)
        } else {
            // v22 = 4 branch: dx = sign * skip, dy = -tan(v30) * sign * skip
            let v30 = (v18 + 90.0).to_radians();
            let v24 = v15 * skip;
            let v23 = -v30.tan() * v24;
            let step = (skip / v30.cos()).abs();
            (v24, v23, step)
        };

        let src_w = source.width as f32;
        let src_h = source.height as f32;
        let pix_x = 1.0 / src_w;
        let pix_y = 1.0 / src_h;

        // Per-channel falloff per loop iteration. The loop runs 8
        // times so total spike length ≈ step_norm * 8 source pixels.
        let r_step = falloff_r.powf(step_norm);
        let g_step = falloff_g.powf(step_norm);
        let b_step = falloff_b.powf(step_norm);

        // Energy normalization. The HLSL loop sums 8 taps with
        // weights (1, f, f², …, f⁷) — total weight per channel
        // is the geometric sum (1 - f⁸) / (1 - f). Halo's
        // `copy_accumulation_target` presumably normalizes via
        // `bling_intensity`, but we don't have visibility into that
        // formula and the unnormalized spike amplifies ~7× which
        // smears the whole image. Pre-scaling `initial_color` by
        // `intensity / sum_weights` makes each pass output ≈
        // `intensity × source` amplitude (per channel). Pass 2
        // gets `intensity_2 = 1.0` so it just preserves pass 1's
        // amplitude — pass 1 owns the bling_intensity scaling.
        let geom_sum = |f: f32| -> f32 {
            if (1.0 - f).abs() < 1e-6 {
                8.0
            } else {
                (1.0 - f.powi(8)) / (1.0 - f)
            }
        };
        let sum_r = geom_sum(r_step);
        let sum_g = geom_sum(g_step);
        let sum_b = geom_sum(b_step);

        let uniforms = SpikeBlurUniforms {
            source_pixel_size: [pix_x, pix_y, src_w, src_h],
            offset_delta: [0.0, 0.0, dx_pixels * pix_x, dy_pixels * pix_y],
            initial_color: [intensity / sum_r, intensity / sum_g, intensity / sum_b, 0.0],
            delta_color: [r_step, g_step, b_step, 0.0],
        };
        shared
            .queue
            .write_buffer(&self.spike_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let pipeline = match (use_linear_y, additive) {
            (true, false) => &self.pipelines.spike_blur_h,
            (true, true) => &self.pipelines.spike_blur_h_add,
            (false, false) => &self.pipelines.spike_blur_v,
            (false, true) => &self.pipelines.spike_blur_v_add,
        };

        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bgl_1tex,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.spike_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // For pass-1 (default blend) we Clear; pass-2 (additive) Loads
        // so each spike's contribution accumulates.
        let load = if additive {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        };

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dest.view,
                resolve_target: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
    }

    fn clear_surface(
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let _ = shared;
        let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_surface"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    /// Single-tap copy. Halo uses `apply_binary_op_ex` shader-id 0
    /// (raw passthrough) for this; we use a dedicated tiny shader
    /// since none of the explicit_shaders[] entries match exactly.
    fn copy_step(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        source: &wgpu::TextureView,
        dest: &BloomSurface,
    ) {
        // `postprocess_copy.wgsl::default_ps` samples at
        // `texcoord * ps_postprocess_scale.xy` (engine `copy.hlsl:13`
        // verbatim). A raw 1:1 passthrough needs `scale.xy = (1, 1)` —
        // passing `(0, 0)` collapses every dest pixel onto source's
        // `(0, 0)` texel, silently destroying the bloom pyramid output
        // (this method is called from step 7 / bloom_finalcopy).
        self.upload_uniforms(
            shared,
            [dest.width as f32, dest.height as f32],
            [1.0, 1.0, 0.0, 0.0],
        );
        self.run_pipeline_1tex(
            shared, encoder,
            label,
            &self.pipelines.postprocess_copy,
            source,
            &dest.view,
            wgpu::Color::TRANSPARENT,
        );
    }

    /// `c_screen_postprocess::gaussian_blur_fixed @ 0x1806b68b0`.
    /// Body verbatim from dllcache:
    ///   copy(11, target, temp, bilinear, clamp, 1, 1, 1, 1);
    ///   resolve_surface(temp, 0, 0, 0);                  // PC: no-op
    ///   set_shader_constant(0x330000, 5, kernel_vertical);
    ///   copy(34, temp, target, bilinear, clamp, scale_r, scale_g, scale_b, scale_a);
    pub fn gaussian_blur_fixed(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        target: &BloomSurface,
        temp: &BloomSurface,
        scale: [f32; 4],
    ) {
        encoder.push_debug_group("gaussian_blur_fixed");

        // copy(11, target, temp, ...) — shader 11 = blur_11_horizontal.
        // Hardcoded Pascal-row-10 weights. Pixel_size derives from the
        // source surface (target).
        self.upload_uniforms(
            shared,
            [target.width as f32, target.height as f32],
            [0.0; 4],
        );
        self.run_pipeline_1tex(
            shared, encoder,
            "blur_11_horizontal",
            &self.pipelines.blur_11_horizontal,
            &target.view,
            &temp.view,
            wgpu::Color::TRANSPARENT,
        );

        // copy(34, temp, target, ..., scale) — shader 34 = kernel_5,
        // but we use blur_11_vertical (same hardcoded Pascal kernel).
        self.upload_uniforms(
            shared,
            [temp.width as f32, temp.height as f32],
            scale,
        );
        self.run_pipeline_1tex(
            shared, encoder,
            "blur_11_vertical",
            &self.pipelines.blur_11_vertical,
            &temp.view,
            &target.view,
            wgpu::Color::TRANSPARENT,
        );

        encoder.pop_debug_group();
    }

    /// `c_screen_postprocess::apply_binary_op_ex @ 0x1806b66b0`. Two-input
    /// fullscreen postprocess shader dispatch.
    ///
    /// Engine body: set RT0 = dest, set explicit shader, bind source_0
    /// + source_1 with per-sampler filter/address mode, write
    /// `pixel_size = 1/source_0.{w,h}` to cbuffer slot `0x420000` and
    /// `scale` to `0x420001`, draw a fullscreen quad, then clear
    /// sampler textures 0 + 1.
    ///
    /// Wgpu adaptation: we resolve the explicit_shader_index to a
    /// concrete `&RenderPipeline` outside this method (caller knows
    /// which shader they want); pixel_size + scale go into our shared
    /// `PostprocessUniforms` cbuffer; sampler bindings are baked into
    /// pre-allocated `sampler_bilinear` / `sampler_point` (address_mode
    /// is currently clamp-only — matches all Halo bloom callers).
    /// Render pass uses LoadOp::Load (engine doesn't clear the dest).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_binary_op_ex(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source_0: &wgpu::TextureView,
        source_0_size: [u32; 2],
        source_1: &wgpu::TextureView,
        dest: &wgpu::TextureView,
        filter_mode: wgpu::FilterMode,
        _address_mode: wgpu::AddressMode,
        scale: [f32; 4],
    ) {
        self.upload_uniforms(
            shared,
            [source_0_size[0] as f32, source_0_size[1] as f32],
            scale,
        );
        let sampler = match filter_mode {
            wgpu::FilterMode::Linear => &self.sampler_bilinear,
            wgpu::FilterMode::Nearest => &self.sampler_point,
        };
        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apply_binary_op_ex"),
            layout: &self.bgl_2tex,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source_0) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(source_1) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: self.uniform_buffer.as_entire_binding() },
            ],
        });
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("apply_binary_op_ex"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Step 3 of `c_screen_postprocess::downsample_generate` —
    /// runs explicit shader 35 (`exposure_downsample`) to compress
    /// the 1/16-res HDR (intensity in `.a`) into a single half-float
    /// average-luminance value, written into ring slot `target_slot`
    /// (engine surface `28 + target_slot`).
    ///
    /// Engine body (verbatim):
    /// ```text
    /// set_sampler_texture(1, default_textures[7]);          // auto_exposure_weight
    /// set_sampler_address_mode(1, clamp, clamp, clamp);
    /// set_sampler_filter_mode(1, bilinear);
    /// copy(35, source, target, bilinear, clamp,
    ///      camera->m_auto_exposure_sensitivity, 1.0, 1.0, 1.0);
    /// update_sampled_exposure(camera->m_exposure, target);
    /// ```
    ///
    /// `update_sampled_exposure` push to `m_render_target_queue` is
    /// the caller's responsibility (Phase D `c_exposure`).
    pub fn exposure_downsample_step(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        source_size: [u32; 2],
        target_slot: usize,
        auto_exposure_sensitivity: f32,
    ) {
        let dest = &self.aux_exposure[target_slot];
        let uniforms = PostprocessUniforms {
            pixel_size: [
                1.0 / source_size[0] as f32,
                1.0 / source_size[1] as f32,
                0.0, 0.0,
            ],
            scale: [auto_exposure_sensitivity, 1.0, 1.0, 1.0],
            intensity_vector: [0.0; 4],
            dark_color_multiplier: [0.0; 4],
        };
        shared.queue.write_buffer(
            &self.exposure_downsample_uniform, 0,
            bytemuck::bytes_of(&uniforms),
        );

        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exposure_downsample"),
            layout: &self.bgl_2tex,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.auto_exposure_weight_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.auto_exposure_weight_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: self.exposure_downsample_uniform.as_entire_binding() },
            ],
        });
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("exposure_downsample"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dest.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        rp.set_pipeline(&self.pipelines.exposure_downsample);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Read-only accessor for ring-slot textures. Phase C readback
    /// uses this to issue `copy_texture_to_buffer`.
    pub fn aux_exposure_texture(&self, slot: usize) -> &wgpu::Texture {
        &self.aux_exposure[slot].tex
    }

    /// `aux_tiny` (1/8-res Rgba16Float bloom-pyramid output, with
    /// per-texel luminance in `.a`) — the source surface fed into
    /// shader 35 by `exposure_downsample_step`. Engine equivalent:
    /// the `dest_buffer_mini` arg of `downsample_generate` (surface
    /// 27 = aux_star, dual-purposed as the 1/8 res downsample
    /// before bling overwrites it; we keep aux_tiny separate).
    pub fn aux_tiny_view(&self) -> &wgpu::TextureView {
        &self.aux_tiny.view
    }
    pub fn aux_tiny_size(&self) -> [u32; 2] {
        [self.aux_tiny.width, self.aux_tiny.height]
    }

    /// Thin proxy for the engine class-static
    /// `c_exposure::m_next_render_target` increment in
    /// `downsample_generate @ 0x1806B5C00:51-55`:
    /// ```c
    /// v8 = c_exposure::m_next_render_target + 28;       // current slot's surface
    /// v9 = c_exposure::m_next_render_target + 1;
    /// if (v9 >= 8) v9 = 0;
    /// c_exposure::m_next_render_target = v9;            // advance for next frame
    /// ```
    /// State lives on `Exposure::next_render_target()` (module-scope
    /// atomic, engine-shape singleton). This method just delegates.
    pub fn next_exposure_ring_slot(&mut self) -> u32 {
        crate::halo::render::camera_fx_settings::Exposure::next_render_target()
    }

    /// Schedule a GPU→CPU readback of ring slot `slot` into its
    /// dedicated staging buffer. Must be called inside the same
    /// command encoder that ran `exposure_downsample_step`, before
    /// `queue.submit`.
    ///
    /// Returns `true` if the copy was encoded; `false` if the slot was
    /// skipped because the previous map_async on it hasn't completed
    /// (caller should not invoke `begin_exposure_map` for skipped slots).
    ///
    /// State transitions:
    ///   Ready    — previous read was skipped; unmap + reset to Idle
    ///              before encoding.
    ///   InFlight — previous map_async still pending; SKIP this frame
    ///              (encoding into a mapped buffer would fail wgpu
    ///              `Queue::submit` validation with "Buffer is still
    ///              mapped"). Caller falls back to previous frame's
    ///              exposure sample. 8-slot ring + ≤3-frame map
    ///              latency makes this rare on dense scenes, but it's
    ///              the only safe response — we can't unmap a buffer
    ///              whose map_async hasn't completed.
    ///   Idle     — proceed normally.
    pub fn schedule_exposure_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
    ) -> bool {
        let state = &self.exposure_slot_state[slot];
        let current = state.load(Ordering::Acquire);
        // Diagnostic: count total schedule attempts + IN_FLIGHT skips
        // so we can measure how often the async readback is stalling
        // auto-exposure. Engine D3D11 Map(DO_NOT_WAIT) typically completes
        // by next-frame use; wgpu's per-resource fence is missing so
        // skips here mean auto-exposure operates on stale data.
        if std::env::var_os("PROTOMORPH_DEBUG_EXPOSURE").is_some() {
            use std::sync::atomic::{AtomicU64, Ordering as Ord};
            static TOTAL: AtomicU64 = AtomicU64::new(0);
            static SKIPS: AtomicU64 = AtomicU64::new(0);
            static LAST_LOG: AtomicU64 = AtomicU64::new(0);
            let total = TOTAL.fetch_add(1, Ord::Relaxed) + 1;
            if current == EXPOSURE_SLOT_IN_FLIGHT {
                SKIPS.fetch_add(1, Ord::Relaxed);
            }
            if total - LAST_LOG.load(Ord::Relaxed) >= 60 {
                LAST_LOG.store(total, Ord::Relaxed);
                let s = SKIPS.load(Ord::Relaxed);
                let pct = (s as f32 / total as f32) * 100.0;
                eprintln!(
                    "[exposure-skip] total={total} skipped={s} ({pct:.2}%)",
                );
            }
        }
        if current == EXPOSURE_SLOT_READY {
            // Drop the value — caller didn't read this cycle.
            self.exposure_staging_buffers[slot].unmap();
            state.store(EXPOSURE_SLOT_IDLE, Ordering::Release);
        } else if current == EXPOSURE_SLOT_IN_FLIGHT {
            // map_async still pending — buffer is mapped from wgpu's
            // POV. Skip this frame; auto-exposure reuses last sample.
            return false;
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.aux_exposure[slot].tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.exposure_staging_buffers[slot],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // wgpu requires bytes_per_row aligned to
                    // COPY_BYTES_PER_ROW_ALIGNMENT (= 256). Our
                    // texel is only 2 bytes (R16Float); the rest
                    // of the row is padding the staging buffer
                    // tolerates.
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        true
    }

    /// Start an async map for slot `slot`. Call AFTER `queue.submit`
    /// for the encoder that ran `schedule_exposure_readback(slot)`.
    /// The callback flips state to `Ready`; `try_read_exposure_sample`
    /// then drains it without blocking.
    ///
    /// Engine equivalent: D3D11 driver implicitly tracks a fence per
    /// resource; `Map(D3D11_MAP_FLAG_DO_NOT_WAIT)` returns nullptr
    /// until the fence settles. wgpu has no per-resource sync, so we
    /// use `map_async` + a callback-set atomic.
    pub fn begin_exposure_map(&self, slot: usize) {
        let state = &self.exposure_slot_state[slot];
        // If something already mapped/in-flight, don't double-schedule.
        if state.load(Ordering::Acquire) != EXPOSURE_SLOT_IDLE {
            return;
        }
        state.store(EXPOSURE_SLOT_IN_FLIGHT, Ordering::Release);
        let cb_state = state.clone();
        self.exposure_staging_buffers[slot].slice(..).map_async(
            wgpu::MapMode::Read,
            move |result| {
                cb_state.store(
                    if result.is_ok() {
                        EXPOSURE_SLOT_READY
                    } else {
                        EXPOSURE_SLOT_IDLE
                    },
                    Ordering::Release,
                );
            },
        );
    }

    /// Non-blocking read of slot `slot`. Returns `Some(stops)` if the
    /// map callback has fired and the value is decoded; `None` if the
    /// GPU hasn't finished yet (caller should reuse last frame's value).
    ///
    /// Drives `device.poll(Poll)` so map callbacks can fire even
    /// without other API activity. `Poll` is non-blocking (vs `Wait`
    /// which drains the entire device queue per gfx-rs/wgpu#4589).
    pub fn try_read_exposure_sample(
        &self,
        shared: &SharedResources,
        slot: usize,
    ) -> Option<f32> {
        // Drive callbacks. Cheap if there's nothing pending.
        let _ = shared.device.poll(wgpu::PollType::Poll);
        let state = &self.exposure_slot_state[slot];
        if state.load(Ordering::Acquire) != EXPOSURE_SLOT_READY {
            return None;
        }
        let buffer = &self.exposure_staging_buffers[slot];
        let view = buffer.slice(..).get_mapped_range();
        let bits = u16::from_le_bytes([view[0], view[1]]);
        drop(view);
        buffer.unmap();
        state.store(EXPOSURE_SLOT_IDLE, Ordering::Release);
        Some(half::f16::from_bits(bits).to_f32())
    }

    /// `c_screen_postprocess::restore_surface_to_EDRAM @ 0x1806b6120`.
    /// Xbox 360-specific — restores a tiled surface from main memory
    /// back into the GPU's on-die EDRAM. PC has no EDRAM; the dllcache
    /// build is x64 (PC-targeted) but retains the function for
    /// API-shape compatibility — it amounts to a tile-resolve no-op
    /// outside the 360 build path.
    ///
    /// We keep the method as a structural marker so the orchestrator
    /// port can call it engine-faithfully; the body is a no-op.
    /// Callers in the dllcache: `c_screenshot_render_engine::copy_composited_tile_to_display`,
    /// `initialize_render_display` — both screenshot paths we don't
    /// exercise.
    pub fn restore_surface_to_EDRAM(&self, _surface_label: &str, _explicit_shader_index: i32) {
        // No-op on PC.
    }

    /// `c_screen_postprocess::gaussian_blur @ 0x1806b69b0`. Two-pass
    /// blur using fixed 11-tap kernels (shader 11 horizontal, 12
    /// vertical). The `horizontal_blur_size` / `vertical_blur_size`
    /// args exist in the engine signature but are unused in this body
    /// — the kernel weights are baked into the shaders. Caller passes
    /// blur_size for ordering / debugging only.
    ///
    /// `target_surface` ↔ `temp_surface` ping-pong: H pass copies
    /// target → temp, V pass copies temp → target. Final result lives
    /// in `target_surface`.
    pub fn gaussian_blur(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        temp: &wgpu::TextureView,
        target_size: [u32; 2],
        _horizontal_blur_size: f32,
        _vertical_blur_size: f32,
    ) {
        encoder.push_debug_group("gaussian_blur");
        // Shader 11 — blur_11_horizontal.
        self.copy(
            shared, encoder, &self.pipelines.blur_11_horizontal,
            target, target_size, temp,
            wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge,
            [1.0, 1.0, 1.0, 1.0],
        );
        // Shader 12 — blur_11_vertical.
        self.copy(
            shared, encoder, &self.pipelines.blur_11_vertical,
            temp, target_size, target,
            wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge,
            [1.0, 1.0, 1.0, 1.0],
        );
        encoder.pop_debug_group();
    }

    /// `c_screen_postprocess::blur_display @ 0x1806b6aa0`. GUI/menu
    /// utility — blurs the current display contents into the bloom
    /// scratch surface for use as a backdrop. Engine flow:
    ///   1. stretch_rect display → albedo
    ///   2. resolve_surface(albedo)
    ///   3. copy(shader 39, albedo → aux_bloom)
    ///   4. resolve_surface(aux_bloom)
    ///   5. gaussian_blur(aux_bloom, aux_exposure_5)
    ///   6. resolve_surface(aux_bloom)
    ///   7. return e_surface_aux_bloom (= 25)
    ///
    /// Used by `render_bitmap` for blurred GUI panel backgrounds.
    /// Protomorph has no GUI surfaces yet — this is a stub that
    /// returns `None` and logs once. The hook lands when GUI rendering
    /// (CHUD / menu / pause overlay) gets ported.
    pub fn blur_display(
        &self,
        _horizontal_blur_size: f32,
        _vertical_blur_size: f32,
    ) -> Option<&wgpu::TextureView> {
        // Engine returns `e_surface_aux_bloom` (= 25). When GUI
        // rendering is ported, return `Some(&self.aux_bloom.view)`
        // after running the steps above.
        None
    }

    /// `c_screen_postprocess::copy @ 0x1806b64c0`. Single-input fullscreen
    /// postprocess shader dispatch — same plumbing as `apply_binary_op_ex`
    /// but only one source texture. Used when the postprocess pixel
    /// shader needs only a single sampled input (e.g. tone curves,
    /// gamma adjust, single-tap copies).
    ///
    /// Engine body sets RT0=dest, source as sampler 0, writes
    /// pixel_size = 1/source.{w,h} + scale, draws fullscreen quad,
    /// clears sampler texture 0. Wgpu version uses bgl_1tex.
    #[allow(clippy::too_many_arguments)]
    pub fn copy(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::TextureView,
        source_size: [u32; 2],
        dest: &wgpu::TextureView,
        filter_mode: wgpu::FilterMode,
        _address_mode: wgpu::AddressMode,
        scale: [f32; 4],
    ) {
        self.upload_uniforms(
            shared,
            [source_size[0] as f32, source_size[1] as f32],
            scale,
        );
        let sampler = match filter_mode {
            wgpu::FilterMode::Linear => &self.sampler_bilinear,
            wgpu::FilterMode::Nearest => &self.sampler_point,
        };
        use crate::halo::render::bind_group_cache::{key_from_ptrs, ptr_addr};
        let bg = self.cached_bg(
            key_from_ptrs(&[ptr_addr(source), ptr_addr(sampler), ptr_addr(&self.bgl_1tex)]),
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("copy"),
                layout: &self.bgl_1tex,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buffer.as_entire_binding() },
                ],
            }),
        );
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, bg.as_ref(), &[]);
        rp.draw(0..3, 0..1);
    }

    /// `c_screen_postprocess::blit @ 0x1806b6280`. Variant of `copy`
    /// that draws a sub-rect: takes `source_texture_rect` (UV region
    /// to sample from) and `dest_texture_rect` (pixel region of dest
    /// to write into) — equivalent to `draw_screen_quad_with_texture_transform`.
    ///
    /// `dest_texture_rect` becomes a viewport on the wgpu render pass.
    /// `source_texture_rect` would normally feed a UV transform
    /// uniform, but our existing postprocess pipelines sample with the
    /// full UV range — supporting rect-blit needs a shader change to
    /// read a UV-transform cbuffer slot. For now this method asserts
    /// that the source rect spans `[0, 1]²`; non-fullscreen source
    /// rects are deferred until a `blit_uv_transform.wgsl` lands.
    #[allow(clippy::too_many_arguments)]
    pub fn blit(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::TextureView,
        source_size: [u32; 2],
        dest: &wgpu::TextureView,
        filter_mode: wgpu::FilterMode,
        address_mode: wgpu::AddressMode,
        scale: [f32; 4],
        source_rect_uv: [f32; 4],   // (u0, v0, u1, v1)
        dest_rect_px: [u32; 4],     // (x0, y0, x1, y1)
        dest_size: [u32; 2],
    ) {
        // v1 limitation: non-fullscreen source rect needs a UV-transform
        // cbuffer + matching shader. Fall through to `copy` when source
        // covers the full texture; otherwise log + skip the dispatch.
        let full_source = source_rect_uv == [0.0, 0.0, 1.0, 1.0];
        if !full_source {
            eprintln!(
                "blit: source_rect_uv != [0,0,1,1] (got {:?}) — UV transform path not yet ported, skipping",
                source_rect_uv,
            );
            return;
        }
        self.upload_uniforms(
            shared,
            [source_size[0] as f32, source_size[1] as f32],
            scale,
        );
        let sampler = match filter_mode {
            wgpu::FilterMode::Linear => &self.sampler_bilinear,
            wgpu::FilterMode::Nearest => &self.sampler_point,
        };
        let _ = address_mode;
        use crate::halo::render::bind_group_cache::{key_from_ptrs, ptr_addr};
        let bg = self.cached_bg(
            key_from_ptrs(&[ptr_addr(source), ptr_addr(sampler), ptr_addr(&self.bgl_1tex)]),
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit"),
                layout: &self.bgl_1tex,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buffer.as_entire_binding() },
                ],
            }),
        );
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        // dest_rect_px → viewport. Engine `draw_screen_quad_with_texture_transform`
        // restricts the rasterizer to the dest pixel region.
        let _ = dest_size;
        let vx = dest_rect_px[0] as f32;
        let vy = dest_rect_px[1] as f32;
        let vw = (dest_rect_px[2] - dest_rect_px[0]) as f32;
        let vh = (dest_rect_px[3] - dest_rect_px[1]) as f32;
        rp.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, bg.as_ref(), &[]);
        rp.draw(0..3, 0..1);
    }

    /// `c_screen_postprocess::apply_binary_op @ 0x1804f38f0`. Thin
    /// wrapper for `apply_binary_op_ex` with `(point, clamp)` sampler
    /// state — engine's default for screenshot/bloom paths that don't
    /// need bilinear filtering.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_binary_op(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source_0: &wgpu::TextureView,
        source_0_size: [u32; 2],
        source_1: &wgpu::TextureView,
        dest: &wgpu::TextureView,
        scale: [f32; 4],
    ) {
        self.apply_binary_op_ex(
            shared, encoder, pipeline,
            source_0, source_0_size, source_1, dest,
            wgpu::FilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
            scale,
        );
    }

    fn upload_uniforms(
        &self,
        shared: &SharedResources,
        size: [f32; 2],
        scale: [f32; 4],
    ) {
        let u = PostprocessUniforms {
            pixel_size: [1.0 / size[0], 1.0 / size[1], size[0], size[1]],
            scale,
            intensity_vector: [0.299, 0.587, 0.114, 0.0],
            dark_color_multiplier: [1.0, 0.0, 0.0, 0.0],
        };
        shared.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }

    fn run_pipeline_2tex(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        source_a: &wgpu::TextureView,
        source_b: &wgpu::TextureView,
        dest: &wgpu::TextureView,
        clear: wgpu::Color,
    ) {
        use crate::halo::render::bind_group_cache::{key_from_ptrs, ptr_addr};
        let bg = self.cached_bg(
            key_from_ptrs(&[ptr_addr(source_a), ptr_addr(source_b), ptr_addr(&self.bgl_2tex)]),
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bgl_2tex,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source_a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(source_b) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                    wgpu::BindGroupEntry { binding: 4, resource: self.uniform_buffer.as_entire_binding() },
                ],
            }),
        );
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, bg.as_ref(), &[]);
        rp.draw(0..3, 0..1);
    }

    fn run_pipeline_1tex(
        &self,
        shared: &SharedResources,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::TextureView,
        dest: &wgpu::TextureView,
        clear: wgpu::Color,
    ) {
        use crate::halo::render::bind_group_cache::{key_from_ptrs, ptr_addr};
        let bg = self.cached_bg(
            key_from_ptrs(&[ptr_addr(source), ptr_addr(&self.sampler_bilinear), ptr_addr(&self.bgl_1tex)]),
            || shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.bgl_1tex,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_bilinear) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buffer.as_entire_binding() },
                ],
            }),
        );
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dest, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, bg.as_ref(), &[]);
        rp.draw(0..3, 0..1);
    }
}
