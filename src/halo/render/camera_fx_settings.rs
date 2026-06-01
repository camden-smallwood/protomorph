//! Mirror of `Ares/source/render/camera_fx_settings.h` — postprocess
//! parameter source.
//!
//! The render path reads camera-fx values out of two related types:
//!   - `c_camera_fx_settings` (368B) — artist-authored values from the
//!     `cfxs` tag (or scenario-default / cluster-palette overrides).
//!     Static "what should the postprocess look like" data.
//!   - `c_camera_fx_values` (536B) — runtime state that interpolates
//!     toward the active settings each frame and exposes the final
//!     numbers the postprocess shader reads (`m_exposure`,
//!     `m_bloom_point`, `m_bloom_inherent`, etc.).
//!
//! `s_render_game_state::s_player_window` (render_game_state.h)
//! carries one of each per player_window (4 slots for splitscreen).
//!
//! Update logic (`c_camera_fx_values::update`,
//! `s_real_parameter::update`, the `c_exposure` autoexposure pipeline)
//! is gameplay → render glue that lands when its callers are ported
//! (Phase 9 finish + cfxs tag walker). The structs below are
//! layout-faithful so the runtime code can be filled in later without
//! re-shaping data.

use glam::Vec3;

// ---------------------------------------------------------------------------
// s_scripted_exposure (52B) — global script-driven exposure override.
// ---------------------------------------------------------------------------

/// `s_scripted_exposure` (camera_fx_settings.h:24-54, 52B).
#[derive(Debug, Clone, Copy)]
pub struct ScriptedExposure {
    pub flags: u32,                                  // 0x0
    pub light_intensity_scale: f32,                  // 0x4
    pub exposure_bias: f32,                          // 0x8
    pub exposure_blend: f32,                         // 0xC
    pub exposure_stops: f32,                         // 0x10
    pub original_exposure_stops: f32,                // 0x14
    pub target_exposure_stops: f32,                  // 0x18
    pub original_exposure_blend: f32,                // 0x1C
    pub target_exposure_blend: f32,                  // 0x20
    pub exposure_animation_start_time: i32,          // 0x24
    pub exposure_animation_end_time: i32,            // 0x28
    pub exposure_adapt_instantly_frames: i32,        // 0x2C
    pub disable_autoexposure_frames: i32,            // 0x30
}

impl Default for ScriptedExposure {
    /// `s_scripted_exposure::initialize_for_new_map @ 0x180686C00` — verbatim port.
    /// Engine assigns `m_light_intensity_scale = 1.0` and zeros everything else.
    /// `#[derive(Default)]` would give `light_intensity_scale = 0.0`, which
    /// multiplicatively zeroes every light through `m_lights_view.m_light_intensity_scale`
    /// in `setup_camera_fx_parameters` (engine line 140).
    fn default() -> Self {
        Self {
            flags: 0,
            light_intensity_scale: 1.0,
            exposure_bias: 0.0,
            exposure_blend: 0.0,
            exposure_stops: 0.0,
            original_exposure_stops: 0.0,
            target_exposure_stops: 0.0,
            original_exposure_blend: 0.0,
            target_exposure_blend: 0.0,
            exposure_animation_start_time: 0,
            exposure_animation_end_time: 0,
            exposure_adapt_instantly_frames: 0,
            disable_autoexposure_frames: 0,
        }
    }
}

impl ScriptedExposure {
    pub const FAST_ADAPT_AUTOEXPOSURE_BIT: u32 = 1 << 0;
    pub const DISABLE_AUTOEXPOSURE_BIT: u32 = 1 << 1;
    pub const ANIMATE_EXPOSURE_BIT: u32 = 1 << 2;
    pub const FORCE_DISABLE_AUTOEXPOSURE_BIT: u32 = 1 << 3;

    /// `s_scripted_exposure::render_apply_scripted_exposure @ 0x180686ab0` —
    /// verbatim port. Lerps `default_stops` (= auto-exposure m_exposure)
    /// toward the scripted `exposure_stops` by `exposure_blend`, then adds
    /// `exposure_bias`. With the default zero-initialized state
    /// (`exposure_blend = 0`, `exposure_bias = 0`), returns
    /// `default_stops` unchanged.
    ///
    /// ```c
    /// return (1 - m_exposure_blend) * default_stops
    ///      + m_exposure_blend * m_exposure_stops
    ///      + m_exposure_bias;
    /// ```
    pub fn render_apply_scripted_exposure(&self, default_stops: f32) -> f32 {
        (1.0 - self.exposure_blend) * default_stops
            + self.exposure_blend * self.exposure_stops
            + self.exposure_bias
    }

    /// `s_scripted_exposure::render_allow_autoexposure @ 0x180686C0` —
    /// returns `(this->m_flags & 2) == 0`. When DISABLE_AUTOEXPOSURE
    /// is set, callers SHOULD skip the auto-exposure ring-buffer push
    /// AND the m_exposure blend.
    pub fn render_allow_autoexposure(&self) -> bool {
        (self.flags & Self::DISABLE_AUTOEXPOSURE_BIT) == 0
    }

    /// `s_scripted_exposure::render_fast_exposure_adapt` —
    /// `(this->m_flags & 1) != 0`. Triggers the FAST_ADAPT 0.1/0.9
    /// blend path in `Exposure::calculate_new_value`.
    pub fn render_fast_exposure_adapt(&self) -> bool {
        (self.flags & Self::FAST_ADAPT_AUTOEXPOSURE_BIT) != 0
    }

    /// `s_scripted_exposure::update_scripted_exposure @ 0x180686A00` —
    /// verbatim port. Called once per tick to advance the scripted
    /// exposure animation (when `ANIMATE_EXPOSURE_BIT` is set) and to
    /// manage the `DISABLE_AUTOEXPOSURE_BIT` / `FAST_ADAPT_AUTOEXPOSURE_BIT`
    /// flag bits based on the frame-counter fields. Engine `dt`
    /// parameter is unused — timing is driven by `game_time_get()`
    /// (integer tick counter, 30 Hz canonical).
    ///
    /// With default-zero state (no HSC scripts running) this method
    /// short-circuits at every branch: ANIMATE bit off, blend stays 0,
    /// counters stay 0 → flags untouched, no work done. Becomes active
    /// when HSC opcodes (`render_set_exposure`, `render_fast_exposure_adapt`,
    /// `render_disable_autoexposure`) populate the struct — at which
    /// point this method's advance shapes how those overrides decay.
    ///
    /// Smoothstep formula: `t² * (3 - 2t)` for the t fraction. Engine
    /// applies it as `t * t * (3 - 2t)` written awkwardly as
    /// `(t) * ((3 - t*2) * t)`.
    pub fn update_scripted_exposure(&mut self, game_time_ticks: i32) {
        // Engine line 14: if ANIMATE_EXPOSURE bit is set
        if (self.flags & Self::ANIMATE_EXPOSURE_BIT) != 0 {
            let start_time = self.exposure_animation_start_time;
            let end_time = self.exposure_animation_end_time;
            let duration = end_time - start_time;
            if game_time_ticks >= end_time || duration <= 0 {
                // Engine line 19-26: animation complete. Snap to target,
                // lock autoexposure for 3 frames, clear ANIMATE bit.
                self.exposure_stops = self.target_exposure_stops;
                self.exposure_blend = self.target_exposure_blend;
                self.flags &= !Self::ANIMATE_EXPOSURE_BIT;
                self.disable_autoexposure_frames = 3;
            } else {
                // Engine line 28-37: smoothstep interpolate.
                let elapsed = (game_time_ticks - start_time) as f32;
                let t = elapsed / duration as f32;
                let smoothed = t * t * (3.0 - 2.0 * t);
                self.exposure_stops = (1.0 - smoothed) * self.original_exposure_stops
                    + smoothed * self.target_exposure_stops;
                self.exposure_blend = (1.0 - smoothed) * self.original_exposure_blend
                    + smoothed * self.target_exposure_blend;
            }
        }
        // Engine line 39-42: DISABLE_AUTOEXPOSURE bit gating.
        // Set when blend is mid-animation, FORCE_DISABLE bit is on, or
        // the disable-frames counter is still draining.
        if self.exposure_blend > 0.0
            || (self.flags & Self::FORCE_DISABLE_AUTOEXPOSURE_BIT) != 0
            || self.disable_autoexposure_frames > 0
        {
            self.flags |= Self::DISABLE_AUTOEXPOSURE_BIT;
        } else {
            self.flags &= !Self::DISABLE_AUTOEXPOSURE_BIT;
        }
        // Engine line 44-45: decrement disable counter.
        if self.disable_autoexposure_frames > 0 {
            self.disable_autoexposure_frames -= 1;
        }
        // Engine line 46-55: FAST_ADAPT bit follows the
        // adapt-instantly-frames counter; counter ticks down each frame.
        if self.exposure_adapt_instantly_frames <= 0 {
            self.flags &= !Self::FAST_ADAPT_AUTOEXPOSURE_BIT;
        } else {
            self.flags |= Self::FAST_ADAPT_AUTOEXPOSURE_BIT;
            self.exposure_adapt_instantly_frames -= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter sub-types (SsaoParameter, ColorGradingParameter, Lightshafts,
// BrightnessContrast, Hslv, ColorizeEffect, Cmyb, Cmy, SelectiveColor,
// ColorBalance) used to live here as a "1:1 engine layout mirror" before
// `blam_tags::camera_fx_settings` carried full parsers. Deleted 2026-05-20
// (commit removing dead sub-type scaffolding) — they were never read by
// the runtime, and the corresponding `CameraFxValues::{ssao, color_grading,
// lightshafts}` fields had zero consumers. Authoritative tag-parser
// shapes now live in `blam_tags::camera_fx_settings`
// (`SsaoBlock`, `ColorGradingBlock`, `LightshaftsBlock`, etc.); when a
// runtime consumer ships it can either use those types directly or
// declare a slimmer protomorph-side view.

// ---------------------------------------------------------------------------
// c_camera_fx_settings::s_parameter flag bits.
// ---------------------------------------------------------------------------
//
// The cfxs nested parameter types (`s_real_parameter`,
// `s_real_instant_parameter`, `s_color_parameter`, `s_word_parameter`,
// `s_real_exposure_parameter`) all share the same 16-bit flag field
// (`m_flags`). The shapes themselves live in `blam_tags::camera_fx_settings`
// (`ScalarParameter`, `InstantScalarParameter`, etc.) — the L2/L3 update
// helpers in this file consume those types directly.
//
// `ParameterFlags` is a unit-struct namespace for the bit constants.

pub struct ParameterFlags;

impl ParameterFlags {
    /// Bit 0 — `use default (ignore these values)`. The 4-level fallback
    /// chooser treats the parameter as absent and walks to the next source.
    pub const USE_DEFAULT_BIT: u16 = 1 << 0;
    /// Bit 1 — `s_real_parameter::calculate_new_value` scales `m_blend_limit`
    /// by `|old|` (relative cap rather than absolute).
    pub const BLEND_LIMIT_RELATIVE_BIT: u16 = 1 << 1;
    /// Bit 2 — auto-adjust target. `s_real_exposure_parameter` reads the
    /// auto-exposure sensor sample instead of using `m_target` directly.
    pub const AUTO_BIT: u16 = 1 << 2;
    /// Bit 3 — `double sided star` (bling cosmetic).
    pub const DOUBLE_SIDED_STAR_BIT: u16 = 1 << 3;
    /// Bit 4 — `fixed` (UI hint, not consumed by the runtime math).
    pub const FIXED_BIT: u16 = 1 << 4;
}

// ---------------------------------------------------------------------------
// c_exposure (304B) — autoexposure sliding-window history.
// ---------------------------------------------------------------------------

/// `c_exposure` (IDA-verified 2026-05-08, dllcache size 276B).
///
/// Autoexposure runs by sampling the GPU-rendered HDR luminance into
/// a small queue (`m_render_target_queue[3]`) of ring-slot indices,
/// reading them back 3 frames later. The CPU keeps a 60-entry
/// history (`m_exposure_buffer`) for the adapt-from-min/max-historic
/// hysteresis check inside `calculate_new_value`.
///
/// IDA struct layout @ 0x180686F50 (`get_type_info c_exposure`):
/// ```text
///   0x000  m_prev_exposure        : float
///   0x004  m_exposure             : float
///   0x008  m_exposure_velocity    : float
///   0x00C  m_exposure_buffer[60]  : float[60]   (240 bytes)
///   0x0FC  m_buffer_index         : int
///   0x100  m_render_target_queue[3]: int[3]      (surface indices)
///   0x10C  m_render_target_queue_size  : int
///   0x110  m_render_target_queue_index : int
///   total = 276 bytes
/// ```
///
/// Field semantics:
/// - `prev_exposure` / `exposure` / `exposure_velocity` are in **log2
///   stops** — `calculate_new_value` adds/subtracts brightness in
///   log space, and the final `view_exposure` formula raises 2 to it.
/// - `render_target_queue` stores `ring_slot` indices in [0..8) (a
///   layer of indirection over the engine's surface enum 28..35).
/// - `render_target_queue_index` points to the **next slot to
///   overwrite** in the FIFO. When `queue_size == 3`, this is also
///   the slot holding the OLDEST entry — the one
///   `calculate_new_value` reads.
#[derive(Debug, Clone, Copy)]
pub struct Exposure {
    pub prev_exposure: f32,                        // 0x000
    pub exposure: f32,                             // 0x004
    pub exposure_velocity: f32,                    // 0x008
    pub exposure_buffer: [f32; 60],                // 0x00C
    pub buffer_index: i32,                         // 0x0FC
    pub render_target_queue: [i32; 3],             // 0x100
    pub render_target_queue_size: i32,             // 0x10C
    pub render_target_queue_index: i32,            // 0x110
}

impl Default for Exposure {
    fn default() -> Self {
        Self {
            prev_exposure: 0.0,
            exposure: 0.0,
            exposure_velocity: 0.0,
            exposure_buffer: [0.0; 60],
            buffer_index: 0,
            render_target_queue: [-1; 3],
            render_target_queue_size: 0,
            render_target_queue_index: 0,
        }
    }
}

/// Engine `game_seconds_to_ticks_real(seconds) = seconds *
/// game_time_globals->tick_rate`. Halo 3 tick_rate is canonically 30 Hz.
const EXPOSURE_HISTORY_TICK_RATE: f32 = 30.0;

/// Module-scope mirror of `c_exposure::m_next_render_target` (engine
/// class-level static — `c_screen_postprocess::downsample_generate @
/// 0x1806B5C00:51-55`). Single counter shared across every
/// `c_exposure` instance in the process. Used by `downsample_generate`
/// to pick the next 1×1 R16Float ring slot (0..7) for the current
/// frame's auto-exposure sample.
///
/// Reset via `Exposure::reset_next_render_target()` (engine-equivalent
/// of zeroing the static on `initialize_for_new_map`).
static EXPOSURE_M_NEXT_RENDER_TARGET: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

impl Exposure {
    /// Engine `_surface_aux_exposure_0` surface-enum value.
    /// `c_screen_postprocess::downsample_generate @ 0x1806B5C00:51`
    /// computes the ring-slot's surface index as
    /// `v8 = c_exposure::m_next_render_target + 28`. The 8 ring slots
    /// occupy engine surface enum positions 28..35. The FIFO
    /// `m_render_target_queue` stores these surface indices (not raw
    /// slot indices); consumers translate back to a slot via subtraction.
    pub const AUX_EXPOSURE_SURFACE_BASE: i32 = 28;

    /// Convert a surface enum value pulled from `m_render_target_queue`
    /// back to a 0..7 slot index for indexing into protomorph's
    /// `ScreenPostprocess::aux_exposure[]` array. Returns `None` for
    /// the engine sentinel `-1`.
    pub fn slot_from_surface_index(surface_index: i32) -> Option<usize> {
        if surface_index < Self::AUX_EXPOSURE_SURFACE_BASE {
            return None;
        }
        let slot = (surface_index - Self::AUX_EXPOSURE_SURFACE_BASE) as usize;
        if slot < 8 { Some(slot) } else { None }
    }

    /// Translate a 0..7 slot index to the engine surface enum value
    /// stored in `m_render_target_queue`. Inverse of `slot_from_surface_index`.
    pub fn surface_index_from_slot(slot: u32) -> i32 {
        Self::AUX_EXPOSURE_SURFACE_BASE + slot as i32
    }

    /// Increment + return the engine class-static
    /// `c_exposure::m_next_render_target` (cycles 0..8). Verbatim port
    /// of `downsample_generate @ 0x1806B5C00:51-55`. Returns the slot
    /// the CURRENT frame writes into; the static then advances to
    /// (slot+1) % 8 for the next frame.
    pub fn next_render_target() -> u32 {
        use std::sync::atomic::Ordering;
        let slot = EXPOSURE_M_NEXT_RENDER_TARGET.load(Ordering::Relaxed);
        EXPOSURE_M_NEXT_RENDER_TARGET.store((slot + 1) % 8, Ordering::Relaxed);
        slot
    }

    /// Reset `m_next_render_target` to 0. Engine equivalent of zeroing
    /// the class-static during `s_render_game_state::initialize_for_new_map`
    /// (engine doesn't explicitly call this — the static persists across
    /// maps in practice — but protomorph exposes the reset for test
    /// determinism and scenario teardown symmetry).
    pub fn reset_next_render_target() {
        EXPOSURE_M_NEXT_RENDER_TARGET.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// `c_exposure::set_exposure @ 0x180687410` — verbatim port.
    /// ```c
    /// this->m_exposure = initial_exposure;
    /// this->m_exposure_velocity = 0.0;
    /// for (i = 60; i; --i) *m_exposure_buffer++ = initial_exposure;
    /// ```
    /// Note: does NOT touch `m_prev_exposure`, `m_buffer_index`, or
    /// `m_render_target_queue[3]` (engine leaves them at their prior values).
    /// Callers that need the FIFO cleared must invoke
    /// `reset_render_target_queue` separately, mirroring engine's pattern in
    /// `s_render_game_state::initialize_for_new_map`.
    pub fn set_exposure(&mut self, initial_exposure: f32) {
        self.exposure = initial_exposure;
        self.exposure_velocity = 0.0;
        self.exposure_buffer = [initial_exposure; 60];
    }

    /// `c_exposure::reset_render_target_queue @ 0x1806876B0` — verbatim port.
    /// ```c
    /// *(_QWORD *)&this->m_render_target_queue_size = 0;
    /// ```
    /// (Single 8-byte zero spans the two adjacent i32 fields
    /// `m_render_target_queue_size` + `m_render_target_queue_index`.)
    /// Does NOT clear the queue array contents — the unread slots are
    /// gated by `queue_size` reaching 3 before the consumer reads them.
    pub fn reset_render_target_queue(&mut self) {
        self.render_target_queue_size = 0;
        self.render_target_queue_index = 0;
    }

    /// `c_exposure::update_sampled_exposure @ 0x180687600` — push a
    /// ring-slot index into the FIFO. Engine version takes a surface
    /// index and calls `update_staging` on the surface texture (the
    /// GPU→CPU staging copy). Our caller schedules the readback
    /// separately (Phase C), so this method is the pure FIFO push.
    pub fn update_sampled_exposure(&mut self, ring_slot: i32) {
        let idx = self.render_target_queue_index as usize;
        self.render_target_queue[idx] = ring_slot;
        self.render_target_queue_index = (self.render_target_queue_index + 1) % 3;
        if self.render_target_queue_size < 3 {
            self.render_target_queue_size += 1;
        }
    }

    /// `c_exposure::get_min_historic_exposure @ 0x180687480` — scan
    /// the most recent `seconds * tick_rate` entries (clamped to
    /// [1, 60]) of `m_exposure_buffer` and return the minimum.
    pub fn get_min_historic_exposure(&self, seconds: f32) -> f32 {
        let ticks = (seconds * EXPOSURE_HISTORY_TICK_RATE) as i32;
        let count = ticks.max(1).min(60);
        let mut idx = (self.buffer_index - count).rem_euclid(60) as usize;
        let mut result = f32::MAX;
        for _ in 0..count {
            let v = self.exposure_buffer[idx];
            if v < result { result = v; }
            idx = (idx + 1) % 60;
        }
        result
    }

    /// `c_exposure::get_max_historic_exposure @ 0x180687540` — same
    /// as `get_min_historic_exposure` but returns the max. Engine's
    /// initial sentinel is `-FLT_MAX`.
    pub fn get_max_historic_exposure(&self, seconds: f32) -> f32 {
        let ticks = (seconds * EXPOSURE_HISTORY_TICK_RATE) as i32;
        let count = ticks.max(1).min(60);
        let mut idx = (self.buffer_index - count).rem_euclid(60) as usize;
        let mut result = f32::MIN;
        for _ in 0..count {
            let v = self.exposure_buffer[idx];
            if v >= result { result = v; }
            idx = (idx + 1) % 60;
        }
        result
    }

    /// `c_camera_fx_settings::s_real_exposure_parameter::calculate_new_value
    /// @ 0x180686F50` — verbatim port with the engine signature.
    ///
    /// Engine prototype (verified via dllcache decompile):
    /// ```c
    /// void calculate_new_value(
    ///     s_real_exposure_parameter *this,
    ///     c_exposure                *exposure,
    ///     char                       flags,
    ///     double                     exposure_target,
    ///     float                      auto_exposure_screen_brightness,
    ///     float                      exposure_min,
    ///     float                      exposure_max);
    /// ```
    /// All five "value" args are overrides that the cluster_palette
    /// merge in `c_camera_fx_values::update` may have rewritten before
    /// the call. `params` is read only for `m_blend_speed` and
    /// `m_auto_exposure_delay`.
    ///
    /// `actual_brightness` is the protomorph-side substitute for the
    /// engine's synchronous `lock_read` of the 3-frame-old GPU sample
    /// (we run that readback async via `update_sampled_exposure`):
    /// `Some(half-decoded f32)` when the FIFO is full and the slot is
    /// valid; `None` otherwise (engine `v10 = 0.0` branch).
    ///
    /// Hysteresis (the trickiest piece):
    /// - Brightening (`m_exposure < target`): blend ONLY if `m_exposure`
    ///   is below the historic min over the `auto_exposure_delay` window.
    ///   Otherwise hold (velocity=0).
    /// - Darkening (`m_exposure > target`): blend ONLY if `m_exposure`
    ///   is above the historic max over the same window. Otherwise hold.
    /// This prevents flickering when the target oscillates inside the
    /// recent history range.
    ///
    /// Not ported (deliberately — scripted-exposure path is out of L1-L6
    /// scope per `project_lighting_track_phases.md`):
    /// - `scripted_exposure_game_globals->m_flags & 2` (history-replace).
    /// - `scripted_exposure_game_globals->m_flags & 1` (FAST_ADAPT 0.1/0.9
    ///   blend short-circuit).
    /// At default-zero scripted globals both gates are off — current code
    /// matches the engine's default behavior.
    pub fn calculate_new_value(
        &mut self,
        params: &blam_tags::camera_fx_settings::ExposureBlock,
        flags: u16,
        exposure_target: f32,
        auto_exposure_screen_brightness: f32,
        exposure_min: f32,
        exposure_max: f32,
        actual_brightness: Option<f32>,
        auto_exposure_lock: bool,
        scripted: &ScriptedExposure,
    ) {
        // Engine: `v9 = exposure_target` is the artist-authored
        // static target; with the AUTO_BIT set, replaced by
        // `prev + screen_brightness_target - actual_brightness`.
        let mut v9 = exposure_target;
        if (flags & ParameterFlags::AUTO_BIT) != 0 {
            let v10 = actual_brightness.unwrap_or(0.0);
            v9 = self.prev_exposure + auto_exposure_screen_brightness - v10;
        }
        let v9_pre_clamp = v9;

        // Capture pre-update value for next frame's prev.
        self.prev_exposure = self.exposure;

        // Clamp to [exposure_min, exposure_max].
        v9 = v9.clamp(exposure_min, exposure_max);

        // Push into the 60-frame ring. Engine line 64-71 gates this on
        // `(scripted_exposure_game_globals->m_flags & 2) == 0` — i.e.,
        // skip when DISABLE_AUTOEXPOSURE bit is set. Engine wraps the
        // gate in `render_allow_autoexposure()`. Default-zero state →
        // allow == true → push always happens.
        let allow_autoexposure = scripted.render_allow_autoexposure();
        if allow_autoexposure {
            self.exposure_buffer[self.buffer_index as usize] = v9;
            self.buffer_index = (self.buffer_index + 1) % 60;
        }

        // Engine line 72-81 — FAST_ADAPT short-circuit. When both gates
        // are right (DISABLE_AUTOEXPOSURE off AND FAST_ADAPT on), engine
        // does a fixed 10/90 blend toward v9 and returns. Used by
        // scripted exposure transitions (cutscenes, `render_set_exposure`
        // HSC) that want exposure to converge in a few frames instead of
        // the velocity-clamped seconds.
        if allow_autoexposure && scripted.render_fast_exposure_adapt() {
            self.exposure = 0.1 * self.exposure + 0.9 * v9;
            return;
        }

        let m_exp = self.exposure;
        let historic_min = self.get_min_historic_exposure(params.auto_exposure_delay);
        let historic_max = self.get_max_historic_exposure(params.auto_exposure_delay);
        let should_blend = if m_exp < v9 {
            m_exp < historic_min
        } else if m_exp > v9 {
            m_exp > historic_max
        } else {
            false
        };

        // PROTOMORPH_DEBUG_EXPOSURE=1 — log calculate_new_value internals
        // once per second (60 ticks). Useful for diagnosing why auto-
        // exposure isn't converging or behaves unexpectedly. Off by
        // default (no overhead).
        if std::env::var("PROTOMORPH_DEBUG_EXPOSURE").map(|s| s == "1").unwrap_or(false) {
            static TRACE_TICK: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            let t = TRACE_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if t % 60 == 0 {
                eprintln!(
                    "[exposure-trace] flags=0x{:04x} target={:.3} auto_b={:.4} min={:.2} max={:.2} \
                     blend_speed={:.4} delay={:.4} actual_b={:?} prev={:.4} m_exp={:.4} \
                     v9_pre={:.4} v9_clamped={:.4} hist_min={:.4} hist_max={:.4} blend={}",
                    flags,
                    exposure_target,
                    auto_exposure_screen_brightness,
                    exposure_min,
                    exposure_max,
                    params.blend_speed,
                    params.auto_exposure_delay,
                    actual_brightness,
                    self.prev_exposure,
                    m_exp,
                    v9_pre_clamp,
                    v9,
                    historic_min,
                    historic_max,
                    should_blend,
                );
            }
        }
        let _ = v9_pre_clamp; let _ = historic_max;

        if !should_blend {
            self.exposure_velocity = 0.0;
            return;
        }

        // Velocity-clamped blend toward v9. Engine
        // `calculate_new_value @ 0x180686F50:102-107`:
        // ```c
        // m_blend_speed = this->m_blend_speed;
        // v24 = |m_exposure - v9|;
        // if (m_blend_speed > v24 * 0.05) m_blend_speed = v24 * 0.05;
        // if (!c_screen_postprocess::x_settings->m_auto_exposure_lock
        //     && (scripted_exposure_game_globals->m_flags & 2) == 0)
        //     exposure->m_exposure = (1 - m_blend_speed) * m_exposure
        //                          + m_blend_speed * v9;
        // ```
        // BOTH gates skip the write but the velocity-clamped math still
        // runs. velocity field already cleared in the HOLD early-return.
        let mut blend_speed = params.blend_speed;
        let abs_delta = (m_exp - v9).abs();
        if blend_speed > abs_delta * 0.05 {
            blend_speed = abs_delta * 0.05;
        }
        if !auto_exposure_lock && allow_autoexposure {
            self.exposure = (1.0 - blend_speed) * m_exp + blend_speed * v9;
        }
    }
}

// ---------------------------------------------------------------------------
// L2: parameter blend math — verbatim ports of the s_*_parameter::update
// family. These are pure math; L3 will call them per-frame from the master
// `c_camera_fx_values::update` blender.
// ---------------------------------------------------------------------------

/// `c_camera_fx_settings::s_real_parameter::calculate_new_value @ 0x180688640`
/// — velocity-clamped lerp with relative-cap option.
///
/// Engine body (after disambiguating Hex-Rays' SIMD `_xmm` masks for `|x|`
/// and unary minus):
/// ```text
/// new   = (1 - blend_speed) * old + blend_speed * target
/// if blend_limit < 0: return new                          // unclamped
/// limit = (flags & 2) ? blend_limit * |old| : blend_limit  // BLEND_LIMIT_RELATIVE_BIT
/// delta = new - old
/// if |delta| <= limit: return new                          // small step
/// if delta > 0: return old + limit                          // clamp positive overstep
/// else:         return old - limit                          // clamp negative overstep
/// ```
pub fn calc_new_value(
    param: &blam_tags::camera_fx_settings::ScalarParameter,
    old: f32,
    target: f32,
) -> f32 {
    let blend_speed = param.blend_speed;
    let new_value = (1.0 - blend_speed) * old + blend_speed * target;

    let blend_limit_raw = param.max_change;
    if blend_limit_raw < 0.0 {
        return new_value;
    }

    let limit = if (param.flags & ParameterFlags::BLEND_LIMIT_RELATIVE_BIT) != 0 {
        blend_limit_raw * old.abs()
    } else {
        blend_limit_raw
    };

    let delta = new_value - old;
    if delta.abs() <= limit {
        return new_value;
    }
    if delta > 0.0 {
        old + limit
    } else {
        old - limit
    }
}

/// `s_real_parameter::update(value) @ 0x180688A20` —
/// `*value = calculate_new_value(this, *value, m_target)`.
pub fn update_real(
    param: &blam_tags::camera_fx_settings::ScalarParameter,
    value: &mut f32,
) {
    *value = calc_new_value(param, *value, param.value);
}

/// `s_real_parameter::update(value, target_override) @ 0x180688A50` —
/// `*value = calculate_new_value(this, *value, target_override)`. Used by
/// `c_camera_fx_values::update` for `bloom_inherent` and `bloom_intensity`
/// when the cluster_palette overrides supply a different target.
pub fn update_real_with_override(
    param: &blam_tags::camera_fx_settings::ScalarParameter,
    value: &mut f32,
    target_override: f32,
) {
    *value = calc_new_value(param, *value, target_override);
}

/// `s_real_instant_parameter::update @ 0x180688A10` — snap to target.
/// `*value = m_target`.
pub fn update_instant(
    param: &blam_tags::camera_fx_settings::InstantScalarParameter,
    value: &mut f32,
) {
    *value = param.value;
}

/// `s_word_parameter::update @ 0x180688A70` — snap to target.
/// `*value = m_target`.
pub fn update_word(
    param: &blam_tags::camera_fx_settings::WordParameter,
    value: &mut u16,
) {
    *value = param.value;
}

/// `s_color_parameter::update @ 0x180688980` — fixed 5/95 lerp per channel.
/// ```c
/// value[i] = 0.05 * m_color[i] + 0.95 * value[i]
/// ```
/// `m_color` is the raw authored color — the `use_default` bit is consumed
/// by the upstream 4-level fallback chooser (it picks a *different* settings
/// source when set), not by this update step.
pub fn update_color(
    param: &blam_tags::camera_fx_settings::ColorParameter,
    value: &mut Vec3,
) {
    value.x = 0.05 * param.color.red   + 0.95 * value.x;
    value.y = 0.05 * param.color.green + 0.95 * value.y;
    value.z = 0.05 * param.color.blue  + 0.95 * value.z;
}

/// `s_real_exposure_parameter::update @ 0x1806889D0` — thin engine wrapper
/// around `Exposure::calculate_new_value`. Keeps the symmetric per-parameter
/// `update` API so L3's master blender mirrors the engine's call shape.
#[allow(clippy::too_many_arguments)]
pub fn update_real_exposure(
    param: &blam_tags::camera_fx_settings::ExposureBlock,
    exposure: &mut Exposure,
    flags: u16,
    target: f32,
    auto_exposure_screen_brightness: f32,
    min: f32,
    max: f32,
    actual_brightness: Option<f32>,
    auto_exposure_lock: bool,
    scripted: &ScriptedExposure,
) {
    exposure.calculate_new_value(
        param, flags, target, auto_exposure_screen_brightness, min, max,
        actual_brightness, auto_exposure_lock, scripted,
    );
}

// ---------------------------------------------------------------------------
// L3: 4-level fallback chooser (`choose_valid_parameter` — inlined per
// parameter inside the engine's `c_camera_fx_values::update`).
// ---------------------------------------------------------------------------

/// Trait that abstracts the `flags` field common to every cfxs sub-parameter.
/// Bit 0 (USE_DEFAULT_BIT) signals "ignore this source, fall to the next".
pub trait UseDefaultFlag {
    fn flags_word(&self) -> u16;

    fn use_default(&self) -> bool {
        (self.flags_word() & ParameterFlags::USE_DEFAULT_BIT) != 0
    }
}

impl UseDefaultFlag for blam_tags::camera_fx_settings::ScalarParameter {
    fn flags_word(&self) -> u16 { self.flags }
}
impl UseDefaultFlag for blam_tags::camera_fx_settings::InstantScalarParameter {
    fn flags_word(&self) -> u16 { self.flags }
}
impl UseDefaultFlag for blam_tags::camera_fx_settings::ColorParameter {
    fn flags_word(&self) -> u16 { self.flags }
}
impl UseDefaultFlag for blam_tags::camera_fx_settings::WordParameter {
    fn flags_word(&self) -> u16 { self.flags }
}
impl UseDefaultFlag for blam_tags::camera_fx_settings::ExposureBlock {
    fn flags_word(&self) -> u16 { self.flags }
}

/// 4-level fallback chooser, mirroring the inlined per-parameter walk in
/// `c_camera_fx_values::update @ 0x180687CB0`. Engine pattern (line 110-122
/// for `auto_exposure_sensitivity`, repeated per parameter):
/// ```c
/// p = &script->X;
/// if ((p->flags & 1) != 0) {
///     if (v10) p = &cluster->X;
///     if ((p->flags & 1) != 0) {
///         if (scenario) p = &scenario->X;
///         if ((p->flags & 1) != 0 && default) p = &default->X;
///     }
/// }
/// ```
/// `v10` is the resolved cfxs from `cluster_palette_entry.camera_fx_tag` —
/// nullable. The fallback is "stuck" at the previous level when its source
/// is None: e.g., if `cluster` is None we keep `p = script` and the inner
/// `(p->flags & 1)` check falls through correctly.
pub fn choose_valid<'a, P: UseDefaultFlag>(
    script: &'a P,
    cluster: Option<&'a P>,
    scenario: Option<&'a P>,
    default: Option<&'a P>,
) -> &'a P {
    let mut chosen = script;
    if chosen.use_default() {
        if let Some(c) = cluster { chosen = c; }
        if chosen.use_default() {
            if let Some(s) = scenario { chosen = s; }
            if chosen.use_default() {
                if let Some(d) = default { chosen = d; }
            }
        }
    }
    chosen
}

// ---------------------------------------------------------------------------
// L3: cluster_palette overlay (deferred to L3a — placeholder type).
// ---------------------------------------------------------------------------

/// `structure_camera_fx_palette_entry` (engine 48B) — the per-cluster cfxs
/// override entry stored in sbsp's `cluster_camera_fx_palettes` block.
///
/// Carries (a) a `cfxs` tag reference whose resolved struct overlays the
/// cluster level of the chooser (`v10` in the engine), and (b) override
/// scalars gated by `flags` bits. The override fields fire only when the
/// EXPOSURE chooser walked away from `script_settings` (engine line 83
/// `cluster_palette_entry && v13 != script_settings`).
///
/// Plumbing (sbsp walker + starting-cluster resolution) is L3a; for L3 main
/// scope this struct exists only as a placeholder so `update`'s signature
/// matches the engine.
#[derive(Debug, Clone)]
pub struct ClusterCameraFxPaletteOverrides {
    /// `flags` byte.
    ///   Bit 0 (`flags & 1`) → `forced_exposure` overrides exposure target,
    ///                         clears AUTO_BIT (becomes fixed-stops mode).
    ///   Bit 1 (`flags & 2`) → `forced_auto_exposure_brightness` overrides
    ///                         the auto-exposure screen-brightness target,
    ///                         sets AUTO_BIT.
    ///   Bit 2 (`flags & 4`) → `exposure_min` / `exposure_max` override the
    ///                         clamp range.
    ///   Bit 3 (`flags & 8`) → `inherent_bloom` + `bloom_intensity` override
    ///                         the bloom params.
    pub flags: u8,
    /// Engine offset 0x18 — read on `flags & 1`.
    pub forced_exposure: f32,
    /// Engine offset 0x1C — read on `flags & 2`. Distinct from
    /// `forced_exposure`; both are present in the 48B palette entry as
    /// two separate floats. The previous protomorph version reused
    /// `forced_exposure` for both paths — fixed 2026-05-20.
    pub forced_auto_exposure_brightness: f32,
    pub exposure_min: f32,
    pub exposure_max: f32,
    pub inherent_bloom: f32,
    pub bloom_intensity: f32,
}

// ---------------------------------------------------------------------------
// L3: default-script-settings constructor.
// ---------------------------------------------------------------------------

/// Reproduce `c_camera_fx_settings::set_defaults(this, /*use_default=*/1)
/// @ 0x1806871A0` for the per-window `m_camera_fx_settings` storage. Every
/// parameter is created with `USE_DEFAULT_BIT` set so the chooser walks past
/// it. Scripted-exposure overlays (out of L1-L6 scope) would later mutate
/// this struct in-place; without scripts active it stays at this state.
pub fn defaulted_script_settings() -> blam_tags::camera_fx_settings::CameraFxSettings {
    use blam_tags::camera_fx_settings::*;
    let scalar_default = ScalarParameter {
        flags: ParameterFlags::USE_DEFAULT_BIT,
        ..Default::default()
    };
    let instant_default = InstantScalarParameter {
        flags: ParameterFlags::USE_DEFAULT_BIT,
        ..Default::default()
    };
    let color_default = ColorParameter {
        flags: ParameterFlags::USE_DEFAULT_BIT,
        ..Default::default()
    };
    let word_default = WordParameter {
        flags: ParameterFlags::USE_DEFAULT_BIT,
        ..Default::default()
    };

    let mut s = CameraFxSettings::default();
    // Engine `set_defaults` writes flags = `use_default | 6` for m_exposure
    // (sets BLEND_LIMIT_RELATIVE + AUTO bits) and `use_default` (=1) for
    // every other parameter.
    s.exposure.flags = ParameterFlags::USE_DEFAULT_BIT
        | ParameterFlags::BLEND_LIMIT_RELATIVE_BIT
        | ParameterFlags::AUTO_BIT;
    s.exposure.blend_speed = 0.05;
    s.exposure.maximum_change = 0.05;
    s.auto_exposure_sensitivity = instant_default;
    s.auto_exposure_anti_bloom = instant_default;
    s.bloom_point = scalar_default;
    s.bloom_inherent = scalar_default;
    s.bloom_intensity = scalar_default;
    s.bloom_large_color = color_default;
    s.bloom_medium_color = color_default;
    s.bloom_small_color = color_default;
    s.bling_intensity = scalar_default;
    s.bling_size = scalar_default;
    s.bling_angle_deg = scalar_default;
    s.bling_count = word_default;
    s.self_illum_preferred = scalar_default;
    s.self_illum_scale = scalar_default;
    // ssao / lightshafts / color_grading: blam-tags walker shape is
    // `Option<...>`. Engine `set_defaults` populates these inline with
    // use_default=1 + tuned default values, but for the chooser their
    // semantics are equivalent to None (skipped). Leave as None.
    s
}

impl CameraFxValues {
    /// Master per-frame parameter blender — port of
    /// `c_camera_fx_values::update @ 0x180687CB0` (verbatim within L1-L6
    /// scope; ssao/color_grading/lightshafts memcpy + g_fColorGradingBlendFactor
    /// advance deferred — they currently flow through the existing
    /// `screen_postprocess.camera_fx` snapshot path).
    ///
    /// Engine signature:
    /// ```c
    /// c_camera_fx_settings *update(
    ///     c_camera_fx_values *this,
    ///     c_camera_fx_settings *script_settings,
    ///     const structure_camera_fx_palette_entry *cluster_palette_entry,
    ///     c_camera_fx_settings *scenario_settings,
    ///     c_camera_fx_settings *default_settings,
    ///     float exposure_boost);
    /// ```
    /// Returns the chosen `s_real_exposure_parameter*` (cached as
    /// `g_exposure_params` for debug HUD). The chosen exposure source is
    /// also the "did we walk past script?" gate for cluster_palette
    /// override fields on bloom_inherent/bloom_intensity.
    ///
    /// Protomorph signature splits engine's single `cluster_palette_entry`
    /// pointer into its two derived inputs:
    /// - `cluster_settings` — engine's `v10` (resolved cfxs from the
    ///   palette entry's tag ref). Independently None in v1 (L3a).
    /// - `cluster_palette_overrides` — engine's `cluster_palette_entry`
    ///   (flags + forced_*). Also None in v1.
    ///
    /// `actual_brightness` is the protomorph-side substitute for the
    /// engine's synchronous `lock_read` of the auto-exposure GPU sample
    /// (Phase D async-readback workaround, see `Exposure::calculate_new_value`).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        script_settings: &blam_tags::camera_fx_settings::CameraFxSettings,
        cluster_settings: Option<&blam_tags::camera_fx_settings::CameraFxSettings>,
        cluster_palette_overrides: Option<&ClusterCameraFxPaletteOverrides>,
        scenario_settings: Option<&blam_tags::camera_fx_settings::CameraFxSettings>,
        default_settings: Option<&blam_tags::camera_fx_settings::CameraFxSettings>,
        exposure_boost: f32,
        actual_brightness: Option<f32>,
        auto_exposure_lock: bool,
        scripted: &ScriptedExposure,
    ) {
        // Engine line 46: `this->m_exposure_boost = exposure_boost;`
        self.exposure_boost = exposure_boost;

        // === EXPOSURE (engine lines 63-109) ===
        let chosen_exp = choose_valid(
            &script_settings.exposure,
            cluster_settings.map(|c| &c.exposure),
            scenario_settings.map(|s| &s.exposure),
            default_settings.map(|d| &d.exposure),
        );
        let chose_other_than_script = !std::ptr::eq(
            chosen_exp as *const _,
            &script_settings.exposure as *const _,
        );

        let mut flags = chosen_exp.flags;
        let mut target = chosen_exp.exposure;
        let mut auto_target = chosen_exp.auto_exposure_screen_brightness;
        let mut min = chosen_exp.minimum;
        let mut max = chosen_exp.maximum;

        // Cluster_palette override fields (engine lines 83-101). Gated on
        // `cluster_palette && v13 != script_settings`.
        if let Some(o) = cluster_palette_overrides {
            if chose_other_than_script {
                if (o.flags & 1) != 0 {
                    // Engine: `m_flags = m_flags & 0xFB` (clear AUTO_BIT = bit 2).
                    flags &= !ParameterFlags::AUTO_BIT;
                    target = o.forced_exposure;
                }
                if (o.flags & 2) != 0 {
                    // Engine: `m_flags = m_flags | 4` (set AUTO_BIT).
                    // Reads the SEPARATE float at offset 0x1C, not the
                    // forced_exposure at 0x18 (engine struct has both).
                    flags |= ParameterFlags::AUTO_BIT;
                    auto_target = o.forced_auto_exposure_brightness;
                }
                if (o.flags & 4) != 0 {
                    min = o.exposure_min;
                    max = o.exposure_max;
                }
            }
        }

        update_real_exposure(
            chosen_exp, &mut self.exposure, flags, target, auto_target, min, max,
            actual_brightness, auto_exposure_lock, scripted,
        );

        // === auto_exposure_sensitivity (engine lines 110-123) ===
        let chosen = choose_valid(
            &script_settings.auto_exposure_sensitivity,
            cluster_settings.map(|c| &c.auto_exposure_sensitivity),
            scenario_settings.map(|s| &s.auto_exposure_sensitivity),
            default_settings.map(|d| &d.auto_exposure_sensitivity),
        );
        update_instant(chosen, &mut self.auto_exposure_sensitivity);

        // === exposure_anti_bloom (engine lines 124-137) ===
        let chosen = choose_valid(
            &script_settings.auto_exposure_anti_bloom,
            cluster_settings.map(|c| &c.auto_exposure_anti_bloom),
            scenario_settings.map(|s| &s.auto_exposure_anti_bloom),
            default_settings.map(|d| &d.auto_exposure_anti_bloom),
        );
        update_instant(chosen, &mut self.exposure_anti_bloom);

        // === bloom_point (engine lines 138-154) ===
        let chosen = choose_valid(
            &script_settings.bloom_point,
            cluster_settings.map(|c| &c.bloom_point),
            scenario_settings.map(|s| &s.bloom_point),
            default_settings.map(|d| &d.bloom_point),
        );
        update_real(chosen, &mut self.bloom_point);

        // === bloom_inherent (engine lines 155-174) — has cluster_palette flag-8 override ===
        let chosen = choose_valid(
            &script_settings.bloom_inherent,
            cluster_settings.map(|c| &c.bloom_inherent),
            scenario_settings.map(|s| &s.bloom_inherent),
            default_settings.map(|d| &d.bloom_inherent),
        );
        let mut tgt = chosen.value;
        if let Some(o) = cluster_palette_overrides {
            if chose_other_than_script && (o.flags & 8) != 0 {
                tgt = o.inherent_bloom;
            }
        }
        update_real_with_override(chosen, &mut self.bloom_inherent, tgt);

        // === bloom_intensity (engine lines 175-194) — has cluster_palette flag-8 override ===
        let chosen = choose_valid(
            &script_settings.bloom_intensity,
            cluster_settings.map(|c| &c.bloom_intensity),
            scenario_settings.map(|s| &s.bloom_intensity),
            default_settings.map(|d| &d.bloom_intensity),
        );
        let mut tgt = chosen.value;
        if let Some(o) = cluster_palette_overrides {
            if chose_other_than_script && (o.flags & 8) != 0 {
                tgt = o.bloom_intensity;
            }
        }
        update_real_with_override(chosen, &mut self.bloom_intensity, tgt);

        // === bling_intensity (engine lines 195-211) ===
        let chosen = choose_valid(
            &script_settings.bling_intensity,
            cluster_settings.map(|c| &c.bling_intensity),
            scenario_settings.map(|s| &s.bling_intensity),
            default_settings.map(|d| &d.bling_intensity),
        );
        update_real(chosen, &mut self.bling_intensity);

        // === bling_size (engine lines 212-228) ===
        let chosen = choose_valid(
            &script_settings.bling_size,
            cluster_settings.map(|c| &c.bling_size),
            scenario_settings.map(|s| &s.bling_size),
            default_settings.map(|d| &d.bling_size),
        );
        update_real(chosen, &mut self.bling_size);

        // === bling_angle (engine lines 229-245) ===
        let chosen = choose_valid(
            &script_settings.bling_angle_deg,
            cluster_settings.map(|c| &c.bling_angle_deg),
            scenario_settings.map(|s| &s.bling_angle_deg),
            default_settings.map(|d| &d.bling_angle_deg),
        );
        update_real(chosen, &mut self.bling_angle);

        // === bling_count (engine lines 246-259) ===
        let chosen = choose_valid(
            &script_settings.bling_count,
            cluster_settings.map(|c| &c.bling_count),
            scenario_settings.map(|s| &s.bling_count),
            default_settings.map(|d| &d.bling_count),
        );
        update_word(chosen, &mut self.bling_count);

        // === bloom_large/medium/small_color (engine lines 260-316) ===
        let chosen = choose_valid(
            &script_settings.bloom_large_color,
            cluster_settings.map(|c| &c.bloom_large_color),
            scenario_settings.map(|s| &s.bloom_large_color),
            default_settings.map(|d| &d.bloom_large_color),
        );
        update_color(chosen, &mut self.bloom_large_color);

        let chosen = choose_valid(
            &script_settings.bloom_medium_color,
            cluster_settings.map(|c| &c.bloom_medium_color),
            scenario_settings.map(|s| &s.bloom_medium_color),
            default_settings.map(|d| &d.bloom_medium_color),
        );
        update_color(chosen, &mut self.bloom_medium_color);

        let chosen = choose_valid(
            &script_settings.bloom_small_color,
            cluster_settings.map(|c| &c.bloom_small_color),
            scenario_settings.map(|s| &s.bloom_small_color),
            default_settings.map(|d| &d.bloom_small_color),
        );
        update_color(chosen, &mut self.bloom_small_color);

        // === self_illum_preferred (engine lines 317-333) ===
        let chosen = choose_valid(
            &script_settings.self_illum_preferred,
            cluster_settings.map(|c| &c.self_illum_preferred),
            scenario_settings.map(|s| &s.self_illum_preferred),
            default_settings.map(|d| &d.self_illum_preferred),
        );
        update_real(chosen, &mut self.self_illum_preferred);

        // === self_illum_scale (engine lines 334-351) ===
        let chosen = choose_valid(
            &script_settings.self_illum_scale,
            cluster_settings.map(|c| &c.self_illum_scale),
            scenario_settings.map(|s| &s.self_illum_scale),
            default_settings.map(|d| &d.self_illum_scale),
        );
        update_real(chosen, &mut self.self_illum_scale);
        let v36 = self.self_illum_scale;

        // === m_self_illum_exposure (engine lines 352-360) ===
        // ```c
        // self_illum_exposure = (((1 - exposure_blend) * m_exposure
        //                       + exposure_blend * exposure_stops)
        //                       + exposure_bias
        //                       - self_illum_preferred) * self_illum_scale
        //                       + self_illum_preferred
        // ```
        // Scripted-exposure globals are out of scope (default 0). With
        // exposure_blend = 0, exposure_bias = 0:
        //   self_illum_exposure = (m_exposure - self_illum_preferred) * self_illum_scale
        //                       + self_illum_preferred
        let m_exposure = self.exposure.exposure;
        let scripted_blended = m_exposure; // (1-0)*m_exposure + 0*0 = m_exposure
        let with_bias = scripted_blended; // + 0 = m_exposure
        self.self_illum_exposure =
            (with_bias - self.self_illum_preferred) * v36 + self.self_illum_preferred;

        // === ssao / color_grading / lightshafts (engine lines 361-433) ===
        // DEFERRED: these copy/memcpy paths still flow through the existing
        // `screen_postprocess.camera_fx` snapshot path (CameraFxBloomDefaults
        // handles ssao/lightshafts; update_color_grading reads its own
        // snapshot). L4-L6 retire those snapshots when these consumers move
        // to per-frame reads from `CameraFxValues`.
        //
        // `g_fColorGradingBlendFactor` advance (engine lines 400-416) is
        // already handled per-frame inside `screen_postprocess`.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::camera_fx_settings::{
        ColorParameter as TagColor, InstantScalarParameter, ScalarParameter, WordParameter as TagWord,
    };
    use blam_tags::math::RealRgbColor;

    fn rp(value: f32, max_change: f32, blend_speed: f32, flags: u16) -> ScalarParameter {
        ScalarParameter { flags, value, max_change, blend_speed }
    }

    #[test]
    fn calc_new_value_unclamped_when_blend_limit_negative() {
        // m_blend_limit < 0 → ignore the cap, plain lerp.
        let p = rp(10.0, -1.0, 0.5, 0);
        let v = calc_new_value(&p, 0.0, 10.0);
        assert!((v - 5.0).abs() < 1e-6);
    }

    #[test]
    fn calc_new_value_clamps_overstep() {
        // blend_speed=1.0 → new = target = 10.0; |delta| = 10 > limit 2.
        // Engine clamps to old + limit = 0 + 2 = 2.0.
        let p = rp(10.0, 2.0, 1.0, 0);
        let v = calc_new_value(&p, 0.0, 10.0);
        assert!((v - 2.0).abs() < 1e-6);
    }

    #[test]
    fn calc_new_value_clamps_negative_overstep() {
        // Going from 5.0 toward -10.0 with limit 1.5 → clamp to 5 - 1.5 = 3.5.
        let p = rp(-10.0, 1.5, 1.0, 0);
        let v = calc_new_value(&p, 5.0, -10.0);
        assert!((v - 3.5).abs() < 1e-6);
    }

    #[test]
    fn calc_new_value_relative_blend_limit_scales_by_old_abs() {
        // BLEND_LIMIT_RELATIVE_BIT (=2) → limit *= |old|.
        // old=4, blend_limit_raw=0.25 → limit = 1.0.
        let p = rp(20.0, 0.25, 1.0, ParameterFlags::BLEND_LIMIT_RELATIVE_BIT);
        let v = calc_new_value(&p, 4.0, 20.0);
        assert!((v - 5.0).abs() < 1e-6); // 4 + 1.0
    }

    #[test]
    fn calc_new_value_small_step_returns_lerp() {
        // |delta| <= limit → no clamping, return the lerp.
        let p = rp(10.0, 100.0, 0.5, 0);
        let v = calc_new_value(&p, 0.0, 10.0);
        assert!((v - 5.0).abs() < 1e-6);
    }

    #[test]
    fn update_real_uses_param_target() {
        let p = rp(8.0, -1.0, 0.25, 0);
        let mut v = 0.0;
        update_real(&p, &mut v);
        // (1 - 0.25) * 0 + 0.25 * 8 = 2.0
        assert!((v - 2.0).abs() < 1e-6);
    }

    #[test]
    fn update_real_with_override_ignores_param_target() {
        let p = rp(99.0, -1.0, 0.5, 0);
        let mut v = 0.0;
        update_real_with_override(&p, &mut v, 4.0);
        // (1 - 0.5) * 0 + 0.5 * 4 = 2.0  (would be 49.5 if param.value were used)
        assert!((v - 2.0).abs() < 1e-6);
    }

    #[test]
    fn update_instant_snaps() {
        let p = InstantScalarParameter { flags: 0, value: 7.0 };
        let mut v = 1.5;
        update_instant(&p, &mut v);
        assert_eq!(v, 7.0);
    }

    #[test]
    fn update_word_snaps() {
        let p = TagWord { flags: 0, value: 4 };
        let mut v: u16 = 0;
        update_word(&p, &mut v);
        assert_eq!(v, 4);
    }

    #[test]
    fn update_color_5_95_lerp() {
        let p = TagColor {
            flags: 0,
            color: RealRgbColor { red: 1.0, green: 0.0, blue: 0.5 },
        };
        let mut v = Vec3::new(0.0, 0.4, 0.0);
        update_color(&p, &mut v);
        // x: 0.05*1 + 0.95*0 = 0.05
        // y: 0.05*0 + 0.95*0.4 = 0.38
        // z: 0.05*0.5 + 0.95*0 = 0.025
        assert!((v.x - 0.05).abs() < 1e-6);
        assert!((v.y - 0.38).abs() < 1e-6);
        assert!((v.z - 0.025).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// c_camera_fx_values (536B) — what the postprocess shader reads.
// ---------------------------------------------------------------------------

/// `c_camera_fx_values` (camera_fx_settings.h:296-335, 536B).
///
/// Runtime-resolved postprocess values. Each frame `update()` blends
/// toward the active `c_camera_fx_settings` (which itself comes from
/// scripted-exposure ∪ cluster-palette ∪ scenario ∪ defaults). The
/// resolved scalars/colors here are the ones the shader cbuffer
/// contains.
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraFxValues {
    pub exposure: Exposure,                        // 0x0   (304B)
    pub exposure_boost: f32,                       // 0x130
    pub auto_exposure_sensitivity: f32,            // 0x134
    pub exposure_anti_bloom: f32,                  // 0x138
    pub bloom_point: f32,                          // 0x13C
    pub bloom_inherent: f32,                       // 0x140
    pub bloom_intensity: f32,                      // 0x144
    pub bling_intensity: f32,                      // 0x148
    pub bling_size: f32,                           // 0x14C
    pub bling_angle: f32,                          // 0x150
    pub bling_count: u16,                          // 0x154
    pub _pad: [u8; 2],                             // padding to align color
    pub bloom_large_color: Vec3,                   // 0x158
    pub bloom_medium_color: Vec3,                  // 0x164
    pub bloom_small_color: Vec3,                   // 0x170
    pub self_illum_preferred: f32,                 // 0x17C
    pub self_illum_scale: f32,                     // 0x180
    pub self_illum_exposure: f32,                  // 0x184
    // Engine `c_camera_fx_values` also carries `ssao` (16B), `color_grading`
    // (80B), and `lightshafts` (44B) at offsets 0x188 / 0x198 / 0x1E8.
    // Omitted from protomorph — those parameters flow through dedicated
    // snapshots / cbuffers (see `screen_postprocess.rs` ssao/lightshafts
    // upload paths and `color_grading.rs`); nothing reads the
    // c_camera_fx_values copy. Re-add per-block when an engine-faithful
    // consumer needs the runtime-blended values directly.
}

impl CameraFxValues {
    /// `c_camera_fx_values::get_render_exposure @ 0x18068E3E0` — verbatim port.
    /// Returns the linear post-tone-curve exposure multiplier the
    /// postprocess shader reads via `g_exposure.r`.
    ///
    /// ```c
    /// v1 = render_apply_scripted_exposure(scripted_exposure_game_globals,
    ///                                     m_exposure.m_exposure)
    ///    + c_camera_fx_values::g_exposure_stops
    ///    + m_exposure_boost;
    /// return pow(2, v1)
    ///      * c_screen_postprocess::x_settings_internal.m_tone_curve_white_point
    ///      * 0.66943294;
    /// ```
    ///
    /// Engine reads three globals — `scripted_exposure_game_globals`
    /// (52B singleton), `c_camera_fx_values::g_exposure_stops`
    /// (class-level static mutated by `g_render_set_exposure` HSC), and
    /// `tone_curve_white_point` (from postprocess x_settings_internal).
    /// Protomorph passes them as parameters so the caller threads in
    /// the right per-frame snapshot.
    pub fn get_render_exposure(
        &self,
        scripted: &ScriptedExposure,
        g_exposure_stops: f32,
        tone_curve_white_point: f32,
    ) -> f32 {
        let v1 = scripted.render_apply_scripted_exposure(self.exposure.exposure)
            + g_exposure_stops
            + self.exposure_boost;
        v1.exp2() * tone_curve_white_point * 0.66943294
    }

    /// `c_camera_fx_values::get_display_exposure_stops @ 0x1806887C0` —
    /// returns the exposure stops directly (post-blend, pre-pow2). Used
    /// by debug HUD readouts. Engine reads `m_exposure.m_exposure`.
    pub fn get_display_exposure_stops(&self) -> f32 {
        self.exposure.exposure
    }

    /// `c_camera_fx_values::get_illum_render_scale @ 0x18068E390` —
    /// verbatim port. Returns the self-illumination exposure scale the
    /// shader reads via `g_alt_exposure.r`.
    ///
    /// ```c
    /// v1 = m_self_illum_exposure
    ///    - render_apply_scripted_exposure(scripted_exposure_game_globals,
    ///                                     m_exposure.m_exposure);
    /// return pow(2, v1);
    /// ```
    ///
    /// The `m_self_illum_exposure` field is computed inside
    /// `c_camera_fx_values::update` (L3) as
    /// `(m_exposure - self_illum_preferred) * self_illum_scale + self_illum_preferred`.
    /// Protomorph threads `scripted` from renderer-level state so HSC
    /// exposure overrides reach this scalar correctly.
    pub fn get_illum_render_scale(&self, scripted: &ScriptedExposure) -> f32 {
        let v1 = self.self_illum_exposure
            - scripted.render_apply_scripted_exposure(self.exposure.exposure);
        v1.exp2()
    }

    /// Engine-symmetric init mirroring `c_camera_fx_values::set @ 0x180682B40`
    /// (the standalone variant; bytes are identical to the inlined per-window
    /// snap inside `s_render_game_state::initialize_for_new_map @ 0x180682790`).
    ///
    /// ```c
    /// c_exposure::set_exposure(&this->m_exposure, settings->m_exposure.m_target);
    /// c_exposure::reset_render_target_queue(&this->m_exposure);
    /// this->m_auto_exposure_sensitivity = settings->m_auto_exposure_sensitivity.m_target;
    /// this->m_exposure_anti_bloom       = settings->m_exposure_anti_bloom.m_target;
    /// this->m_bloom_point     = settings->m_bloom_point.m_target;
    /// this->m_bloom_inherent  = settings->m_bloom_inherent.m_target;
    /// this->m_bloom_intensity = settings->m_bloom_intensity.m_target;
    /// this->m_bloom_large_color  = settings->m_bloom_large_color.m_color;
    /// this->m_bloom_medium_color = settings->m_bloom_medium_color.m_color;
    /// this->m_bloom_small_color  = settings->m_bloom_small_color.m_color;
    /// this->m_bling_intensity = settings->m_bling_intensity.m_target;
    /// this->m_bling_angle     = settings->m_bling_angle.m_target;
    /// this->m_bling_count     = settings->m_bling_count.m_target;
    /// ```
    ///
    /// Engine deliberately does NOT snap `m_bling_size`, `m_self_illum_*`,
    /// `m_ssao`, `m_color_grading`, or `m_lightshafts` — those are filled
    /// per-frame by `c_camera_fx_values::update`. Field defaults stand until then.
    ///
    /// Takes the blam-tags walker shape directly: `.value` is the engine
    /// `m_target` of `s_real_parameter`, `.effective_color()` is `m_color`
    /// (with the use_default bit honored).
    pub fn default_init_from(
        &mut self,
        settings: &blam_tags::camera_fx_settings::CameraFxSettings,
    ) {
        self.exposure.set_exposure(settings.exposure.exposure);
        self.exposure.reset_render_target_queue();
        self.auto_exposure_sensitivity = settings.auto_exposure_sensitivity.value;
        self.exposure_anti_bloom = settings.auto_exposure_anti_bloom.value;
        self.bloom_point = settings.bloom_point.value;
        self.bloom_inherent = settings.bloom_inherent.value;
        self.bloom_intensity = settings.bloom_intensity.value;
        // Engine `c_camera_fx_values::set` does a direct struct copy of
        // `m_color` (no `use_default → (1,1,1)` substitution). For most
        // cfxs the raw color happens to match the would-be substitute,
        // but if it doesn't, engine renders with raw and so do we.
        let large = settings.bloom_large_color.color;
        self.bloom_large_color = Vec3::new(large.red, large.green, large.blue);
        let medium = settings.bloom_medium_color.color;
        self.bloom_medium_color = Vec3::new(medium.red, medium.green, medium.blue);
        let small = settings.bloom_small_color.color;
        self.bloom_small_color = Vec3::new(small.red, small.green, small.blue);
        self.bling_intensity = settings.bling_intensity.value;
        self.bling_angle = settings.bling_angle_deg.value;
        self.bling_count = settings.bling_count.value;
    }
}
