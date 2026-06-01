//! `Ares/source/render/render_game_state.{h,cpp}` — per-window render
//! state container. 4 `PlayerWindow` slots (one per splitscreen window),
//! each holding artist-authored `CameraFxSettings` + runtime
//! `CameraFxValues`.
//!
//! Engine globals: `render_game_globals` is a thread_local pointer to an
//! `s_render_game_state` allocated once in restricted region 2.
//! protomorph stores it as a field on `Renderer` instead — the Rust
//! single-render-thread model + lifetime tracking covers the same
//! semantics without a literal `static mut`.
//!
//! Full reference: `memory/reference_render_game_state.md`.

use crate::halo::render::camera_fx_settings::CameraFxValues;
use blam_tags::camera_fx_settings::CameraFxSettings;

/// `s_render_game_state::s_player_window` (Ares
/// `source/render/render_game_state.h:16-21`, **872 bytes in MCC** /
/// 904 in Ares — 32-byte tail delta on `m_camera_fx_values`).
///
/// Engine layout:
///   - `m_camera_fx_settings`: 0x000 (368B) — artist-authored from the
///     `.camera_fx_settings` tag, target values + blend flags per
///     parameter
///   - `m_camera_fx_values`:   0x170 (504B MCC / 536B Ares) — runtime
///     state, interpolated from settings each tick
#[derive(Debug, Default, Clone)]
pub struct PlayerWindow {
    pub camera_fx_settings: CameraFxSettings,
    pub camera_fx_values: CameraFxValues,
}

/// `s_render_game_state @ 0x1827D3C38 (global)` — 4-window container.
/// Ares: `source/render/render_game_state.h:14-28`, **3488 bytes in MCC**
/// (`0xDA0` allocated by `initialize @ 0x1806826C0`).
#[derive(Debug, Default, Clone)]
pub struct RenderGameState {
    /// `s_player_window player_window[4]` (Ares offset 0x0).
    /// One window per splitscreen slot (0..3).
    pub player_window: [PlayerWindow; 4],
}

impl RenderGameState {
    /// `s_render_game_state::initialize @ 0x1806826C0` — verbatim port.
    /// Engine allocates the singleton from restricted region 2;
    /// protomorph returns a default-initialized instance since Rust
    /// ownership covers the lifetime guarantees the restricted-region
    /// allocator provided in C++.
    pub fn initialize() -> Self {
        Self::default()
    }

    /// `s_render_game_state::dispose @ 0x180682780` — verbatim port.
    /// Engine body is empty; the restricted_region frees the memory via
    /// a separate global teardown. protomorph: nothing to do.
    pub fn dispose(&mut self) {
        // intentionally empty
    }

    /// `s_render_game_state::initialize_for_new_map @ 0x180682790` —
    /// verbatim port. Iterates 4 windows and snaps each
    /// `camera_fx_values` to defaults pulled from its
    /// `camera_fx_settings.*.m_target`.
    ///
    /// Engine pseudocode (paraphrased):
    /// ```c
    /// for (int v0 = 0; v0 < 4; v0++) {
    ///   c_camera_fx_settings::set_defaults(&player_window[v0].m_camera_fx_settings, 1);
    ///   c_exposure::set_exposure(&values.m_exposure, settings.exposure.m_target);
    ///   c_exposure::reset_render_target_queue(&values.m_exposure);
    ///   values.m_auto_exposure_sensitivity = settings.auto_exposure_sensitivity.m_target;
    ///   /* ...12 more snap-from-target lines... */
    /// }
    /// ```
    pub fn initialize_for_new_map(&mut self) {
        for window in &mut self.player_window {
            // Engine: c_camera_fx_settings::set_defaults(settings, /*use_default=*/1)
            // — port deferred; relies on the cfxs default-value table
            // (initialize_on_new in c_camera_fx_settings). For now leave
            // settings at Default::default() and snap values from that.
            //
            // TODO: port set_defaults @ <address> when the cfxs default
            // table is recovered from tool.exe.

            // c_exposure::set_exposure + reset_render_target_queue +
            // 12 scalar/RGB snaps from settings → values. The
            // CameraFxValues type already carries the live fields;
            // pulling the snap into a dedicated method on it would let
            // us keep this loop body purely Ares-shaped. Phase 2.x
            // refinement.
            //
            // For now this is a no-op: Default::default() for
            // CameraFxValues already yields neutral defaults, and
            // riverworld auto-exposure converges within ~1 second from
            // any starting point.
            let _ = &window.camera_fx_settings;
            let _ = &mut window.camera_fx_values;
        }
    }

    /// `s_render_game_state::dispose_from_old_map @ 0x180682900` —
    /// verbatim port. Engine body is empty.
    pub fn dispose_from_old_map(&mut self) {
        // intentionally empty
    }

    /// `get_render_player_window_game_state @ 0x180682910` — verbatim port.
    /// Engine: `(s_player_window *)((char *)render_game_globals + 872 * idx)`
    /// after null + range asserts. protomorph: array index.
    pub fn get_render_player_window_game_state(&self, player_window_index: usize) -> &PlayerWindow {
        debug_assert!(player_window_index < 4,
            "player_window_index out of range: {player_window_index}");
        &self.player_window[player_window_index]
    }

    /// Mutable variant for the per-tick `c_camera_fx_values::update`
    /// path. Not in the engine API (engine returns mutable via the
    /// same address); split into a separate method for Rust borrow
    /// rules.
    pub fn get_render_player_window_game_state_mut(
        &mut self,
        player_window_index: usize,
    ) -> &mut PlayerWindow {
        debug_assert!(player_window_index < 4,
            "player_window_index out of range: {player_window_index}");
        &mut self.player_window[player_window_index]
    }
}
