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
}
