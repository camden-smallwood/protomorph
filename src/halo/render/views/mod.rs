//! Subset of `Ares/source/render/views/` actually wired through
//! protomorph. Most of the View hierarchy (`c_first_person_view`,
//! `c_lightmap_shadows_view`, `c_occlusion_view`,
//! `c_reflection_view`, `c_texture_camera_view`, `c_ui_view`,
//! `c_fullscreen_view`, `c_world_view`, abstract `c_view`) was
//! ported as zero-call-site stub modules and removed in the
//! 2026-05-09 audit. `c_player_view` + `c_lights_view` are the
//! only views with live render paths today.

pub mod lights_view;
pub mod render_player_view;

pub use lights_view::LightsView;
pub use render_player_view::PlayerView;
