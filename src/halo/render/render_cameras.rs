//! Mirror of `Ares/source/render/render_cameras.h`.
//!
//! Halo separates the per-view camera state from the per-frame
//! viewport / projection math.

use glam::Vec2;

#[derive(Debug, Clone, Copy, Default)]
pub struct RealRectangle2d {
    pub min: Vec2,
    pub max: Vec2,
}
