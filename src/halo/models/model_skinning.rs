//! Mirror of `Ares/source/models/model_skinning.h` (58 lines).
//!
//! Halo's GPU-side skinning matrix layout. The runtime computes
//! per-frame skinning matrices from object node matrices + the
//! render_model's region/permutation selection, packs them into
//! `s_model_skinning`, and uploads to the vertex-shader skinning
//! cbuffer.

use glam::Vec4;

/// `s_model_skinning_matrix` (h:13-19, 48B). 3-row × 4-col affine —
/// matches the GPU upload format.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ModelSkinningMatrix {
    pub x: Vec4,
    pub y: Vec4,
    pub z: Vec4,
}

/// `s_model_skinning_region` (h:21-25, 4B). One per region — points
/// at this region's matrix range in the parent `ModelSkinning`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelSkinningRegion {
    pub first_matrix_index: i16,
    pub matrix_count: u16,
}

/// `s_model_skinning` (h:28-34, 66068B). Per-render-model skinning
/// matrix pool. Halo: fixed 16-region × 1375-matrix array (sized
/// for the largest possible character). v1: keep the layout-faithful
/// fixed-size for accurate uploads.
#[derive(Debug, Clone, Copy)]
pub struct ModelSkinning {
    pub base_node_matrix_count: u16,
    pub region_count: u16,
    pub regions: [ModelSkinningRegion; 16],
    /// Halo's 1375 matrix cap is the max nodes × max regions
    /// covering every shipped character.
    pub matrices: [ModelSkinningMatrix; 1375],
}

impl Default for ModelSkinning {
    fn default() -> Self {
        Self {
            base_node_matrix_count: 0,
            region_count: 0,
            regions: [ModelSkinningRegion::default(); 16],
            matrices: [ModelSkinningMatrix::default(); 1375],
        }
    }
}

/// `render_model_get_skinning_size_in_bytes @ h:40`.
pub fn render_model_get_skinning_size_in_bytes(skinning_matrix_count: i64) -> i64 {
    skinning_matrix_count * std::mem::size_of::<ModelSkinningMatrix>() as i64
}
