//! Mirror of `Ares/source/cubemaps/cubemap_sample.h` (45 lines).
//!
//! Cubemap sampling state. Each cluster owns a fixed-index cubemap
//! (the env probe) and the runtime walks the camera position to find
//! the nearest one for environment mapping. `c_dynamic_cubemap_sample`
//! keeps a current + last cubemap reference and blends between them
//! when crossing cluster boundaries — a soft transition instead of a
//! hard pop when the camera moves into a new cluster.
//!
//! The render path reads `m_current`/`m_last`/`m_blend_factor` and
//! sets the env-probe cbuffer slot accordingly. v1 protomorph ships
//! a single env probe (no cluster-driven dynamics) — this struct
//! lays the canonical shape down for when cluster-aware probes land.

use crate::halo::structures::clusters::ClusterReference;
use glam::Vec3;

/// `s_cubemap_sample` (cubemap_sample.h:14-20, 6B).
#[derive(Debug, Clone, Copy, Default)]
pub struct CubemapSample {
    pub cluster_reference: ClusterReference, // 0x0  (4B in our widened ClusterReference)
    pub cluster_cubemap_index: i16,          // 0x2 (Halo offset; layout-tracked in comment)
    pub bitmap_index: i16,                   // 0x4
}

/// `c_dynamic_cubemap_sample` (cubemap_sample.h:22-34, 16B).
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamicCubemapSample {
    pub current: CubemapSample, // 0x0
    pub last: CubemapSample,    // 0x6
    pub blend_factor: f32,      // 0xC
}

impl DynamicCubemapSample {
    /// `c_dynamic_cubemap_sample::set_defaults` (cubemap_sample.h:32).
    pub fn set_defaults(&mut self) {
        *self = Self::default();
    }

    /// `c_dynamic_cubemap_sample::update_cubemap_sample
    /// @ cubemap_sample.h:31`. Drives `blend_factor` toward 1.0 at
    /// `interpolation_speed` per call. When it hits 1.0 the `last`
    /// reference can be discarded (current fully replaces it).
    pub fn update_cubemap_sample(&mut self, interpolation_speed: f32) {
        if self.blend_factor < 1.0 {
            self.blend_factor = (self.blend_factor + interpolation_speed).min(1.0);
        }
    }

    /// `c_dynamic_cubemap_sample::search_for_cubemap_sample
    /// @ cubemap_sample.h:30`. Walks visible clusters by distance to
    /// `current_position` and replaces `m_current` if it finds one
    /// at least `minimum_percent_closer` better than the existing
    /// pick.
    ///
    /// v1 stub — returns false (no replacement) until cluster
    /// occupancy is tracked. The walker body lands when the BSP
    /// cluster-cubemap index lookup is wired (Phase 11+).
    pub fn search_for_cubemap_sample(
        &mut self,
        _current_position: Vec3,
        _minimum_percent_closer: f32,
    ) -> bool {
        false
    }
}

/// `c_dynamic_cubemap_sample::get_cluster_cubemap_index
/// @ cubemap_sample.h:28`. Halo: looks up the cluster's cubemap
/// index from `g_render_structure_globals`. Stub returns -1 until
/// the per-cluster cubemap index list lands.
pub fn get_cluster_cubemap_index(_cluster_reference: ClusterReference) -> i32 {
    -1
}
