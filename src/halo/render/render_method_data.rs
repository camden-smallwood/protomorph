//! `c_render_method_data` + `c_lighting_interface` — Rust mirrors of
//! Halo 3's per-pass render-state globals.
//!
//! These are the static globals that the dllcache extern dispatch
//! reads from. Written by:
//! - `render_method_submit_data` (per draw — `lightmap_*`, `cubemap_*`,
//!   `change_colors`, `context*`)
//! - `setup_default_lighting` + `select_instance_entry_point` (per
//!   pass / per instance — `dominant_light_*`, `sh_probe_*`)
//! - `c_player_view::render_albedo` (per pass — engine cbuffer
//!   zero-inits at slots 0x130000, 0x140000)
//!
//! Read by:
//! - `render_method_submit_externs` — walks the 49-extern table and
//!   populates engine cbuffers from these fields per draw
//!
//! See `reference_h3_extern_table.md` (auto-memory) for the full
//! extern → field mapping.
//!
//! v1 scope: data containers only — no GPU upload yet. Phase D3.8
//! adds the bind groups; Phase D3.9 adds the per-draw extern walker.
//!
//! Source: `Ares/source/render_methods/render_method_submit.h`,
//! `lighting_interface.h`. dllcache `setup_default_lighting`.

use glam::{Vec3, Vec4};

/// Packs the cluster reference Halo uses across structure code:
/// `bsp_index:8 | cluster_index:8` in a u16.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusterRef {
    pub bsp_index: u8,
    pub cluster_index: u8,
}

impl ClusterRef {
    pub const NONE: Self = Self { bsp_index: 0xFF, cluster_index: 0xFF };
    pub fn is_none(self) -> bool {
        self.bsp_index == 0xFF && self.cluster_index == 0xFF
    }
}

/// 4-channel color used for change_colors + emblem palette.
#[derive(Debug, Clone, Copy, Default)]
pub struct Color4 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color4 {
    pub const ZERO: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
}

/// Mirrors `c_render_method_data` static globals (per-draw state).
///
/// Field naming verbatim from `Ares/source/render_methods/render_method_submit.h`.
/// Externs read these by name through the routing dispatch.
#[derive(Debug, Clone, Default)]
pub struct CRenderMethodData {
    /// `m_valid` — guards against draws before submit_data is called.
    pub valid: bool,

    /// `m_lightmap_bsp_index` — index into `g_render_structure_globals.lightmap_bsp_data[16]`.
    /// -1 = no lightmap (default-lighting fallback).
    pub lightmap_bsp_index: i16,

    /// `m_lightmap_cluster_reference` — which cluster's lightmap data
    /// the current draw uses. For instance draws this is the cluster
    /// the instance is contained by.
    pub lightmap_cluster_reference: ClusterRef,

    /// `m_lightmap_instance_index` — for instance draws, which entry
    /// in `bsp_lightmap_data->instances[]` to read for per-instance
    /// SH / lightmap UV stream lookup.
    pub lightmap_instance_index: i16,

    /// `m_cubemap_blend_factor` — lerp weight between
    /// `cubemap_dynamic_0` and `cubemap_dynamic_1` for fading clusters.
    pub cubemap_blend_factor: f32,

    /// `m_cubemap_dynamic_0_*` — primary dynamic cubemap.
    pub cubemap_dynamic_0_index: i16,
    pub cubemap_dynamic_0_cluster_reference: ClusterRef,

    /// `m_cubemap_dynamic_1_*` — secondary cubemap (for cross-fade).
    pub cubemap_dynamic_1_index: i16,
    pub cubemap_dynamic_1_cluster_reference: ClusterRef,

    /// `m_change_colors[4]` — primary, secondary, tertiary, quaternary
    /// color palette. BSP draws set all to zero; objects override per
    /// variant. Read by externs 15..18.
    pub change_colors: [Color4; 4],
    // m_context + m_context_interface — emblem callback, deferred
    // (UI-only path; gameplay codes 631..636 are object-side).
}

impl CRenderMethodData {
    /// Reset to per-frame defaults. Mirrors what `c_player_view::render`
    /// does pre-pass — zeroes the user cbuffer bases and invalidates
    /// the per-draw cache.
    pub fn pass_pre_init(&mut self) {
        self.valid = false;
        self.lightmap_bsp_index = -1;
        self.lightmap_cluster_reference = ClusterRef::NONE;
        self.lightmap_instance_index = -1;
        self.cubemap_blend_factor = 0.0;
        self.cubemap_dynamic_0_index = -1;
        self.cubemap_dynamic_0_cluster_reference = ClusterRef::NONE;
        self.cubemap_dynamic_1_index = -1;
        self.cubemap_dynamic_1_cluster_reference = ClusterRef::NONE;
        self.change_colors = [Color4::ZERO; 4];
    }

    /// Mirrors `render_method_submit_data` for cluster draws — sets
    /// the lightmap target cluster and clears object-only fields.
    pub fn submit_cluster(&mut self, bsp_index: u8, cluster_ref: ClusterRef) {
        self.valid = true;
        self.lightmap_bsp_index = bsp_index as i16;
        self.lightmap_cluster_reference = cluster_ref;
        self.lightmap_instance_index = -1;
        self.change_colors = [Color4::ZERO; 4];
    }

    /// Mirrors `render_method_submit_data` for instance draws.
    pub fn submit_instance(
        &mut self,
        bsp_index: u8,
        instance_index: i16,
        instance_cluster_ref: ClusterRef,
    ) {
        self.valid = true;
        self.lightmap_bsp_index = bsp_index as i16;
        self.lightmap_cluster_reference = instance_cluster_ref;
        self.lightmap_instance_index = instance_index;
        self.change_colors = [Color4::ZERO; 4];
    }
}

