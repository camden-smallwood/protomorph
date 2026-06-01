//! Mirror of `Ares/source/scenario/scenario_lightmap_definitions.h`
//! (133 lines).
//!
//! Halo's "scenario_lightmap" tag is the per-scenario lightmap data
//! authored by Bungie's lightmapper. The render path reads:
//!
//!   - `s_scenario_lightmap` (88B) — top-level. Owns per-bsp data,
//!     object per-vertex blocks, lightmap_objects, airprobes,
//!     scenery_probes, device_probes.
//!   - `s_scenario_lightmap_bsp_data` (436B) — per-BSP. Contains
//!     compression vectors, the lightprobe atlas reference, the
//!     dominant-light intensity bitmap, plus the embedded
//!     `s_render_geometry imported_geometry` (the chart-split
//!     vertex stream we render BSPs from per
//!     `feedback_lightmap_geometry_source.md`).
//!   - per-instance/cluster/scenery probe blocks (4-8B each).
//!   - airprobes / scenery_probes / device_probes (72-92B each, all
//!     containing `half_rgb_lightprobe_with_dominant_light`).
//!
//! Loaders that decode the tag into runtime GPU resources stay in
//! `halo/scenario/loader.rs`; this file only carries the type
//! shapes so render code can refer to them by canonical names.

use crate::halo::geometry::lightprobe::HalfRgbLightprobeWithDominantLight;
use crate::halo::geometry::render_geometry::RenderGeometry;
use glam::Vec3;

/// Halo: `SCENARIO_LIGHTMAP_DEFINITION_COMPRESSION_VECTORS = 18`
/// (scenario_lightmap_definitions.h:17).
pub const SCENARIO_LIGHTMAP_COMPRESSION_VECTORS: usize = 18;

/// `c_object_identifier` — opaque 8-byte handle. Halo packs as a
/// (game-state-index, salt) pair. We mirror layout-faithfully.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObjectIdentifier(pub u64);

/// `s_scenario_airprobe_info` (scenario_lightmap_definitions.h:24-30,
/// 20B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioAirprobeInfo {
    pub position: Vec3,
    pub name: i32,
    pub manual_bsp_flags: u16,
    pub _pad: u16,
}

/// `s_scenario_lightmap_pervertex_data` (16B).
#[derive(Debug, Clone, Default)]
pub struct ScenarioLightmapPervertexData {
    pub lightprobe_data: Vec<u8>, // s_tag_block lightprobe_data
    pub lightprobe_runtime_vertex_buffer_index: i32,
}

/// `s_scenario_lightmap_entity_data` (4B). Base for cluster /
/// instance variants — they extend it.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioLightmapEntityData {
    pub lightprobe_texture_array_index: i16,
    pub pervertex_block_index: i16,
}

/// `s_scenario_lightmap_cluster_data` (4B). Same as base.
pub type ScenarioLightmapClusterData = ScenarioLightmapEntityData;

/// `s_scenario_lightmap_instance_data` (8B). Adds a probe_index.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioLightmapInstanceData {
    pub entity: ScenarioLightmapEntityData, // 0x0  (4B)
    pub probe_index: i16,                   // 0x4
    pub _pad: u16,                          // 0x6
}

/// `s_scenario_lightmap_airprobe` (92B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioLightmapAirprobe {
    pub airprobe_info: ScenarioAirprobeInfo,
    pub probe: HalfRgbLightprobeWithDominantLight,
}

/// `s_scenario_lightmap_scenery_probe` (80B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioLightmapSceneryProbe {
    pub object_id: ObjectIdentifier,
    pub probe: HalfRgbLightprobeWithDominantLight,
}

/// `s_scenario_lightmap_device_machine_probe` (84B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioLightmapDeviceMachineProbe {
    pub position: Vec3,
    pub probe: HalfRgbLightprobeWithDominantLight,
}

/// `s_scenario_lightmap_device_machine_probe_data` (44B).
#[derive(Debug, Clone, Default)]
pub struct ScenarioLightmapDeviceMachineProbeData {
    pub object_id: ObjectIdentifier,
    pub bounds: [f32; 6], // real_rectangle3d
    pub probes: Vec<ScenarioLightmapDeviceMachineProbe>,
}

/// `s_scenario_lightmap_bsp_data` (436B).
///
/// Top-level per-BSP lightmap data. Carries:
///   - 18× compression vectors for vertex-position decompression
///   - lightprobe atlas + dominant-light intensity texture references
///   - per-vertex blocks (one per mesh)
///   - per-cluster + per-instance + single-probe blocks
///   - the embedded chart-split `imported_geometry` we render from
#[derive(Debug, Clone, Default)]
pub struct ScenarioLightmapBspData {
    pub flags: u16,                                              // 0x0
    pub bsp_reference_index: i16,                                // 0x2
    pub structure_bsp_import_checksum: i32,                       // 0x4
    pub compression_vectors: [Vec3; SCENARIO_LIGHTMAP_COMPRESSION_VECTORS], // 0x8
    /// Tag reference to the lightprobe-textures bitmap group.
    pub lightprobe_textures_bitmap_group_path: String,           // 0xE0
    /// Tag reference to the dominant-light-intensity bitmap group.
    pub dominant_light_intensity_bitmap_group_path: String,      // 0xF0
    pub bsp_pervertex_blocks: Vec<ScenarioLightmapPervertexData>, // 0x100
    pub clusters: Vec<ScenarioLightmapClusterData>,              // 0x10C
    pub instances: Vec<ScenarioLightmapInstanceData>,            // 0x118
    pub single_probes: Vec<HalfRgbLightprobeWithDominantLight>,  // 0x124
    pub imported_geometry: RenderGeometry,                       // 0x130
}

/// `s_scenario_lightmap` (88B). Top-level scenario lightmap tag
/// struct.
#[derive(Debug, Clone, Default)]
pub struct ScenarioLightmap {
    pub job_guid: i32,
    pub lightmap_bsps: Vec<ScenarioLightmapBspData>,
    pub object_pervertex_blocks: Vec<ScenarioLightmapPervertexData>,
    pub lightmap_objects: Vec<u8>, // opaque tag block — gameplay only
    pub airprobes: Vec<ScenarioLightmapAirprobe>,
    pub scenery_probes: Vec<ScenarioLightmapSceneryProbe>,
    pub device_probes: Vec<ScenarioLightmapDeviceMachineProbeData>,
    pub errors: Vec<u8>,
}
