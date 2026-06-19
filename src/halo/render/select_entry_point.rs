//! `select_instance_entry_point` — line-by-line port from dllcache
//! `@ 0x180691340`.
//!
//! Per-instance picker: takes the entry-point requested by the
//! material's rmt2 (always `static_lighting_per_pixel` for opaque BSP
//! shaders) and downgrades to per-vertex / SH / PRT-quadratic based on
//! what lightmap data is available for the structure_instance.
//!
//! Decision tree (from dllcache):
//! ```text
//!   if input != per_pixel: pass through
//!   elif lightmap_data && !force_default && instance_data:
//!     entry = bsp_lightmap_data.instances[i]
//!     if entry.pervertex_block_index == -1:
//!       if entry.probe_block_index >= 0:
//!         upload single_probe → SH (or PRT-quadratic if mesh has SH4 stream)
//!       elif structure_instance.lightmap_texcoord_block_index >= 0
//!            && per_instance_lightmap_texcoords[block].vertex_buffer_index != -1:
//!         bind per-instance UV stream → per_pixel
//!       else:
//!         fall through to SH-fallback
//!     else:
//!       bind per-vertex SH stream → per_vertex
//!       (or fall through if vertex buffer is missing)
//!   else:
//!     SH-fallback: setup_default_lighting + entry = SH
//! ```
//!
//! Two side effects in the engine that callers must replicate:
//! 1. **Probe upload**: when SH path picks a single_probe, dequantize
//!    and upload to engine cbuffer slot 0x30 + dominant light to slot
//!    0x24. (Engine: `calculate_and_set_ravi_constants` +
//!    `set_object_dominant_light_direction_and_intensity`.)
//! 2. **Vertex stream override**: per_vertex binds the per-vertex SH
//!    stream; per_pixel (with override) binds the per-instance UV
//!    stream. (Engine: `c_rasterizer::set_vertex_stream`.)

use crate::halo::render_methods::HaloEntryPoint;
use blam_tags::scenario_lightmap::{LightmapBspData, LightmapInstanceEntry};

/// Output of the picker: which entry point, plus the data the caller
/// needs to set up the per-instance lighting state.
#[derive(Debug, Clone, Copy)]
pub struct SelectedEntry {
    pub entry: HaloEntryPoint,
    /// When `Some`, the caller must upload this probe to the lighting
    /// cbuffer (slot 0x24 dominant + slot 0x30 SH probe) before the
    /// draw. When `None`, the existing `c_lighting_interface` state is
    /// used (set via `setup_default_lighting` or a previous instance).
    pub probe_index: Option<usize>,
    /// When `Some`, the caller binds this per-instance UV stream as
    /// the geometry vertex buffer (replaces the def-mesh's shared
    /// buffer for per-pixel instance lightmaps).
    pub override_vertex_stream: VertexStreamOverride,
    /// When true, the caller must call `setup_default_lighting` before
    /// the draw. Mirrors the engine's "fall back to default" path.
    pub default_lighting_fallback: bool,
    /// `true` iff the engine's PRT activation gate fired —
    /// `structure_instance.lightmapping_policy == 2 (single-probe)`
    /// AND `mesh.vertex_buffer_indices[3] != 0xFFFF`. The engine maps
    /// this to `_entry_point_static_lighting_prt_quadratic` (which gets
    /// remapped per-mesh at `render_mesh_part_default @ 0x18069EBC0`).
    /// When this flag is set, `entry` is `StaticPrtAmbient` (the only
    /// PRT variant present in the MCC corpus). For diagnostics this
    /// mirrors `entry == StaticPrtAmbient` but is kept as a separate
    /// field so future Linear/Quadratic shipping can flip the entry
    /// variant without re-deriving "PRT branch took" elsewhere.
    pub prt_branch_taken: bool,
}

/// Which kind of vertex stream override applies, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStreamOverride {
    /// No override — use the def-mesh's shared vertex buffer.
    None,
    /// Bind the per-instance lightmap-UV stream
    /// (`lightmap.imported_geometry.per_instance_lightmap_texcoords[block_idx]`).
    PerInstanceUv { block_index: i16 },
    /// Bind the per-vertex SH stream
    /// (`lightmap.bsp_per_vertex_data[pervertex_block_index].vertex_buffer_index`).
    /// We don't ship a pipeline that reads it yet — recorded for
    /// future expansion.
    PerVertexSh { pervertex_block_index: i16 },
}

/// `select_instance_entry_point @ 0x180691340`.
///
/// `requested` is the entry point coming from the rmt2/rmsh (typically
/// `StaticPerPixel` for opaque BSP shaders). `bsp_lightmap_data` is
/// the active BSP's lightmap tag data (or `None` if the BSP has no
/// lightmap, equivalent to `g_render_structure_globals.lightmap_bsp_data[i]`
/// being null).
///
/// `lightmapping_policy` is the sbsp's
/// `structure_instance.lightmapping_policy` — only consulted to upgrade
/// SH → PRT-quadratic when the mesh has a quadratic SH stream.
///
/// `mesh_has_prt_quadratic_stream` mirrors
/// `mesh->vertex_buffer_indices[3] != 0xFFFF` from the engine.
pub fn select_instance_entry_point(
    requested: HaloEntryPoint,
    bsp_lightmap_data: Option<&LightmapBspData>,
    structure_instance_index: i32,
    lightmap_texcoord_block_index: i16,
    lightmapping_policy: blam_tags::structure_bsp::InstancedGeometryLightmappingPolicy,
    mesh_has_prt_quadratic_stream: bool,
    has_per_instance_uv_stream: impl Fn(i16) -> bool,
    has_pervertex_runtime_buffer: impl Fn(i16) -> bool,
) -> SelectedEntry {
    let mut out = SelectedEntry {
        entry: requested,
        probe_index: None,
        override_vertex_stream: VertexStreamOverride::None,
        default_lighting_fallback: false,
        prt_branch_taken: false,
    };

    // Only static_lighting_per_pixel gets downgraded; everything else
    // (sky, shadow_apply, transparent, …) passes through.
    if !matches!(requested, HaloEntryPoint::StaticPerPixel) {
        return out;
    }

    let Some(lbsp) = bsp_lightmap_data else {
        // No lightmap data on this BSP — full fallback.
        out.entry = HaloEntryPoint::StaticSh;
        out.default_lighting_fallback = true;
        return out;
    };

    // Look up the instance entry.
    let entry: Option<&LightmapInstanceEntry> =
        lbsp.instances.get(structure_instance_index as usize);

    let Some(entry) = entry else {
        // No lightmap row for this instance. Empty `instances` = this BSP
        // has no instance lightmap (expected). A *populated* table missing
        // this index is an index-space mismatch worth surfacing.
        if !lbsp.instances.is_empty() {
            eprintln!(
                "[lightmap] instance {structure_instance_index} has no lightmap entry \
                 (table has {} rows) — falling back to default SH lighting",
                lbsp.instances.len(),
            );
        }
        out.entry = HaloEntryPoint::StaticSh;
        out.default_lighting_fallback = true;
        return out;
    };

    if entry.pervertex_block_index == -1 {
        // No per-vertex block. Try single probe → SH (or PRT-quadratic).
        if entry.probe_block_index >= 0
            && (entry.probe_block_index as usize) < lbsp.probes.len()
        {
            out.probe_index = Some(entry.probe_block_index as usize);
            // Engine: `lightmapping_policy == 2 && mesh->vertex_buffer_indices[3] != 0xFFFF`
            // upgrades to `_entry_point_static_lighting_prt_quadratic`
            // (which is then remapped by `render_mesh_part_default` to
            // the actual ambient/linear/quadratic variant). Sampled MCC
            // H3 corpus is 100% Ambient where present, so we emit
            // `StaticPrtAmbient` directly. Linear/Quadratic would need
            // their own entry-point variants + shaders.
            out.prt_branch_taken = lightmapping_policy
                == blam_tags::structure_bsp::InstancedGeometryLightmappingPolicy::SingleProbe
                && mesh_has_prt_quadratic_stream;
            out.entry = if out.prt_branch_taken {
                HaloEntryPoint::StaticPrtAmbient
            } else {
                HaloEntryPoint::StaticSh
            };
            return out;
        }

        // No single probe. Try per-instance UV stream → keep per_pixel.
        if lightmap_texcoord_block_index >= 0
            && has_per_instance_uv_stream(lightmap_texcoord_block_index)
        {
            out.entry = HaloEntryPoint::StaticPerPixel;
            out.override_vertex_stream =
                VertexStreamOverride::PerInstanceUv { block_index: lightmap_texcoord_block_index };
            return out;
        }
    } else if (entry.pervertex_block_index as usize) < lbsp.bsp_per_vertex_data.len() {
        // Has a per-vertex SH block. The engine (`select_instance_entry_point
        // @ 0x180691340`) reads the runtime vertex-buffer index `v22` from
        // byte offset 12 of the 16-byte pervertex block
        // (`LightmapPerVertexBlock.vertex_buffer_index`) and branches THREE
        // ways — we previously collapsed it to two and sent the first case
        // to the flat default-lighting fallback:
        //
        //   v22 == -1            → `goto LABEL_43` with the entry STILL
        //                          `_entry_point_static_lighting_per_pixel`.
        //                          i.e. render PER-PIXEL off the mesh's own
        //                          lightmap UVs (no per-instance stream
        //                          override) — NOT the flat SH fallback.
        //   v22 != -1, buf ok    → `_entry_point_static_lighting_per_vertex`
        //   v22 != -1, buf null  → falls through to the SH default fallback.
        let block = &lbsp.bsp_per_vertex_data[entry.pervertex_block_index as usize];
        if block.vertex_buffer_index == -1 {
            // Engine returns per_pixel here (the radial floor panels on
            // guardian land in this case — per-vertex block authored but no
            // runtime SH stream, so the engine uses the per-pixel atlas).
            out.entry = HaloEntryPoint::StaticPerPixel;
            return out;
        }
        if has_pervertex_runtime_buffer(entry.pervertex_block_index) {
            // Engine-faithful path: dedicated `static_sh_per_vertex`
            // entry shader reads SH from a SECONDARY vertex buffer
            // and evaluates per-fragment.
            out.entry = HaloEntryPoint::StaticShPerVertex;
            out.override_vertex_stream = VertexStreamOverride::PerVertexSh {
                pervertex_block_index: entry.pervertex_block_index,
            };
            return out;
        }
    }

    // Nothing matched — SH fallback with default lighting.
    out.entry = HaloEntryPoint::StaticSh;
    out.default_lighting_fallback = true;
    out
}
