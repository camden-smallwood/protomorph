//! Per-BSP GPU resources — vertex/index buffers per mesh + materials.
//!
//! Mirrors what `g_render_structure_globals.render_geometry[bsp_index]`
//! holds at runtime in dllcache. For protomorph we upload from the
//! loose tag's `per mesh temporary[i]/raw vertices/indices` once at
//! scenario load (the cache-format `api_resource` is null in loose tags).
//!
//! Materials are uploaded via the existing
//! [`crate::halo::render_methods::pipeline_cache`] (per-VariantKey
//! pipeline + cbuffer + texture bind group). Different from per-character
//! `GpuMaterial` only in that BSP draws don't need a model matrix or
//! bone array.

use std::path::Path;
use std::rc::Rc;

use blam_tags::paths::resolve_tag_path;
use wgpu::util::DeviceExt;

use crate::halo::geometry::ModelVertex;
use crate::halo::render::shared::SharedResources;
use crate::halo::render_methods::material_bindings::{fallback_view_for_param, sample_intent_for_param};
use crate::halo::render_methods::materials::{
    InlinePixelFormat, InlineTexture, MaterialData, MaterialTextureUsage,
};
use crate::halo::render_methods::pipeline_cache::{HaloPipelineCache, VariantArtifacts};
use crate::halo::render_methods::serializer::serialize as serialize_material;
use crate::halo::render_methods::HaloEntryPoint;
use crate::halo::render_methods::animated::AnimatedMaterial;

use glam::Vec3;
use half::f16;

pub struct BspMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub parts: Vec<BspMeshPart>,
    /// `Some` when the mesh contains at least one water-surface part
    /// (`s_part.flags & _part_is_water_surface`). Holds the per-instance
    /// streams the engine packs at cache build (per
    /// `reference_dllcache_water_pipeline.md`): 156 byte/inst control
    /// points + 72 byte/inst lighting attrs. Each "instance" is one
    /// source water triangle (3 control points). The water draw uses
    /// these alongside the engine-static `g_water_tessellation_*`
    /// barycentric stream + IB.
    pub water: Option<WaterMeshBuffers>,
    /// `Some` when the mesh has a baked PRT Ambient transfer stream
    /// (per-vertex grayscale, one `f32` per vertex). Bound to slot 1
    /// for `HaloEntryPoint::StaticPrtAmbient` instance draws. Mirrors
    /// the engine's slot-2 vertex stream produced by Reach
    /// `create_prt_vertex_buffer @ 0x82E080F0`; see
    /// `project_research_per_mesh_prt_2026_05_11.md`.
    pub prt_ambient_buffer: Option<wgpu::Buffer>,
}

/// Per-mesh water instance streams — packed at BSP load from the
/// tag's `raw_water_data` block. `instance_count` = source water
/// triangle count. Stride 156 for `control_points` (mirrors D3D
/// slot 1) and 72 for `lighting_attrs` (mirrors D3D slot 3) per
/// `Ares/source/rasterizer/rasterizer_resource_definitions.cpp:41`.
///
/// `parts` ranges let the water renderer dispatch a separate draw
/// per source part — different rmw materials on the same mesh route
/// to different option-driven shader pipelines (Phase A4).
pub struct WaterMeshBuffers {
    pub control_points: wgpu::Buffer,
    pub lighting_attrs: wgpu::Buffer,
    pub instance_count: u32,
    pub parts: Vec<WaterMeshPartRange>,
    /// World-space AABB of all control-point positions in this mesh.
    /// Used by `WaterRenderer` to detect when the camera is below a
    /// water surface (`g_is_underwater` in the engine — drives the
    /// underwater-fog branch in `entry_water_shading.wgsl` per
    /// `water_shading_fx.hlsl:1042`). Engine resolves this via
    /// per-water-volume containment; the AABB approximation is
    /// sufficient for the riverworld basin case.
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// One water-flagged source part's instance range within a
/// [`WaterMeshBuffers`]. `material_index` points into the parent
/// `BspGpu::materials` (an rmw [`BspMaterial`]).
#[derive(Debug, Clone, Copy)]
pub struct WaterMeshPartRange {
    pub material_index: u16,
    pub instance_start: u32,
    pub instance_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct BspMeshPart {
    pub material_index: u16,
    pub index_start: u32,
    pub index_count: u32,
    /// `e_geometry_part_type` (Ares `geometry_definitions_new.h:25`):
    /// 0=opaque_not_drawn, 1=opaque_shadow_only, 2=opaque_shadow_casting,
    /// 3=opaque_non_shadowing, 4=transparent, 5=lightmap_only.
    /// Used by `submit_visibility` to compute cluster_part.flags
    /// (`is_transparent ? 4 : (part_type & 3)`); the render passes mask
    /// against `flags` to filter parts.
    pub part_type: i8,
    /// Model-space transparent sort centroid — `RenderMeshPart.sort_position
    /// .position`. BSP geometry is world-space (identity transform), so for
    /// cluster parts this is already the world centroid; instance parts
    /// transform it by the instance matrix. `None` when unauthored. Drives
    /// the back-to-front transparent sort `z_sort` (engine
    /// `c_transparency_renderer::add_element`).
    pub sort_centroid: Option<[f32; 3]>,
    /// Model-space transparent sort PLANE — `RenderMeshPart.sort_position
    /// .plane` `(i, j, k, d)`. Fed to the engine transparent sort's stage-2
    /// BSP plane sort (`c_transparency_renderer::sort_plane_and_point`).
    /// `None` when unauthored. Combined with `sort_radius > 0.0001` and
    /// `|n| > 0.5`, this makes the element `use_plane` (engine `add_element`).
    pub sort_plane: Option<[f32; 4]>,
    /// `RenderMeshPart.sort_position.radius` — the authored sort-quad radius.
    /// Engine `add_element` uses it to gate `use_plane`, scale `importance`,
    /// and size the 9 anchor points. `0.0` when unauthored.
    pub sort_radius: f32,
}

pub struct BspMaterial {
    /// Pipeline for cluster mesh draws — `static_per_pixel_ps` body.
    /// `None` when the render_method subclass has no WGSL port (rmw,
    /// rmd, rmhg, etc.). Materials with `None` pipelines still
    /// classify + queue to transparency; the (currently stubbed)
    /// transparency render will skip them with a counter when their
    /// pipelines aren't built.
    pub artifacts: Option<Rc<VariantArtifacts>>,
    /// Pipeline for instance mesh draws — `static_lighting_sh` body.
    /// `None` for the same reason as `artifacts`.
    pub artifacts_sh: Option<Rc<VariantArtifacts>>,
    /// Pipeline for instance draws that have per-vertex SH —
    /// `static_sh_per_vertex_ps` body. Reads SH from a SECONDARY
    /// vertex buffer (`PerVertexShVertex`) instead of the cbuffer.
    /// Engine-faithful path for `pervertex_block_index >= 0`
    /// instances/clusters; selected by
    /// `select_instance_entry_point` when the GPU per-vertex buffer
    /// is built.
    pub artifacts_sh_per_vertex: Option<Rc<VariantArtifacts>>,
    /// Pipeline for the PRT Ambient instance draw —
    /// `_entry_point_static_lighting_prt_ambient` body. Reads a
    /// per-vertex UNORM transfer scalar from a secondary vertex buffer
    /// (`PrtAmbientVertex`, slot 1) and attenuates the SH probe's
    /// indirect contribution at the fragment. Built for every material
    /// whose subclass has a WGSL pipeline, but only used by draws the
    /// instance picker routes to `HaloEntryPoint::StaticPrtAmbient`.
    pub artifacts_prt_ambient: Option<Rc<VariantArtifacts>>,
    /// Pipeline for the FIRST geometry pass — `_entry_point_albedo`
    /// body. Outputs to the `_surface_albedo` G-buffer + the normal
    /// G-buffer; no lighting / atmosphere / exposure. Engine call
    /// site: `c_player_view::render_albedo @ 0x18068abc0`.
    pub artifacts_albedo: Option<Rc<VariantArtifacts>>,
    /// Bind groups for `artifacts` (per_pixel pipeline), indexed by
    /// scenario cubemap probe index. Length 1 when the BSP has no
    /// cubemap atlas (all clusters share the rasg default cube);
    /// length = `scenario.cubemaps[].len()` when an atlas is loaded —
    /// each entry holds the bind group with that probe's cube view
    /// substituted into the `dynamic_environment_map_*` slots. Mirrors
    /// the engine's `c_render_method::submit_dynamic_cubemap` per-draw
    /// rebind keyed on `c_dynamic_cubemap_sample::m_current.m_bitmap_index`.
    /// Look up at draw time with `cluster_to_probe[cluster_idx]`.
    pub bind_group: Option<Vec<wgpu::BindGroup>>,
    /// Per-probe bind groups for `artifacts_sh` (sh pipeline). Layouts
    /// can differ between entry-point variants when slot ordering
    /// changes. See [`Self::bind_group`] for indexing.
    pub bind_group_sh: Option<Vec<wgpu::BindGroup>>,
    /// Per-probe bind groups for `artifacts_sh_per_vertex` (per-vertex
    /// SH pipeline). See [`Self::bind_group`].
    pub bind_group_sh_per_vertex: Option<Vec<wgpu::BindGroup>>,
    /// Per-probe bind groups for `artifacts_prt_ambient` (PRT Ambient
    /// pipeline). See [`Self::bind_group`].
    pub bind_group_prt_ambient: Option<Vec<wgpu::BindGroup>>,
    /// Per-probe bind groups for `artifacts_albedo` (albedo G-buffer
    /// pipeline). See [`Self::bind_group`].
    pub bind_group_albedo: Option<Vec<wgpu::BindGroup>>,
    pub shader_name: String,
    /// FOURCC of the source rm** tag — used by `submit_visibility` to
    /// route transparent subclasses (rmw water, rmd decals, etc.) to
    /// the transparency queue instead of the opaque cluster_parts list.
    pub render_method_group_tag: u32,
    /// Engine-faithful transparency: mirrors
    /// `c_render_method::get_is_transparent @ 0x1806917c0` (which reads
    /// a flag from `s_render_method_postprocess_definition`). For
    /// rmsh, the postprocess flag is computed from the rmdf
    /// `blend_mode` option choice — opaque=false, anything else=true.
    /// For rmd/rmhg/etc. always true. **rmw is NOT transparent here**
    /// — water has its own pre-transparency dispatch
    /// (`c_water_renderer::render_shading`); see `is_water` below.
    pub is_transparent: bool,
    /// `true` when the material's render_method subclass is `rmw`.
    /// Water draws in `c_water_renderer::render_shading`
    /// (pre-transparency, two sub-passes, instanced patch rendering
    /// with `_vertex_type_water`), separately from cluster_parts /
    /// transparency. Per `reference_dllcache_water_pipeline.md`.
    pub is_water: bool,
    /// Transparent draw-order bucket — the render_method's authored
    /// `sort layer` mapped to `TransparentSortLayer`. Primary key of the
    /// back-to-front transparent sort. `Normal` for opaque/stub materials.
    pub sort_layer: crate::halo::render::render_transparents::TransparentSortLayer,
}

impl BspMaterial {
    /// `true` when the WGSL pipelines for this material's subclass
    /// are ported and built. `false` for transparent subclasses we
    /// haven't ported (rmw, rmd, rmhg, etc.) — the classifier still
    /// queues them, the draw path skips on missing pipeline.
    pub fn is_renderable(&self) -> bool {
        self.artifacts.is_some()
    }
}

/// Render_method subclasses that are ALWAYS transparent regardless
/// of any per-material option — alpha-blended-by-construction.
/// `c_render_method::get_is_transparent` returns true unconditionally
/// for these. `submit_visibility` routes them to
/// `c_transparency_renderer::add_element` rather than
/// `g_render_structure_globals.render_cluster_parts`.
///
/// rmsh / rmtr / rmfl are NOT here — their transparency depends on
/// the chosen `blend_mode` rmdf option (or, for rmfl, the
/// `alpha_test` discard convention which keeps them opaque-pass).
///
/// rmw is also NOT here — water draws in its own pre-transparency
/// pass via `c_water_renderer::render_shading` (per
/// `reference_dllcache_water_pipeline.md`), gated on
/// `s_part.flags & _part_is_water_surface`. See `is_water_subclass`.
pub fn is_always_transparent_subclass(group_tag: u32) -> bool {
    matches!(
        &group_tag.to_be_bytes(),
        b"rmd " | b"rmhg" | b"rmp " | b"rmb " | b"rmco" | b"rmlv"
    )
}

/// `true` when the material's render_method subclass is water (rmw).
/// Water materials route through `c_water_renderer` (own pre-transparency
/// pass with `_vertex_type_water` instanced patch rendering) instead of
/// the rmsh-style `render_cluster_parts` or transparency dispatch.
/// See `reference_dllcache_water_pipeline.md`.
pub fn is_water_subclass(group_tag: u32) -> bool {
    &group_tag.to_be_bytes() == b"rmw "
}

/// `true` when the WGSL pipeline assembler has bodies for this
/// subclass AND the subclass renders through the BSP-mesh static-lighting
/// entry points (StaticPerPixel / StaticSh / StaticShPerVertex /
/// StaticPrtAmbient / Albedo). `assemble()` panics on unported subclasses
/// (no silent fallbacks per `feedback_wgsl_must_mirror_hlsl.md`), so we
/// gate pipeline construction on this. Materials with unported subclasses
/// still get loaded + classified (so `submit_visibility` can route
/// them to transparency); their `BspMaterial.artifacts` stays `None`.
///
/// `rmd ` is intentionally excluded: rmd is a decal subclass with its
/// own vertex layout (`RasterizerVertexWorld`, 44 B) and dedicated entry
/// points (`HaloEntryPoint::Decal` / `DecalAlbedo`). It's drawn through
/// `c_decal::render` via `render_decals.rs`, not the BSP-mesh path.
/// Riverworld's `sbsp.materials` list happens to contain an rmd slot;
/// pairing its decal WGSL with the BSP-mesh `ModelVertex` layout would
/// mismatch at `@location(4)` (binormal `vec4<f32>` vs node_indices
/// `Uint8x4`).
/// Classify a render_method subclass into one of three buckets:
///   - `true`  — protomorph has WGSL bodies; pipeline_cache builds.
///   - `false` — handled by a dedicated renderer (water/decal) that
///               sidesteps the pipeline_cache; no panic.
///   - panic  — neither ported nor recognized as externally-rendered;
///               surfaces missing work instead of silently dropping
///               the material at draw time.
///
/// Anything new appearing in a BSP material slot must be classified
/// explicitly here. Per `feedback_wgsl_must_mirror_hlsl.md`: silent
/// fallbacks hide what needs porting.
pub fn subclass_has_wgsl_pipeline(group_tag: u32) -> bool {
    let bytes = group_tag.to_be_bytes();
    match &bytes {
        // Subclasses we build WGSL pipelines for via
        // `render_methods/mod.rs::assemble`. `rmcs` shares the rmsh
        // rmdf body — dispatched at the `b"rmsh" | b"rmcs" | b"rmhg"`
        // branch. Deadlock's `deadlock_net` / `deadlock_netb` chain-
        // link fence shaders are rmcs and live here.
        b"rmsh" | b"rmcs" | b"rmtr" | b"rmfl" | b"rm  " | b"rmhg" => true,
        // Subclasses handled by dedicated renderers — water shading
        // routes through `WaterRenderer`, decals through the BSP's
        // own decal system. BSP material slots for these legitimately
        // carry `artifacts: None`; the alt renderer reads them through
        // a separate code path (see `bsp.water_material_data`).
        b"rmw " | b"rmd " => false,
        // Anything else is unrecognized. Engine subclasses we haven't
        // accounted for: `rmsk` (skin), `rmct` (cortana), `rmp `
        // (particle), `rmb ` (beam), `rmco` (contrail), `rmlv`
        // (light_volume). When one of these surfaces in a BSP material
        // slot, we want the missing-port to PANIC instead of silently
        // dropping the mesh at draw time.
        _ => {
            let ascii = bytes
                .iter()
                .map(|&b| if (0x20..=0x7E).contains(&b) { b as char } else { '?' })
                .collect::<String>();
            panic!(
                "[protomorph] BSP material has unknown render_method subclass \
                 '{ascii}' (0x{group_tag:08X}). Add it to the WGSL pipeline gate \
                 if portable, or to the external-renderer arm if it routes \
                 through a dedicated renderer. \
                 (per feedback_wgsl_must_mirror_hlsl.md — silent fallbacks hide \
                 what needs porting.)"
            );
        }
    }
}

/// Engine-faithful transparency classification — mirrors
/// `c_render_method::get_is_transparent @ 0x1806917c0`.
///
/// For BSP-relevant subclasses:
///   - `rmsh` — depends on the chosen `blend_mode` rmdf option
///     (`opaque` → opaque pass; `additive` / `alpha_blend` /
///     `multiplicative` / etc. → transparency pass)
///   - `rmtr` — terrain, always opaque
///   - `rmfl` — foliage, alpha-tested in opaque pass (`discard`)
///   - `rmw `, `rmd `, `rmhg`, `rmp `, `rmb `, `rmco`, `rmlv` —
///     always transparent
///
fn classify_transparency(
    group_tag: u32,
    choices: &crate::halo::render_methods::CategoryChoices,
) -> bool {
    if is_always_transparent_subclass(group_tag) {
        return true;
    }
    if &group_tag.to_be_bytes() == b"rmsh" {
        // rmsh has 13 categories; "blend_mode" is one of them.
        // Look up its chosen value — opaque is the only opaque one.
        let blend_mode = choices.get("blend_mode").unwrap_or("opaque");
        return blend_mode != "opaque";
    }
    // rmtr, rmfl — opaque.
    false
}

pub struct BspGpu {
    pub meshes: Vec<BspMesh>,
    /// Per-material slot — `None` for materials whose render_method
    /// subclass we haven't ported (rmd, rmw, rmhg, etc.). The draw
    /// loop skips parts whose material is `None` rather than rendering
    /// through a silent fallback shader (per
    /// `feedback_wgsl_must_mirror_hlsl.md`).
    pub materials: Vec<Option<BspMaterial>>,
    pub cluster_mesh_indices: Vec<i16>,
    pub bsp_index_in_scenario: usize,

    // Instance support — `instances[i]` corresponds to
    // `sbsp.instanced_geometry_instances[i]`. Each instance points
    // (via `definition_index`) at one of `definitions[]`, which
    // resolves to a `mesh_index` into [`Self::meshes`].
    pub instances: Vec<BspInstanceRuntime>,
    pub definitions: Vec<BspInstanceDefinitionRuntime>,
    /// Per-instance bind group at @group(1) — uniform buffer holding
    /// the instance's world matrix. Mirrors Halo's
    /// `render_mesh_submit_rigid_matrix` (which uploads to VS slot 12,
    /// the "skinning" slot) but using our existing model_bgl.
    pub instance_model_bgs: Vec<wgpu::BindGroup>,
    /// Single shared 1-bone identity bind group for instance draws —
    /// instances are RIGID (no skinning), the shader's bone[0] = identity.
    pub instance_nm_bg: Option<wgpu::BindGroup>,
    /// Per-placement vertex buffer override — `Some(buf)` when the
    /// placement has a per-instance lightmap UV stream from the
    /// lightmap tag. The buffer is a copy of the def-mesh's vertex
    /// pool with `lightmap_texcoord` replaced by the per-placement
    /// UVs. `None` placements reuse the def-mesh's shared buffer
    /// (which carries the lightmap tag's authoring UVs — typically
    /// the canonical/zero unwrap).
    pub instance_vertex_buffers: Vec<Option<wgpu::Buffer>>,
    /// Per-instance per-vertex SH vertex buffer — `Some(buf)` when the
    /// instance has `pervertex_block_index >= 0` AND the lightmap
    /// `bsp_per_vertex_data[block_idx].lightprobe_data` count matches
    /// the def-mesh's vertex count. `None` placements fall back to the
    /// per-instance averaged probe via `instance_lighting_offsets`
    /// (existing bridge).
    ///
    /// 64 bytes/vertex (`PerVertexShVertex`). Bound as the SECOND
    /// vertex buffer at draw time when `instance_selections[i].entry`
    /// is `StaticShPerVertex`.
    pub instance_per_vertex_sh_buffers: Vec<Option<wgpu::Buffer>>,
    /// `select_instance_entry_point @ 0x180691340` output per instance.
    /// Computed once at load from the lightmap tag. Drives pipeline
    /// + vertex-stream choice at draw time.
    pub instance_selections: Vec<crate::halo::render::select_entry_point::SelectedEntry>,
    /// Per-instance dynamic-offset into the global `engine_lighting_buffer`
    /// (shared.rs). For SH-path instances with a `probe_index`, the BSP
    /// load step writes the dequantized probe to a unique slot and stores
    /// the byte offset here. Default-lighting instances get `0` (frame
    /// default written by `setup_default_lighting`).
    pub instance_lighting_offsets: Vec<u32>,
    /// Per-cluster dynamic-offset into `engine_lighting_buffer`. `0` means
    /// the cluster is per-pixel-lit (uses the atlas via static_per_pixel).
    /// Non-zero means the cluster has authored per-vertex SH (cluster
    /// `pervertex_block_index >= 0` in lightmap data); we average those
    /// per-vertex probes into a single ravi probe and route the cluster's
    /// parts through the StaticSh pipeline at this offset. Bridge until
    /// the engine-faithful per-vertex vertex-stream pipeline ships
    /// (see reference_bsp_per_vertex_sh_stream.md).
    pub cluster_lighting_offsets: Vec<u32>,
    /// Per-cluster per-vertex SH vertex buffer — `Some(buf)` when the
    /// cluster has `pervertex_block_index >= 0` AND the lightmap
    /// `bsp_per_vertex_data[block_idx].lightprobe_data` count matches
    /// the cluster mesh's vertex count. `None` clusters use the
    /// averaged-probe bridge via `cluster_lighting_offsets`. Engine
    /// `_entry_point_static_lighting_per_vertex` (idx=3) interpolates
    /// these probes per-vertex and evaluates ravi at the fragment
    /// normal — preserves spatial variance the averaging bridge loses.
    pub cluster_per_vertex_sh_buffers: Vec<Option<wgpu::Buffer>>,
    /// Per-cluster dynamic-offset into `simple_lights_buffer`. `0` =
    /// the empty slot (no lights). Non-zero = a per-cluster slot
    /// pre-baked at scenario load with the up-to-8 stli lights nearest
    /// to that cluster's bounds. Engine equivalent:
    /// `c_lights_view::add_simple_light_to_draw_list` is called per
    /// cluster part inside `c_structure_renderer::render_cluster_mesh_part`,
    /// giving each cluster its own light set (stable across camera
    /// motion, no inter-frame popping).
    pub cluster_simple_lights_offsets: Vec<u32>,
    /// Resolved `MaterialData` for EVERY distinct `rmw` material on this
    /// BSP, paired with its `material_index` into `sbsp.materials` (the
    /// same index each `WaterMeshPart` carries). docks has two
    /// (`dock_water_1` + `dock_water_2`); riverworld has one.
    /// WaterRenderer builds a per-material slot from each at scenario
    /// load and selects by `material_index` per water part.
    pub water_materials: Vec<(u16, MaterialData)>,
    /// Per-cluster scenario cubemap probe selection. `cluster_to_probe[i]`
    /// is the index into `scenario.cubemaps[]` (and
    /// `<scenario>_<bsp>_cubemaps.bitmap` atlas slices) chosen for
    /// cluster `i`. Mirrors the engine's `cluster_cubemaps[]` runtime
    /// cache that MCC strips from the tag corpus — we synthesize it at
    /// load by nearest-3D-distance from cluster centroid to scenario
    /// probe position. Used at draw time to pick the right `BspMaterial`
    /// bind group (`bind_group[cluster_to_probe[cluster_idx] as usize]`).
    /// Empty when the BSP has no atlas; in that case material bind
    /// groups have a single entry with the rasg default cube.
    pub cluster_to_probe: Vec<u16>,

    /// P6.5 — per-frame animated cbuffer re-upload state, one entry per
    /// BSP material with `is_animated=true`. Engine equivalent:
    /// `update_constants @ 0x180685300` walks the material's overlay
    /// list each frame. Protomorph re-resolves the full cbuffer at
    /// the current time and writes through to each `props_buffer`.
    /// Empty when the BSP has no animated materials.
    pub animated_materials: Vec<AnimatedMaterial>,
}

#[derive(Debug, Clone, Copy)]
pub struct BspInstanceRuntime {
    pub definition_index: i16,
    /// Composed world matrix (scale × basis(forward, left, up) +
    /// position). `forward/left/up` are read from the sbsp
    /// `instanced geometry instances[i]` block (a 3x3 orthonormal
    /// basis); `scale` is the instance's uniform scale.
    pub world_matrix: glam::Mat4,
}

#[derive(Debug, Clone, Copy)]
pub struct BspInstanceDefinitionRuntime {
    pub mesh_index: i16,
}

/// Per-BSP cubemap routing — atlas cube views (one per scenario probe)
/// + cluster-to-probe assignment. Engine path:
/// `c_dynamic_cubemap_sample::search_for_cubemap_sample_in_cluster`
/// (MCC dllcache @ 0x1806e0ff0) walks `structure_cluster.cluster_cubemaps[]`
/// at runtime, picks nearest probe by 3D distance with hysteresis
/// crossfade. MCC strips `cluster_cubemaps[]` so we synthesize the
/// nearest-probe assignment at scenario load and bake it into per-probe
/// material bind groups; runtime selection becomes a no-op (only one
/// candidate per cluster).
pub struct BspCubemapRouting {
    /// One cube `TextureView` per scenario probe, indexed 1:1 with
    /// `scenario.cubemaps[]`. Sourced from
    /// `<scenario_dir>/<scenario>_<bsp>_cubemaps.bitmap` whose
    /// `bitmaps[i]` block is also 1:1.
    pub probe_views: Vec<wgpu::TextureView>,
    /// `cluster_to_probe[cluster_idx]` = scenario probe index used by
    /// that cluster. Assigned at scenario load by nearest-distance
    /// from cluster centroid to scenario probe position.
    pub cluster_to_probe: Vec<u16>,
}

impl BspGpu {
    pub fn from_loaded_bsp(
        shared: &SharedResources,
        pipeline_cache: &mut HaloPipelineCache,
        sampler_cache: &mut crate::halo::rasterizer::dx11::SamplerStateCache,
        bsp: &crate::halo::scenario::loader::LoadedBsp,
        tags_root: &Path,
        next_probe_slot: &mut u32,
        cubemap_routing: Option<&BspCubemapRouting>,
    ) -> Self {
        let mut materials: Vec<Option<BspMaterial>> = Vec::with_capacity(bsp.sbsp.materials.len());
        let mut water_materials: Vec<(u16, MaterialData)> = Vec::new();
        // P6.5 — collected during material walk; one entry per animated
        // material. Empty when the BSP has no TagFunction-time materials.
        let mut animated_materials: Vec<AnimatedMaterial> = Vec::new();
        for (mi, mat) in bsp.sbsp.materials.iter().enumerate() {
            let mat_data = if mat.render_method.is_empty() {
                // Genuinely unauthored material slot — engine-faithful
                // no-op stub. (Distinct from a load FAILURE, which
                // panics below per the no-silent-fallback policy.)
                MaterialData::stub_for_shader(&format!("bsp_mat_{mi}"))
            } else {
                let shader_abs = resolve_tag_path(tags_root, &mat.render_method, mat.render_method_extension());
                let try_load = crate::halo::loader::load_shader_material_public(
                    &shader_abs,
                    tags_root,
                    mat.render_method.split('\\').last().unwrap_or("bsp_mat"),
                );
                if let Some(mat_data) = try_load {
                    mat_data
                } else {
                    // Engine fallback: when a referenced shader tag is
                    // missing on disk (some s3d_* MCC ports drop a tag
                    // file from the build), substitute `shaders\invalid.shader`
                    // — the canonical "tag not present" placeholder that
                    // ships with the MCC tag corpus. Stops the BSP load
                    // dying on a single broken material reference.
                    let invalid_abs = resolve_tag_path(tags_root, "shaders\\invalid", "shader");
                    crate::halo::loader::load_shader_material_public(
                        &invalid_abs, tags_root, "invalid",
                    )
                    .unwrap_or_else(|| panic!(
                        "[protomorph] failed to load BSP material #{mi} '{}' at {} \
                         AND the `shaders\\invalid.shader` fallback also failed to load. \
                         Render-method walker returned None for both.",
                        mat.render_method, shader_abs.display(),
                    ))
                }
            };
            let stub;
            let choices = if mat_data.category_choices.is_empty() {
                stub = crate::halo::render_methods::pipeline_cache::stub_choices();
                &stub
            } else {
                &mat_data.category_choices
            };
            let is_transparent = classify_transparency(mat.render_method_group_tag, choices);
            let is_water = is_water_subclass(mat.render_method_group_tag);

            // Pipeline build is gated on whether the WGSL assembler
            // has bodies for this subclass — engine-honest "we
            // classify everything, we draw what we can". Unported
            // subclasses still land in transparency via `is_transparent`;
            // the draw path will skip on `artifacts.is_none()` until
            // their WGSL is ported.
            let (
                artifacts,
                artifacts_sh,
                artifacts_sh_per_vertex,
                artifacts_prt_ambient,
                artifacts_albedo,
                bind_group,
                bind_group_sh,
                bind_group_sh_per_vertex,
                bind_group_prt_ambient,
                bind_group_albedo,
            ) = if subclass_has_wgsl_pipeline(mat.render_method_group_tag) {
                let artifacts = pipeline_cache.ensure(
                    shared,
                    &mat_data.resolved,
                    &mat_data.cbuffer,
                    choices,
                    HaloEntryPoint::StaticPerPixel,
                    tags_root,
                );
                let artifacts_sh = pipeline_cache.ensure(
                    shared,
                    &mat_data.resolved,
                    &mat_data.cbuffer,
                    choices,
                    HaloEntryPoint::StaticSh,
                    tags_root,
                );
                let artifacts_sh_pv = pipeline_cache.ensure(
                    shared,
                    &mat_data.resolved,
                    &mat_data.cbuffer,
                    choices,
                    HaloEntryPoint::StaticShPerVertex,
                    tags_root,
                );
                let artifacts_prt_amb = pipeline_cache.ensure(
                    shared,
                    &mat_data.resolved,
                    &mat_data.cbuffer,
                    choices,
                    HaloEntryPoint::StaticPrtAmbient,
                    tags_root,
                );
                let artifacts_albedo = pipeline_cache.ensure(
                    shared,
                    &mat_data.resolved,
                    &mat_data.cbuffer,
                    choices,
                    HaloEntryPoint::Albedo,
                    tags_root,
                );
                // Per-probe bind groups — engine mirrors
                // `c_render_method::submit_dynamic_cubemap` per draw,
                // we bake one bind group per probe at load time and
                // pick by `cluster_to_probe[cluster_idx]` at draw.
                // No atlas → length 1 (rasg default cube fallback).
                //
                // Material textures + props cbuffer are uploaded ONCE
                // per variant by `resolve_material_resources`; the
                // per-probe loop only swaps the cube override view.
                // Without this hoist, every probe re-uploads every
                // material's diffuse/bump/etc. (10-100K redundant
                // texture uploads per BSP → VRAM bomb).
                // P6.5 — collect per-variant props_buffer handles when
                // the material is animated, so the per-frame updater can
                // queue.write_buffer fresh time-resolved bytes into all
                // of them. wgpu::Buffer is Arc-cloned so this is cheap.
                let mut animated_buffers: Vec<wgpu::Buffer> = Vec::new();
                // Capture the variant artifacts' layout for the animated
                // serialize path. All variants of a given material share
                // the same layout (engine emits one cb13 layout per rmsh+
                // rmt2 pair), so any variant's layout works; we grab the
                // first one we see.
                let mut animated_layout: Option<
                    crate::halo::render_methods::cbuffer::CbufferLayout,
                > = None;
                let mut build_per_probe = |arts: &VariantArtifacts| -> Vec<wgpu::BindGroup> {
                    let resources =
                        resolve_material_resources(shared, &mat_data, arts, sampler_cache);
                    if mat_data.is_animated {
                        animated_buffers.push(resources.props_buffer.clone());
                        if animated_layout.is_none() {
                            animated_layout = Some(arts.layout.clone());
                        }
                    }
                    match cubemap_routing {
                        Some(routing) if !routing.probe_views.is_empty() => routing
                            .probe_views
                            .iter()
                            .map(|view| {
                                build_material_bind_group(
                                    shared,
                                    arts,
                                    &resources,
                                    Some(view),
                                )
                            })
                            .collect(),
                        _ => vec![build_material_bind_group(
                            shared, arts, &resources, None,
                        )],
                    }
                };
                let bg = build_per_probe(&artifacts);
                let bg_sh = build_per_probe(&artifacts_sh);
                let bg_sh_pv = build_per_probe(&artifacts_sh_pv);
                let bg_prt_amb = build_per_probe(&artifacts_prt_amb);
                let bg_albedo = build_per_probe(&artifacts_albedo);
                // P6.5 — record the animation state if any variant
                // contributed buffers. Source must be present for
                // re-resolution; `is_animated` is only set when
                // `render_method_source.is_some()` (loader.rs guards),
                // so the unwrap is safe under invariant.
                if !animated_buffers.is_empty() {
                    if let (Some(source), Some(layout)) =
                        (mat_data.render_method_source.as_ref(), animated_layout)
                    {
                        // Snapshot the load-time cbuffer bytes (full —
                        // including phantom slots from
                        // `loader::append_param_phantom_slots` and
                        // `pad_terrain_cbuffer`). Used per-frame as the
                        // fallback so phantom-slot values survive the
                        // rmt2-only re-resolve. Without this, scalars
                        // like `fresnel_curve_steepness`,
                        // `normal_specular_power`, terrain blend-map
                        // defaults etc. zero out every frame on any
                        // material flagged `is_animated`.
                        let loaded_cbuffer_bytes = mat_data.cbuffer.bytes.clone();
                        animated_materials.push(AnimatedMaterial {
                            source: source.clone(),
                            layout,
                            primary_change_color: mat_data.primary_change_color,
                            secondary_change_color: mat_data.secondary_change_color,
                            loaded_cbuffer_bytes,
                            props_buffers: animated_buffers,
                        });
                    }
                }
                (
                    Some(artifacts),
                    Some(artifacts_sh),
                    Some(artifacts_sh_pv),
                    Some(artifacts_prt_amb),
                    Some(artifacts_albedo),
                    Some(bg),
                    Some(bg_sh),
                    Some(bg_sh_pv),
                    Some(bg_prt_amb),
                    Some(bg_albedo),
                )
            } else {
                (None, None, None, None, None, None, None, None, None, None)
            };

            if is_transparent && artifacts.is_none() {
                let tag = mat.render_method_group_tag.to_be_bytes();
                eprintln!(
                    "[bsp] transparent material #{mi} subclass={:?} shader='{}' — no WGSL pipeline (will be skipped at draw)",
                    std::str::from_utf8(&tag).unwrap_or("???"),
                    mat.render_method,
                );
            }
            materials.push(Some(BspMaterial {
                artifacts,
                artifacts_sh,
                artifacts_sh_per_vertex,
                artifacts_prt_ambient,
                artifacts_albedo,
                bind_group,
                bind_group_sh,
                bind_group_sh_per_vertex,
                bind_group_prt_ambient,
                bind_group_albedo,
                shader_name: mat.render_method.clone(),
                render_method_group_tag: mat.render_method_group_tag,
                is_transparent,
                is_water,
                sort_layer: mat_data
                    .render_method_source
                    .as_ref()
                    .map(|s| {
                        crate::halo::render::render_transparents::TransparentSortLayer::from_global(
                            s.rmsh.sort_layer.get(),
                        )
                    })
                    .unwrap_or(crate::halo::render::render_transparents::TransparentSortLayer::Normal),
            }));
            // Capture EVERY rmw material's resolved data + its
            // material_index so WaterRenderer can build a per-material
            // slot (env_cubemap + reflection/fresnel + wave/foam/
            // global_shape textures + variant pipeline) and route each
            // water part to the right one. docks has 2 (dock_water_1 /
            // dock_water_2); rendering them both is what makes its
            // harbor look right (the single-material path forced every
            // surface onto dock_water_1 → wrong wave size / no foam).
            if is_water {
                water_materials.push((mi as u16, mat_data));
            }
        }

        let mut meshes: Vec<BspMesh> = Vec::with_capacity(bsp.meshes.len());
        for mesh in &bsp.meshes {
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                meshes.push(BspMesh {
                    vertex_buffer: shared.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("bsp_empty_vbuf"),
                        size: 64,
                        usage: wgpu::BufferUsages::VERTEX,
                        mapped_at_creation: false,
                    }),
                    index_buffer: shared.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("bsp_empty_ibuf"),
                        size: 4,
                        usage: wgpu::BufferUsages::INDEX,
                        mapped_at_creation: false,
                    }),
                    parts: Vec::new(),
                    water: None,
                    prt_ambient_buffer: None,
                });
                continue;
            }

            let verts: Vec<ModelVertex> =
                mesh.vertices.iter().map(convert_bsp_vertex).collect();
            let vertex_buffer = shared
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bsp_mesh_vbuf"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let index_buffer = shared
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bsp_mesh_ibuf"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            let parts = mesh
                .parts
                .iter()
                .map(|p| BspMeshPart {
                    material_index: p.material_index,
                    index_start: p.index_start,
                    index_count: p.index_count,
                    part_type: p.part_type,
                    sort_centroid: p
                        .sort_position
                        .map(|sp| [sp.position.x, sp.position.y, sp.position.z]),
                    sort_plane: p
                        .sort_position
                        .map(|sp| [sp.plane.i, sp.plane.j, sp.plane.k, sp.plane.d]),
                    sort_radius: p.sort_position.map(|sp| sp.radius).unwrap_or(0.0),
                })
                .collect();

            // Pack per-mesh water instance streams when the mesh
            // carries `raw_water_data`. Mirrors the offline cache
            // build's per-source-triangle packing into the engine's
            // stride-156 + stride-72 instance buffers — see
            // `reference_dllcache_water_pipeline.md`.
            let water = pack_water_buffers(&shared.device, mesh);

            // PRT Ambient secondary stream (slot 1 at draw time). Empty
            // when the mesh has no baked transfer data; bake math lives
            // in `blam-tags/src/render_model.rs`. Already validated to
            // be `vertices.len()` long when non-empty.
            let prt_ambient_buffer = if mesh.prt_ambient_stream.is_empty() {
                None
            } else {
                Some(shared.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bsp_mesh_prt_ambient"),
                    contents: bytemuck::cast_slice(&mesh.prt_ambient_stream),
                    usage: wgpu::BufferUsages::VERTEX,
                }))
            };

            meshes.push(BspMesh { vertex_buffer, index_buffer, parts, water, prt_ambient_buffer });
        }

        let cluster_mesh_indices: Vec<i16> = bsp.sbsp.clusters.iter().map(|c| c.mesh_index).collect();

        // ---- Instances + definitions ------------------------------
        let definitions: Vec<BspInstanceDefinitionRuntime> = bsp
            .sbsp
            .instance_definitions
            .iter()
            .map(|d| BspInstanceDefinitionRuntime { mesh_index: d.mesh_index })
            .collect();

        let instances: Vec<BspInstanceRuntime> = bsp
            .sbsp
            .instanced_geometry_instances
            .iter()
            .map(|inst| BspInstanceRuntime {
                definition_index: inst.definition_index,
                world_matrix: compose_instance_world_matrix(inst),
            })
            .collect();

        // Per-instance @group(1) bind groups — each holds a uniform
        // buffer with the instance's world matrix. (Matches the
        // existing pipeline's model_bgl layout.)
        let instance_model_bgs = build_instance_bind_groups(shared, &instances);

        // Shared identity 1-bone @group(3) for instances.
        let instance_nm_bg = if instances.is_empty() {
            None
        } else {
            Some(build_identity_nm_bg(shared))
        };

        // Build per-placement vertex buffers when the lightmap tag
        // carries a per-instance UV stream for this placement. Each
        // is a copy of the def-mesh's full vertex pool with
        // lightmap_texcoord replaced from
        // `lightmap.imported_geometry.per_instance_lightmap_texcoords[
        //  structure_instance.lightmap_texcoord_block_index]
        //  .texture_coordinates[k].lightmap texcoord`.
        // Mirrors `select_instance_entry_point @ 0x180691340`'s
        // per-instance vertex stream binding.
        let instance_vertex_buffers: Vec<Option<wgpu::Buffer>> = bsp
            .sbsp
            .instanced_geometry_instances
            .iter()
            .enumerate()
            .map(|(p_idx, inst)| {
                let block_idx = inst.lightmap_texcoord_block_index;
                if block_idx < 0 {
                    return None;
                }
                let uvs = bsp
                    .per_instance_lightmap_uvs
                    .iter()
                    .find(|p| p.block_index == block_idx as usize)?;
                if uvs.uvs.is_empty() {
                    return None;
                }
                let def_idx = inst.definition_index as usize;
                let def = bsp.sbsp.instance_definitions.get(def_idx)?;
                let mesh_idx = def.mesh_index as usize;
                let src_mesh = bsp.meshes.get(mesh_idx)?;
                if src_mesh.vertices.is_empty() {
                    return None;
                }
                if uvs.uvs.len() != src_mesh.vertices.len() {
                    eprintln!(
                        "[bsp_gpu] instance[{p_idx}] block[{}]: uv count {} != mesh verts {}",
                        block_idx, uvs.uvs.len(), src_mesh.vertices.len(),
                    );
                    // Length mismatch — skip override rather than mis-zip.
                    return None;
                }
                let mut verts: Vec<ModelVertex> = src_mesh
                    .vertices
                    .iter()
                    .map(convert_bsp_vertex)
                    .collect();
                for (vi, uv) in uvs.uvs.iter().enumerate() {
                    let u = half::f16::from_f32(uv.x);
                    let v = half::f16::from_f32(uv.y);
                    verts[vi].lightmap_texcoord = [u.to_bits(), v.to_bits()];
                }
                Some(
                    shared
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("bsp_instance_vbuf"),
                            contents: bytemuck::cast_slice(&verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                )
            })
            .collect();

        // Per-instance per-vertex SH vertex buffers. Engine-faithful
        // path for `pervertex_block_index >= 0` instances: each
        // vertex carries its own SH probe, the rasterizer
        // interpolates, and PS evaluates ravi at fragment normal.
        // Replaces the per-instance averaging bridge (which loses
        // spatial variance across the instance mesh).
        //
        // Built when:
        //   - lightmap tag has the instance's `pervertex_block_index`
        //   - the per-vertex block's probe count matches the def-mesh
        //     vertex count (i.e. one probe per vertex)
        // Otherwise `None` and the picker falls back to the averaging
        // bridge or default lighting.
        use crate::halo::geometry::PerVertexShVertex;
        let mut n_pv_sh_built = 0usize;
        let instance_per_vertex_sh_buffers: Vec<Option<wgpu::Buffer>> = bsp
            .sbsp
            .instanced_geometry_instances
            .iter()
            .enumerate()
            .map(|(i, inst)| {
                let lbsp_ref = bsp.lightmap.as_ref()?;
                let entry = lbsp_ref.instances.get(i)?;
                if entry.pervertex_block_index < 0 {
                    return None;
                }
                let pv_block = lbsp_ref
                    .bsp_per_vertex_data
                    .get(entry.pervertex_block_index as usize)?;
                if pv_block.lightprobe_data.is_empty() {
                    return None;
                }
                let def = bsp
                    .sbsp
                    .instance_definitions
                    .get(inst.definition_index as usize)?;
                let mesh = bsp.meshes.get(def.mesh_index as usize)?;
                if pv_block.lightprobe_data.len() != mesh.vertices.len() {
                    // Probe-vs-vertex count mismatch — skip rather
                    // than mis-zip. Falls through to the averaging
                    // bridge.
                    return None;
                }
                let probes: Vec<PerVertexShVertex> = pv_block
                    .lightprobe_data
                    .iter()
                    .map(|p| PerVertexShVertex::from_probe(&p.dequantize()))
                    .collect();
                n_pv_sh_built += 1;
                Some(
                    shared
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("bsp_instance_per_vertex_sh"),
                            contents: bytemuck::cast_slice(&probes),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                )
            })
            .collect();
        if n_pv_sh_built > 0 {
            eprintln!(
                "[bsp_gpu] BSP[{}] uploaded {} per-instance per-vertex SH buffers",
                bsp.scenario_bsp_index, n_pv_sh_built,
            );
        }

        // PRT Ambient bake stats — how many def-meshes carry a
        // pre-baked PRT vertex stream from `per_mesh_prt_data[i].mesh
        // pca data`. See `project_research_per_mesh_prt_2026_05_11.md`.
        let n_prt_meshes = bsp.meshes.iter().filter(|m| !m.prt_ambient_stream.is_empty()).count();
        let prt_stream_bytes: usize = bsp.meshes.iter().map(|m| m.prt_ambient_stream.len() * 4).sum();
        if n_prt_meshes > 0 {
            eprintln!(
                "[bsp_gpu] BSP[{}] PRT Ambient bake: {} meshes, {} KB total stream",
                bsp.scenario_bsp_index, n_prt_meshes, prt_stream_bytes / 1024,
            );
        }

        // `select_instance_entry_point @ 0x180691340` per instance —
        // deterministic from (lightmap, sbsp), so cache at load time.
        use crate::halo::render::select_entry_point::{
            select_instance_entry_point, SelectedEntry, VertexStreamOverride,
        };
        use crate::halo::render_methods::HaloEntryPoint;
        let lbsp = bsp.lightmap.as_ref();
        let instance_selections: Vec<SelectedEntry> = bsp
            .sbsp
            .instanced_geometry_instances
            .iter()
            .enumerate()
            .map(|(i, inst)| {
                let has_uv = |block_index: i16| -> bool {
                    bsp.per_instance_lightmap_uvs
                        .iter()
                        .any(|p| p.block_index == block_index as usize && !p.uvs.is_empty())
                };
                // Per-vertex SH GPU buffer present? Closure scopes
                // by instance index since the buffer is keyed there;
                // engine-faithful per-vertex SH path is now enabled
                // when this returns true.
                let inst_idx = i;
                let has_pv = |_pv_block_index: i16| -> bool {
                    instance_per_vertex_sh_buffers
                        .get(inst_idx)
                        .map(|b| b.is_some())
                        .unwrap_or(false)
                };
                // PRT-quadratic stream gate: the engine looks at
                // `mesh->vertex_buffer_indices[3] != 0xFFFF` on the def
                // mesh. blam-tags surfaces this as
                // `RenderMesh::has_prt_vertex_stream`. See
                // `project_research_per_mesh_prt_2026_05_11.md`.
                let mesh_has_prt_stream = bsp
                    .sbsp
                    .instance_definitions
                    .get(inst.definition_index as usize)
                    .and_then(|def| bsp.meshes.get(def.mesh_index as usize))
                    .map(|m| m.has_prt_vertex_stream)
                    .unwrap_or(false);
                select_instance_entry_point(
                    HaloEntryPoint::StaticPerPixel,
                    lbsp,
                    i as i32,
                    inst.lightmap_texcoord_block_index,
                    inst.lightmapping_policy.get(),
                    mesh_has_prt_stream,
                    has_uv,
                    has_pv,
                )
            })
            .collect();

        // Histogram for visibility — burns one log on load.
        let (mut n_pp, mut n_sh, mut n_pv, mut n_fall, mut n_prt) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for s in &instance_selections {
            match s.entry {
                HaloEntryPoint::StaticPerPixel => n_pp += 1,
                HaloEntryPoint::StaticSh => {
                    if s.default_lighting_fallback { n_fall += 1 } else { n_sh += 1 }
                }
                HaloEntryPoint::StaticShPerVertex => n_pv += 1,
                HaloEntryPoint::StaticPrtAmbient => n_prt += 1,
                HaloEntryPoint::Albedo => {} // not picked by select_instance_entry_point
                HaloEntryPoint::StaticVertexColor => {} // sky-only; BSP instances never select it
                HaloEntryPoint::Decal | HaloEntryPoint::DecalAlbedo | HaloEntryPoint::DecalObject => {} // decals don't use the instance entry-point selector
            }
        }
        // How many would qualify for per-vertex SH if we shipped the pipeline?
        let n_pv_eligible = lbsp
            .map(|l| {
                bsp.sbsp
                    .instanced_geometry_instances
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        l.instances
                            .get(*i)
                            .map(|e| e.pervertex_block_index >= 0)
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0);
        // Cross-check the picker's `PerInstanceUv` choice against the
        // load-time per-instance vertex-buffer build. Both branch on the
        // same predicate (`per_instance_lightmap_uvs[block_index]` exists
        // and non-empty); a mismatch would mean the picker says "use
        // per-instance UVs" but draw time can't, or vice-versa.
        let n_picker_uv_override = instance_selections
            .iter()
            .filter(|s| matches!(s.override_vertex_stream, VertexStreamOverride::PerInstanceUv { .. }))
            .count();
        let n_loadtime_uv_override = instance_vertex_buffers
            .iter()
            .filter(|vb| vb.is_some())
            .count();
        debug_assert_eq!(
            n_picker_uv_override, n_loadtime_uv_override,
            "picker `PerInstanceUv` count ({n_picker_uv_override}) drifted from \
             load-time per-instance VB count ({n_loadtime_uv_override}); see \
             render/select_entry_point.rs and bsp_gpu.rs:475 for the two predicates",
        );
        eprintln!(
            "[bsp_gpu] BSP[{}] instance picker: per_pixel={} sh={} sh_per_vertex={} \
             prt_ambient={} fallback={} (per_vertex_eligible={}, uv_override={}, total={})",
            bsp.scenario_bsp_index,
            n_pp, n_sh, n_pv, n_prt, n_fall,
            n_pv_eligible, n_picker_uv_override, instance_selections.len(),
        );

        // Per-instance probe upload — two sources:
        //   1. SH path with `probe_index = Some(idx)` → dequantize
        //      `lbsp.probes[idx]` (order-3, full SH)
        //   2. Per-vertex-eligible fallbacks → average all per-vertex
        //      probes from `lbsp.bsp_per_vertex_data[pv_idx]` into a
        //      single order-2 probe (zero-padded to order-3). Bridge
        //      until we ship a true per-vertex pipeline. Per-vertex
        //      probes have NO authored direction; we derive one from
        //      the L1 SH bands (same path as `find_dominant_light_direction`).
        // Both routes write to `engine_lighting_buffer` at a unique
        // slot; the structure_renderer rebinds group 0 with that
        // dynamic offset per draw.
        use crate::halo::lighting::{
            pack_ravi_constants, GpuEngineDominantLight, GpuEngineLightingPs,
        };
        use crate::halo::render::lighting_interface::find_dominant_light_direction;
        use crate::halo::render::shared::{ENGINE_LIGHTING_ENTRIES, ENGINE_LIGHTING_STRIDE};
        let mut instance_lighting_offsets: Vec<u32> =
            vec![0u32; instance_selections.len()];
        let mut n_avg_per_vertex = 0usize;
        if let Some(lbsp) = lbsp {
            for (i, sel) in instance_selections.iter().enumerate() {
                // Route 1: explicit single-probe upload.
                if let Some(probe_idx) = sel.probe_index {
                    let Some(probe) = lbsp.probes.get(probe_idx) else { continue };
                    if *next_probe_slot >= ENGINE_LIGHTING_ENTRIES {
                        eprintln!(
                            "[bsp_gpu] WARNING: engine_lighting_buffer slot exhausted ({} max)",
                            ENGINE_LIGHTING_ENTRIES,
                        );
                        continue;
                    }
                    let dq = probe.dequantize();
                    let ravi_payload = GpuEngineLightingPs {
                        ravi: pack_ravi_constants(
                            &dq.red_terms, &dq.green_terms, &dq.blue_terms, 1.0,
                        ),
                    };
                    let dom_payload = GpuEngineDominantLight {
                        direction: [
                            dq.dominant_light_direction[0],
                            dq.dominant_light_direction[1],
                            dq.dominant_light_direction[2],
                            1.0,
                        ],
                        intensity: [
                            dq.dominant_light_intensity[0],
                            dq.dominant_light_intensity[1],
                            dq.dominant_light_intensity[2],
                            0.0,
                        ],
                    };
                    let offset = *next_probe_slot * ENGINE_LIGHTING_STRIDE;
                    shared.queue.write_buffer(
                        &shared.engine_lighting_buffer,
                        offset as u64,
                        bytemuck::bytes_of(&ravi_payload),
                    );
                    shared.queue.write_buffer(
                        &shared.engine_dominant_light_buffer,
                        offset as u64,
                        bytemuck::bytes_of(&dom_payload),
                    );
                    instance_lighting_offsets[i] = offset;
                    *next_probe_slot += 1;
                    continue;
                }

                // Route 2: per-vertex-eligible cbuffer fallback. Averages all
                // per-vertex probes into one SH probe and uploads to the
                // lighting cbuffer. Only consumed when the per-vertex GPU
                // buffer failed to build (vertex count mismatch) — the picker
                // then falls back to `StaticSh` which reads from the cbuffer.
                // When `instance_per_vertex_sh_buffers[i].is_some()`, the
                // `StaticShPerVertex` pipeline reads SH from the per-vertex
                // VB and ignores the cbuffer, so the upload would be wasted.
                let Some(entry) = lbsp.instances.get(i) else { continue };
                if entry.pervertex_block_index < 0 {
                    continue;
                }
                if instance_per_vertex_sh_buffers
                    .get(i)
                    .map(|b| b.is_some())
                    .unwrap_or(false)
                {
                    continue;
                }
                let Some(pv_block) = lbsp
                    .bsp_per_vertex_data
                    .get(entry.pervertex_block_index as usize)
                else {
                    continue;
                };
                if pv_block.lightprobe_data.is_empty() {
                    continue;
                }
                if *next_probe_slot >= ENGINE_LIGHTING_ENTRIES {
                    eprintln!(
                        "[bsp_gpu] WARNING: engine_lighting_buffer slot exhausted ({} max) \
                         — per-vertex SH probe upload skipped",
                        ENGINE_LIGHTING_ENTRIES,
                    );
                    continue;
                }
                let n = pv_block.lightprobe_data.len() as f32;
                let inv_n = 1.0 / n;
                let mut sh_r = [0.0_f32; 9];
                let mut sh_g = [0.0_f32; 9];
                let mut sh_b = [0.0_f32; 9];
                let mut intensity = [0.0_f32; 3];
                for pv_probe in &pv_block.lightprobe_data {
                    let dq = pv_probe.dequantize();
                    for k in 0..4 {
                        sh_r[k] += dq.red_terms[k] * inv_n;
                        sh_g[k] += dq.green_terms[k] * inv_n;
                        sh_b[k] += dq.blue_terms[k] * inv_n;
                    }
                    intensity[0] += dq.dominant_light_intensity[0] * inv_n;
                    intensity[1] += dq.dominant_light_intensity[1] * inv_n;
                    intensity[2] += dq.dominant_light_intensity[2] * inv_n;
                }
                // Derive dominant direction from L1 bands (same recipe
                // as `find_dominant_light_direction` — luma-weighted
                // collapse + normalize).
                let dom_dir = find_dominant_light_direction(&sh_r, &sh_g, &sh_b);
                let ravi_payload = GpuEngineLightingPs {
                    ravi: pack_ravi_constants(&sh_r, &sh_g, &sh_b, 1.0),
                };
                let dom_payload = GpuEngineDominantLight {
                    direction: [dom_dir.x, dom_dir.y, dom_dir.z, 1.0],
                    intensity: [intensity[0], intensity[1], intensity[2], 0.0],
                };
                let offset = *next_probe_slot * ENGINE_LIGHTING_STRIDE;
                shared.queue.write_buffer(
                    &shared.engine_lighting_buffer,
                    offset as u64,
                    bytemuck::bytes_of(&ravi_payload),
                );
                shared.queue.write_buffer(
                    &shared.engine_dominant_light_buffer,
                    offset as u64,
                    bytemuck::bytes_of(&dom_payload),
                );
                instance_lighting_offsets[i] = offset;
                *next_probe_slot += 1;
                n_avg_per_vertex += 1;
            }
        }
        if n_avg_per_vertex > 0 {
            eprintln!(
                "[bsp_gpu] BSP[{}] uploaded {} per-vertex-averaged probes \
                 (cbuffer fallback for instances where per-vertex VB build failed)",
                bsp.scenario_bsp_index, n_avg_per_vertex,
            );
        }

        // Per-cluster SH bridge — same pattern as the per-instance bridge
        // above. Halo's lightmap baker authors per-vertex SH for some
        // clusters (water, transparents, low-detail areas) instead of
        // baking into the lightprobe atlas. The engine binds those as a
        // SECONDARY VERTEX STREAM at draw time (see
        // `reference_bsp_per_vertex_sh_stream.md`); we approximate by
        // averaging the per-vertex SH probes into a single cluster SH
        // and routing the cluster's parts through StaticSh at a
        // per-cluster dynamic offset. Loses spatial variance across the
        // cluster but lights the geometry correctly (waterfall on
        // riverworld renders properly with this — verified visually).
        let cluster_count = bsp.sbsp.clusters.len();
        let mut cluster_lighting_offsets: Vec<u32> = vec![0u32; cluster_count];
        let mut n_cluster_pv = 0usize;
        if let Some(lbsp) = lbsp {
            for cluster_idx in 0..cluster_count {
                let Some(entry) = lbsp.clusters.get(cluster_idx) else { continue };
                if entry.pervertex_block_index < 0 {
                    continue;
                }
                let Some(pv_block) = lbsp
                    .bsp_per_vertex_data
                    .get(entry.pervertex_block_index as usize)
                else {
                    continue;
                };
                if pv_block.lightprobe_data.is_empty() {
                    continue;
                }
                if *next_probe_slot >= ENGINE_LIGHTING_ENTRIES {
                    eprintln!(
                        "[bsp_gpu] WARNING: engine_lighting_buffer slot exhausted at cluster {}",
                        cluster_idx,
                    );
                    break;
                }
                let n = pv_block.lightprobe_data.len() as f32;
                let inv_n = 1.0 / n;
                let mut sh_r = [0.0_f32; 9];
                let mut sh_g = [0.0_f32; 9];
                let mut sh_b = [0.0_f32; 9];
                let mut intensity = [0.0_f32; 3];
                for pv_probe in &pv_block.lightprobe_data {
                    let dq = pv_probe.dequantize();
                    for k in 0..4 {
                        sh_r[k] += dq.red_terms[k] * inv_n;
                        sh_g[k] += dq.green_terms[k] * inv_n;
                        sh_b[k] += dq.blue_terms[k] * inv_n;
                    }
                    intensity[0] += dq.dominant_light_intensity[0] * inv_n;
                    intensity[1] += dq.dominant_light_intensity[1] * inv_n;
                    intensity[2] += dq.dominant_light_intensity[2] * inv_n;
                }
                let dom_dir = find_dominant_light_direction(&sh_r, &sh_g, &sh_b);
                let ravi_payload = GpuEngineLightingPs {
                    ravi: pack_ravi_constants(&sh_r, &sh_g, &sh_b, 1.0),
                };
                let dom_payload = GpuEngineDominantLight {
                    direction: [dom_dir.x, dom_dir.y, dom_dir.z, 1.0],
                    intensity: [intensity[0], intensity[1], intensity[2], 0.0],
                };
                let offset = *next_probe_slot * ENGINE_LIGHTING_STRIDE;
                shared.queue.write_buffer(
                    &shared.engine_lighting_buffer,
                    offset as u64,
                    bytemuck::bytes_of(&ravi_payload),
                );
                shared.queue.write_buffer(
                    &shared.engine_dominant_light_buffer,
                    offset as u64,
                    bytemuck::bytes_of(&dom_payload),
                );
                cluster_lighting_offsets[cluster_idx] = offset;
                *next_probe_slot += 1;
                n_cluster_pv += 1;
            }
        }
        if n_cluster_pv > 0 {
            eprintln!(
                "[bsp_gpu] BSP[{}] uploaded {} per-cluster SH probes (per-vertex-lit clusters; \
                 routes through StaticSh at draw time)",
                bsp.scenario_bsp_index, n_cluster_pv,
            );
        }

        // Per-cluster per-vertex SH vertex buffers — engine-faithful
        // path for `pervertex_block_index >= 0` clusters whose probe
        // count matches the cluster mesh's vertex count. When `Some`,
        // the cluster routes through `StaticShPerVertex` at draw time
        // and the per-vertex SH stream is interpolated by the
        // rasterizer (preserves spatial variance the averaging bridge
        // loses).
        let mut n_cluster_pv_buffers = 0usize;
        let cluster_per_vertex_sh_buffers: Vec<Option<wgpu::Buffer>> = (0..cluster_count)
            .map(|cluster_idx| {
                let lbsp_ref = lbsp?;
                let entry = lbsp_ref.clusters.get(cluster_idx)?;
                if entry.pervertex_block_index < 0 {
                    return None;
                }
                let pv_block = lbsp_ref
                    .bsp_per_vertex_data
                    .get(entry.pervertex_block_index as usize)?;
                if pv_block.lightprobe_data.is_empty() {
                    return None;
                }
                let cluster = bsp.sbsp.clusters.get(cluster_idx)?;
                if cluster.mesh_index < 0 {
                    return None;
                }
                let mesh = bsp.meshes.get(cluster.mesh_index as usize)?;
                if pv_block.lightprobe_data.len() != mesh.vertices.len() {
                    // Probe-vs-vertex count mismatch — skip rather
                    // than mis-zip. Falls through to the averaging
                    // bridge at `cluster_lighting_offsets[i]`.
                    return None;
                }
                use crate::halo::geometry::PerVertexShVertex;
                let probes: Vec<PerVertexShVertex> = pv_block
                    .lightprobe_data
                    .iter()
                    .map(|p| PerVertexShVertex::from_probe(&p.dequantize()))
                    .collect();
                n_cluster_pv_buffers += 1;
                Some(
                    shared
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("bsp_cluster_per_vertex_sh"),
                            contents: bytemuck::cast_slice(&probes),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                )
            })
            .collect();
        if n_cluster_pv_buffers > 0 {
            eprintln!(
                "[bsp_gpu] BSP[{}] uploaded {} per-cluster per-vertex SH buffers \
                 (engine-faithful StaticShPerVertex route for cluster meshes)",
                bsp.scenario_bsp_index, n_cluster_pv_buffers,
            );
        }

        Self {
            meshes,
            materials,
            cluster_mesh_indices,
            bsp_index_in_scenario: bsp.scenario_bsp_index,
            instances,
            definitions,
            instance_model_bgs,
            instance_nm_bg,
            instance_vertex_buffers,
            instance_per_vertex_sh_buffers,
            instance_selections,
            instance_lighting_offsets,
            cluster_lighting_offsets,
            cluster_per_vertex_sh_buffers,
            // Default — populated by `bake_cluster_simple_lights` at
            // scenario load once `scenario_lights` is resolved.
            cluster_simple_lights_offsets: vec![0u32; cluster_count],
            water_materials,
            cluster_to_probe: cubemap_routing
                .map(|r| r.cluster_to_probe.clone())
                .unwrap_or_default(),
            animated_materials,
        }
    }

    /// P6.5 — Re-resolve all animated cb13 user cbuffers at the
    /// per-frame `time` and write the fresh bytes into each cached
    /// `props_buffer`.
    ///
    /// **Divergence from engine, documented:** engine `update_constants
    /// @ 0x180685300` walks `rmsh.m_overlays` and updates ONLY the
    /// byte ranges that hold animated slots — leaves static slots
    /// alone. Protomorph re-resolves and rewrites the whole cbuffer.
    /// Functionally equivalent (static slots evaluate to the same
    /// value at any T, so the bytes match) but does extra work.
    ///
    /// The shortcut is intentional: `blam_tags::is_cbuffer_animated`
    /// uses an empirical t=0/t=1 byte comparison instead of parsing
    /// overlay metadata, and this update path mirrors that choice. If
    /// per-overlay-walk fidelity is required (e.g., cbuffer-pool
    /// pipeline-stall concerns on Xbox-era hardware), file a P6.6
    /// follow-up to port the engine's incremental path.
    pub fn update_animated_cbuffers(&self, queue: &wgpu::Queue, time_seconds: f32) {
        for animated in &self.animated_materials {
            animated.write_at_time(queue, time_seconds);
        }
    }
}

/// Compose a per-instance world matrix from the sbsp's
/// `instanced geometry instances[i]` fields. Halo stores instances as
/// a uniform scale + 3x3 orthonormal basis (forward/left/up rows) +
/// position. The runtime's `render_mesh_submit_rigid_matrix` premultiplies
/// scale into the basis: `out = (scale * R | t)` then ships to VS slot 12.
fn compose_instance_world_matrix(
    inst: &blam_tags::structure_bsp::BspInstance,
) -> glam::Mat4 {
    use glam::{Mat4, Vec3, Vec4};
    let scale = inst.scale;
    // forward/left/up are world-space ROWS of the rotation. Halo's
    // basis convention: forward = +X, left = +Y, up = +Z.
    let f = Vec3::new(inst.forward.i, inst.forward.j, inst.forward.k) * scale;
    let l = Vec3::new(inst.left.i, inst.left.j, inst.left.k) * scale;
    let u = Vec3::new(inst.up.i, inst.up.j, inst.up.k) * scale;
    let p = Vec3::new(inst.position.x, inst.position.y, inst.position.z);
    Mat4::from_cols(
        Vec4::new(f.x, f.y, f.z, 0.0),
        Vec4::new(l.x, l.y, l.z, 0.0),
        Vec4::new(u.x, u.y, u.z, 0.0),
        Vec4::new(p.x, p.y, p.z, 1.0),
    )
}

fn build_instance_bind_groups(
    shared: &SharedResources,
    instances: &[BspInstanceRuntime],
) -> Vec<wgpu::BindGroup> {
    use crate::halo::geometry::ModelUniforms;
    let mut out = Vec::with_capacity(instances.len());
    for (i, inst) in instances.iter().enumerate() {
        let uniforms = ModelUniforms { model: inst.world_matrix.to_cols_array_2d() };
        let bytes = bytemuck::bytes_of(&uniforms);
        let mut padded = vec![0u8; shared.model_stride];
        padded[..bytes.len()].copy_from_slice(bytes);
        let buffer = shared.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("bsp_instance_model_uniform"),
                contents: &padded,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bsp_instance_model_bg"),
            layout: &shared.model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(bytes.len() as u64),
                }),
            }],
        });
        out.push(bg);
        let _ = (i, buffer);
    }
    out
}

fn build_identity_nm_bg(shared: &SharedResources) -> wgpu::BindGroup {
    let identity_mat = glam::Mat4::IDENTITY.to_cols_array();
    let bytes = bytemuck::cast_slice(&identity_mat);
    let mut padded = vec![0u8; shared.node_matrices_stride];
    padded[..bytes.len()].copy_from_slice(bytes);
    let buffer = shared.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("bsp_instance_nm_buffer"),
            contents: &padded,
            usage: wgpu::BufferUsages::STORAGE,
        },
    );
    let bg = shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bsp_instance_nm_bg"),
        layout: &shared.node_matrices_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(shared.node_matrices_stride as u64),
            }),
        }],
    });
    let _ = buffer;
    bg
}

fn convert_bsp_vertex(v: &blam_tags::render_model::RenderVertex) -> ModelVertex {
    use crate::halo::geometry::{oct_encode_snorm8, pack_tangent_with_sign, pack_weights_unorm8};
    let normal = Vec3::new(v.normal.i, v.normal.j, v.normal.k);
    let texcoord_u = f16::from_f32(v.texcoord.x);
    let texcoord_v = f16::from_f32(v.texcoord.y);
    let raw_tangent = Vec3::new(v.tangent.i, v.tangent.j, v.tangent.k);
    let raw_binormal = Vec3::new(v.binormal.i, v.binormal.j, v.binormal.k);
    let (tangent_dir, binormal_dir) = if raw_tangent.length_squared() > 0.0001 {
        (raw_tangent.normalize(), raw_binormal.normalize_or_zero())
    } else {
        let basis = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let stub = (basis - normal * basis.dot(normal)).normalize_or_zero();
        (stub, normal.cross(stub))
    };
    let lm_u = f16::from_f32(v.lightmap_texcoord.x);
    let lm_v = f16::from_f32(v.lightmap_texcoord.y);
    ModelVertex {
        position: [v.position.x, v.position.y, v.position.z],
        texcoord: [texcoord_u.to_bits(), texcoord_v.to_bits()],
        normal: oct_encode_snorm8(normal),
        _pad: [0, 0],
        tangent: pack_tangent_with_sign(tangent_dir, normal, binormal_dir),
        node_indices: [0, 0, 0, 0],
        node_weights: pack_weights_unorm8([1.0, 0.0, 0.0, 0.0]),
        lightmap_texcoord: [lm_u.to_bits(), lm_v.to_bits()],
    }
}

/// Mirror of `Renderer::upload_model_data`'s inline texture upload —
/// kept local until we have a shared `GpuTextureBuilder` extracted.
/// Same layout: `tex.data` is one concat blob holding the full mip
/// pyramid for each layer back-to-back.
pub fn upload_inline_texture(
    shared: &SharedResources,
    tex: &InlineTexture,
    intent: crate::halo::render::SampleIntent,
) -> wgpu::TextureView {
    use crate::halo::render::resolve_wgpu_format;
    let format = resolve_wgpu_format(tex.format, tex.encoding, intent);
    // BC formats need block-aligned texture extents: wgpu rejects
    // `create_texture` whose width/height aren't multiples of 4 for any
    // BC1..BC7 format. Halo's bitmap loader keeps the authored dims
    // (e.g. 474×474 for a sparse bitm), so round up for the GPU
    // allocation while preserving the conceptual size everywhere else.
    // See `feedback_wgpu_bc_mip_alignment.md` — the same rounding the
    // mip-level `write_texture` already applies, but at create-time.
    let (tex_width, tex_height) = if tex.format.is_compressed() {
        ((tex.width + 3) & !3, (tex.height + 3) & !3)
    } else {
        (tex.width, tex.height)
    };
    let texture = shared.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bsp_inline_texture"),
        size: wgpu::Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: tex.layers,
        },
        mip_level_count: tex.mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Halo cubemap byte layout per dllcache `bitmap_cube_map_address @
    // 0x1804ea600`: **MIP-MAJOR**, with all 6 faces vertically stacked
    // at each mip. Pixel offset = `v5 + (y + width * face_index) * width
    // + x` where `v5 += 6 * width²` per preceding mip. So the storage
    // layout is:
    //   mip 0: [face 0 mip 0][face 1 mip 0]...[face 5 mip 0]
    //   mip 1: [face 0 mip 1][face 1 mip 1]...[face 5 mip 1]
    //   ...
    // Faces are in HALO INTERNAL order [+X, +Y, -X, -Y, +Z, -Z]; at
    // upload time the engine remaps to D3D order
    // [+X, -X, +Y, -Y, +Z, -Z] via
    // `c_cubemap::face_D3D_mapping_table = [0, 2, 1, 3, 4, 5]` inside
    // `c_d3d_resource_allocator::create_cube_texture @ 0x1806c1b00`.
    //
    // 2D / 2D-array / 3D bitmaps appear to use FACE-MAJOR layout
    // (each layer/slice's full mip chain contiguous) — that's how the
    // current upload reads them, and the wave_displacement_array /
    // lightprobe_atlas paths render correctly. Only the `is_cube`
    // branch needs the mip-major reinterpretation.
    const CUBE_FACE_D3D_MAPPING: [u32; 6] = [0, 2, 1, 3, 4, 5];
    if tex.is_cube {
        // Mip-major source iteration matching Halo's storage.
        let mut mip_offset = 0usize;
        for mip in 0..tex.mip_count {
            let mw = (tex.width >> mip).max(1);
            let mh = (tex.height >> mip).max(1);
            let face_bytes = tex.format.level_bytes(mw, mh);
            let (extent_w, extent_h, rows, bytes_per_row);
            if tex.format.is_compressed() {
                let bw = ((mw + 3) / 4).max(1);
                let bh = ((mh + 3) / 4).max(1);
                extent_w = bw * 4;
                extent_h = bh * 4;
                rows = bh;
                bytes_per_row = bw * tex.format.block_bytes();
            } else {
                extent_w = mw;
                extent_h = mh;
                rows = mh;
                bytes_per_row = mw * tex.format.bytes_per_pixel();
            }
            for source_face in 0..tex.layers {
                let dst_layer = if (source_face as usize) < CUBE_FACE_D3D_MAPPING.len() {
                    CUBE_FACE_D3D_MAPPING[source_face as usize]
                } else {
                    source_face
                };
                let off = mip_offset + (source_face as usize) * face_bytes;
                shared.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: dst_layer },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.data[off..off + face_bytes],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rows),
                    },
                    wgpu::Extent3d {
                        width: extent_w,
                        height: extent_h,
                        depth_or_array_layers: 1,
                    },
                );
            }
            mip_offset += (tex.layers as usize) * face_bytes;
        }
    } else {
        // 2D / 2D-array / 3D: face-major (each layer's full mip chain
        // contiguous).
        let mut offset = 0usize;
        for layer in 0..tex.layers {
            for mip in 0..tex.mip_count {
                let mw = (tex.width >> mip).max(1);
                let mh = (tex.height >> mip).max(1);
                let level_bytes = tex.format.level_bytes(mw, mh);
                let (extent_w, extent_h, rows, bytes_per_row);
                if tex.format.is_compressed() {
                    let bw = ((mw + 3) / 4).max(1);
                    let bh = ((mh + 3) / 4).max(1);
                    extent_w = bw * 4;
                    extent_h = bh * 4;
                    rows = bh;
                    bytes_per_row = bw * tex.format.block_bytes();
                } else {
                    extent_w = mw;
                    extent_h = mh;
                    rows = mh;
                    bytes_per_row = mw * tex.format.bytes_per_pixel();
                }
                shared.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.data[offset..offset + level_bytes],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(rows),
                    },
                    wgpu::Extent3d {
                        width: extent_w,
                        height: extent_h,
                        depth_or_array_layers: 1,
                    },
                );
                offset += level_bytes;
            }
        }
    }
    let view_dimension = if tex.is_cube {
        // 6-layer array → cube view.
        wgpu::TextureViewDimension::Cube
    } else if tex.layers > 1 {
        // Multi-layer non-cube → D2Array. Halo's `type=array` bitmaps
        // (e.g. wave_displacement_array, lightprobe_atlas) land here:
        // depth>1 in the bitm tag means "stack of slices" addressed by
        // an integer index, sampled in WGSL as `texture_2d_array<f32>`.
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    };
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(view_dimension),
        ..Default::default()
    })
}

/// Return the loaded `TextureView` for engine externs whose source is
/// a global (rasg-loaded) texture rather than per-material data.
/// Currently routes the 3 Cook-Torrance pre-integrated BRDF LUTs from
/// `rasterizer_globals.material_textures[]`. Returns `None` for any
/// extern not handled here — caller falls through to the per-material
/// bitmap lookup or fallback view.
fn engine_extern_view(
    shared: &SharedResources,
    ext: blam_tags::render_method::RenderMethodExtern,
) -> Option<wgpu::TextureView> {
    use blam_tags::render_method::RenderMethodExtern as E;
    use blam_tags::rasterizer_globals::MaterialTexture as MT;
    let slot = match ext {
        E::TextureCookTorranceCc0236 => MT::CookTorranceCc0236,
        E::TextureCookTorranceDd0236 => MT::CookTorranceDd0236,
        E::TextureCookTorranceC78d78 => MT::CookTorranceC78d78,
        _ => return None,
    };
    Some(shared.fallback_textures.material_texture_view(slot).clone())
}

/// One material's per-variant resources resolved once: every material
/// texture uploaded exactly once, props cbuffer built once. Per-probe
/// bind-group construction reuses these by reference and only swaps
/// the cube override view, so we don't re-upload the same diffuse /
/// bump / detail textures N-probes × 4-variants times per material.
struct ResolvedMaterialResources {
    /// `(slot, view-or-cube-marker)` per binding. `Fixed` views are
    /// already uploaded; `CubeOverridable` slots take the per-probe
    /// cube view at bind-group-build time (or the rasg default when
    /// the BSP has no probe atlas).
    slots: Vec<(u32, ResolvedSlotView)>,
    /// Per-material props cbuffer — same bytes for every probe.
    props_buffer: wgpu::Buffer,
    /// One sampler per unique `SamplerKey` in
    /// `variant.sampler_map.unique_keys` — parallel-indexed. Each
    /// `BitmapBinding` references its sampler via the dedup map's
    /// `binding_to_key_index` lookup. Engine equivalent: D3D11
    /// `d3d11_sampler_state_cache::get` per texture stage.
    per_key_samplers: Vec<wgpu::Sampler>,
}

enum ResolvedSlotView {
    /// View is fixed across probes (material's own bitmap, or a
    /// non-cube fallback). Holds the actual TextureView.
    Fixed(wgpu::TextureView),
    /// Cube extern slot with no material bitmap — at bind-group build
    /// time, substitute the probe's cube view, or fall back to the
    /// rasg `DefaultDynamicCubeMap` when no probe override is in play.
    CubeOverridable,
}

/// Resolve every material texture + props cbuffer once. This is the
/// only place that calls `upload_inline_texture`, so it runs exactly
/// once per (material, variant) pair regardless of probe count.
fn resolve_material_resources(
    shared: &SharedResources,
    mat: &MaterialData,
    artifacts: &VariantArtifacts,
    sampler_cache: &mut crate::halo::rasterizer::dx11::SamplerStateCache,
) -> ResolvedMaterialResources {
    use blam_tags::render_method::RenderMethodExtern;
    let bindings = &artifacts.bindings;
    let mut slots: Vec<(u32, ResolvedSlotView)> = Vec::with_capacity(bindings.textures.len());
    for tb in &bindings.textures {
        // Engine-managed extern textures take precedence over the
        // material's own bitmap lookup. Cook-Torrance BRDF LUTs are
        // sourced from `rasterizer_globals.material_textures[]` and
        // routed by extern enum (HLSL extern_texture_mode 24/25/26).
        if let Some(ext) = tb.source_extern {
            if let Some(view) = engine_extern_view(shared, ext) {
                slots.push((tb.slot, ResolvedSlotView::Fixed(view)));
                continue;
            }
        }
        let resolved = match (mat.find_texture_by_name(&tb.name), tb.is_cube) {
            // Binding expects a cube but the rmop targets a 2D bitmap.
            // Engine pixl bytecode locks the register dimension to cube
            // (HLSL `PARAM_SAMPLER_CUBE(environment_map)`), so trying
            // to bind the 2D view trips `dimension D2 vs Cube` wgpu
            // validation. Fall through to the cube fallback view (the
            // 1×1 black cube — same neutral the runtime would resolve
            // when no probe override is in play). Seen on warehouse +
            // similar maps that author 2D bitmaps on cube slots.
            (Some(tex), true) if !tex.is_cube => {
                ResolvedSlotView::Fixed(fallback_view_for_param(shared, &tb.name, true))
            }
            // Material carries its own bitmap — engine path:
            // `c_render_method_submit_set_texture` reads the rmsh's
            // own texture bind. Upload exactly once.
            (Some(tex), _) => ResolvedSlotView::Fixed(upload_inline_texture(
                shared,
                tex,
                sample_intent_for_param(&tb.name),
            )),
            // Cube extern slot with no material bitmap — defer to
            // per-probe build (substitute probe view or rasg default).
            (None, true) => ResolvedSlotView::CubeOverridable,
            // Non-cube slot with no material bitmap — fixed fallback.
            (None, false) => {
                ResolvedSlotView::Fixed(fallback_view_for_param(shared, &tb.name, false))
            }
        };
        slots.push((tb.slot, resolved));
    }

    let props_bytes = serialize_material(mat, &artifacts.layout);
    // P6.5 — COPY_DST so the per-frame animated cbuffer updater can
    // queue.write_buffer fresh bytes each frame. Non-animated materials
    // pay a tiny memory overhead (the COPY_DST allocation tracking) for
    // a unified buffer-creation path.
    let props_buffer = shared
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bsp_material_props"),
            contents: &props_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    // Per-key samplers — one per unique `SamplerKey` in the artifacts'
    // sampler dedup map. Each `BitmapBinding`'s authored
    // `BitmapAddressMode` / `BitmapFilterMode` is now honored via the
    // shared dedupe binding it points at. Engine equivalent:
    // `d3d11_sampler_state_cache::get` per texture stage.
    let per_key_samplers: Vec<wgpu::Sampler> = artifacts
        .sampler_map
        .unique_keys
        .iter()
        .map(|&k| sampler_cache.get(&shared.device, k))
        .collect();

    ResolvedMaterialResources {
        slots,
        props_buffer,
        per_key_samplers,
    }
}

/// Build a per-probe bind group from already-resolved material
/// resources. This is cheap — no GPU uploads, just a BindGroup
/// descriptor referencing pre-existing views/buffers. Called once
/// per probe per variant.
fn build_material_bind_group(
    shared: &SharedResources,
    artifacts: &VariantArtifacts,
    resources: &ResolvedMaterialResources,
    cube_override: Option<&wgpu::TextureView>,
) -> wgpu::BindGroup {
    let bindings = &artifacts.bindings;

    // Materialize each slot's view (cheap Arc clone for Fixed). Cube
    // extern slots take the per-probe atlas cube (`cube_override`) when
    // the BSP has a cubemap atlas, else the 1×1 black cube. (Env math +
    // RGBW decode are faithful; exposure now applies at composite, so
    // the prior black-cube workaround is being re-measured — see F.)
    let views: Vec<wgpu::TextureView> = resources
        .slots
        .iter()
        .map(|(_, resolved)| match resolved {
            ResolvedSlotView::Fixed(v) => v.clone(),
            ResolvedSlotView::CubeOverridable => cube_override
                .cloned()
                .unwrap_or_else(|| shared.fallback_textures.black_cube_view.clone()),
        })
        .collect();

    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(
        resources.slots.len() + resources.per_key_samplers.len() + 2,
    );
    for ((slot, _), view) in resources.slots.iter().zip(views.iter()) {
        entries.push(wgpu::BindGroupEntry {
            binding: *slot,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    // Per-key samplers — one per unique `SamplerKey`, bound at the
    // `s_dedupe_K` slots emitted by `emit_wgsl_per_binding`.
    for (idx, sampler) in resources.per_key_samplers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: bindings.per_binding_sampler_slot(idx),
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: bindings.per_binding_cbuffer_slot(&artifacts.sampler_map),
        // Path B: cbuffer slot is dynamic-offset (BGL declares
        // `has_dynamic_offset: true`). One slot per material today;
        // per-object pools land in Phase 5.
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &resources.props_buffer,
            offset: 0,
            size: wgpu::BufferSize::new(resources.props_buffer.size()),
        }),
    });

    shared.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bsp_material_bg"),
        layout: &artifacts.material_bgl,
        entries: &entries,
    })
}

/// Pack per-mesh water instance streams from the BSP's `raw_water_data`
/// block (parsed into `mesh.water_data` by blam-tags). Each
/// `RawWaterTriangle` becomes one instance of:
///
/// - **156 bytes (stream 0)** — `WaterControlPointInstance`: 3 control
///   points worth of position + texcoord + tangent + binormal +
///   lightmap_uv, packed across 10 attributes per
///   `Ares/source/rasterizer/rasterizer_resource_definitions.cpp:41`.
/// - **72 bytes (stream 2)** — `WaterLightingInstance`: 3 control
///   points worth of `local_info` + `base_texcoord`, packed across 5
///   attributes (NORMAL0..4).
///
/// Each control point has two indices (already resolved by
/// blam-tags via the Reach `create_mesh_water_vertex_buffer` walk):
/// - `regular_idx` → `mesh.vertices[regular_idx]` (regular pool —
///   position/texcoord/tangent/binormal/lightmap_uv).
/// - `water_idx` → `mesh.water_data.vertices[water_idx]` (append pool
///   — local_info / base_texcoord).
///
/// See `reference_water_vertex_buffer_build.md`.
fn pack_water_buffers(
    device: &wgpu::Device,
    mesh: &blam_tags::RenderMesh,
) -> Option<WaterMeshBuffers> {
    use crate::halo::render::render_water::{
        WaterControlPointInstance, WaterLightingInstance,
    };

    let wd = mesh.water_data.as_ref()?;
    if wd.triangles.is_empty() || wd.vertices.is_empty() {
        return None;
    }

    // Diagnostic: dump base_texcoord range across the append pool.
    // If .x/.y exceed [0,1], the Repeat sampler tiles the river-map
    // bitmap multiple times across the lake → grid artifact.
    {
        let (mut mn_x, mut mx_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut mn_y, mut mx_y) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut mn_z, mut mx_z) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &wd.vertices {
            let (x, y, z) = (v.base_texcoord.x, v.base_texcoord.y, v.base_texcoord.z);
            if x < mn_x { mn_x = x; }
            if x > mx_x { mx_x = x; }
            if y < mn_y { mn_y = y; }
            if y > mx_y { mx_y = y; }
            if z < mn_z { mn_z = z; }
            if z > mx_z { mx_z = z; }
        }
        let oob_x = mn_x < 0.0 || mx_x > 1.0;
        let oob_y = mn_y < 0.0 || mx_y > 1.0;
        eprintln!(
            "[water-bt] {} water verts, {} tris: base_texcoord x=[{mn_x:.4}, {mx_x:.4}]{} y=[{mn_y:.4}, {mx_y:.4}]{} z=[{mn_z:.4}, {mx_z:.4}]",
            wd.vertices.len(),
            wd.triangles.len(),
            if oob_x { " ⚠OOB" } else { "" },
            if oob_y { " ⚠OOB" } else { "" },
        );

        // local_info.x is the per-scenario wave-amplitude scale that
        // multiplies displacement at engine HLSL line 447
        // (`displacement *= IN.local_info.x`). Riverworld uses 3.178
        // across all verts; docks/etc. may pick different values.
        // local_info.y is per-vertex water_depth (drives bankalpha=depth
        // + globalshape=depth fades).
        let (mut li_x_mn, mut li_x_mx) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut li_y_mn, mut li_y_mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &wd.vertices {
            if v.local_info.x < li_x_mn { li_x_mn = v.local_info.x; }
            if v.local_info.x > li_x_mx { li_x_mx = v.local_info.x; }
            if v.local_info.y < li_y_mn { li_y_mn = v.local_info.y; }
            if v.local_info.y > li_y_mx { li_y_mx = v.local_info.y; }
        }
        eprintln!(
            "[water-bt]   local_info: x=[{li_x_mn:.4}, {li_x_mx:.4}] (amp scale) \
             y=[{li_y_mn:.4}, {li_y_mx:.4}] (water_depth)",
        );
    }

    // Diagnostic: dump lightmap_texcoord range across the regular mesh
    // vertices that water triangles index into. Engine HLSL samples
    // `lightprobe_texture_array` at `lm_tex.xy` for the SH DC term that
    // becomes water `lightmap_intensity` (water_shading_fx.hlsl:769).
    // If the range is all-zero / outside the atlas's valid charts, the
    // SH sample returns 0 → water_color collapses → river renders
    // transparent. Cross-reference with `riverworld_lightmap_..._lp_array_dxt5`
    // alpha layout (16% of texels have BC3 alpha=0).
    {
        let mut idxs = std::collections::HashSet::<u32>::new();
        for tri in &wd.triangles {
            for cp in &tri.control_points {
                idxs.insert(cp.regular_idx as u32);
            }
        }
        let (mut mn_x, mut mx_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut mn_y, mut mx_y) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut zero = 0usize;
        for &i in &idxs {
            let i = i as usize;
            if i >= mesh.vertices.len() { continue; }
            let lm = &mesh.vertices[i].lightmap_texcoord;
            if lm.x == 0.0 && lm.y == 0.0 { zero += 1; }
            if lm.x < mn_x { mn_x = lm.x; }
            if lm.x > mx_x { mx_x = lm.x; }
            if lm.y < mn_y { mn_y = lm.y; }
            if lm.y > mx_y { mx_y = lm.y; }
        }
        eprintln!(
            "[water-bt]   water regular-vert lightmap_uv: \
             x=[{mn_x:.4}, {mx_x:.4}] y=[{mn_y:.4}, {mx_y:.4}] \
             ({}/{} verts at (0,0))",
            zero, idxs.len(),
        );
    }

    let mut control_points: Vec<WaterControlPointInstance> =
        Vec::with_capacity(wd.triangles.len());
    let mut lighting_attrs: Vec<WaterLightingInstance> =
        Vec::with_capacity(wd.triangles.len());
    let mut aabb_min = [f32::INFINITY; 3];
    let mut aabb_max = [f32::NEG_INFINITY; 3];

    for tri in &wd.triangles {
        let cp = &tri.control_points;
        // Sanity-check indices.
        let max_reg = cp[0].regular_idx.max(cp[1].regular_idx).max(cp[2].regular_idx) as usize;
        let max_water = cp[0].water_idx.max(cp[1].water_idx).max(cp[2].water_idx) as usize;
        if max_reg >= mesh.vertices.len() || max_water >= wd.vertices.len() {
            continue;
        }

        let v0 = &mesh.vertices[cp[0].regular_idx as usize];
        let v1 = &mesh.vertices[cp[1].regular_idx as usize];
        let v2 = &mesh.vertices[cp[2].regular_idx as usize];

        // Track AABB across all control-point positions.
        for v in [v0, v1, v2] {
            for i in 0..3 {
                let p = [v.position.x, v.position.y, v.position.z][i];
                if p < aabb_min[i] { aabb_min[i] = p; }
                if p > aabb_max[i] { aabb_max[i] = p; }
            }
        }

        let a0 = &wd.vertices[cp[0].water_idx as usize];
        let a1 = &wd.vertices[cp[1].water_idx as usize];
        let a2 = &wd.vertices[cp[2].water_idx as usize];

        // Stream 0 — control-point packing. The HLSL packs three CP's
        // worth of position+tex+tan+bin+lm into 10 attributes (8 vec4
        // POSITION + 1 vec4 TEXCOORD0 + 1 vec3 TEXCOORD1) per the
        // engine's interleave; field order matches
        // `water_fx.hlsl:188-208`.
        control_points.push(WaterControlPointInstance {
            // POSITION0: pos1.xyz, tc1.x
            pos1_tc1x: [v0.position.x, v0.position.y, v0.position.z, v0.texcoord.x],
            // POSITION1: tc1.y, tan1.xyz
            tc1y_tan1: [v0.texcoord.y, v0.tangent.i, v0.tangent.j, v0.tangent.k],
            // POSITION2: bin1.xyz, lm1.x
            bin1_lm1x: [v0.binormal.i, v0.binormal.j, v0.binormal.k, v0.lightmap_texcoord.x],
            // POSITION3: lm1.y, pos2.xyz
            lm1y_pos2: [v0.lightmap_texcoord.y, v1.position.x, v1.position.y, v1.position.z],
            // POSITION4: tc2.xy, tan2.xy
            tc2_tan2xy: [v1.texcoord.x, v1.texcoord.y, v1.tangent.i, v1.tangent.j],
            // POSITION5: tan2.z, bin2.xyz
            tan2z_bin2: [v1.tangent.k, v1.binormal.i, v1.binormal.j, v1.binormal.k],
            // POSITION6: lm2.xy, pos3.xy
            lm2_pos3xy: [v1.lightmap_texcoord.x, v1.lightmap_texcoord.y, v2.position.x, v2.position.y],
            // POSITION7: pos3.z, tc3.xy, tan3.x
            pos3z_tc3_tan3x: [v2.position.z, v2.texcoord.x, v2.texcoord.y, v2.tangent.i],
            // TEXCOORD0: tan3.yz, bin3.xy
            tan3yz_bin3xy: [v2.tangent.j, v2.tangent.k, v2.binormal.i, v2.binormal.j],
            // TEXCOORD1: bin3.z, lm3.xy
            bin3z_lm3: [v2.binormal.k, v2.lightmap_texcoord.x, v2.lightmap_texcoord.y],
        });

        // Stream 2 — lighting packing. NORMAL0..4 carry three CP's
        // worth of local_info (li) + base_texcoord (bt).
        lighting_attrs.push(WaterLightingInstance {
            // NORMAL0: li1.xyz, bt1.x
            li1_bt1x: [a0.local_info.x, a0.local_info.y, a0.local_info.z, a0.base_texcoord.x],
            // NORMAL1: bt1.yz, li2.xy
            bt1yz_li2xy: [a0.base_texcoord.y, a0.base_texcoord.z, a1.local_info.x, a1.local_info.y],
            // NORMAL2: li2.z, bt2.xyz
            li2z_bt2: [a1.local_info.z, a1.base_texcoord.x, a1.base_texcoord.y, a1.base_texcoord.z],
            // NORMAL3: li3.xyz, bt3.x
            li3_bt3x: [a2.local_info.x, a2.local_info.y, a2.local_info.z, a2.base_texcoord.x],
            // NORMAL4: bt3.yz
            bt3yz: [a2.base_texcoord.y, a2.base_texcoord.z],
        });
    }

    if control_points.is_empty() {
        return None;
    }

    // Build per-part ranges from the blam-tags walker's per-part
    // triangle ranges. Each part's `mesh_part_index` resolves to a
    // `RenderMeshPart.material_index` (the rmw material in the parent
    // BSP material list).
    let parts: Vec<WaterMeshPartRange> = wd
        .parts
        .iter()
        .filter_map(|wp| {
            let mp = mesh.parts.get(wp.mesh_part_index as usize)?;
            Some(WaterMeshPartRange {
                material_index: mp.material_index,
                instance_start: wp.triangle_start,
                instance_count: wp.triangle_count,
            })
        })
        .collect();

    let control_points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bsp_water_control_points"),
        contents: bytemuck::cast_slice(&control_points),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let lighting_attrs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bsp_water_lighting_attrs"),
        contents: bytemuck::cast_slice(&lighting_attrs),
        usage: wgpu::BufferUsages::VERTEX,
    });
    Some(WaterMeshBuffers {
        control_points: control_points_buffer,
        lighting_attrs: lighting_attrs_buffer,
        instance_count: control_points.len() as u32,
        parts,
        aabb_min,
        aabb_max,
    })
}

