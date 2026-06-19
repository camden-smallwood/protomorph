//! Mirror of `Ares/source/render/render_objects.{h,cpp}` (151 lines
//! / 1284 lines) + dllcache c_object_renderer bodies.
//!
//! `c_object_renderer` is a static-method namespace class — Halo
//! Pattern A. Owns the per-frame `s_render_object_globals` (417KB
//! struct of fixed arrays). Three responsibilities:
//!
//!   1. submit_visibility — populate object_render_contexts[] from
//!      visibility data (currently the legacy `render_list` Vec until
//!      F12's c_visible_items walker is wired).
//!   2. render_object_contexts(entry_point, mesh_part_mask) — walk
//!      contexts in marker range, dispatch each via
//!      render_object_context.
//!   3. render_albedo / render_static_lighting / render_lights /
//!      render_shadows_generate / render_shadows_apply — pass
//!      orchestrators that set rasterizer state then call
//!      render_object_contexts with the appropriate entry_point.
//!
//! Halo's marker stack (`marker_index`,
//! `object_render_context_starting_index[6]`) lets reflection /
//! lightmap_shadow / occlusion sub-views push their own context
//! ranges without disturbing the primary view. v1 ships a single-
//! marker version; multi-marker support lands when sub-views
//! actually use it.

use crate::halo::objects::ObjectIndex;
use crate::halo::render::render_method_data::CRenderMethodData;
use crate::halo::render::shared::{FrameContext, SharedResources};
use crate::halo::render::GpuModel;
use crate::halo::render_methods::HaloEntryPoint;
use crate::halo::visibility::LodTransparency;

// ---------------------------------------------------------------------------
// Constants — pool sizes from Halo
// ---------------------------------------------------------------------------

/// `g_render_object_globals.object_render_contexts` is a
/// `c_static_array<s_object_render_context, 1024>`. Halo enforces
/// the 1024 cap via assert.
pub const K_MAX_OBJECT_RENDER_CONTEXTS: usize = 1024;

/// `g_render_object_globals.context_mesh_parts` is a
/// `c_static_array<s_context_mesh_part, 8192>`.
pub const K_MAX_CONTEXT_MESH_PARTS: usize = 8192;

/// `g_render_object_globals.object_render_context_starting_index[6]` —
/// 6-deep marker stack.
pub const K_MAX_OBJECT_MARKERS: usize = 6;

// ---------------------------------------------------------------------------
// e_entry_point (rasterizer_constants.h:9-30) — full Halo set
// ---------------------------------------------------------------------------

/// `e_entry_point` (rasterizer_constants.h:9-30). Full Halo entry-
/// point enum. Render passes choose which one to dispatch via
/// `c_object_renderer::render_object_contexts(entry_point, ...)`.
///
/// Our material-side `HaloEntryPoint` (in `render_methods/`) only
/// has the two we've ported (StaticSh, StaticPerPixel); the
/// full enum here is the canonical surface so `render_albedo`
/// etc. can declare what they pass without lying about our subset.
/// `to_material_entry_point()` maps the canonical → our subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum EntryPoint {
    Default = 0,
    Albedo = 1,
    StaticLightingDefault = 2,
    StaticLightingPerPixel = 3,
    StaticLightingPerVertex = 4,
    StaticLightingSh = 5,
    StaticLightingPrtAmbient = 6,
    StaticLightingPrtLinear = 7,
    StaticLightingPrtQuadratic = 8,
    DynamicLighting = 9,
    ShadowGenerate = 10,
    ShadowApply = 11,
    ActiveCamo = 12,
    LightmapDebugMode = 13,
    VertexColorLighting = 14,
    WaterTessellation = 15,
    WaterShading = 16,
    DynamicLightingCinematic = 17,
}

// ---------------------------------------------------------------------------
// e_render_object_mesh_part_flags (render_objects.h:14-28)
// ---------------------------------------------------------------------------

/// `e_render_object_mesh_part_flags` — the mesh-part flag bits the
/// `mesh_part_mask` argument tests against. Halo uses bit-and to
/// filter parts per-pass (e.g. shadow pass tests `shadow_casting`,
/// transparent pass tests `transparent`, etc.).
pub mod render_object_mesh_part_flags {
    pub const LIT: u32 = 1 << 0;
    pub const DECAL: u32 = 1 << 5;
}

// ---------------------------------------------------------------------------
// e_submit_visibility_flags (render_objects.h:30-37)
// ---------------------------------------------------------------------------

pub mod submit_visibility_flags {}

// ---------------------------------------------------------------------------
// s_object_render_context (render_objects.h:69-77, 24B)
// ---------------------------------------------------------------------------

/// `s_object_render_context` (render_objects.h:69-77, 24B).
///
/// Halo packs `first_context_mesh_part_index:16 |
/// mesh_part_count:15 | use_special_alpha_constant:1` into a single
/// u32 (offset 0x0). Rust idiom = three named fields.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectRenderContext {
    pub first_context_mesh_part_index: u16,    // 0x0  bits 0-15
    pub mesh_part_count: u16,                  // 0x0  bits 16-30 (15 bits)
    pub use_special_alpha_constant: bool,      // 0x0  bit 31
    pub lod_transparency: LodTransparency,     // 0x4  (4B)
    /// Halo: `render_lighting *lighting` — pointer into the
    /// per-frame lighting pool. We store the pool index instead.
    /// `-1` means no per-object lighting (fall back to default).
    pub lighting_index: i32,                   // 0x8 (was ptr)
    /// Halo: `s_shader_extern_info *shader_extern_info` — pointer
    /// into the per-frame shader-extern pool. We store the pool
    /// index. `-1` means no per-object shader externs.
    pub shader_extern_info_index: i32,         // 0x10 (was ptr)
}

// ---------------------------------------------------------------------------
// s_context_mesh_part (render_objects.h:80-90, 48B)
// ---------------------------------------------------------------------------

/// `s_context_mesh_part` (render_objects.h:80-90, 48B).
#[derive(Debug, Clone, Copy, Default)]
pub struct ContextMeshPart {
    pub flags: u32,                             // 0x0
    /// Halo: `s_render_object_info *info` (pointer). We store an
    /// index into the per-frame `s_render_object_info` pool.
    pub info_index: i32,                        // 0x8 (was ptr)
    /// Halo: `unsigned long *subpart_bitvector` (pointer). We store
    /// an index into a per-frame bitvector pool.
    pub subpart_bitvector_index: i32,           // 0x10 (was ptr)
    pub part_index: u16,                        // 0x18
    pub mesh_index: u16,                        // 0x1A
    pub region_index: u16,                      // 0x1C
    /// Padding 0x1E-0x1F (alignment to 0x20).
    pub _pad: u16,
    pub render_method_template_index: i32,      // 0x20
    /// Halo: `s_shield_impact_parameters const *shield_impact_parameters`.
    /// `-1` for no shield impact.
    pub shield_impact_parameters_index: i32,    // 0x28 (was ptr)
}

// ---------------------------------------------------------------------------
// s_render_object_globals (render_objects.cpp embedded, 417864B)
// ---------------------------------------------------------------------------

/// `s_render_object_globals` (render_objects.cpp:embedded, 417864B
/// in Halo). Storage for per-frame contexts + mesh parts + marker
/// stack.
///
/// Halo uses `c_static_array<T, N>` fixed arrays for cache
/// locality on Xbox 360 (where dynamic alloc per-frame is
/// expensive). PC we use `Vec` since we don't have the same
/// preallocation pressure.
#[derive(Debug, Default)]
pub struct RenderObjectGlobals {
    pub object_render_contexts: Vec<ObjectRenderContext>,
    pub object_render_context_count: i32,
    pub context_mesh_parts: Vec<ContextMeshPart>,
    pub total_context_mesh_part_count: i32,
    pub currently_adding_to_context: bool,
    pub marker_index: i32,
    pub object_render_context_starting_index: [i32; K_MAX_OBJECT_MARKERS],
    pub context_mesh_part_starting_index: [i32; K_MAX_OBJECT_MARKERS],
}

// ---------------------------------------------------------------------------
// c_object_renderer
// ---------------------------------------------------------------------------

/// `c_object_renderer` (render_objects.h:93-132). Halo-style static
/// namespace class — owned by the Renderer (Pattern C).
#[derive(Debug, Default)]
pub struct ObjectRenderer {
    pub state: RenderObjectGlobals,
}

impl ObjectRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `c_object_renderer::reset @ 0x1806e1670`. Clears all per-frame
    /// state.
    pub fn reset(&mut self) {
        self.state.object_render_contexts.clear();
        self.state.object_render_context_count = 0;
        self.state.context_mesh_parts.clear();
        self.state.total_context_mesh_part_count = 0;
        self.state.currently_adding_to_context = false;
        self.state.marker_index = -1;
        self.state.object_render_context_starting_index = [0; K_MAX_OBJECT_MARKERS];
        self.state.context_mesh_part_starting_index = [0; K_MAX_OBJECT_MARKERS];
    }

    /// `c_object_renderer::push_marker @ 0x1806e17b0`. Records the
    /// current context + mesh-part counts so `pop_marker` can
    /// truncate back to them.
    pub fn push_marker(&mut self) {
        self.state.marker_index += 1;
        let idx = self.state.marker_index as usize;
        assert!(idx < K_MAX_OBJECT_MARKERS, "object marker stack overflow");
        self.state.object_render_context_starting_index[idx] =
            self.state.object_render_context_count;
        self.state.context_mesh_part_starting_index[idx] =
            self.state.total_context_mesh_part_count;
    }

    /// `c_object_renderer::begin_object_render_context
    /// @ 0x1806e18f0`. Returns the new context index. Halo asserts
    /// `!currently_adding_to_context` and the 1024 cap.
    pub fn begin_object_render_context(&mut self) -> i32 {
        assert!(
            !self.state.currently_adding_to_context,
            "begin_object_render_context: already adding"
        );
        assert!(
            self.state.object_render_context_count < K_MAX_OBJECT_RENDER_CONTEXTS as i32,
            "object_render_contexts full"
        );
        let idx = self.state.object_render_context_count;
        self.state.object_render_contexts.push(ObjectRenderContext {
            first_context_mesh_part_index: self.state.total_context_mesh_part_count as u16,
            mesh_part_count: 0,
            use_special_alpha_constant: false,
            lod_transparency: LodTransparency::OPAQUE,
            lighting_index: -1,
            shader_extern_info_index: -1,
        });
        self.state.object_render_context_count += 1;
        self.state.currently_adding_to_context = true;
        idx
    }

    /// `c_object_renderer::end_object_render_context @ 0x1806e19c0`.
    pub fn end_object_render_context(&mut self) {
        assert!(
            self.state.currently_adding_to_context,
            "end_object_render_context: not currently adding"
        );
        self.state.currently_adding_to_context = false;
    }

    /// Append a mesh-part to the in-progress context.
    /// Mirrors `add_context_mesh_part @ 0x1806e1a20`. Returns the
    /// added mesh-part index.
    pub fn add_context_mesh_part(
        &mut self,
        flags: u32,
        info_index: i32,
        mesh_index: u16,
        part_index: u16,
        region_index: u16,
        render_method_template_index: i32,
    ) -> i32 {
        assert!(
            self.state.currently_adding_to_context,
            "add_context_mesh_part: no active context"
        );
        assert!(
            self.state.total_context_mesh_part_count < K_MAX_CONTEXT_MESH_PARTS as i32,
            "context_mesh_parts full"
        );
        let idx = self.state.total_context_mesh_part_count;
        self.state.context_mesh_parts.push(ContextMeshPart {
            flags,
            info_index,
            subpart_bitvector_index: -1,
            part_index,
            mesh_index,
            region_index,
            _pad: 0,
            render_method_template_index,
            shield_impact_parameters_index: -1,
        });
        self.state.total_context_mesh_part_count += 1;

        // Bump the active context's count.
        if let Some(ctx) = self.state.object_render_contexts.last_mut() {
            ctx.mesh_part_count += 1;
        }
        idx
    }

    /// `c_object_renderer::render_object_contexts(entry_point,
    /// mesh_part_mask) @ 0x1806e33b0`.
    ///
    /// Walks contexts from the current marker's starting index to
    /// `object_render_context_count`. For each, walks its mesh
    /// parts and dispatches `render_object_context_mesh_part` when
    /// `(part.flags & mesh_part_mask) != 0`.
    ///
    /// State (mirrors dllcache):
    ///   - PIX events around the iteration
    ///   - `c_lighting_interface::setup_object_lighting_for_entry_point`
    ///     per context (sets per-object SH cbuffer)
    ///   - `rasterizer_stipple_set_fade_byte` per context for LOD
    ///     dissolve (we skip — no stipple yet)
    ///
    /// v1: simplified pass-through that calls a draw-callback per
    /// mesh-part. The callback (`draw_mesh_part`) lives on
    /// `Renderer` since it needs encoder + frame_ctx + bsps + models.
    pub fn render_object_contexts(
        &self,
        entry_point: EntryPoint,
        mesh_part_mask: u32,
        mut draw_mesh_part: impl FnMut(usize, &ContextMeshPart, EntryPoint, u32, &ObjectRenderContext),
    ) {
        let start = if self.state.marker_index >= 0 {
            self.state.object_render_context_starting_index
                [self.state.marker_index as usize] as usize
        } else {
            0
        };
        let end = self.state.object_render_context_count as usize;
        for context_index in start..end {
            let ctx = &self.state.object_render_contexts[context_index];
            let first = ctx.first_context_mesh_part_index as usize;
            let count = ctx.mesh_part_count as usize;
            for i in first..(first + count) {
                let mesh_part = &self.state.context_mesh_parts[i];
                if (mesh_part.flags & mesh_part_mask) != 0 {
                    draw_mesh_part(context_index, mesh_part, entry_point, mesh_part_mask, ctx);
                }
            }
        }
    }

    /// `c_object_renderer::render_albedo @ 0x1806e49e0`. Sets render
    /// state + dispatches contexts with `_entry_point_albedo`.
    ///
    /// Halo state:
    ///   - z_buffer_mode = write
    ///   - cull_mode = cw
    ///   - color_write_enable(0/1, 15) — both MRTs full
    ///   - begin/end_render_mesh_part_only (PIX scope)
    ///
    /// v1: state is implicit in our render-pass setup (we already
    /// open the lighting_base + depth render pass with the right
    /// Operations); just walks the contexts.
    pub fn render_albedo(
        &self,
        mesh_part_flags: u32,
        draw: impl FnMut(usize, &ContextMeshPart, EntryPoint, u32, &ObjectRenderContext),
    ) {
        self.render_object_contexts(EntryPoint::Albedo, mesh_part_flags, draw);
    }

    /// `c_object_renderer::render_static_lighting @ 0x1806e4a80`.
    /// Halo dispatches with
    /// `_entry_point_static_lighting_prt_quadratic` (Halo's default
    /// for static lighting; PRT quadratic is the highest-quality
    /// path).
    pub fn render_static_lighting(
        &self,
        mesh_part_flags: u32,
        draw: impl FnMut(usize, &ContextMeshPart, EntryPoint, u32, &ObjectRenderContext),
    ) {
        self.render_object_contexts(
            EntryPoint::StaticLightingPrtQuadratic,
            mesh_part_flags,
            draw,
        );
    }

    /// `c_object_renderer::render_albedo_decals @ 0x1806E4A60` — engine
    /// body is one line:
    /// ```c
    /// render_object_contexts(_entry_point_default, 32);
    /// ```
    /// where `32 = 1 << _render_object_mesh_part_decal_bit` filters to
    /// only mesh parts authored as "decal-style geometry" (sticker
    /// overlays on weapons, blood splats on bipeds, sensor textures
    /// on machinery, etc.). Dispatched from `c_player_view::render_albedo
    /// @ 0x18068ad36` AFTER sky and BEFORE `c_decal_system::render_all`,
    /// under `_z_buffer_mode_read` (depth test only, no write — see
    /// `feedback_decal_no_depth_write`).
    ///
    /// **Note:** `submit_visibility_from_render_list` does not yet OR
    /// the DECAL bit into per-part `flags` (we only translate
    /// `e_geometry_part_type` via `part_type_to_flags` which maps to
    /// the STRUCTURE flag set — LIT/SHADOW_CASTING/TRANSPARENT, no
    /// DECAL). Engine sets the DECAL bit deeper in
    /// `submit_object_hierarchy_recursive` based on per-mesh-part
    /// authoring that we haven't surfaced yet. Until that lands, this
    /// dispatch is a no-op — included so the call site exists in the
    /// engine-faithful order and so the visibility-side gap is the
    /// only remaining work for object decals.
    pub fn render_albedo_decals(
        &self,
        draw: impl FnMut(usize, &ContextMeshPart, EntryPoint, u32, &ObjectRenderContext),
    ) {
        self.render_object_contexts(
            EntryPoint::Default,
            render_object_mesh_part_flags::DECAL,
            draw,
        );
    }

}

/// Submit visibility — populate object_render_contexts from the
/// renderer's `render_list`. Mirrors
/// `c_object_renderer::submit_visibility @ 0x1806e16d0` in shape;
/// the body is simpler since we don't have the full
/// c_visible_items walker yet (one context per object, one
/// mesh-part per (mesh, part) pair).
///
/// Sits as a free function so the caller can borrow the renderer's
/// model + render_list without juggling lifetimes through
/// `&mut ObjectRenderer`.
pub fn submit_visibility_from_render_list(
    renderer: &mut ObjectRenderer,
    transparency: &mut crate::halo::render::render_transparents::TransparencyRenderer,
    render_list: &[(ObjectIndex, usize)],
    models: &[GpuModel],
    objects: &crate::halo::objects::ObjectStore,
) {
    use crate::halo::render::render_transparents::{
        TransparentDispatch, TransparentSortLayer,
    };
    use crate::halo::render::structure_renderer::part_type_to_flags;
    renderer.reset();
    renderer.push_marker();

    for (object_slot, &(obj_idx, model_idx)) in render_list.iter().enumerate() {
        let Some(model) = models.get(model_idx) else { continue };
        // Object world transform — MUST be the same transform the render path
        // uses: `obj.header_index` → datum world matrix (NOT the object-STORE
        // index `obj_idx.0`). `world_matrix()` is keyed by the header/datum
        // index, where parent-marker attachment wrote the transform (e.g. the
        // sky-marker-attached waterfall). Passing the store index read a
        // DIFFERENT object's datum, so the sort centroid disagreed with the
        // rendered geometry — the waterfall's centroid landed at ~[-178] while
        // its geometry rendered ~200 units away, making it flip in front of the
        // glass. Mirror `render_player_view`'s model-uniform world exactly.
        let obj = objects.get(obj_idx);
        let obj_world = Some(
            obj.header_index
                .and_then(crate::halo::objects::object_header_data::world_matrix)
                .unwrap_or_else(|| obj.model_matrix()),
        );
        renderer.begin_object_render_context();
        for (mesh_idx, mesh) in model.meshes.iter().enumerate() {
            for (part_idx, part) in mesh.parts.iter().enumerate() {
                // Engine-faithful per-part flags from `e_geometry_part_type`.
                // `c_object_renderer::submit_visibility` does the same mapping
                // — opaque_shadow_casting=2 → LIT|SHADOW_CASTING; transparent=4
                // and lightmap_only=5 → 0 (filtered out by the mask gate in
                // `render_object_contexts`); transparent parts route through
                // `c_transparency_renderer` instead.
                let mat = model.materials.get(part.material_index);
                let material_is_transparent =
                    mat.map(|m| m.is_transparent).unwrap_or(false);
                let material_is_decal_object =
                    mat.map(|m| m.artifacts_decal_object.is_some()).unwrap_or(false);

                // Engine `c_object_renderer::submit_object_mesh_parts @
                // 0x1806E1BF0`:
                //   is_decal = c_render_method::get_is_decal(material);
                //   if (is_decal)  flags = (flags & ~LIT) | DECAL;
                // → decal mesh parts dispatch ONLY through
                // `render_albedo_decals` (DECAL mask), not SL or
                // transparency.
                let flags = if material_is_decal_object {
                    render_object_mesh_part_flags::DECAL
                } else {
                    part_type_to_flags(part.part_type)
                };

                // Engine: a part routes to the transparency pool if EITHER
                // `part_type==4 (transparent)` OR the material's blend_mode
                // is non-opaque. Some FX scenery (waterfalls, man cannons,
                // tower pulses) ship parts marked `opaque_shadow_casting`
                // but with an alpha-blend rmsh material — register them
                // here so the transparency renderer dispatches them.
                if flags == 0 || material_is_transparent {
                    if material_is_transparent {
                        // Engine `submit_object_mesh_parts @ 0x1806E1BF0`:
                        //   origin = global_origin3d;                       // CONSTANT
                        //   if (sorting_index != -1 && !(mesh.flags & 2))   // 2=UseRegionIndexForSorting
                        //       origin = sorting_position.position;
                        //   ... offset = region_index * 0.1;
                        // So meshes flagged UseRegionIndexForSorting (or lacking
                        // a sorting position) collapse to a CONSTANT centroid
                        // (global_origin3d → the object origin after the region
                        // matrix) and order by region_index*0.1, while others use
                        // their authored sorting position. This holds co-located
                        // transparent layers (waterfall liquid/sheet/mist) in a
                        // FIXED composite order instead of flipping as their
                        // per-mesh centroids cross under camera motion.
                        // `[0,0,0]` (model space) is global_origin3d → maps to the
                        // object origin via `obj_world`.
                        let model_centroid = if !mesh.use_region_index_for_sorting {
                            part.sort_centroid.unwrap_or([0.0, 0.0, 0.0])
                        } else {
                            [0.0, 0.0, 0.0]
                        };
                        let world_centroid = match obj_world {
                            Some(m) => m
                                .transform_point3(glam::Vec3::from(model_centroid))
                                .to_array(),
                            None => model_centroid,
                        };
                        // Region tiebreak that orders co-located layers. Engine
                        // value is `+region_index * 0.1`, but our transparent
                        // sort resolves the polarity REVERSED (same compensation
                        // the sky region-order needs, see
                        // reference_transparent_sort_engine): NEGATIVE offset
                        // makes the LOWER region draw on top. The waterfall's
                        // regions are authored front-to-back (mist=0 front …
                        // liquid=2 back), so `-region*0.1` keeps the frothy
                        // mist/sheet composited OVER the liquid. (Root polarity
                        // convention vs engine tracked for the Phase-2 sort pass.)
                        let region_offset = -(mesh.region_index as f32) * 0.1;
                        transparency.add_element(
                            world_centroid,
                            None,
                            0.0,
                            region_offset,
                            mat.map(|m| m.sort_layer)
                                .unwrap_or(TransparentSortLayer::Normal),
                            TransparentDispatch::Object {
                                object_slot: object_slot as u32,
                                mesh_index: mesh_idx as u16,
                                part_index: part_idx as u16,
                                probe_index: obj.cubemap_probe_index,
                            },
                        );
                    }
                    continue;
                }
                renderer.add_context_mesh_part(
                    flags,
                    -1,
                    mesh_idx as u16,
                    part_idx as u16,
                    0,
                    -1,
                );
            }
        }
        renderer.end_object_render_context();
    }
}

/// Per-mesh-part draw — what the `draw_mesh_part` callback does in
/// `render_object_contexts`. Mirrors the legacy `draw_models` body
/// but operating on a single ContextMeshPart at a time.
///
/// Bound to a specific `(rpass, frame_ctx, render_list, models,
/// shared)` tuple via closure capture at call sites.
///
/// Cross-call state for `record_mesh_part_draw` to elide redundant
/// per-object bind-group calls. wgpu doesn't elide redundant
/// `set_bind_group` (gpuweb#116); the engine's D3D11 path has
/// `g_texture_dirty` bitmasks + driver state-filter doing it twice.
/// We mirror the engine's first layer here.
#[derive(Default)]
pub struct ObjectPassBindings {
    /// Last-bound dynamic offset for camera_bind_group (= per-object
    /// engine_lighting offset). `None` = first call in pass.
    last_lighting_offset: Option<u32>,
    /// Last-bound dynamic offset for model_bind_group (per-object
    /// `object_slot * model_stride`).
    last_model_offset: Option<u32>,
    /// Last-bound dynamic offset for node_matrices_bind_group.
    last_nm_offset: Option<u32>,
    /// Last camera bind group (albedo vs SL has different shared
    /// bg). Pointer comparison.
    last_camera_bg: Option<*const wgpu::BindGroup>,
}

impl ObjectPassBindings {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ObjectRenderer {
    pub fn record_mesh_part_draw<'a>(
        &self,
        rpass: &mut wgpu::RenderPass<'a>,
        shared: &'a SharedResources,
        ctx: &'a FrameContext<'a>,
        mesh_part: &ContextMeshPart,
        object_slot: usize,
        entry_point: HaloEntryPoint,
        bindings: &mut ObjectPassBindings,
    ) {
        let Some(&(obj_idx, model_idx)) = ctx.render_list.get(object_slot) else { return };
        let Some(gpu_model) = ctx.models.get(model_idx) else { return };
        let Some(mesh) = gpu_model.meshes.get(mesh_part.mesh_index as usize) else { return };
        let Some(part) = mesh.parts.get(mesh_part.part_index as usize) else { return };

        let model_offset = (object_slot * shared.model_stride) as u32;
        let nm_offset = (object_slot * shared.node_matrices_stride) as u32;

        // L4 — per-object engine_lighting bind. Engine equivalent:
        // `c_lighting_interface::setup_object_lighting_for_entry_point @
        // 0x1806A9AA0` calls `calculate_and_set_ravi_constants(sh, scale)`
        // and `set_object_dominant_light_direction_and_intensity(...)` to
        // write the per-object cbuffer. We pre-bake the resolved sample
        // at scenario load (see `Renderer::bake_scenery_lighting_offsets`)
        // and switch via dynamic offset here.
        let lighting_offset = ctx.game.objects.get(obj_idx).engine_lighting_offset
            .unwrap_or(crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET);
        // Diag: PROTOMORPH_DIAG_DRAW_OFFSET=<offset> matches a specific
        // dynamic-offset value (read from bake-time log) and prints
        // EVERY draw call that uses it. Lets us confirm the bake-time
        // slot is the one actually bound for the placement we're
        // tracking. Cached: only fires when set, no allocation on hot path.
        {
            use std::sync::OnceLock;
            static FILTER: OnceLock<Option<u32>> = OnceLock::new();
            let target = FILTER.get_or_init(|| {
                std::env::var("PROTOMORPH_DIAG_DRAW_OFFSET").ok().and_then(|s| {
                    let t = s.trim();
                    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                        u32::from_str_radix(hex, 16).ok()
                    } else {
                        t.parse::<u32>().ok()
                    }
                })
            });
            if let Some(t) = *target {
                if lighting_offset == t {
                    eprintln!(
                        "[diag-draw] entry={entry_point:?} obj_idx={obj_idx:?} model_idx={model_idx} mesh={} part={} lighting_offset=0x{lighting_offset:x}",
                        mesh_part.mesh_index, mesh_part.part_index,
                    );
                }
            }
            // PROTOMORPH_DIAG_DRAW_OBJ_IDX=<n>: one-shot log of EVERY
            // draw per unique (entry, model, mesh, part) tuple for the
            // matching obj_idx. Diff target for "is my object actually
            // being drawn" + "with what lighting_offset" investigations.
            if let Some(t) = std::env::var("PROTOMORPH_DIAG_DRAW_OBJ_IDX").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
                if obj_idx.0 == t {
                    use std::sync::Mutex;
                    use std::sync::OnceLock;
                    use std::collections::HashSet;
                    static SEEN: OnceLock<Mutex<HashSet<(u32, u32, u32, u32)>>> = OnceLock::new();
                    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
                    let key = (entry_point as u32, model_idx as u32, mesh_part.mesh_index as u32, mesh_part.part_index as u32);
                    let mut guard = seen.lock().unwrap();
                    if guard.insert(key) {
                        eprintln!(
                            "[diag-draw-obj] entry={entry_point:?} obj_idx={} model_idx={model_idx} mesh={} part={} lighting_offset=0x{lighting_offset:x}",
                            obj_idx.0, mesh_part.mesh_index, mesh_part.part_index,
                        );
                    }
                }
            }
        }
        // DecalObject runs INSIDE `render_albedo` (writes
        // `_surface_accum_HDR` + normal G-buffer as RT0/RT1), so it
        // MUST use `camera_bind_group` (placeholder views at slots
        // 10/11). Binding `camera_bind_group_sl` here would source the
        // real `_surface_accum_HDR` view as a RESOURCE while the same
        // render pass holds it as COLOR_TARGET — wgpu validation
        // RESOURCE/COLOR_TARGET conflict.
        let camera_bg = match entry_point {
            HaloEntryPoint::Albedo | HaloEntryPoint::DecalObject => &shared.camera_bind_group,
            _ => &shared.camera_bind_group_sl,
        };
        // Dirty-tracked: skip set_bind_group(0/1/3) when offsets +
        // camera_bg pointer haven't changed since the last
        // record_mesh_part_draw within this pass. Most parts of the
        // same object share these — engine's `commit_state` does the
        // same dirty-bit elision.
        let camera_bg_ptr = camera_bg as *const _;
        if bindings.last_lighting_offset != Some(lighting_offset)
            || bindings.last_camera_bg != Some(camera_bg_ptr)
        {
            rpass.set_bind_group(
                0,
                camera_bg,
                &[
                    lighting_offset,
                    // atmosphere @ binding 2 — objects use the cluster default
                    // (object rmsh shaders don't author per-shader fog overrides;
                    // the override is sky/effect-side). The sky pass below
                    // rebinds with each sky material's resolved offset.
                    crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                    crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                    // dominant_light @ binding 13 mirrors `lighting_offset`.
                    lighting_offset,
                ],
            );
            bindings.last_lighting_offset = Some(lighting_offset);
            bindings.last_camera_bg = Some(camera_bg_ptr);
        }
        if bindings.last_model_offset != Some(model_offset) {
            rpass.set_bind_group(1, &shared.model_bind_group, &[model_offset]);
            bindings.last_model_offset = Some(model_offset);
        }
        if bindings.last_nm_offset != Some(nm_offset) {
            rpass.set_bind_group(3, &shared.node_matrices_bind_group, &[nm_offset]);
            bindings.last_nm_offset = Some(nm_offset);
        }

        // Per-part filtering happens at submit_visibility time
        // (`submit_visibility_from_render_list` skips parts with
        // `part_type_to_flags(part.part_type) == 0`). Render-time only
        // hits parts the engine-faithful classifier already approved.
        // Per-mesh PRT activation, mirroring `render_mesh_part_default
        // @ 0x18069EBC0`'s engine remap. When the mesh carries a PRT
        // Ambient transfer stream AND the material's PRT pipeline
        // compiled (rmsh-family only), upgrade the lit-pass entry to
        // `StaticPrtAmbient` and bind the per-mesh secondary VB. Albedo
        // pass is unchanged — PRT only affects lighting evaluation.
        let mut bind_prt_stream = false;
        if let Some(material) = gpu_model.materials.get(part.material_index) {
            let use_prt = !matches!(entry_point, HaloEntryPoint::Albedo | HaloEntryPoint::DecalObject)
                && mesh.prt_ambient_buffer.is_some()
                && material.artifacts_prt_ambient.is_some()
                && material.bind_group_prt_ambient.is_some();
            // Entry-point-specific resolution. DecalObject requires the
            // dedicated rmd-shaped artifacts + a dynamic offset for the
            // `decal_constants` UBO; Albedo / SL paths use the static
            // bind group with no dynamic offset.
            enum BindPath<'a> {
                NoOffset(&'a wgpu::BindGroup),
                DecalObject(&'a wgpu::BindGroup),
            }
            // Helper: compute the per-instance cbuffer slot offset.
            // Path B: `slot_size = align_up(layout.total_size, 256)`;
            // offset = `instance_within_model * slot_size`. Static
            // materials always read slot 0 (per-frame writer only
            // updates slot 0); animated-per-object materials in Phase 5
            // will write each slot with that instance's `ObjectBoundContext`.
            let cbuffer_slot_offset = |layout: &crate::halo::render_methods::cbuffer::CbufferLayout| -> u32 {
                let slot_size = (layout.total_size as usize)
                    .next_multiple_of(crate::halo::render_methods::pipeline_cache::MATERIAL_SLOT_ALIGNMENT)
                    as u32;
                let instance = ctx.game.objects.get(obj_idx).instance_within_model;
                instance * slot_size
            };
            let (pipeline, bind_path, cbuffer_offset) = match entry_point {
                HaloEntryPoint::Albedo => match (
                    material.artifacts_albedo.as_ref(),
                    material.bind_group_albedo.as_ref(),
                ) {
                    (Some(a), Some(b)) => (
                        &a.pipeline,
                        BindPath::NoOffset(b),
                        cbuffer_slot_offset(&a.layout),
                    ),
                    _ => (
                        &material.artifacts.pipeline,
                        BindPath::NoOffset(&material.bind_group),
                        cbuffer_slot_offset(&material.artifacts.layout),
                    ),
                },
                HaloEntryPoint::DecalObject => match (
                    material.artifacts_decal_object.as_ref(),
                    material.bind_group_decal_object.as_ref(),
                ) {
                    (Some(a), Some(b)) => (
                        &a.pipeline,
                        BindPath::DecalObject(b),
                        cbuffer_slot_offset(&a.layout),
                    ),
                    // Material isn't rmd-on-object — skip the draw
                    // rather than dispatch with the wrong pipeline.
                    _ => return,
                },
                _ if use_prt => {
                    bind_prt_stream = true;
                    let arts = material.artifacts_prt_ambient.as_ref().unwrap();
                    (
                        &arts.pipeline,
                        BindPath::NoOffset(material.bind_group_prt_ambient.as_ref().unwrap()),
                        cbuffer_slot_offset(&arts.layout),
                    )
                }
                _ => (
                    &material.artifacts.pipeline,
                    BindPath::NoOffset(&material.bind_group),
                    cbuffer_slot_offset(&material.artifacts.layout),
                ),
            };
            rpass.set_pipeline(pipeline);
            match bind_path {
                // Path B: cbuffer slot dynamic-offset. `cbuffer_offset`
                // points at this placement's slot in the material's pool
                // (`instance_within_model * slot_size`). Static materials
                // only use slot 0; per-instance writes land in Phase 5.
                BindPath::NoOffset(bg) => rpass.set_bind_group(2, bg, &[cbuffer_offset]),
                // BGL has TWO dynamic-offset entries: cbuffer (binding
                // first in BGL order) + decal_constants. Object decals
                // always read decal_constants offset 0 (single default
                // payload — engine `submit_object_mesh_parts` doesn't
                // author per-decal sprite/fade).
                BindPath::DecalObject(bg) => rpass.set_bind_group(2, bg, &[cbuffer_offset, 0u32]),
            }
        } else {
            // Part references a material the model doesn't have — e.g.
            // `inviso_block.render_model` carries collision geometry but ZERO
            // materials because it's an invisible player-blocker the engine
            // never draws. With no material there's no pipeline or material
            // bind group, so skip the draw. Falling through would dispatch
            // with the previously-bound pipeline (the decorator pipeline,
            // since decorators draw first in the albedo pass) while
            // `model_bind_group` is still set at group 1 — a BindGroupLayout
            // mismatch wgpu rejects (the 005_intro crash). Mirrors the
            // DecalObject early-return above.
            return;
        }
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        if bind_prt_stream {
            // SAFETY: `bind_prt_stream` only flips true when the
            // PRT pipeline + bind group + buffer all exist.
            rpass.set_vertex_buffer(1, mesh.prt_ambient_buffer.as_ref().unwrap().slice(..));
        }
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(part.index_start..part.index_start + part.index_count, 0, 0..1);
    }
}

// Suppress "unused" on the c_render_method_data import — it's
// reserved for the per-context shader-extern resolver landing in
// Phase 11+.
#[allow(dead_code)]
fn _link_render_method_data(_: &CRenderMethodData) {}

// ---------------------------------------------------------------------------
// `c_object_renderer::submit_and_render_sky` — dllcache 0x1806E4950
// ---------------------------------------------------------------------------

/// Engine-faithful sky dispatcher. Called from each of the three passes
/// that include sky (albedo, static_lighting, transparents); dispatches
/// by `entry_point_type`. Mirrors:
/// ```text
/// c_object_renderer::push_marker();
/// c_transparency_renderer::push_marker();
/// c_visible_items::push_marker();
/// c_visibility_collection::expand_sky_object(player_window_index);
/// c_object_renderer::submit_visibility(entry_point_type == 2);
/// switch (entry_point_type) {
///   case 0: c_object_renderer::render_albedo(1);          break;
///   case 1: c_object_renderer::render_static_lighting(1); break;
///   case 2: c_transparency_renderer::sort();
///           c_transparency_renderer::render(1, illum_scale); break;
/// }
/// c_visible_items::pop_marker();
/// c_transparency_renderer::pop_marker();
/// c_object_renderer::pop_marker();
/// ```
///
/// Protomorph mirror notes:
/// - We don't yet have the visibility-collection / object_render_contexts
///   pipeline for sky, so we walk `ctx.sky_model.meshes` directly inside
///   the per-type branch (equivalent to the engine after `submit_visibility`
///   has filled object_render_contexts with sky's meshes).
/// - Type 1 (`static_lighting`) is a no-op until the per-material
///   static_lighting pipeline split lands (Phase 7); the call site is
///   wired so the pass-order is engine-faithful.
/// - `_player_window_index` reserved — used by visibility expansion which
///   we haven't ported yet.
pub fn submit_and_render_sky<'rp>(
    entry_point_type: u32,
    _player_window_index: u32,
    rpass: &mut wgpu::RenderPass<'rp>,
    ctx: &'rp FrameContext<'rp>,
    transparency_renderer: &mut crate::halo::render::render_transparents::TransparencyRenderer,
) {
    let Some(sky_model) = ctx.sky_model else { return };

    match entry_point_type {
        0 => {
            // render_albedo(1). Sky meshes drawn opaque with the
            // Albedo entry-point variant — outputs to (`_surface_albedo`,
            // normal G-buffer). Skip transparent materials (route to
            // type=2 path) and any material whose albedo variant
            // didn't compile.
            rpass.set_bind_group(1, &ctx.sky_gpu.model_bind_group, &[0u32]);
            rpass.set_bind_group(3, &ctx.sky_gpu.node_matrices_bind_group, &[0u32]);
            for mesh in &sky_model.meshes {
                rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                rpass.set_index_buffer(
                    mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                for part in &mesh.parts {
                    let Some(material) = sky_model.materials.get(part.material_index)
                    else { continue };
                    if material.is_transparent {
                        continue;
                    }
                    let (Some(artifacts), Some(bind_group)) = (
                        material.artifacts_albedo.as_ref(),
                        material.bind_group_albedo.as_ref(),
                    ) else { continue };
                    rpass.set_pipeline(&artifacts.pipeline);
                    // Path B: cbuffer dynamic-offset, offset 0 for static sky materials.
                    rpass.set_bind_group(2, bind_group, &[0u32]);
                    rpass.draw_indexed(
                        part.index_start..part.index_start + part.index_count,
                        0, 0..1,
                    );
                }
            }
        }
        1 => {
            // render_static_lighting(1). Engine routes per-mesh between
            // `_entry_point_vertex_color_lighting` (sky_dome_simple) and
            // `_entry_point_static_lighting_sh` based on `mesh->flags & 1`
            // (`render_mesh_part_default @ 0x18069EBC0:64-82`).
            //
            // **PROTOMORPH CONSTRAINT:** sky_dome_simple reads per-vertex
            // vert_color from the streamed `_vertex_buffer_usage_vert_color`
            // buffer (`vertex_buffer_indices[2]`), which is BAKED at cache-
            // build time by tool.exe from a lighting computation — NOT
            // present in the source tag's `raw_vertex.vertex_color` field
            // (that field reads as ~(1.0,1.0,1.0) default for every
            // snowbound sky mesh). Until protomorph can reproduce tool.exe's
            // vert_color bake (or reads from a baked .map cache), routing
            // any sky mesh through sky_dome_simple produces uniform bright
            // `(1,1,1) × g_exposure` output instead of the authored gradient.
            //
            // Workaround (non-engine-faithful but visually correct given the
            // data): draw EVERY sky mesh through its material's textured
            // `static_sh` pipeline. The material's albedo bitmap (planet,
            // sundisk, mountains, etc.) carries the per-pixel detail that
            // would otherwise come from the missing vert_color stream. This
            // matches pre-8f79e7c behavior. The `SkyGpu.pipeline_*` variants
            // and `entry_sky_dome_simple.wgsl` remain in the codebase as
            // dormant infrastructure for when the vert_color bake lands.
            //
            // Engine `c_lighting_interface::setup_default_lighting @
            // 0x1806AA190` runs at the TOP of `render_static_lighting @
            // 0x1806E4A80:7` (gated by `rasterizer_force_default_lighting
            // @ 0x180664DF0` — `render_default_lighting` global, true in
            // shipped builds). It writes 4 cbuffer slots:
            //   0x240000   dominant_light_direction
            //   0x240001   dominant_light_intensity
            //   0x2E0000   ravi[10] SH (10×vec4)
            //   0x2A0000   ravi[10] SH duplicate
            // Source: `g_globals_items.skies[0]` → object → defn →
            // model_defn → render_model lightprobe at offsets +61/+77/+93.
            // Protomorph mirror: `LightingInterface` (lighting_interface.rs)
            // packs all three into one struct, populated at scenario load
            // from the sky tag's `default_lightprobe` (mod.rs:1106-1123).
            // Re-binding all 3 dynamic offsets to `_DEFAULT_OFFSET` here
            // resets to the sky probe — required because the prior BSP
            // cluster pass left group 0 with a per-cluster SH offset.
            rpass.set_bind_group(
                0,
                &ctx.shared.camera_bind_group_sl,
                &[
                    crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                    // atmosphere @ binding 2 — default; rebound per sky material
                    // below to honor its per-shader `Custom fog setting index`.
                    crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                    crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                    crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                ],
            );
            rpass.set_bind_group(1, &ctx.sky_gpu.model_bind_group, &[0u32]);
            rpass.set_bind_group(3, &ctx.sky_gpu.node_matrices_bind_group, &[0u32]);
            // Single loop over all sky meshes in tag order. Engine SL
            // pass filters by `mesh_part.flags & mesh_part_mask=1` which
            // skips transparent parts (those have flags=4, drawn through
            // submit_and_render_sky(2) via the transparency pool).
            for mesh in &sky_model.meshes {
                rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                // Bind the per-vertex baked color stream at slot 1 when
                // present (the `_entry_point_vertex_color_lighting` path).
                if let Some(vc) = mesh.vert_color_buffer.as_ref() {
                    rpass.set_vertex_buffer(1, vc.slice(..));
                }
                rpass.set_index_buffer(
                    mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                for part in &mesh.parts {
                    let Some(material) = sky_model.materials.get(part.material_index)
                    else { continue };
                    if material.is_transparent {
                        continue;
                    }
                    // Per-shader atmosphere override (engine
                    // `render_method_submit_volatile_per_shader`): rebind group 0
                    // with THIS sky material's atmosphere-table offset. Construct's
                    // skydome/clouds/planetoid/etc. resolve to `fog_waterfall`
                    // (the bright daytime haze); without this they'd inherit the
                    // dim cluster-default `fog_sky` (g=1 → mie inscatter zeroed).
                    rpass.set_bind_group(
                        0,
                        &ctx.shared.camera_bind_group_sl,
                        &[
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                            material.atmosphere_offset,
                            crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                        ],
                    );
                    // Engine `render_mesh_part_default @ 0x18069EBC0:64-82`:
                    // a vc-bit mesh routes to `_entry_point_vertex_color_lighting`
                    // (`static_per_vertex_color_ps` = albedo×vert_color +
                    // self_illum), NOT `static_lighting_sh`. Use it when the
                    // mesh carries the vert_color stream AND the material
                    // compiled the variant; else fall back to static_sh.
                    let vc = mesh.vert_color_buffer.is_some();
                    if let (true, Some(arts), Some(bg)) = (
                        vc,
                        material.artifacts_vertex_color.as_ref(),
                        material.bind_group_vertex_color.as_ref(),
                    ) {
                        rpass.set_pipeline(&arts.pipeline);
                        rpass.set_bind_group(2, bg, &[0u32]);
                    } else {
                        rpass.set_pipeline(&material.artifacts.pipeline);
                        // Path B: cbuffer dynamic-offset, offset 0.
                        rpass.set_bind_group(2, &material.bind_group, &[0u32]);
                    }
                    rpass.draw_indexed(
                        part.index_start..part.index_start + part.index_count,
                        0, 0..1,
                    );
                }
            }
        }
        2 => {
            // c_transparency_renderer::sort() + render(1, illum_scale).
            // Engine push_marker scopes a sky-only batch into the
            // transparency pool; sort + render walk only that batch;
            // pop_marker truncates the pool back to the regular scope.
            use crate::halo::render::render_transparents::TransparentDispatch;
            let diag_sky = std::env::var_os("PROTOMORPH_DIAG_SKY_SORT").is_some() && {
                use std::sync::atomic::{AtomicU32, Ordering};
                static F: AtomicU32 = AtomicU32::new(0);
                F.fetch_add(1, Ordering::Relaxed) == 0
            };
            let eye: glam::Vec3 = ctx.game.camera.position.into();
            let fwd: glam::Vec3 = ctx.game.camera.forward.into();
            // Engine `global_origin3d` — constant for all sky parts (see the
            // add_element comment below). Using the camera position makes the
            // view-forward depth term zero, so intra-sky order is driven
            // purely by the per-region offset, exactly as the engine.
            let sky_origin = eye.to_array();
            // Diagnostic: PROTOMORPH_SKY_SKIP="8,9" skips those sky mesh
            // indices in the transparent pass — used to isolate which mesh
            // is occluding the planetoid (mesh 7).
            let sky_skip: Vec<u16> = std::env::var("PROTOMORPH_SKY_SKIP")
                .ok()
                .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
                .unwrap_or_default();
            transparency_renderer.push_marker();
            for (mi, mesh) in sky_model.meshes.iter().enumerate() {
                if sky_skip.contains(&(mi as u16)) {
                    continue;
                }
                for (pi, part) in mesh.parts.iter().enumerate() {
                    let Some(material) = sky_model.materials.get(part.material_index)
                    else { continue };
                    if !material.is_transparent {
                        if diag_sky {
                            eprintln!(
                                "[diag-sky-mat] mesh={mi} part={pi} mat={} blend={} OPAQUE(skip)",
                                part.material_index, material.blend_mode
                            );
                        }
                        continue;
                    }
                    // Sky meshes do NOT sort by their authored centroid.
                    // Every sky mesh has the `has transparent sorting plane`
                    // flag (mesh flags bit1) set, so engine
                    // `submit_object_mesh_parts @ 0x1806E1BF0` leaves
                    // `origin = *global_origin3d` (constant for all sky
                    // parts) INSTEAD of reading the part's sorting position.
                    // `add_transparent_mesh_part @ 0x1806E3EB0` then sets
                    // z_sort = |dot(origin-cam, fwd)| - (region*0.1 + 100000).
                    // With `origin` constant the depth term is identical for
                    // every sky part, so the ONLY differentiator is
                    // region_index*0.1 — the sky draws back-to-front in
                    // REGION order, not 3D depth. This sky model's regions map
                    // 1:1 to meshes in order (region i → mesh i), so
                    // region_index == mi.
                    //
                    // We pass the constant `sky_origin` (depth term → 0) and
                    // `offset = -mi*0.1`; sort() yields z_sort = +mi*0.1,
                    // ordering HIGHER mesh index first (back) → mesh 0 last
                    // (front). This puts matte_waypoint (mesh 9) BEHIND the
                    // planetoid (mesh 7) so it stops drawing on top of the
                    // other sky transparents (verified flying near the
                    // skybox). [Reversed from region-ascending after visual
                    // check — the engine's region/offset polarity for sky
                    // draws higher regions first.]
                    if diag_sky {
                        eprintln!(
                            "[diag-sky-mat] mesh={mi} part={pi} mat={} blend={} TRANSPARENT",
                            part.material_index, material.blend_mode
                        );
                    }
                    // Sky passes a NULL sort plane ⇒ point-only; the
                    // -mi*0.1 offset gives the region-index ordering.
                    transparency_renderer.add_element(
                        sky_origin,
                        None,
                        0.0,
                        -(mi as f32) * 0.1,
                        material.sort_layer,
                        TransparentDispatch::SkyMeshPart {
                            mesh_index: mi as u16,
                            part_index: pi as u16,
                        },
                    );
                }
            }
            transparency_renderer.sort(eye.to_array(), fwd.to_array());
            if std::env::var_os("PROTOMORPH_DIAG_SKY_SORT").is_some() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) % 120 == 0 {
                    transparency_renderer.debug_dump_sorted(eye.to_array(), "sky");
                }
            }
            transparency_renderer.render(rpass, ctx);
            transparency_renderer.pop_marker();
        }
        // Engine `c_player_view::submit_and_render_sky` dispatches only
        // on the 3 entry_point_type values (0/1/2). Type 1 is the
        // static_lighting no-op pending Phase 7 — handle explicitly so
        // a contract-violating value (4, 5, …) surfaces as a panic
        // instead of silently rendering nothing.
        n => panic!(
            "[protomorph] submit_and_render_sky: unsupported entry_point_type {n}. \
             Engine `c_player_view::submit_and_render_sky` only dispatches 0/1/2."
        ),
    }
}
