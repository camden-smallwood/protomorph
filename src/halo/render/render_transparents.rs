//! `c_transparency_renderer` — global transparent-element pool +
//! sort + dispatch.
//!
//! Mirrors `Ares/source/render/render_transparents.{h,cpp}` and its
//! dllcache implementation:
//!   - declarations: render_transparents.h:53-97
//!   - render body: `c_transparency_renderer::render(bool depth_test)
//!     @ 0x1806CDB70` (dllcache)
//!   - player-view orchestrator: `c_player_view::render_transparents
//!     @ 0x18068b3b0` (dllcache; implemented on `PlayerView` here)
//!
//! Halo's design: every transparent surface (particles, decals, water,
//! object transparents, BSP transparent parts, halograms, etc.) calls
//! `add_element()` during visibility submit, registering a centroid
//! + plane + sort_layer + a render callback. Then `sort()` does a
//! BSP-style spatial sort and `render()` walks the sorted indices and
//! dispatches each callback.
//!
//! v1 simplifications:
//!   - Sort is a plain layer-then-z stable sort (Halo's
//!     `transparent_layer_and_z_sort_proc`); the recursive
//!     plane/point spatial split (`sort_plane_and_point` /
//!     `group_plane_and_point_of_sublist`) is deferred — it only
//!     matters when transparents intersect, which our v1 content
//!     doesn't have.
//!   - Dispatch goes through a Rust enum (`TransparentDispatch`)
//!     instead of an `fn(*const c_void, long)` callback. Each
//!     element carries enough data to draw itself.
//!   - Active-camo path stubbed (set_using_active_camo /
//!     set_active_camo_bounds match Halo signatures but no body).
//!   - k_max_number_of_rendered_transparents = 1024 enforced at
//!     `add_element` (Halo asserts; we silently drop with a log).
//!
//! References:
//!   - reference_halo_frame_pipeline.md — render_transparents in pass order
//!   - render_view.h:343 — `c_player_view::render_transparents()`

/// `e_transparent_sort_layer` (render_transparents.h:13-20).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TransparentSortLayer {
    Invalid = 0,
    Pre = 1,
    Normal = 2,
    Post = 3,
}

/// `e_transparent_sort_method` (render_transparents.h:22-28). v1 uses
/// only `PointQsort` — BSP and plane-qsort variants TBD with the full
/// recursive sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TransparentSortMethod {
    Bsp = 0,
    PointQsort = 1,
    PointPlaneQsort = 2,
}

/// What a sorted transparent element actually does when its turn comes
/// in `render()`. Replaces Halo's
/// `void (*render_callback)(void const *user_data, long user_context)`
/// — Rust avoids the type-erased pointer dance in favour of a dispatch
/// enum. Each variant carries the parameters the corresponding draw
/// path needs.
///
/// Variants land as their canonical render paths come online:
///   - `BspClusterPart` / `BspInstancePart` — `c_structure_renderer`
///     transparent variants (Phase 12 brings rmw/rmd/rmhg shaders).
///   - `Object` — `c_object_renderer::render_object_transparent`
///     (Phase 11).
///   - `Particle` / `Effect` — effects pipeline (Phase 12).
#[derive(Debug, Clone, Copy)]
pub enum TransparentDispatch {
    /// BSP cluster mesh transparent part. World-space (identity model
    /// matrix). Material slot may currently be `None` until the rm**
    /// subclass shader lands; render() skips Nones.
    BspClusterPart {
        bsp_index: u8,
        cluster_index: u16,
        mesh_index: u16,
        part_index: u16,
    },
    /// BSP instance-mesh transparent part. Carries the placement's
    /// world matrix via `structure_instance_index`.
    BspInstancePart {
        bsp_index: u8,
        structure_instance_index: u16,
        mesh_index: u16,
        part_index: u16,
    },
    /// Object transparent (vehicles, characters with transparent
    /// materials). Wired by Phase 11 / `c_object_renderer`.
    Object {
        object_slot: u32,
        mesh_index: u16,
        part_index: u16,
    },
    /// Sky model transparent part. Engine
    /// `c_object_renderer::submit_and_render_sky @ 0x1806E4950` queues
    /// sky transparents into the same `transparency_renderer` pool
    /// where they sort with BSP/object transparents. Drawn with the
    /// sky's dedicated camera-anchored model_u + identity
    /// node_matrices (parallax-locked) — see `render_sky::SkyGpu`.
    SkyMeshPart {
        mesh_index: u16,
        part_index: u16,
    },
}

/// `s_transparent_types` (render_transparents.h:32-44, 0xB0 bytes).
/// We omit `anchor_points[9]` and `plane` for v1 — they're inputs to
/// the recursive plane-qsort which we don't run yet. Halo also stores
/// the sort_layer + z_sort + centroid + importance + the callback
/// triple — we mirror those.
#[derive(Debug, Clone, Copy)]
pub struct TransparentTypes {
    /// Halo: `bool use_plane;` — false means point-only sort, true
    /// means use the plane equation. Stays `false` until we port the
    /// plane-qsort.
    pub use_plane: bool,
    /// Halo: `float z_sort;` — view-space depth used as the primary
    /// sort key WITHIN a layer.
    pub z_sort: f32,
    pub centroid: [f32; 3],
    pub sort_layer: TransparentSortLayer,
    /// Halo: `float importance;` — controls debug overlay verbosity;
    /// not used by sort math.
    pub importance: f32,
    /// What to dispatch when this element's turn comes up.
    pub dispatch: TransparentDispatch,
}

/// `s_transparency_marker` (render_transparents.h:47-51, 2 bytes).
/// Halo uses these for nested visibility passes (e.g. reflection
/// view re-uses the same global transparent pool). `push_marker`
/// records `m_total_transparent_count`; `pop_marker` truncates back
/// to that count.
#[derive(Debug, Clone, Copy, Default)]
pub struct TransparencyMarker {
    pub starting_transparent_index: u16,
}

/// Halo: `k_max_number_of_transparency_markers = 6`
/// (render_transparents.h:58).
pub const K_MAX_NUMBER_OF_TRANSPARENCY_MARKERS: usize = 6;

/// Halo: `k_max_number_of_rendered_transparents = 1024`
/// (render_transparents.h:59). Enforced at `add_element`.
pub const K_MAX_NUMBER_OF_RENDERED_TRANSPARENTS: usize = 1024;

/// `c_transparency_renderer` (render_transparents.h:53-97).
///
/// Halo stores a single global instance (`g_transparency_renderer
/// @ 0x182592E57`); we own it on the `Renderer`.
#[derive(Debug)]
pub struct TransparencyRenderer {
    /// `c_static_array<s_transparent_types, 1024> transparents`. Append
    /// order = registration order; `sorted_order` indexes into this.
    transparents: Vec<TransparentTypes>,
    /// `c_sorter<s_transparent_types, 1024> transparent_sorted_order`.
    /// Filled by `sort()`; iterated by `render()`. Stores indices
    /// into `transparents`.
    sorted_order: Vec<u16>,
    /// `s_transparency_marker m_markers[6]`.
    markers: [TransparencyMarker; K_MAX_NUMBER_OF_TRANSPARENCY_MARKERS],
    /// `long m_current_marker_index` — `-1` when no marker active.
    current_marker_index: i32,
    /// `long m_total_transparent_count` — Halo tracks this separately
    /// from `transparents.size()` so debug overlays can report it
    /// without re-reading the static array. We keep it for parity.
    total_transparent_count: i32,
    /// Active-camo state (set/cleared by `set_using_active_camo` /
    /// `set_active_camo_bounds`). Stubbed v1.
    using_active_camo: bool,
    needs_active_camo_ldr_resolve: bool,
    /// Diagnostic toggle. When `false`, `render` is a no-op. Used to
    /// isolate visual artifacts that might be coming from post-water
    /// transparency draws (which run AFTER the underwater_fog pass
    /// and therefore appear unfogged).
    pub render_enabled: bool,
}

impl TransparencyRenderer {
    pub fn new() -> Self {
        Self {
            transparents: Vec::with_capacity(K_MAX_NUMBER_OF_RENDERED_TRANSPARENTS),
            sorted_order: Vec::with_capacity(K_MAX_NUMBER_OF_RENDERED_TRANSPARENTS),
            markers: [TransparencyMarker::default(); K_MAX_NUMBER_OF_TRANSPARENCY_MARKERS],
            current_marker_index: -1,
            total_transparent_count: 0,
            using_active_camo: false,
            needs_active_camo_ldr_resolve: false,
            render_enabled: true,
        }
    }

    /// `c_transparency_renderer::reset @ 0x1806CCEA0`. Pseudocode:
    ///   m_current_marker_index = -1;
    ///   m_total_transparent_count = 0;
    ///   transparent_sorted_order.m_count = 0;
    pub fn reset(&mut self) {
        self.current_marker_index = -1;
        self.total_transparent_count = 0;
        self.transparents.clear();
        self.sorted_order.clear();
    }

    /// `c_transparency_renderer::push_marker @ 0x1806CCEC0`. Increments
    /// `m_current_marker_index` and records the current count.
    pub fn push_marker(&mut self) {
        self.current_marker_index += 1;
        assert!(
            (self.current_marker_index as usize) < K_MAX_NUMBER_OF_TRANSPARENCY_MARKERS,
            "transparency marker stack overflow",
        );
        self.markers[self.current_marker_index as usize] = TransparencyMarker {
            starting_transparent_index: self.total_transparent_count as u16,
        };
    }

    /// `c_transparency_renderer::pop_marker @ 0x1806CCF40`. Truncates
    /// the transparent list back to the marker's starting index.
    pub fn pop_marker(&mut self) {
        assert!(self.current_marker_index >= 0, "pop_marker without push");
        let start = self.markers[self.current_marker_index as usize].starting_transparent_index
            as usize;
        self.transparents.truncate(start);
        self.total_transparent_count = start as i32;
        self.current_marker_index -= 1;
    }

    /// `c_transparency_renderer::add_element @ 0x1806CCFD0`. Halo:
    ///   bool add_element(centroid, plane, offset, sort_layer,
    ///       render_callback, user_data, user_context, radius);
    /// Returns false if the static array is full.
    ///
    /// v1: skips plane / offset / radius / importance — those feed
    /// the recursive plane-qsort. Replace `render_callback` triple
    /// with a `TransparentDispatch` enum.
    pub fn add_element(
        &mut self,
        centroid: [f32; 3],
        z_sort: f32,
        sort_layer: TransparentSortLayer,
        dispatch: TransparentDispatch,
    ) -> bool {
        if self.transparents.len() >= K_MAX_NUMBER_OF_RENDERED_TRANSPARENTS {
            return false;
        }
        self.transparents.push(TransparentTypes {
            use_plane: false,
            z_sort,
            centroid,
            sort_layer,
            importance: 0.0,
            dispatch,
        });
        self.total_transparent_count += 1;
        true
    }

    /// `c_transparency_renderer::sort @ 0x1806CD620`. Halo: per-layer
    /// then BSP-style spatial sort.
    ///
    /// **Marker-aware** — operates on `[m_markers[m_current_marker_index]
    /// .starting_transparent_index..m_total_transparent_count]` only.
    /// Earlier batches (already sorted) are left untouched. Engine
    /// dispatches sort+render in pairs per marker scope (`submit_and_render_sky(2)`
    /// pushes a sky-only marker, sorts + renders that, pops; the
    /// surrounding `render_transparents` then sorts + renders the
    /// outer regular-transparent batch).
    ///
    /// v1: stable sort by `(sort_layer, z_sort)` mirroring
    /// `transparent_layer_and_z_sort_proc @ 0x1806CE870`. Higher
    /// z_sort = farther from camera = drawn first (back-to-front).
    pub fn sort(&mut self) {
        let start = self.current_batch_start();
        let end = self.total_transparent_count as usize;
        if end <= start {
            return;
        }
        // Grow sorted_order so [0..end] is addressable. Earlier indices
        // (from prior sort calls) are preserved verbatim.
        if self.sorted_order.len() < end {
            self.sorted_order.resize(end, 0);
        }
        for i in start..end {
            self.sorted_order[i] = i as u16;
        }
        let elements = &self.transparents;
        self.sorted_order[start..end].sort_by(|&a, &b| {
            let ea = &elements[a as usize];
            let eb = &elements[b as usize];
            ea.sort_layer
                .cmp(&eb.sort_layer)
                .then(eb.z_sort.partial_cmp(&ea.z_sort).unwrap_or(std::cmp::Ordering::Equal))
        });
    }

    /// Starting index of the current marker scope, or 0 when no marker
    /// is active. Matches engine `m_markers[m_current_marker_index]
    /// .starting_transparent_index` semantics.
    fn current_batch_start(&self) -> usize {
        if self.current_marker_index < 0 {
            0
        } else {
            self.markers[self.current_marker_index as usize].starting_transparent_index as usize
        }
    }

    /// `c_transparency_renderer::set_using_active_camo @ 0x1806CDDF0`.
    /// Halo: `m_using_active_camo = 1`.
    pub fn set_using_active_camo(&mut self) {
        self.using_active_camo = true;
    }

    /// `c_transparency_renderer::set_active_camo_bounds @ 0x1806CDEA0`.
    /// Captures the resolve bounds and clears the active-camo flag
    /// (Halo: marks `m_needs_active_camo_ldr_resolve` so the LDR
    /// composite picks the snapshot up later).
    pub fn set_active_camo_bounds(&mut self) {
        if self.using_active_camo {
            self.needs_active_camo_ldr_resolve = true;
            self.using_active_camo = false;
        }
    }

    /// Sorted indices into `transparents` for the CURRENT marker scope.
    /// Iterated by `render()`. Earlier batches (already rendered) are
    /// skipped — engine `c_transparency_renderer::render` walks
    /// `[m_markers[m_current_marker_index].starting_transparent_index
    /// ..m_total_transparent_count]`.
    pub fn sorted_elements(&self) -> impl Iterator<Item = &TransparentTypes> + '_ {
        let start = self.current_batch_start();
        let end = self.total_transparent_count as usize;
        let range = if end > start && self.sorted_order.len() >= end {
            start..end
        } else {
            0..0
        };
        self.sorted_order[range]
            .iter()
            .map(move |&i| &self.transparents[i as usize])
    }

    /// Number of registered transparents (Halo:
    /// `m_total_transparent_count`).
    pub fn total_count(&self) -> usize {
        self.total_transparent_count as usize
    }

    /// `c_transparency_renderer::render(depth_test=1) @ 0x1806CDB70`.
    /// Walks the CURRENT marker scope's `sorted_order` slice and
    /// dispatches each `TransparentDispatch` variant. Render state per
    /// dllcache:
    ///   - depth-test on, depth-write off (read-only depth)
    ///   - color-write RGB only
    ///   - cull off
    ///   - target = lighting_base (HDR composite) + read-only depth
    ///
    /// **Caller-managed rpass.** Engine `render` operates on whatever
    /// targets are bound at the rasterizer level (set up by the
    /// surrounding `c_player_view::render_transparents`). To mirror
    /// that and to allow multiple sort+render pairs inside one rpass
    /// (sky-only batch then regular batch — see `submit_and_render_sky(2)`),
    /// we take an existing `&mut RenderPass` instead of opening one.
    ///
    /// Materials with `artifacts.is_none()` (rmw/rmd/rmhg/etc. — WGSL
    /// not yet ported per `subclass_has_wgsl_pipeline`) skip with no
    /// substitute shader.
    pub fn render<'rp>(
        &self,
        rpass: &mut wgpu::RenderPass<'rp>,
        ctx: &'rp crate::halo::render::shared::FrameContext<'rp>,
    ) {
        if !self.render_enabled {
            return;
        }
        let start = self.current_batch_start();
        let end = self.total_transparent_count as usize;
        if end <= start {
            return;
        }
        let (Some(identity_model_bg), Some(identity_nm_bg)) =
            (ctx.structure_renderer.identity_model_bg.as_ref(), ctx.structure_renderer.identity_nm_bg.as_ref())
        else {
            return;
        };

        rpass.set_bind_group(
            0,
            &ctx.shared.camera_bind_group_sl,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );

        for element in self.sorted_elements() {
            match element.dispatch {
                TransparentDispatch::BspClusterPart {
                    bsp_index,
                    cluster_index,
                    mesh_index,
                    part_index,
                } => {
                    let Some(bsp) = ctx.structure_renderer.bsps.get(bsp_index as usize) else { continue };
                    let Some(mesh) = bsp.meshes.get(mesh_index as usize) else { continue };
                    let Some(mesh_part) = mesh.parts.get(part_index as usize) else { continue };
                    let Some(material_slot) =
                        bsp.materials.get(mesh_part.material_index as usize)
                    else {
                        continue;
                    };
                    let Some(material) = material_slot.as_ref() else { continue };
                    // Per-cluster lightmap policy. See
                    // structure_renderer::render_cluster_mesh_part for
                    // the full rationale — clusters with per-vertex SH
                    // (waterfall on riverworld) route through StaticSh
                    // at a per-cluster dynamic offset because the atlas
                    // sample at their UVs returns ~0.
                    let cluster_offset = bsp
                        .cluster_lighting_offsets
                        .get(cluster_index as usize)
                        .copied()
                        .unwrap_or(0);
                    let use_sh = cluster_offset != 0;
                    let (artifacts_opt, bind_groups_opt) = if use_sh {
                        (material.artifacts_sh.as_ref(), material.bind_group_sh.as_ref())
                    } else {
                        (material.artifacts.as_ref(), material.bind_group.as_ref())
                    };
                    let (Some(artifacts), Some(bind_groups)) = (artifacts_opt, bind_groups_opt) else {
                        continue;
                    };
                    // Engine-faithful per-cluster cubemap pick — see
                    // `structure_renderer::render_cluster_mesh_part`.
                    let probe_idx = bsp
                        .cluster_to_probe
                        .get(cluster_index as usize)
                        .copied()
                        .unwrap_or(0) as usize;
                    let Some(bind_group) =
                        bind_groups.get(probe_idx).or_else(|| bind_groups.first())
                    else {
                        continue;
                    };
                    let simple_lights_offset = bsp
                        .cluster_simple_lights_offsets
                        .get(cluster_index as usize)
                        .copied()
                        .unwrap_or(crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET);
                    let lighting_offset = if use_sh {
                        cluster_offset
                    } else {
                        crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET
                    };
                    rpass.set_bind_group(
                        0,
                        &ctx.shared.camera_bind_group_sl,
                        // 3rd entry = dominant_light @ binding 13.
                        &[lighting_offset, simple_lights_offset, lighting_offset],
                    );
                    rpass.set_pipeline(&artifacts.pipeline);
                    rpass.set_bind_group(1, identity_model_bg, &[0u32]);
                    // Path B: cbuffer slot is dynamic; offset 0 for static materials.
                    rpass.set_bind_group(2, bind_group, &[0u32]);
                    rpass.set_bind_group(3, identity_nm_bg, &[0u32]);
                    rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    rpass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(
                        mesh_part.index_start..mesh_part.index_start + mesh_part.index_count,
                        0,
                        0..1,
                    );
                }
                TransparentDispatch::BspInstancePart {
                    bsp_index,
                    structure_instance_index,
                    mesh_index,
                    part_index,
                } => {
                    let Some(bsp) = ctx.structure_renderer.bsps.get(bsp_index as usize) else { continue };
                    let Some(mesh) = bsp.meshes.get(mesh_index as usize) else { continue };
                    let Some(mesh_part) = mesh.parts.get(part_index as usize) else { continue };
                    let Some(material_slot) =
                        bsp.materials.get(mesh_part.material_index as usize)
                    else {
                        continue;
                    };
                    let Some(material) = material_slot.as_ref() else { continue };
                    let Some(inst_model_bg) = bsp
                        .instance_model_bgs
                        .get(structure_instance_index as usize)
                    else {
                        continue;
                    };
                    let Some(inst_nm_bg) = bsp.instance_nm_bg.as_ref() else { continue };
                    let (Some(artifacts), Some(bind_groups)) = (
                        material.artifacts.as_ref(),
                        material.bind_group.as_ref(),
                    ) else {
                        continue;
                    };
                    // Instance probe selection — todo: per-instance
                    // nearest-probe lookup. For now bind probe[0].
                    let Some(bind_group) = bind_groups.first() else { continue };
                    rpass.set_pipeline(&artifacts.pipeline);
                    rpass.set_bind_group(1, inst_model_bg, &[0u32]);
                    // Path B: cbuffer slot is dynamic; offset 0 for static materials.
                    rpass.set_bind_group(2, bind_group, &[0u32]);
                    rpass.set_bind_group(3, inst_nm_bg, &[0u32]);
                    let placement_vb = bsp
                        .instance_vertex_buffers
                        .get(structure_instance_index as usize)
                        .and_then(|o| o.as_ref())
                        .unwrap_or(&mesh.vertex_buffer);
                    rpass.set_vertex_buffer(0, placement_vb.slice(..));
                    rpass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(
                        mesh_part.index_start..mesh_part.index_start + mesh_part.index_count,
                        0,
                        0..1,
                    );
                }
                TransparentDispatch::Object { object_slot, mesh_index, part_index } => {
                    // Object transparent (vehicles, FX scenery — waterfalls,
                    // man cannons, tower pulses). Bridges the object render
                    // list into the transparency pool. Mirrors the BSP-side
                    // pattern in render_objects::record_mesh_part_draw.
                    let Some(&(_obj_idx, model_idx)) =
                        ctx.render_list.get(object_slot as usize)
                    else { continue };
                    let Some(gpu_model) = ctx.models.get(model_idx) else { continue };
                    let Some(mesh) = gpu_model.meshes.get(mesh_index as usize) else { continue };
                    let Some(part) = mesh.parts.get(part_index as usize) else { continue };
                    let Some(material) = gpu_model.materials.get(part.material_index) else { continue };
                    let model_offset =
                        ((object_slot as usize) * ctx.shared.model_stride) as u32;
                    let nm_offset =
                        ((object_slot as usize) * ctx.shared.node_matrices_stride) as u32;
                    rpass.set_pipeline(&material.artifacts.pipeline);
                    rpass.set_bind_group(1, &ctx.shared.model_bind_group, &[model_offset]);
                    // Path B: cbuffer slot dynamic; offset 0.
                    rpass.set_bind_group(2, &material.bind_group, &[0u32]);
                    rpass.set_bind_group(3, &ctx.shared.node_matrices_bind_group, &[nm_offset]);
                    rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    rpass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(
                        part.index_start..part.index_start + part.index_count,
                        0, 0..1,
                    );
                }
                TransparentDispatch::SkyMeshPart { mesh_index, part_index } => {
                    // Engine `c_transparency_renderer::render` walks the
                    // transparent pool firing per-item `render_callback`s.
                    // For sky mesh parts the callback is
                    // `c_object_renderer::render_transparent_object_mesh_part
                    // @ 0x1806E4050` which calls
                    // `render_object_context_mesh_part(...,
                    //   entry_point = _entry_point_static_lighting_prt_quadratic,
                    //   mesh_part_mask = 4 /*transparent*/)`.
                    // Then `render_mesh_part_default @ 0x18069EBC0:64-82`
                    // REMAPS the entry point per mesh:
                    //   if (mesh->flags & 1)   → _entry_point_vertex_color_lighting
                    //                            (= sky_dome_simple shader)
                    //   else                   → _entry_point_static_lighting_sh
                    // The render_method's authored blend_mode (the rmsh
                    // category choice) becomes the pipeline's blend state.
                    // Verified: planet_huge.render_method_template's
                    // `available entry points` flags include
                    // `vertex color lighting` (bit 14) — so the engine
                    // really does compile sky_dome_simple for these
                    // transparent sky materials.
                    //
                    // **DATA-BLOCKED:** Engine-faithful would route
                    // vc-bit meshes through SkyGpu's per-blend-mode
                    // pipeline (sky_dome_simple shader). However,
                    // blam-tags currently reads vert_color from
                    // `raw_vertex.vertex_color` (the offline-vert
                    // block) which is ~(1.0,1.0,1.0) default for ALL
                    // snowbound sky meshes — the real per-vertex sky
                    // gradient lives in the streamed vertex buffer at
                    // `vertex_buffer_indices[2]` (engine
                    // `_vertex_buffer_usage_vert_color`) which
                    // blam-tags doesn't yet expose. Routing through
                    // sky_dome_simple here produces uniform bright
                    // white for every transparent sky mesh (planet,
                    // sun, clouds, horizons) — visually worse than the
                    // material's textured `static_sh` variant. Until
                    // blam-tags surfaces the real vert_color stream,
                    // we keep the (non-engine-faithful) routing
                    // through the material's pipeline so the planet
                    // bitmap, sundisk bitmap, etc. are at least
                    // sampled. See [[feedback_sky_vert_color_buffer_bug]].
                    let Some(sky_model) = ctx.sky_model else { continue };
                    let Some(mesh) = sky_model.meshes.get(mesh_index as usize) else { continue };
                    let Some(part) = mesh.parts.get(part_index as usize) else { continue };
                    let Some(material) = sky_model.materials.get(part.material_index) else { continue };
                    // Re-bind group 0 with `ENGINE_LIGHTING_DEFAULT_OFFSET`
                    // so the shader's ravi[10] cbuffer reads the sky's
                    // lightprobe SH (populated at scenario load from the
                    // scenario's sky tag via `lighting_interface` at
                    // mod.rs:1106-1123). Engine equivalent: the
                    // setup_default_lighting() call at the top of
                    // `render_transparent_object_mesh_part` (via
                    // `setup_object_lighting_for_entry_point` →
                    // setup_default_lighting fallback).
                    rpass.set_bind_group(
                        0,
                        &ctx.shared.camera_bind_group_sl,
                        &[
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                            crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                        ],
                    );
                    rpass.set_pipeline(&material.artifacts.pipeline);
                    rpass.set_bind_group(1, &ctx.sky_gpu.model_bind_group, &[0u32]);
                    // Path B: cbuffer slot dynamic; offset 0.
                    rpass.set_bind_group(2, &material.bind_group, &[0u32]);
                    rpass.set_bind_group(3, &ctx.sky_gpu.node_matrices_bind_group, &[0u32]);
                    rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    rpass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(
                        part.index_start..part.index_start + part.index_count,
                        0, 0..1,
                    );
                }
            }
        }
    }
}

impl Default for TransparencyRenderer {
    fn default() -> Self {
        Self::new()
    }
}
