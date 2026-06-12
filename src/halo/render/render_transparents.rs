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

impl TransparentSortLayer {
    /// Map the render_method's authored `GlobalSortLayer` to the runtime
    /// transparent sort bucket. Discriminants match 1:1 (invalid=0,
    /// pre[-pass]=1, normal=2, post[-pass]=3) — this is the engine's
    /// `sort layer` → `e_transparent_sort_layer` carry. Defaults to
    /// `Normal` for `Invalid` (the engine asserts non-invalid in `sort()`).
    pub fn from_global(g: blam_tags::render_method::GlobalSortLayer) -> Self {
        use blam_tags::render_method::GlobalSortLayer as G;
        match g {
            G::PrePass => TransparentSortLayer::Pre,
            G::PostPass => TransparentSortLayer::Post,
            G::Normal | G::Invalid => TransparentSortLayer::Normal,
        }
    }
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
        /// Object's resolved dynamic-env cubemap probe index (engine
        /// `c_dynamic_cubemap_sample`). Selects `GpuMaterial::
        /// bind_group_probes[idx]` when the material has per-probe cube
        /// variants; `None` → the rasg-default `bind_group`.
        probe_index: Option<u16>,
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
    /// A single particle EMITTER of a single live instance (engine
    /// `c_particle_emitter::submit @0x180568FA0` registers one element PER
    /// EMITTER at `get_position_world`; `render_callback` draws only that
    /// emitter's rows). `draw_index` indexes
    /// `EffectStore::emitter_draws`; the draw reads its batch + per-row
    /// spans and dispatches `ParticleGpu::draw_emitter`. Replaces the old
    /// per-BATCH `Particle` whose single averaged centroid mis-sorted
    /// multi-instance systems (s3d_turf's 11 steam vents share one batch).
    EmitterInstance {
        draw_index: u32,
    },
    /// Patchy-fog fullscreen composite. Engine `c_player_view::
    /// queue_patchy_fog @ 0x18068BFB0` registers it with `radius = 0`
    /// (point sort) and `offset = transparent_sort_distance` at a
    /// centroid that same distance along camera forward — so its
    /// `z_sort` resolves to ~0 and it draws LAST within its layer:
    /// every transparent/particle in the layer composites first and
    /// gets fogged over. Drawn via `PatchyFogPass::draw` (prepared
    /// once per frame before the transparents pass).
    PatchyFog,
}

/// `s_transparent_types` (render_transparents.h:32-44, 0xB0 bytes).
/// We carry everything the engine sort consumes EXCEPT the 9
/// `anchor_points`: those exist only to make the plane-vs-plane test
/// (`plane_location_testing`) robust for near-coplanar planes, and for
/// construct's discrete water/glass planes that test reduces to the
/// `radius → 0` limit (a single centroid-vs-plane side test). We use the
/// centroid test in [`TransparencyRenderer::sort`]; see that method.
#[derive(Debug, Clone, Copy)]
pub struct TransparentTypes {
    /// Halo: `bool use_plane;` — false means point-only sort, true means
    /// this element participates in the stage-2 BSP plane sort. Engine
    /// `add_element`: true iff a plane was supplied AND `radius > 1e-4`
    /// AND `|plane.n| > 0.5`. Computed in `sort()` (we defer it there
    /// with `z_sort`/`importance` since the camera is per-frame).
    pub use_plane: bool,
    /// Camera-FACED plane (engine `add_element` flips the authored plane
    /// so the camera sits on its positive side). Valid iff `use_plane`.
    /// Computed in `sort()` from `raw_plane`.
    pub plane: [f32; 4],
    /// Authored sort plane `(i, j, k, d)` as registered (model→world
    /// already applied by the caller). `None` when the part has no
    /// authored sorting position. Feeds `use_plane` + `plane`.
    pub raw_plane: Option<[f32; 4]>,
    /// Authored sort-quad radius (`SortingPosition.radius`). Gates
    /// `use_plane` and scales `importance`.
    pub radius: f32,
    /// Halo: `float z_sort;` — view-space depth used as the primary
    /// sort key WITHIN a layer. Computed in `sort()` from `centroid`,
    /// the camera, and `offset` (engine `add_element` stores it at
    /// registration; we defer it to `sort()` since the camera is fixed
    /// per frame and not every registration site has it).
    pub z_sort: f32,
    /// Halo `add_element` `offset` param — SUBTRACTED from the
    /// view-forward depth before comparison. `c_object_renderer::
    /// add_transparent_mesh_part @ 0x1806E3EB0` passes
    /// `region_index*0.1 (+100000 for sky)`; the +100000 is a constant
    /// within the sky's own marker batch (cancels in comparison), the
    /// `region*0.1` a per-region tiebreak. 0 for paths without a bias.
    pub offset: f32,
    pub centroid: [f32; 3],
    pub sort_layer: TransparentSortLayer,
    /// Halo: `float importance;` — engine `add_element` sets it to
    /// `radius² · |plane.n · camera_forward| / dist(centroid, camera)`
    /// for plane elements (0 for points). The stage-2 sort picks the
    /// highest-importance plane in a sublist as the BSP splitter.
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
    /// `raw_plane` + `radius` are the authored sorting plane/radius
    /// (already transformed to world by the caller for instances). They
    /// feed the stage-2 BSP plane sort; `None`/`0.0` → point-only element
    /// (`use_plane == false`). `use_plane`, the camera-faced `plane`,
    /// `z_sort`, and `importance` are all resolved in [`Self::sort`]
    /// (the camera is per-frame and not every caller has it). Replaces
    /// the engine `render_callback` triple with a `TransparentDispatch`.
    pub fn add_element(
        &mut self,
        centroid: [f32; 3],
        raw_plane: Option<[f32; 4]>,
        radius: f32,
        offset: f32,
        sort_layer: TransparentSortLayer,
        dispatch: TransparentDispatch,
    ) -> bool {
        if self.transparents.len() >= K_MAX_NUMBER_OF_RENDERED_TRANSPARENTS {
            return false;
        }
        self.transparents.push(TransparentTypes {
            use_plane: false,
            plane: [0.0; 4],
            raw_plane,
            radius,
            z_sort: 0.0,
            offset,
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
    /// Stable sort by `(sort_layer, z_sort)` mirroring
    /// `transparent_layer_and_z_sort_proc @ 0x1806CE870`. Higher
    /// z_sort = farther along the view axis = drawn first (back-to-front).
    ///
    /// `z_sort` is **view-forward depth**, not euclidean distance —
    /// engine `add_element @ 0x1806CCFD0` computes
    /// `abs(dot(centroid - camera_pos, camera_forward)) - offset`.
    /// Euclidean distance ranks off-axis elements (e.g. a sky nebula to
    /// the side) as farther than they really are in screen depth, which
    /// flips their order against on-axis elements (the planet). All
    /// object/sky parts have `use_plane == false` (engine passes a NULL
    /// plane + 0 radius from `add_transparent_mesh_part`), so the
    /// recursive plane-qsort never runs for them — this metric is the
    /// only thing ordering them within a layer.
    pub fn sort(&mut self, camera_pos: [f32; 3], camera_forward: [f32; 3]) {
        let start = self.current_batch_start();
        let end = self.total_transparent_count as usize;
        if end <= start {
            return;
        }
        let f = camera_forward;
        // Per-element resolve (engine does this in add_element; we defer
        // it because the camera is per-frame and not every caller has it):
        //   z_sort    = |(centroid - camera_pos) · camera_forward| - offset
        //   use_plane = raw_plane present AND radius > 1e-4 AND |n| > 0.5
        //   plane     = raw_plane flipped so the camera is on its + side
        //   importance= radius² · |plane.n · forward| / dist(centroid, cam)
        for i in start..end {
            let e = &mut self.transparents[i];
            let c = e.centroid;
            let dx = c[0] - camera_pos[0];
            let dy = c[1] - camera_pos[1];
            let dz = c[2] - camera_pos[2];
            e.z_sort = (dx * f[0] + dy * f[1] + dz * f[2]).abs() - e.offset;

            e.use_plane = false;
            e.plane = [0.0; 4];
            e.importance = 0.0;
            if let Some(p) = e.raw_plane {
                let n_len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                if e.radius > 0.000_1 && n_len > 0.5 {
                    e.use_plane = true;
                    // Face the plane toward the camera (engine add_element).
                    let side = p[0] * camera_pos[0]
                        + p[1] * camera_pos[1]
                        + p[2] * camera_pos[2]
                        - p[3];
                    let faced = if side >= 0.0 {
                        p
                    } else {
                        [-p[0], -p[1], -p[2], -p[3]]
                    };
                    e.plane = faced;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.000_1);
                    let face_dot =
                        (faced[0] * f[0] + faced[1] * f[1] + faced[2] * f[2]).abs();
                    e.importance = (e.radius * e.radius * face_dot) / dist;
                }
            }
        }
        // Grow sorted_order so [0..end] is addressable. Earlier indices
        // (from prior sort calls) are preserved verbatim.
        if self.sorted_order.len() < end {
            self.sorted_order.resize(end, 0);
        }
        for i in start..end {
            self.sorted_order[i] = i as u16;
        }
        // Stage 1: stable sort by (sort_layer ASC, z_sort desc) —
        // `transparent_layer_and_z_sort_proc @ 0x1806CE870`, verified
        // against BOTH engine sort paths (the debug bubble sort and the
        // production `qsort_2byte`/`templated_sort<short>::qsort`). The
        // proc returns true ⟺ `a` sorts to a HIGHER index = drawn LATER
        // = on top:
        //   diff layer: proc = (layer_a > layer_b) → higher layer on top
        //   same layer: proc = (z_a > z_b)         → farther on top (?)
        // So the engine sorts LAYERS ASCENDING: pre-pass(1) drawn first
        // (bottom), normal(2), post(3) last (top). The patchy-fog tag
        // authors `transparent_sort_layer = pre-pass`, so the engine
        // draws it BEFORE normal transparents — it does NOT cover the
        // railing/drips; those blend via their own per-shader atmosphere
        // inscatter, not by the fog pass painting over them.
        //
        // NOTE the z stays DESCENDING here (farther first = back-to-front)
        // — protomorph's deliberate, user-verified (construct planetoid)
        // divergence from the binary's literal ascending-z partition,
        // because back-to-front is what correct alpha-over needs and the
        // engine's stage-2 plane sort (which we mostly gate off) is what
        // makes its front-to-back partition come out right in practice.
        {
            let elements = &self.transparents;
            self.sorted_order[start..end].sort_by(|&a, &b| {
                let ea = &elements[a as usize];
                let eb = &elements[b as usize];
                ea.sort_layer.cmp(&eb.sort_layer).then(
                    eb.z_sort
                        .partial_cmp(&ea.z_sort)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
        }
        // Stage 2: per sort-layer run, the recursive BSP plane sort
        // (`group_plane_and_point_of_sublist` + `sort_plane_and_point`).
        //
        // GATED OFF by default (PROTOMORPH_PLANE_SORT to enable). The
        // algorithm is engine-faithful, but it only sorts correctly when the
        // sublist's plane elements are GOOD spatial separators for the whole
        // sublist. On construct the big waterfall sheets read with NO authored
        // plane (`plane = [0,0,0,0]` in the loose tag) so they participate as
        // planeless POINTS, and the only available splitter is a small
        // horizontal glass pane (z≈5.3) — using it to partition the distant
        // waterfall scatters far segments to the "camera side" and draws them
        // OVER near glass (a milky veil). Plain `z_sort` (stage 1) orders these
        // correctly (far-first). Re-enable once the waterfall's authored sort
        // plane is recovered (suspected loose-tag parse gap: radius is present
        // but the plane reads zero) so it splits as a plane, not a point.
        if std::env::var_os("PROTOMORPH_PLANE_SORT").is_some() {
            let mut run_start = start;
            while run_start < end {
                let layer =
                    self.transparents[self.sorted_order[run_start] as usize].sort_layer;
                let mut run_end = run_start + 1;
                while run_end < end
                    && self.transparents[self.sorted_order[run_end] as usize].sort_layer
                        == layer
                {
                    run_end += 1;
                }
                if run_end - run_start >= 2 {
                    self.plane_sort_sublist(run_start, run_end);
                }
                run_start = run_end;
            }
        }
    }

    /// `group_plane_and_point_of_sublist @ 0x1806CDEE0` followed by the
    /// recursive `sort_plane_and_point @ 0x1806CE070`. Partitions the
    /// `sorted_order[start..end]` run (a single sort-layer) into
    /// plane-bearing elements (`use_plane`) followed by point elements,
    /// then BSP-sorts them so the result is a stable back-to-front order
    /// independent of camera rotation. Operates in place on `sorted_order`.
    fn plane_sort_sublist(&mut self, start: usize, end: usize) {
        // Group: planes first (preserving stage-1 order), then points.
        let mut grouped: Vec<u16> = Vec::with_capacity(end - start);
        let mut points: Vec<u16> = Vec::with_capacity(end - start);
        for &oi in &self.sorted_order[start..end] {
            if self.transparents[oi as usize].use_plane {
                grouped.push(oi);
            } else {
                points.push(oi);
            }
        }
        let plane_count = grouped.len();
        grouped.extend_from_slice(&points);
        let out = self.sort_plane_and_point(&grouped, plane_count);
        self.sorted_order[start..end].copy_from_slice(&out);
    }

    /// Recursive BSP split of one sort-layer sublist. `items` is
    /// `[plane elements..., point elements...]` (length = sublist size);
    /// `plane_count` is how many leading entries carry a usable plane.
    /// Mirrors `sort_plane_and_point @ 0x1806CE070`:
    ///   - pick the highest-`importance` plane as the split plane,
    ///   - partition the other planes + all points to the behind/front
    ///     side of it (points by `dot(centroid, n) - d` sign; planes by
    ///     their centroid, the `radius → 0` limit of the engine's
    ///     9-anchor `plane_location_testing` — exact for non-intersecting
    ///     planes, which is every discrete water/glass surface here),
    ///   - emit `[behind..., split, front...]`, recurse on each side.
    /// Camera-independent ⇒ stable under rotation. Returns the reordered
    /// index list (back-to-front).
    fn sort_plane_and_point(&self, items: &[u16], plane_count: usize) -> Vec<u16> {
        if items.len() <= 1 || plane_count == 0 {
            // No splitter: leave the stage-1 (z_sort) order untouched.
            return items.to_vec();
        }
        // Choose the splitter: highest-importance plane element.
        let mut split_pos = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &oi) in items[..plane_count].iter().enumerate() {
            let imp = self.transparents[oi as usize].importance;
            if imp > best {
                best = imp;
                split_pos = i;
            }
        }
        let split = items[split_pos];
        let sp = self.transparents[split as usize].plane;

        // Signed side of a centroid w.r.t. the (camera-faced) split plane.
        // < 0 = behind (away from camera) = drawn first.
        let side = |oi: u16| -> f32 {
            let c = self.transparents[oi as usize].centroid;
            sp[0] * c[0] + sp[1] * c[1] + sp[2] * c[2] - sp[3]
        };

        let mut left_planes: Vec<u16> = Vec::new();
        let mut right_planes: Vec<u16> = Vec::new();
        for (i, &oi) in items[..plane_count].iter().enumerate() {
            if i == split_pos {
                continue;
            }
            if side(oi) < 0.0 {
                left_planes.push(oi);
            } else {
                right_planes.push(oi);
            }
        }
        let mut left_points: Vec<u16> = Vec::new();
        let mut right_points: Vec<u16> = Vec::new();
        for &oi in &items[plane_count..] {
            if side(oi) < 0.0 {
                left_points.push(oi);
            } else {
                right_points.push(oi);
            }
        }

        // Recurse each side, planes-before-points so a side's own
        // splitter is available.
        let left_plane_n = left_planes.len();
        let mut left = left_planes;
        left.extend_from_slice(&left_points);
        let left_sorted = self.sort_plane_and_point(&left, left_plane_n);

        let right_plane_n = right_planes.len();
        let mut right = right_planes;
        right.extend_from_slice(&right_points);
        let right_sorted = self.sort_plane_and_point(&right, right_plane_n);

        // [behind..., split, front...] = back-to-front.
        let mut out = Vec::with_capacity(items.len());
        out.extend_from_slice(&left_sorted);
        out.push(split);
        out.extend_from_slice(&right_sorted);
        out
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

    /// Debug: dump the current marker batch's post-sort draw order with
    /// both the engine metric (view-forward depth, the live `z_sort`)
    /// and the old euclidean distance, so the two can be compared. Front
    /// (drawn last / on top) is the LAST line. Gated by the caller.
    pub fn debug_dump_sorted(&self, camera_pos: [f32; 3], label: &str) {
        eprintln!("[diag-sort] {label}: {} elems (back-to-front; last = on top)", {
            let s = self.current_batch_start();
            let e = self.total_transparent_count as usize;
            e.saturating_sub(s)
        });
        for (rank, el) in self.sorted_elements().enumerate() {
            let c = el.centroid;
            let dx = c[0] - camera_pos[0];
            let dy = c[1] - camera_pos[1];
            let dz = c[2] - camera_pos[2];
            let eucl = (dx * dx + dy * dy + dz * dz).sqrt();
            eprintln!(
                "  [{rank:2}] {:?} layer={:?} centroid=[{:.1},{:.1},{:.1}] z_sort(fwd)={:.2} eucl={:.1} \
                 use_plane={} imp={:.4} plane={:?} r={:.2}",
                el.dispatch, el.sort_layer, c[0], c[1], c[2], el.z_sort, eucl,
                el.use_plane, el.importance, el.plane, el.radius,
            );
        }
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
            &ctx.shared.camera_bind_group_transparent,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                // atmosphere @ binding 2 — default; per-element binds below
                // override it (sky transparents → their custom fog setting).
                crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
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
                    // Empty `cluster_to_probe` = no cubemap atlas → length-1
                    // rasg-default-cube vec, bind slot 0. A populated table
                    // with an OOB cluster/probe index is a real bug → fail loud.
                    let bind_group = if bsp.cluster_to_probe.is_empty() {
                        bind_groups.first()
                    } else {
                        let probe_idx = bsp.cluster_to_probe[cluster_index as usize] as usize;
                        Some(&bind_groups[probe_idx])
                    };
                    let Some(bind_group) = bind_group else { continue };
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
                        &ctx.shared.camera_bind_group_transparent,
                        // [1]=atmosphere @ binding 2 (BSP cluster default);
                        // last = dominant_light @ binding 13.
                        &[
                            lighting_offset,
                            crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                            simple_lights_offset,
                            lighting_offset,
                        ],
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
                    // Per-instance entry/probe selection — engine
                    // `render_instance_mesh_part @ 0x18068FB20` ALWAYS routes
                    // instances through `select_instance_entry_point` (per-vertex
                    // SH / single probe), even for transparents — UNLIKE the
                    // cluster path (`render_transparent_cluster_mesh_part`) which
                    // forces per_pixel. Mirror the opaque
                    // `StructureRenderer::render_instance_mesh_part`. Forcing
                    // StaticPerPixel + default lighting here lit transparent glass
                    // from the empty-atlas → bright-sky fallback → white frost;
                    // the real per-vertex baked SH is dim/local (correct).
                    use crate::halo::render_methods::HaloEntryPoint as Ep;
                    let sel = bsp
                        .instance_selections
                        .get(structure_instance_index as usize)
                        .map(|s| s.entry);
                    let (artifacts_opt, bind_groups_opt) = match sel {
                        Some(Ep::StaticPerPixel) => (material.artifacts.as_ref(), material.bind_group.as_ref()),
                        Some(Ep::StaticShPerVertex) => (
                            material.artifacts_sh_per_vertex.as_ref(),
                            material.bind_group_sh_per_vertex.as_ref(),
                        ),
                        Some(Ep::StaticPrtAmbient) => (
                            material.artifacts_prt_ambient.as_ref(),
                            material.bind_group_prt_ambient.as_ref(),
                        ),
                        _ => (material.artifacts_sh.as_ref(), material.bind_group_sh.as_ref()),
                    };
                    let (Some(artifacts), Some(bind_groups)) = (artifacts_opt, bind_groups_opt) else {
                        continue;
                    };
                    // Instance probe selection — todo: per-instance
                    // nearest-probe lookup. For now bind probe[0].
                    let Some(bind_group) = bind_groups.first() else { continue };
                    let lighting_offset = bsp
                        .instance_lighting_offsets
                        .get(structure_instance_index as usize)
                        .copied()
                        .unwrap_or(crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET);
                    // Rebind group 0 with the per-instance lighting offset +
                    // BSP cluster-default atmosphere (so a preceding SkyMeshPart's
                    // custom fog doesn't leak into this instance).
                    rpass.set_bind_group(
                        0,
                        &ctx.shared.camera_bind_group_transparent,
                        &[
                            lighting_offset,
                            crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                            crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                            lighting_offset,
                        ],
                    );
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
                    // Per-vertex SH stream (slot 1) — StaticShPerVertex only.
                    if matches!(sel, Some(Ep::StaticShPerVertex)) {
                        if let Some(pv_buf) = bsp
                            .instance_per_vertex_sh_buffers
                            .get(structure_instance_index as usize)
                            .and_then(|o| o.as_ref())
                        {
                            rpass.set_vertex_buffer(1, pv_buf.slice(..));
                        }
                    }
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
                TransparentDispatch::Object { object_slot, mesh_index, part_index, probe_index } => {
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
                    // Rebind group 0 with this object material's atmosphere
                    // offset (honors any per-shader fog override; resets a
                    // preceding SkyMeshPart's setting so it doesn't leak here).
                    rpass.set_bind_group(
                        0,
                        &ctx.shared.camera_bind_group_transparent,
                        &[
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                            material.atmosphere_offset,
                            crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                        ],
                    );
                    rpass.set_pipeline(&material.artifacts.pipeline);
                    rpass.set_bind_group(1, &ctx.shared.model_bind_group, &[model_offset]);
                    // Per-cluster dynamic-env cube: pick this object's probe
                    // variant when the material has them (engine
                    // `render_method_submit_dynamic_cubemap`); else the
                    // rasg-default `bind_group`. Path B: cbuffer offset 0.
                    let obj_bind_group = match probe_index {
                        Some(p) if !material.bind_group_probes.is_empty() => material
                            .bind_group_probes
                            .get(p as usize)
                            .unwrap_or(&material.bind_group),
                        _ => &material.bind_group,
                    };
                    rpass.set_bind_group(2, obj_bind_group, &[0u32]);
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
                        &ctx.shared.camera_bind_group_transparent,
                        &[
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                            // Per-shader atmosphere override — transparent sky
                            // backdrop (nebula, planetoid, clouds) resolves to
                            // its `Custom fog setting index` (construct →
                            // fog_waterfall), the bright daytime haze.
                            material.atmosphere_offset,
                            crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                        ],
                    );
                    rpass.set_bind_group(1, &ctx.sky_gpu.model_bind_group, &[0u32]);
                    rpass.set_bind_group(3, &ctx.sky_gpu.node_matrices_bind_group, &[0u32]);
                    rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    // Engine routes vc-bit sky meshes through
                    // `_entry_point_vertex_color_lighting` — albedo×vert_color
                    // + self_illum, NOT static_sh. Use the vertex_color
                    // variant when the mesh carries the stream and the
                    // material compiled it; else fall back to static_sh.
                    let vc = mesh.vert_color_buffer.is_some();
                    if let (true, Some(arts), Some(bg)) = (
                        vc,
                        material.artifacts_vertex_color.as_ref(),
                        material.bind_group_vertex_color.as_ref(),
                    ) {
                        rpass.set_vertex_buffer(
                            1,
                            mesh.vert_color_buffer.as_ref().unwrap().slice(..),
                        );
                        rpass.set_pipeline(&arts.pipeline);
                        rpass.set_bind_group(2, bg, &[0u32]);
                    } else {
                        rpass.set_pipeline(&material.artifacts.pipeline);
                        // Path B: cbuffer slot dynamic; offset 0.
                        rpass.set_bind_group(2, &material.bind_group, &[0u32]);
                    }
                    rpass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(
                        part.index_start..part.index_start + part.index_count,
                        0, 0..1,
                    );
                }
                TransparentDispatch::EmitterInstance { draw_index } => {
                    // Draw exactly this emitter-instance's rows so it
                    // composites at its OWN world depth (engine
                    // `c_particle_emitter::render_callback` →
                    // `c_particle_emitter_gpu::render`). The particle
                    // pipeline binds its OWN group0 (camera + grid +
                    // frame_params + scene-depth, a different layout than
                    // the structure camera group), so restore the structure
                    // group0 afterward for the next BSP/object element
                    // (those arms rely on the once-before-loop binding and
                    // don't rebind group 0 themselves).
                    if let Some(d) = ctx.game.effects.emitter_draws.get(draw_index as usize) {
                        let s = d.span_start as usize;
                        let e = s + d.span_count as usize;
                        if let Some(spans) = ctx.game.effects.emitter_draw_spans.get(s..e) {
                            ctx.particle_gpu.draw_emitter(rpass, d.batch as usize, spans);
                        }
                    }
                    rpass.set_bind_group(
                        0,
                        &ctx.shared.camera_bind_group_transparent,
                        &[
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                            crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                            crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                        ],
                    );
                }
                TransparentDispatch::PatchyFog => {
                    // Fullscreen multiply-add fog composite (engine
                    // `render_patchy_fog_callback`). Sorted at z_sort≈0 so
                    // everything in its layer has already drawn and gets
                    // fogged. Binds its OWN group 0 → restore the structure
                    // camera group afterward (same contract as Particle).
                    ctx.patchy_fog_pass.draw(rpass);
                    rpass.set_bind_group(
                        0,
                        &ctx.shared.camera_bind_group_transparent,
                        &[
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                            crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                            crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                        ],
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
