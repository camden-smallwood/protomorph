//! `StructureRenderer` — Rust idiom of `c_structure_renderer` (Ares
//! `render/render_structure.{h,cpp}`). Owns the BSP-side per-frame
//! state that the engine keeps in `g_render_structure_globals`.
//!
//! Method names mirror the engine (drop the `c_` prefix per project
//! style). State fields drop `m_`/`s_` prefixes.
//!
//! References (dllcache addresses):
//! - `submit_visibility @ 0x18068E860`
//! - `render_albedo @ 0x18068FF90`
//! - `render_static_lighting @ 0x180690060`
//! - `render_cluster_parts @ 0x18068F6A0` (instances first, then clusters)
//! - `render_cluster_mesh_part @ 0x18068F7F0`
//! - `render_instance_mesh_part @ 0x18068FB20`
//!
//! See `reference_h3_structure_render_pipeline.md` for the deep
//! dllcache trace of the call chain.


use crate::halo::render::shared::FrameContext;
use crate::halo::render_methods::HaloEntryPoint;
use crate::halo::structures::BspGpu;

/// `e_render_structure_mesh_part_flags` (Ares `render_structure.h:15`).
/// Bits set on `RenderClusterPart.flags` / `RenderInstancePart.flags`
/// by `submit_visibility`. The pass entry points (`render_albedo`,
/// `render_static_lighting`, `render_transparents`) AND a mask
/// against these to filter parts.
pub mod mesh_part_flags {
    /// `_render_structure_mesh_part_flag_lit_bit` — opaque sub-pass 0
    /// (lit but not shadow-casting).
    pub const LIT: u32 = 1 << 0;
    /// `_render_structure_mesh_part_flag_shadow_casting_bit` — opaque
    /// sub-pass 1 (casts shadows).
    pub const SHADOW_CASTING: u32 = 1 << 1;
    /// `_render_structure_mesh_part_flag_transparent_bit` — bit set
    /// when `c_render_method::get_is_transparent` returns true. The
    /// transparent pass uses `mask = TRANSPARENT`; the opaque passes
    /// use `mask = LIT | SHADOW_CASTING` (= 3) which excludes
    /// transparents AND part_type=0 (`opaque_not_drawn`).
    pub const TRANSPARENT: u32 = 1 << 2;

    /// Mask passed to `render_albedo` + `render_static_lighting` (= 3
    /// in dllcache `render_albedo @ 0x18068FF90`). Renders parts with
    /// `(flags & ALBEDO_MASK) != 0`. Excludes opaque_not_drawn,
    /// transparent, and lightmap_only parts.
    pub const ALBEDO_MASK: u32 = LIT | SHADOW_CASTING;
}

use blam_tags::render_model::GeometryPartType;

/// Map [`GeometryPartType`] (Ares `e_geometry_part_type`) to the runtime
/// `RenderClusterPart.flags` mask bits. The engine does this in the
/// visibility traversal (`s_visible_cluster.flags`); we compute the same
/// mapping here from `part.part_type` directly.
///
/// - `OpaqueShadowOnly` → LIT (lit pass; not in albedo without
///   shadow_casting in the mask)
/// - `OpaqueShadowCasting` / `OpaqueNonShadowing` → LIT | SHADOW_CASTING
/// - `OpaqueNotDrawn` → 0 (never renders)
/// - `Transparent` → 0 (queued via `TransparencyRenderer` instead)
/// - `LightmapOnly` → 0 (offline lightmap baking only — `z_invis_lightquad`)
pub fn part_type_to_flags(part_type: GeometryPartType) -> u32 {
    use mesh_part_flags::*;
    use GeometryPartType::*;
    match part_type {
        OpaqueShadowOnly => LIT,
        OpaqueShadowCasting | OpaqueNonShadowing => LIT | SHADOW_CASTING,
        OpaqueNotDrawn | Transparent | LightmapOnly => 0,
    }
}

/// Engine `part_is_renderable @ 0x18069eb90`:
/// ```c
/// return part->type != 5                       // NOT lightmap_only
///     && part->render_method_index != -1
///     && materials[part->render_method_index].render_method.index != -1;
/// ```
/// `submit_visibility` calls this as a GATE before computing the
/// per-part render-pass flags — a non-renderable part is never added to
/// any pass, *including the transparent pass*. The `render_method` /
/// material validity checks are already handled upstream here (parts
/// with a `None` material slot are skipped), so this only mirrors the
/// `part->type != 5` clause: [`GeometryPartType::LightmapOnly`]
/// (`z_invis_lightquad`) parts are invisible quads baked solely to feed
/// the offline lightmapper and are never drawn at runtime. Without this
/// gate the `is_transparent` shortcut (always true for rmhg/rmw/etc.)
/// would route these invisible quads into the transparent pass and draw
/// them (e.g. s3d_lockout `lct_collision` halogram lightquads rendering
/// as flat grey planes).
#[inline]
pub fn part_is_renderable(part_type: GeometryPartType) -> bool {
    part_type != GeometryPartType::LightmapOnly
}

/// Transform a model-space transparent sort plane `(i, j, k, d)` to world
/// by the instance matrix — engine `matrix4x3_transform_plane` inside
/// `c_structure_renderer::submit_visibility @ 0x18068E860`. Returns `None`
/// for a degenerate (zero-normal / unauthored) plane so the element falls
/// back to point-only sorting. Assumes a rigid + uniform-scale matrix
/// (instance placements are scale × orthonormal-basis + translation): the
/// world normal is the rotated unit normal, and `d` is recomputed through
/// the transformed closest-point-to-origin (`n·d`).
fn transform_sort_plane(m: &glam::Mat4, sort_plane: Option<[f32; 4]>) -> Option<[f32; 4]> {
    let p = sort_plane?;
    let n = glam::Vec3::new(p[0], p[1], p[2]);
    let nl = n.length();
    if nl <= 1e-6 {
        return None;
    }
    let n_unit = n / nl;
    let p_on = n_unit * (p[3] / nl);
    let n_world = m.transform_vector3(n_unit).normalize_or_zero();
    if n_world == glam::Vec3::ZERO {
        return None;
    }
    let p_world = m.transform_point3(p_on);
    Some([n_world.x, n_world.y, n_world.z, n_world.dot(p_world)])
}

/// `s_render_cluster_part` (Ares `render_structure.h:56`, 12 bytes).
#[derive(Debug, Clone, Copy)]
pub struct RenderClusterPart {
    /// Index into `Renderer::bsps`.
    pub bsp_index: u8,
    /// Cluster index within the BSP.
    pub cluster_index: u16,
    /// Mesh index within `BspGpu::meshes`.
    pub mesh_index: u16,
    /// Part index within `BspMesh::parts`.
    pub part_index: u16,
    /// `submit_visibility` sets this to:
    ///   `is_transparent ? TRANSPARENT : (part_type & 3)`
    /// — the render-pass mask gate AND's against this.
    pub flags: u32,
}

/// `s_render_instance_part` (Ares `render_structure.h:65`, 16 bytes).
#[derive(Debug, Clone, Copy)]
pub struct RenderInstancePart {
    pub bsp_index: u8,
    pub structure_instance_index: u16,
    pub mesh_index: u16,
    pub part_index: u16,
    pub alpha_byte: u8,
    pub flags: u32,
}

/// `c_structure_renderer` — owns per-frame BSP draw state.
///
/// Engine has this as a static-method class operating on a global
/// `g_render_structure_globals`; we own the state on the struct and
/// the methods take `&mut self` / `&self` as appropriate.
pub struct StructureRenderer {
    /// `render_cluster_parts` flat list — built each frame by
    /// `submit_visibility`.
    pub cluster_parts: Vec<RenderClusterPart>,
    /// `render_instance_parts` flat list — same.
    pub instance_parts: Vec<RenderInstancePart>,
    /// `g_render_structure_globals.render_geometry[bsp_index]` —
    /// per-BSP runtime data. Loaded by `Renderer::load_bsp` →
    /// `BspGpu::from_loaded_bsp`. Drawn through cluster_parts /
    /// instance_parts after `submit_visibility` populates them.
    pub bsps: Vec<BspGpu>,
    /// Per-BSP cluster→mesh_index lookup table — what
    /// `g_render_structure_globals.render_geometry[i].clusters[c].mesh_index`
    /// would return. Sized one entry per loaded BSP.
    pub bsp_cluster_meshes: Vec<Vec<i16>>,
    /// Per-BSP cluster bounding spheres (center, radius) derived from
    /// `bounds_x/y/z` at scenario load. Used by `submit_visibility` to
    /// skip clusters outside the camera frustum (engine equivalent:
    /// `cluster_build_object_payload @ 0x1807E50B0` pre-bakes per-cluster
    /// cull flags + `obb_visible @ 0x1806BDF50` runtime test).
    pub bsp_cluster_bounds: Vec<Vec<(glam::Vec3, f32)>>,
    /// Per-BSP instance bounding spheres (center, radius). Sourced
    /// directly from `BspInstance::world_bounding_sphere_*` (tag-time
    /// precomputed, mirrors Halo authoring tool output). Used by
    /// `submit_visibility` to skip out-of-frustum instances — the
    /// instance loop is the dominant per-frame cost on instance-heavy
    /// BSPs (e.g. cyberdyne has 834 instances vs guardian's 479,
    /// 5× FPS gap before this cull).
    pub bsp_instance_bounds: Vec<Vec<(glam::Vec3, f32)>>,
    /// Shared identity bind groups for cluster draws — clusters
    /// render at world-coords with no per-instance transform. The
    /// engine has `c_rasterizer::set_shader_constant` shape-2; we use
    /// a shared identity uniform + 1-bone identity node-matrix that
    /// every cluster part binds as model + node-matrices.
    pub identity_model_bg: Option<wgpu::BindGroup>,
    pub identity_nm_bg: Option<wgpu::BindGroup>,
    /// Backing buffers for the identity bind groups (kept alive for
    /// the bind groups' lifetime).
    pub identity_buffers: Option<(wgpu::Buffer, wgpu::Buffer)>,
    /// Next free slot in the global `engine_lighting_buffer` for
    /// per-instance SH probes. Slot 0 is reserved for the
    /// frame-default lighting; slots 1..N hold dequantized
    /// `lbsp.probes[i]` written by `BspGpu::from_loaded_bsp`.
    pub next_probe_slot: u32,
}

impl StructureRenderer {
    pub fn new() -> Self {
        Self {
            cluster_parts: Vec::new(),
            instance_parts: Vec::new(),
            bsps: Vec::new(),
            bsp_cluster_meshes: Vec::new(),
            bsp_cluster_bounds: Vec::new(),
            bsp_instance_bounds: Vec::new(),
            identity_model_bg: None,
            identity_nm_bg: None,
            identity_buffers: None,
            next_probe_slot: 1, // slot 0 = frame default
        }
    }

    /// Reset per-scenario BSP state to the freshly-constructed default
    /// so a second `load_scenario` (map switch) doesn't append/leak onto
    /// the previous map's geometry. A NO-OP on a fresh renderer's first
    /// load: every field below is already empty / `next_probe_slot == 1`,
    /// so the clears are byte-identical to today.
    ///
    /// `bsps`, `bsp_cluster_meshes`, `bsp_cluster_bounds` and
    /// `bsp_instance_bounds` are all repopulated
    /// by `Renderer::upload_bsp` (driven from `game.rs::load_scene` after
    /// `load_scenario` returns). `cluster_parts` / `instance_parts` are
    /// rebuilt every frame by `submit_visibility`, so they're cleared here
    /// only for tidiness. `identity_*` bindings are reinitialized by
    /// `upload_bsp` → `init_bsp_identity_bindings` (which gates on
    /// `is_none()`), so they're left alone — they're shared per-frame
    /// scratch bindings, not per-scenario content.
    pub fn reset(&mut self) {
        self.cluster_parts.clear();
        self.instance_parts.clear();
        self.bsps.clear();
        self.bsp_cluster_meshes.clear();
        self.bsp_cluster_bounds.clear();
        self.bsp_instance_bounds.clear();
        self.next_probe_slot = 1; // slot 0 = frame default
    }

    /// `submit_visibility(submit_transparents) @ 0x18068E860`.
    ///
    /// Walks visible clusters + instances and appends to the flat
    /// draw lists. When a `frustum_planes` slice is provided, each
    /// cluster's pre-baked bounding sphere is tested against the 5-
    /// plane Gribb-Hartmann camera frustum. Engine equivalent uses
    /// `visibility_region_test_sphere` against the PVS-walked region;
    /// the projections walker (Phase E4) is shipped but produces
    /// false-negatives in some configurations, so until that's traced
    /// down the safer cull is the camera frustum.
    ///
    /// `cluster_metadata[bsp_idx][cluster_idx]` gives the mesh index
    /// for that cluster (`-1` skips). Transparent parts are routed
    /// into `transparency`.
    pub fn submit_visibility(
        &mut self,
        transparency: &mut crate::halo::render::render_transparents::TransparencyRenderer,
        frustum_planes: Option<&[glam::Vec4; 5]>,
    ) {
        use crate::halo::render::render_transparents::TransparentDispatch;

        let sphere_in_frustum = |center: glam::Vec3, radius: f32| -> bool {
            let Some(planes) = frustum_planes else { return true; };
            for plane in planes {
                let s = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
                if s < -radius {
                    return false;
                }
            }
            true
        };

        // Transparent submission is NOT view-frustum culled — engine-faithful.
        // `c_structure_renderer::submit_visibility @ 0x18068E860` walks the
        // PVS-visible `c_visible_items` and submits EVERY transparent part with
        // no frustum test, letting the rasterizer clip + the depth test reject
        // anything behind opaque geometry. A bounding-sphere-vs-frustum test
        // (camera-DIRECTION dependent) instead pops transparents (glass,
        // waterfalls) in/out as the sphere center crosses a frustum plane even
        // while geometry is still on screen. So: transparent parts always
        // submit; OPAQUE parts stay frustum-culled (`in_frustum`) for perf.
        // (Until the PVS walker is trusted enough to also bound transparents by
        // visible-cluster set, every transparent BSP part map-wide is submitted;
        // depth handles occlusion. Watch the 1024-element pool cap on very large
        // maps — switch to `submit_visibility_from_items` when PVS is reliable.)
        // DIAGNOSTIC (PROTOMORPH_SKIP_SHADER=substr): drop transparent parts
        // whose shader path contains the substring, to isolate which shader is
        // producing a visible artifact (e.g. =glass_activation).
        let skip_shader = std::env::var("PROTOMORPH_SKIP_SHADER").ok();

        self.cluster_parts.clear();
        self.instance_parts.clear();
        transparency.reset();

        // DIAGNOSTIC (PROTOMORPH_NO_BSP_CLUSTERS=1): skip ALL cluster parts too
        // (pairs with PROTOMORPH_NO_BSP_INSTANCES). With both set + NO_OBJECTS,
        // nothing structural is drawn — if the first-frame freeze SURVIVES, the
        // hang is not in the structure draw path at all (look upstream: lightmap
        // / env-probe / cubemap upload, geometry decode, a non-draw pass).
        let skip_clusters = std::env::var_os("PROTOMORPH_NO_BSP_CLUSTERS").is_some();

        for (bsp_idx, bsp) in self.bsps.iter().enumerate() {
            let bsp_index = bsp_idx as u8;

            // Clusters.
            let cluster_bounds_for_bsp = self.bsp_cluster_bounds.get(bsp_idx);
            if let Some(cluster_meshes) = self.bsp_cluster_meshes.get(bsp_idx) {
                for (cluster_idx, &mesh_idx) in cluster_meshes.iter().enumerate() {
                    if skip_clusters {
                        break;
                    }
                    if mesh_idx < 0 {
                        continue;
                    }
                    // Opaque parts cull on this; transparent parts ignore it
                    // (always submitted — see policy note above). We can't skip
                    // the whole cluster on `!in_frustum` anymore: an off-frustum
                    // cluster may still carry transparent parts we must submit.
                    let in_frustum = cluster_bounds_for_bsp
                        .and_then(|b| b.get(cluster_idx))
                        .map_or(true, |bounds| sphere_in_frustum(bounds.0, bounds.1));
                    let mesh_idx = mesh_idx as usize;
                    let Some(mesh) = bsp.meshes.get(mesh_idx) else { continue };
                    if mesh.parts.is_empty() {
                        continue;
                    }
                    for (part_idx, part) in mesh.parts.iter().enumerate() {
                        let Some(mat_slot) = bsp.materials.get(part.material_index as usize) else {
                            continue;
                        };
                        let Some(mat) = mat_slot.as_ref() else { continue };
                        // Engine `part_is_renderable` gate (BEFORE the
                        // transparency classification): lightmap_only parts
                        // (part_type 5) are never drawn, even for transparent
                        // materials.
                        if !part_is_renderable(part.part_type) {
                            continue;
                        }
                        // Engine: cluster_part.flags =
                        //   is_transparent ? TRANSPARENT : part_type_to_flags(part_type)
                        let flags = if mat.is_transparent {
                            mesh_part_flags::TRANSPARENT
                        } else {
                            part_type_to_flags(part.part_type)
                        };
                        if mat.is_transparent {
                            // BSP geometry is world-space (identity), so the
                            // part's authored sort centroid + plane ARE world —
                            // pass straight through. Submitted UNCONDITIONALLY
                            // (no frustum gate) per engine `submit_visibility`.
                            if std::env::var("PROTOMORPH_DIAG_SORT_PLANE").is_ok() {
                                let n = part.sort_plane.map(|p| {
                                    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
                                });
                                eprintln!(
                                    "[sortplane] cluster bsp{bsp_index} c{cluster_idx} m{mesh_idx} p{part_idx} \
                                     radius={:.4} |n|={:?} plane={:?}",
                                    part.sort_radius, n, part.sort_plane,
                                );
                            }
                            transparency.add_element(
                                part.sort_centroid.unwrap_or([0.0, 0.0, 0.0]),
                                part.sort_plane,
                                part.sort_radius,
                                0.0,
                                mat.sort_layer,
                                TransparentDispatch::BspClusterPart {
                                    bsp_index,
                                    cluster_index: cluster_idx as u16,
                                    mesh_index: mesh_idx as u16,
                                    part_index: part_idx as u16,
                                },
                            );
                            continue;
                        }
                        // Opaque parts: never drawn when off-screen.
                        if !in_frustum {
                            continue;
                        }
                        self.cluster_parts.push(RenderClusterPart {
                            bsp_index,
                            cluster_index: cluster_idx as u16,
                            mesh_index: mesh_idx as u16,
                            part_index: part_idx as u16,
                            flags,
                        });
                    }
                }
            }

            // Instances. Resolution chain mirrors
            // `render_structure_instance` lookup: instance →
            // definitions[instance.definition_index].mesh_index →
            // bsp.meshes[mesh_index].parts[*].
            //
            // DIAGNOSTIC (PROTOMORPH_NO_BSP_INSTANCES=1): skip ALL structure
            // instances and render cluster geometry only. Bisection tool for
            // the huge-BSP first-frame freeze — isolate whether the hang is in
            // cluster draws or the (potentially numerous) instance draws.
            let skip_instances = std::env::var_os("PROTOMORPH_NO_BSP_INSTANCES").is_some();
            let instance_bounds_for_bsp = self.bsp_instance_bounds.get(bsp_idx);
            for (inst_idx, inst) in bsp.instances.iter().enumerate() {
                if skip_instances {
                    break;
                }
                let def_i = inst.definition_index;
                if def_i < 0 {
                    continue;
                }
                // Per-instance frustum cull — tag-precomputed
                // world_bounding_sphere from BspInstance. Gates OPAQUE instance
                // parts only; transparent parts always submit (engine-faithful,
                // see policy note above), so we can't skip the whole instance.
                let in_frustum = instance_bounds_for_bsp
                    .and_then(|b| b.get(inst_idx))
                    .map_or(true, |bounds| sphere_in_frustum(bounds.0, bounds.1));
                let Some(def) = bsp.definitions.get(def_i as usize) else { continue };
                if def.mesh_index < 0 {
                    continue;
                }
                let mesh_idx = def.mesh_index as usize;
                let Some(mesh) = bsp.meshes.get(mesh_idx) else { continue };
                for (part_idx, part) in mesh.parts.iter().enumerate() {
                    let Some(mat_slot) = bsp.materials.get(part.material_index as usize) else {
                        continue;
                    };
                    let Some(mat) = mat_slot.as_ref() else { continue };
                    // Engine `part_is_renderable` gate — skip lightmap_only
                    // (part_type 5) parts even when the material is transparent.
                    if !part_is_renderable(part.part_type) {
                        continue;
                    }
                    let flags = if mat.is_transparent {
                        mesh_part_flags::TRANSPARENT
                    } else {
                        part_type_to_flags(part.part_type)
                    };
                    if mat.is_transparent {
                        // Instance parts are placed by the instance matrix; use
                        // the instance's world bounding-sphere center as the
                        // sort centroid (per-instance back-to-front). Submitted
                        // UNCONDITIONALLY (no frustum gate) per engine
                        // `submit_visibility`; PROTOMORPH_SKIP_SHADER still
                        // isolates a specific shader for diagnostics.
                        let skip = skip_shader
                            .as_deref()
                            .is_some_and(|s| mat.shader_name.contains(s));
                        if !skip {
                            if std::env::var("PROTOMORPH_DIAG_SORT_PLANE").is_ok() {
                                let n = part.sort_plane.map(|p| {
                                    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
                                });
                                eprintln!(
                                    "[sortplane] instance bsp{bsp_index} i{inst_idx} m{mesh_idx} p{part_idx} \
                                     radius={:.4} |n|={:?} plane={:?} shader={}",
                                    part.sort_radius, n, part.sort_plane, mat.shader_name,
                                );
                            }
                            let inst_center = instance_bounds_for_bsp
                                .and_then(|b| b.get(inst_idx))
                                .map(|b| [b.0.x, b.0.y, b.0.z])
                                .unwrap_or([0.0, 0.0, 0.0]);
                            // Instance geometry is placed by the instance
                            // matrix, so the authored (model-space) sort plane
                            // + centroid must be transformed to world — engine
                            // `matrix4x3_transform_plane` / `_transform_point`
                            // in `submit_visibility @ 0x18068E860`.
                            let m = &inst.world_matrix;
                            let world_centroid = part
                                .sort_centroid
                                .map(|c| {
                                    m.transform_point3(glam::Vec3::from(c)).to_array()
                                })
                                .unwrap_or(inst_center);
                            let world_plane = transform_sort_plane(m, part.sort_plane);
                            transparency.add_element(
                                world_centroid,
                                world_plane,
                                part.sort_radius,
                                0.0,
                                mat.sort_layer,
                                TransparentDispatch::BspInstancePart {
                                    bsp_index,
                                    structure_instance_index: inst_idx as u16,
                                    mesh_index: mesh_idx as u16,
                                    part_index: part_idx as u16,
                                },
                            );
                        }
                        continue;
                    }
                    // Opaque parts: never drawn when off-screen.
                    if !in_frustum {
                        continue;
                    }
                    self.instance_parts.push(RenderInstancePart {
                        bsp_index,
                        structure_instance_index: inst_idx as u16,
                        mesh_index: mesh_idx as u16,
                        part_index: part_idx as u16,
                        alpha_byte: 0xFF,
                        flags,
                    });
                }
            }
        }
        // No sort here — engine-faithful: sort happens once per marker
        // scope inside `c_player_view::render_transparents` (sky batch
        // first, then outer regular batch). Sorting here would be
        // overridden anyway since object transparents are added AFTER
        // BSP submit_visibility.
    }

    /// `render_albedo @ 0x18068FF90`. First-pass G-buffer fill — calls
    /// `render_cluster_parts(_entry_point_albedo, ALBEDO_MASK)`. Writes
    /// `_surface_albedo` (RGB + spec_mask) + normal G-buffer; no
    /// lighting / atmosphere / exposure.
    pub fn render_albedo<'a>(
        &self,
        rpass: &mut wgpu::RenderPass<'a>,
        ctx: &'a FrameContext<'a>,
    ) {
        self.render_cluster_parts(
            rpass,
            ctx,
            HaloEntryPoint::Albedo,
            mesh_part_flags::ALBEDO_MASK,
        );
    }

    /// `render_static_lighting @ 0x180690060`. Re-walks the same
    /// geometry sorted by `render_albedo` but with the per-cluster
    /// static-lighting entry-point — outputs the FINAL lit color into
    /// `lighting_base`, sampling albedo/normal from G-buffer when the
    /// shader supports it (currently re-computed inline, see entry_static_*.wgsl).
    pub fn render_static_lighting<'a>(
        &self,
        rpass: &mut wgpu::RenderPass<'a>,
        ctx: &'a FrameContext<'a>,
    ) {
        self.render_cluster_parts(
            rpass,
            ctx,
            HaloEntryPoint::StaticPerPixel,
            mesh_part_flags::ALBEDO_MASK,
        );
    }

    /// `render_cluster_parts(entry_point, mesh_part_mask) @ 0x18068F6A0`.
    /// Walks instance_parts FIRST (per dllcache), then cluster_parts.
    fn render_cluster_parts<'a>(
        &self,
        rpass: &mut wgpu::RenderPass<'a>,
        ctx: &'a FrameContext<'a>,
        entry_point: HaloEntryPoint,
        mesh_part_mask: u32,
    ) {
        let (Some(identity_model_bg), Some(identity_nm_bg)) =
            (self.identity_model_bg.as_ref(), self.identity_nm_bg.as_ref())
        else {
            return;
        };

        // ---- Instances first -------------------------------------
        for inst_part in &self.instance_parts {
            if (mesh_part_mask & inst_part.flags) == 0 {
                continue;
            }
            self.render_instance_mesh_part(rpass, ctx, inst_part, entry_point);
        }

        // ---- Clusters then ---------------------------------------
        // Restore frame-default lighting offset (instances may have
        // set per-instance probes via dynamic offset). Albedo pass
        // binds the placeholder group (writes `_surface_albedo`); all
        // other entry points bind the SL group with REAL G-buffer
        // views at slots 10/11.
        let camera_bg = match entry_point {
            HaloEntryPoint::Albedo => &ctx.shared.camera_bind_group,
            _ => &ctx.shared.camera_bind_group_sl,
        };
        rpass.set_bind_group(
            0,
            camera_bg,
            &[
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
                // atmosphere @ binding 2 — BSP uses the cluster default
                // (per-material overrides are sky/object-side for now).
                crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13 shares the same slot
                // cursor as the ravi cbuffer at binding 1.
                crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET,
            ],
        );
        rpass.set_bind_group(1, identity_model_bg, &[0u32]);
        rpass.set_bind_group(3, identity_nm_bg, &[0u32]);
        for part in &self.cluster_parts {
            if (mesh_part_mask & part.flags) == 0 {
                continue;
            }
            self.render_cluster_mesh_part(rpass, ctx, part, entry_point);
        }
    }

    /// `render_cluster_mesh_part @ 0x18068F7F0`. Called per
    /// cluster_part inside the rpass; we already gated on
    /// `(mesh_part_mask & part.flags) != 0` in the caller.
    fn render_cluster_mesh_part<'a>(
        &self,
        rpass: &mut wgpu::RenderPass<'a>,
        ctx: &'a FrameContext<'a>,
        part: &RenderClusterPart,
        entry_point: HaloEntryPoint,
    ) {
        let Some(bsp) = self.bsps.get(part.bsp_index as usize) else { return };
        let Some(mesh) = bsp.meshes.get(part.mesh_index as usize) else { return };
        let Some(mesh_part) = mesh.parts.get(part.part_index as usize) else { return };
        let Some(material_slot) = bsp.materials.get(mesh_part.material_index as usize) else {
            return;
        };
        let Some(material) = material_slot.as_ref() else { return };

        // Per-cluster lightmap policy: clusters with `pervertex_block_index >= 0`
        // have their SH baked as per-vertex (NOT in the atlas), so the
        // per-pixel atlas sample returns ~0 for those vertices.
        //
        // Engine path: `_entry_point_static_lighting_per_vertex` (idx=3)
        // takes per-vertex SH probes via a SECOND vertex stream and
        // evaluates ravi at fragment normal. We wire that whenever the
        // cluster has `cluster_per_vertex_sh_buffers[i] = Some(buf)`
        // (probe count matched mesh vertex count at load).
        //
        // Fallback when no per-vertex buffer is available: per-cluster
        // averaged-probe bridge via `cluster_lighting_offsets[i]`. Loses
        // spatial variance but keeps visible illumination correct.
        let cluster_offset = bsp
            .cluster_lighting_offsets
            .get(part.cluster_index as usize)
            .copied()
            .unwrap_or(0);
        let cluster_pv_buf = bsp
            .cluster_per_vertex_sh_buffers
            .get(part.cluster_index as usize)
            .and_then(|o| o.as_ref());
        // Engine `render_cluster_mesh_part @ 0x18068F7F0:111` forces
        // BOTH terrain (`rmtr`) AND foliage (`rmfl`) materials to
        // `_entry_point_static_lighting_sh` (the SH-cbuffer path) +
        // calls `c_lighting_interface::setup_default_lighting`,
        // regardless of lightmap atlas availability or per-vertex SH.
        // The engine check is `material.render_method.group_tag == 0x726D666C`
        // (= 'rmfl') in the per_pixel→sh swap chain; foliage shaders
        // don't have lightmap UV coverage, so atlas sampling would
        // return black/garbage and per-vertex SH coverage isn't
        // guaranteed for foliage clusters either. Treating both
        // subclasses the same as terrain is the safe engine-faithful
        // route.
        // Centralized cluster pipeline pick (mirrors the engine's
        // render_cluster_mesh_part @0x18068F7F0: rmtr/rmfl → SH) + protomorph's
        // no-lightmap-BSP → SH routing (loose-tag data gap, e.g. mainmenu, whose
        // empty per-pixel atlas would render StaticPerPixel clusters black). Lives
        // in select_entry_point.rs next to select_instance_entry_point so the
        // cluster + instance lighting selections can't silently diverge again.
        let sel = crate::halo::render::select_entry_point::select_cluster_entry_point(
            material.render_method_group_tag,
            bsp.has_lightmap,
            cluster_offset,
            cluster_pv_buf.is_some(),
            matches!(entry_point, HaloEntryPoint::Albedo),
        );
        let use_per_vertex = sel.use_per_vertex;
        let use_sh = sel.use_sh;

        let (artifacts_opt, bind_groups_opt) = match entry_point {
            HaloEntryPoint::Albedo => (
                material.artifacts_albedo.as_ref(),
                material.bind_group_albedo.as_ref(),
            ),
            _ if use_per_vertex => (
                material.artifacts_sh_per_vertex.as_ref(),
                material.bind_group_sh_per_vertex.as_ref(),
            ),
            _ if use_sh => (
                material.artifacts_sh.as_ref(),
                material.bind_group_sh.as_ref(),
            ),
            _ => (material.artifacts.as_ref(), material.bind_group.as_ref()),
        };
        let (Some(artifacts), Some(bind_groups)) = (artifacts_opt, bind_groups_opt) else {
            // Subclass not ported — material is registered for
            // classification + transparency queueing but has no
            // wgpu pipeline.
            return;
        };
        // Pick the cluster's chosen scenario probe — engine mirrors
        // `c_dynamic_cubemap_sample::search_for_cubemap_sample_in_cluster`
        // result. Fall back to slot 0 if the BSP has no atlas (length-1
        // bind group vec; the rasg default cube is bound).
        // Empty `cluster_to_probe` = this BSP has no cubemap atlas, so
        // `bind_groups` is the length-1 rasg-default-cube vec → bind slot 0.
        // A *populated* table with an OOB cluster index (or a probe index
        // past the atlas) is a real bug → fail loud rather than silently
        // binding the wrong/default cube.
        let bind_group = if bsp.cluster_to_probe.is_empty() {
            bind_groups.first()
        } else {
            let probe_idx = bsp.cluster_to_probe[part.cluster_index as usize] as usize;
            Some(&bind_groups[probe_idx])
        };
        let Some(bind_group) = bind_group else { return };

        // Per-cluster simple_lights offset — engine: each cluster gets
        // its own up-to-8 nearest stli lights, computed once at
        // scenario load and stable across camera motion. Slot 0 =
        // empty (no lights). Falls back to default if per-cluster
        // bake hasn't run yet.
        // Always sized `vec![0; cluster_count]` (0 == SIMPLE_LIGHTS_DEFAULT_OFFSET),
        // so an in-range cluster yields the same value as before; an OOB cluster
        // index is a real bug → fail loud instead of silently binding no lights.
        let simple_lights_offset =
            bsp.cluster_simple_lights_offsets[part.cluster_index as usize];

        // Always bind here (covers both `use_sh = true` overriding the
        // SH probe slot AND the simple_lights per-cluster slot, even
        // when SH lighting falls back to per-pixel atlas).
        //
        // Per-vertex path: the offset still feeds the per-cluster
        // averaged probe (used by the shader for dominant-light
        // direction/intensity); SH evaluation comes from the
        // interpolated vertex stream.
        let lighting_offset = if use_per_vertex || use_sh {
            cluster_offset
        } else {
            crate::halo::render::shared::ENGINE_LIGHTING_DEFAULT_OFFSET
        };
        let camera_bg = match entry_point {
            HaloEntryPoint::Albedo => &ctx.shared.camera_bind_group,
            _ => &ctx.shared.camera_bind_group_sl,
        };
        rpass.set_bind_group(
            0,
            camera_bg,
            // [1]=atmosphere @ binding 2 (BSP cluster default); last entry =
            // dominant_light @ binding 13; shares the ravi cbuffer cursor.
            &[
                lighting_offset,
                crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                simple_lights_offset,
                lighting_offset,
            ],
        );

        rpass.set_pipeline(&artifacts.pipeline);
        // Path B: cbuffer entry on material BGL is dynamic-offset; offset
        // 0 = slot for non-animated / non-per-object materials. Animated
        // per-object materials use slot_idx * slot_size in Phase 5.
        rpass.set_bind_group(2, bind_group, &[0u32]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        // Per-vertex SH stream (slot 1) — bound only for the
        // StaticShPerVertex pipeline (engine `_entry_point_static_
        // lighting_per_vertex` idx=3). The pipeline declared a 2-buffer
        // vertex layout in `pipeline_cache.rs`.
        if use_per_vertex {
            if let Some(pv_buf) = cluster_pv_buf {
                rpass.set_vertex_buffer(1, pv_buf.slice(..));
            }
        }
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(
            mesh_part.index_start..mesh_part.index_start + mesh_part.index_count,
            0,
            0..1,
        );

        // No restore needed — every cluster draw rebinds with its own
        // pair of offsets at the top of this fn.
    }

    /// `render_instance_mesh_part @ 0x18068FB20`. Per dllcache: binds
    /// the instance's world matrix to VS slot 12, picks the per-instance
    /// lighting offset (from `select_instance_entry_point`), and may
    /// override the vertex stream when the lightmap tag carries a
    /// per-instance UV stream.
    fn render_instance_mesh_part<'a>(
        &self,
        rpass: &mut wgpu::RenderPass<'a>,
        ctx: &'a FrameContext<'a>,
        inst_part: &RenderInstancePart,
        entry_point: HaloEntryPoint,
    ) {
        let Some(bsp) = self.bsps.get(inst_part.bsp_index as usize) else { return };
        let Some(mesh) = bsp.meshes.get(inst_part.mesh_index as usize) else { return };
        let Some(mesh_part) = mesh.parts.get(inst_part.part_index as usize) else { return };
        let Some(material_slot) = bsp.materials.get(mesh_part.material_index as usize) else {
            return;
        };
        let Some(material) = material_slot.as_ref() else { return };
        let Some(inst_model_bg) = bsp
            .instance_model_bgs
            .get(inst_part.structure_instance_index as usize)
        else {
            return;
        };
        let Some(inst_nm_bg) = bsp.instance_nm_bg.as_ref() else { return };

        // `select_instance_entry_point` precomputed which pipeline +
        // probe to use for the LIGHTING pass. The albedo G-buffer pass
        // is identical across all instances (no SH/per-vertex variation
        // — it just outputs albedo + normal), so we ignore the selection
        // and use `artifacts_albedo` directly.
        let sel = bsp
            .instance_selections
            .get(inst_part.structure_instance_index as usize)
            .copied();
        let entry_choice = sel.map(|s| s.entry);
        // Defense in depth: the StaticShPerVertex / StaticPrtAmbient pipelines
        // declare a 2nd vertex buffer (slot 1). Only select them when their
        // vb1 source actually exists — otherwise the draw below would bind a
        // 2-buffer pipeline with nothing in slot 1 and wgpu panics ("requires
        // vertex buffer 1 to be set"). The selector is gated the same way at
        // load (bsp_gpu.rs), but guard here too so any future drift degrades
        // to the 1-buffer SH pipeline instead of crashing.
        let pv_present = bsp
            .instance_per_vertex_sh_buffers
            .get(inst_part.structure_instance_index as usize)
            .and_then(|o| o.as_ref())
            .is_some();
        let prt_present = mesh.prt_ambient_buffer.is_some();
        let (artifacts, bind_groups) = match entry_point {
            HaloEntryPoint::Albedo => (
                material.artifacts_albedo.as_ref(),
                material.bind_group_albedo.as_ref(),
            ),
            _ => match entry_choice {
                Some(HaloEntryPoint::StaticPerPixel) => {
                    (material.artifacts.as_ref(), material.bind_group.as_ref())
                }
                Some(HaloEntryPoint::StaticShPerVertex) if pv_present => (
                    material.artifacts_sh_per_vertex.as_ref(),
                    material.bind_group_sh_per_vertex.as_ref(),
                ),
                Some(HaloEntryPoint::StaticPrtAmbient) if prt_present => (
                    material.artifacts_prt_ambient.as_ref(),
                    material.bind_group_prt_ambient.as_ref(),
                ),
                _ => (material.artifacts_sh.as_ref(), material.bind_group_sh.as_ref()),
            },
        };
        let (Some(artifacts), Some(bind_groups)) = (artifacts, bind_groups) else {
            // Subclass not ported — see render_cluster_mesh_part above.
            return;
        };
        // Instance probe selection — todo: per-instance nearest-probe
        // lookup. For now bind probe[0] (or the only entry when no
        // atlas). Engine equivalent would need instance world position
        // → cluster lookup → cluster_to_probe.
        let bind_group = bind_groups.first();
        let Some(bind_group) = bind_group else { return };
        rpass.set_pipeline(&artifacts.pipeline);
        // Path B: cbuffer entry on material BGL is dynamic-offset; offset
        // 0 = slot for non-animated / non-per-object materials. Animated
        // per-object materials use slot_idx * slot_size in Phase 5.
        rpass.set_bind_group(2, bind_group, &[0u32]);
        // Per-instance lighting cbuffer override — engine:
        //   calculate_and_set_ravi_constants(...) +
        //   set_object_dominant_light_direction_and_intensity(...)
        // happen inside `select_instance_entry_point`; here we just
        // pick the dynamic offset.
        // `instance_lighting_offsets` is sized to the instance count, so an
        // OOB instance index is a real bug → fail loud (was a silent `0`).
        let lighting_offset =
            bsp.instance_lighting_offsets[inst_part.structure_instance_index as usize];
        // Per-instance simple_lights — instances live in some cluster;
        // we route them through that cluster's per-cluster slot. v1
        // uses the default empty slot for instances; per-instance
        // light selection is a Phase L2e follow-up.
        let camera_bg = match entry_point {
            HaloEntryPoint::Albedo => &ctx.shared.camera_bind_group,
            _ => &ctx.shared.camera_bind_group_sl,
        };
        rpass.set_bind_group(
            0,
            camera_bg,
            &[
                lighting_offset,
                // atmosphere @ binding 2 — BSP cluster default.
                crate::halo::render::shared::ATMOSPHERE_DEFAULT_OFFSET,
                crate::halo::render::shared::SIMPLE_LIGHTS_DEFAULT_OFFSET,
                // dominant_light @ binding 13 mirrors `lighting_offset`.
                lighting_offset,
            ],
        );
        rpass.set_bind_group(1, inst_model_bg, &[0u32]);
        rpass.set_bind_group(3, inst_nm_bg, &[0u32]);

        let placement_vb = bsp
            .instance_vertex_buffers
            .get(inst_part.structure_instance_index as usize)
            .and_then(|o| o.as_ref())
            .unwrap_or(&mesh.vertex_buffer);
        rpass.set_vertex_buffer(0, placement_vb.slice(..));

        // Per-vertex SH stream (slot 1) — bound only for the
        // StaticShPerVertex pipeline. The pipeline declared 2 vertex
        // buffer layouts in `pipeline_cache.rs`; the second slot
        // expects `PerVertexShVertex` payload from
        // `bsp.instance_per_vertex_sh_buffers[i]`. Other entries use
        // a 1-buffer pipeline and skip this bind.
        if matches!(entry_choice, Some(HaloEntryPoint::StaticShPerVertex)) {
            if let Some(pv_buf) = bsp
                .instance_per_vertex_sh_buffers
                .get(inst_part.structure_instance_index as usize)
                .and_then(|o| o.as_ref())
            {
                rpass.set_vertex_buffer(1, pv_buf.slice(..));
            }
        }

        // PRT Ambient transfer stream (slot 1) — bound only for the
        // StaticPrtAmbient pipeline. Mirrors the engine's per-mesh
        // slot-3-usage / slot-2-stream binding (Ares
        // `rasterizer_resource_definitions.cpp:46`). The stream is
        // per-mesh (shared across instances of the same def), uploaded
        // at BSP load — see `bsp_gpu.rs` `BspMesh::prt_ambient_buffer`.
        if matches!(entry_choice, Some(HaloEntryPoint::StaticPrtAmbient)) {
            if let Some(prt_buf) = mesh.prt_ambient_buffer.as_ref() {
                rpass.set_vertex_buffer(1, prt_buf.slice(..));
            }
        }

        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(
            mesh_part.index_start..mesh_part.index_start + mesh_part.index_count,
            0,
            0..1,
        );
    }
}

impl Default for StructureRenderer {
    fn default() -> Self { Self::new() }
}
