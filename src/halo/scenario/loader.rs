//! Scenario tag loader. Mirrors Ares `scenario_load @ 0x18017AEA0` +
//! `scenario_activate_initial_zone_set @ 0x180...`. Produces a
//! [`LoadedScenario`] holding the scenario tag's data plus everything
//! resolved at activation time:
//! - The active zone set's BSP mask + the resulting list of `LoadedBsp`s
//!   (sbsp tag + per-BSP lightmap data, both walked into runtime form).
//! - All scenario placements grouped by object type with their palette
//!   entries resolved to `.obje` tag paths.
//! - Sky / decorator / cubemap references for downstream passes.
//!
//! What this does NOT do (yet):
//! - Load `.obje` tags for placements (Phase E1 — the c_object lifecycle
//!   spawns runtime objects from these).
//! - Decode mesh vertex/index data from sbsp render geometry (Phase D2).
//! - Resolve PVS visibility data (Phase H).
//! - Stream zone-set switches (Phase J).

use std::path::{Path, PathBuf};

use blam_tags::TagFile;
use blam_tags::geometry::{read_compression_bounds_at, CompressionBounds};
use blam_tags::paths::{derive_tags_root, resolve_tag_path};
use blam_tags::render_model::{
    extract_imported_geometry_meshes, extract_sbsp_render_geometry_meshes, RenderMesh,
};
use blam_tags::scenario::{
    ObjectPlacement, Scenario, SkyReference, StructureBspReference, TagReferencePalette, ZoneSet,
};
use blam_tags::scenario_lightmap::{LightmapBspData, ScenarioLightmap};
use blam_tags::decorator_set::DecoratorSet;
use blam_tags::sky_atmosphere::SkyAtmosphere;
use blam_tags::structure_bsp::StructureBsp;
use blam_tags::structure_lighting_info::StructureLightingInfo;

#[derive(Debug)]
pub enum ScenarioLoadError {
    Read { path: PathBuf, source: blam_tags::TagReadError },
    Parse(String),
    NoZoneSet,
    NoActiveBsps,
    TagsRootNotFound(PathBuf),
}

impl std::fmt::Display for ScenarioLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "read {}: {source}", path.display()),
            Self::Parse(msg) => write!(f, "parse: {msg}"),
            Self::NoZoneSet => write!(f, "scenario has no zone sets"),
            Self::NoActiveBsps => write!(f, "active zone set activates no BSPs"),
            Self::TagsRootNotFound(p) => write!(f, "tags root not found near {}", p.display()),
        }
    }
}

impl std::error::Error for ScenarioLoadError {}

/// One loaded BSP — sbsp tag + matching per-BSP lightmap data + path
/// info for resolving sub-references later.
pub struct LoadedBsp {
    /// Index into the scenario's `structure_bsps[]` block.
    pub scenario_bsp_index: usize,
    /// Path to the on-disk `.scenario_structure_bsp` tag.
    pub sbsp_path: PathBuf,
    /// Decoded sbsp metadata (clusters, instances, materials, mesh
    /// metadata).
    pub sbsp: StructureBsp,
    /// Decoded mesh data — vertex + index buffers per
    /// `sbsp.render_geometry.meshes[i]`. `meshes[i]` aligns 1:1 with
    /// `sbsp.meshes_metadata[i]`. Cluster `mesh_index` and instance
    /// `definition_index` index into here.
    pub meshes: Vec<RenderMesh>,
    /// Per-instance lightmap UV streams from the lightmap tag's
    /// `imported geometry/per_instance_lightmap_texcoords[]`. Indexed
    /// by `instanced_geometry_instances[i].lightmap_texcoord_block_index`.
    /// `None`-shaped placements (block_index < 0) fall back to default
    /// lighting in `select_instance_entry_point`.
    pub per_instance_lightmap_uvs: Vec<blam_tags::PerInstanceLightmapUvs>,
    /// Per-BSP lightmap data — `None` when the scenario has no
    /// per-BSP lightmap chain (rare; most maps have one).
    pub lightmap: Option<LightmapBspData>,
    /// Decoded `.scenario_structure_lighting_info` (`stli`), pointed at
    /// by sbsp's `structure lighting_info^` tag-ref. Carries the
    /// authored generic-light definitions + instances for this BSP —
    /// the source of dynamic point/spot lights consumed by the
    /// `simple_lights` cbuffer. `None` when the tag-ref is empty or
    /// fails to parse (most maps have one; MP maps usually have
    /// 8-50 instances).
    pub lighting_info: Option<StructureLightingInfo>,
}

/// Result of loading a scenario tag — everything the renderer needs to
/// build draw lists.
pub struct LoadedScenario {
    /// Path to the `.scenario` tag we loaded.
    pub scenario_path: PathBuf,
    /// Resolved tags-root directory (for resolving child references).
    pub tags_root: PathBuf,
    /// The decoded scenario tag.
    pub scenario: Scenario,
    /// Currently active zone set index (into `scenario.zone_sets`).
    pub active_zone_set: usize,
    /// BSPs loaded for the active zone set.
    pub active_bsps: Vec<LoadedBsp>,
    /// `.wind` tag resolved per active BSP, parallel to `active_bsps`.
    /// `None` when the BSP's `wind` ref is empty or the tag failed to
    /// load. Engine equivalent: `get_wind_definition @ 0x1809081E0`
    /// reads `scenario.structure_bsp_references[bsp_index].wind` to
    /// resolve the active wind tag per camera position.
    pub active_bsp_winds: Vec<Option<blam_tags::wind::Wind>>,
    /// Globals fallback wind — `globals/default_wind.wind`. Used when
    /// the camera is outside any BSP3D leaf (engine path:
    /// `global_game_globals->default_wind_settings`). Hardcoded path
    /// resolves the same tag the engine resolves — a proper globals
    /// walker is deferred.
    pub default_wind: Option<blam_tags::wind::Wind>,
    /// Atmospheric scattering parameters loaded from
    /// `scenario.atmospheric` → `.sky_atm_parameters`. `None` when the
    /// scenario has no atmosphere reference. Provides Rayleigh/Mie
    /// multipliers, sun pitch/heading, fog colors for
    /// `compute_scattering` in shaders.
    pub atmosphere: Option<SkyAtmosphere>,
    /// Camera fx settings loaded from `scenario.camera_fx_settings`
    /// → `.camera_fx_settings` (`cfxs`). Carries the level's exposure
    /// + bloom + tonemap. Used to compute Halo's `view_exposure`
    /// (typically ~0.67) that's applied as `g_exposure` in shaders.
    pub camera_fx: Option<blam_tags::camera_fx_settings::CameraFxSettings>,
    /// Chocolate mountain table loaded from
    /// `scenario.chocolate_mountain` → `.chocolate_mountain_new`
    /// (`chmt`). Per-object-type minimum-luminance floor; the engine
    /// applies it via
    /// `c_chocalate_moutain::apply_chocalate_mountain_lighting @
    /// 0x1806f9ee0`. `None` when the scenario doesn't author one.
    pub chocolate_mountain: Option<blam_tags::chocolate_mountain::ChocolateMountain>,
    /// Sky-tag default lighting payload. Walks
    /// `scenario.skies[0].sky` (`.scenery`) → `.model` → `.render_model`
    /// and pulls `sky_lights[count-1]` (the dominant sun, per
    /// `get_sun_constants_from_sky @ 0x1803adcb0`) + `default_lightprobe`
    /// (the SH3 probe `setup_default_lighting` uses when the per-cluster
    /// lightmap chain misses). `None` when the chain breaks at any link.
    pub sky_lighting: Option<SkyLighting>,
    /// Loaded `.decorator_set` tags, indexed
    /// `decorator_sets[block_idx][set_idx]` to align with
    /// `scenario.decorators[block_idx].sets[set_idx]`. Each entry is
    /// `Some(tag)` when the .decorator_set parsed cleanly, `None` when
    /// the tag was missing or malformed (placements still loaded; the
    /// renderer just can't draw them without the set definition).
    /// Riverworld typically has one block × 11 sets.
    pub decorator_sets: Vec<Vec<Option<DecoratorSet>>>,
    /// Loaded `.render_model` tags pointed to by each
    /// `decorator_sets[i][j].render_model_path`. Same shape as
    /// `decorator_sets`. `None` when the .decorator_set was missing OR
    /// its referenced render_model failed to load. Each render_model
    /// has 1+ meshes (different LODs / subparts) — the placement's
    /// `type_index` resolves to a `decorator_types[k].mesh` block index
    /// which picks the mesh.
    pub decorator_render_models: Vec<Vec<Option<blam_tags::RenderModel>>>,
    /// Per-decorator-set per-subpart triangle-list slices, parallel to
    /// `decorator_render_models`. Each `DecoratorSubpartGeometry` has
    /// the shared vertex pool + N per-subpart index lists (one per
    /// decorator type / mesh subpart). Drives the per-type draw split
    /// in `c_decorator_renderer`. `None` when the render_model isn't
    /// shaped like a decorator (no per-mesh-temp / no parts / etc).
    pub decorator_subpart_geometry:
        Vec<Vec<Option<blam_tags::render_model::DecoratorSubpartGeometry>>>,
    /// Resolved scenario `lights[]` placements with their `light` tags
    /// loaded. Empty when the scenario has no `lights[]` block, or when
    /// every placement was malformed. Source for the per-light shadow
    /// scheduler (`c_lights_view::submit_visibility_and_render @
    /// 0x1806C6930`) and lens-flare submission
    /// (`light_submit_lens_flares @ 0x18086A850`). Cluster lights baked
    /// into `lighting_info.generic_light_instances` are a separate path
    /// — they don't pass through a `light` tag and don't reach this list.
    pub runtime_lights: Vec<crate::halo::scenario::RuntimeLight>,
    /// Top-level `.scenario_lightmap` (sLdT). Carries scenario-global
    /// airprobes / scenery probes / device-machine probes that the
    /// engine indexes per-object during
    /// `lights_prepare_for_object_static_new`. `None` when the scenario
    /// has no `new lightmaps` reference or the tag failed to load
    /// (campaign maps usually populate this; MP maps often don't, or MCC
    /// strips the airprobes block).
    pub scenario_lightmap: Option<ScenarioLightmap>,
    /// Resolved `.decal_system` tags, one per `scenario.decal_palette[i]`.
    /// `None` for empty refs or load failures (placement renderer falls
    /// back to skip-draw on miss). Mesh projection against BSP geometry
    /// happens in Phase D2 — this vector just holds the decoded
    /// definitions + their `actual shader` for later consumption.
    pub decal_palette: Vec<Option<blam_tags::decal_system::DecalSystem>>,
    /// Per-BSP packed decal mesh storage built by D2b. Populated at
    /// scenario load with one `LoadedBspDecals` slot per
    /// `active_bsps[]` entry. Each per-BSP buffer concatenates the
    /// `rasterizer_vertex_world` output from every placement bound to
    /// that BSP. Consumed by the (D4b) decal pipeline.
    pub decal_renderer: crate::halo::render::render_decals::DecalRenderer,
}

/// SH3 default lightprobe (9 coefficients per channel) as protomorph
/// consumes it. blam-tags' `render_model::DefaultLightprobe` now carries
/// the full 16-coefficient on-disk form (the trailing 7 past SH3's 9
/// are zero); we truncate to the SH3 9 the lighting pipeline expects.
#[derive(Debug, Clone, Default)]
pub struct SkyProbe {
    pub r: [f32; 9],
    pub g: [f32; 9],
    pub b: [f32; 9],
}

impl From<&blam_tags::render_model::DefaultLightprobe> for SkyProbe {
    fn from(p: &blam_tags::render_model::DefaultLightprobe) -> Self {
        let take9 = |a: &[f32; 16]| -> [f32; 9] {
            let mut out = [0.0f32; 9];
            out.copy_from_slice(&a[..9]);
            out
        };
        SkyProbe { r: take9(&p.r), g: take9(&p.g), b: take9(&p.b) }
    }
}

/// Decoded default lighting from the scenario's primary sky tag.
/// Mirrors what dllcache's `get_sun_constants_from_sky` +
/// `setup_default_lighting` extract from `sky.render_model`.
#[derive(Debug, Clone)]
pub struct SkyLighting {
    /// SH-derived dominant-light direction. Currently sourced from
    /// `calculate_dominant_light_from_lightprobe(probe)` — engine's
    /// `setup_default_lighting` route, NOT the `lightgen_lights[last]`
    /// direct read (see `lightgen_sun_*` below).
    pub sun_direction: blam_tags::math::RealVector3d,
    /// SH-derived dominant-light intensity (per channel). Used by
    /// `setup_default_lighting`'s dominant-light feed for object lighting.
    pub sun_intensity: blam_tags::math::RealVector3d,
    /// SH3 probe (9 floats per channel) baked into the sky's render_model.
    /// Read directly from `default lightprobe r/g/b`.
    pub probe: SkyProbe,
    /// Engine `get_sun_constants_from_sky @ 0x1803adcb0` → atmosphere
    /// path. Computed as `sky_lights[count-1].intensity × solid_angle ×
    /// 0.2 × g_render_light_intensity` (we use g_render_light_intensity=1).
    /// Consumed by `c_atmosphere_fog_interface::get_sun_parameters` when
    /// the active atmosphere setting's flag bit 1 ("Override Real Sun
    /// Values") is UNSET — i.e., the atmosphere borrows the sky's sun
    /// instead of authoring its own. `None` when the sky has no
    /// `sky_lights` entries.
    pub lightgen_sun_intensity: Option<blam_tags::math::RealVector3d>,
    /// `sky_lights[count-1].direction` direct read. Same engine path
    /// as `lightgen_sun_intensity`.
    pub lightgen_sun_direction: Option<blam_tags::math::RealVector3d>,
}

impl LoadedScenario {
    /// Load a `.scenario` tag and activate its first zone set, loading
    /// every BSP referenced by `bsp_zone_flags`. Mirrors
    /// `scenario_load → scenario_activate_initial_zone_set`.
    pub fn load(scenario_path: &Path) -> Result<Self, ScenarioLoadError> {
        let tags_root = derive_tags_root(scenario_path)
            .ok_or_else(|| ScenarioLoadError::TagsRootNotFound(scenario_path.to_path_buf()))?;

        let tag = TagFile::read(scenario_path).map_err(|e| ScenarioLoadError::Read {
            path: scenario_path.to_path_buf(),
            source: e,
        })?;
        let scenario = Scenario::from_tag(&tag).map_err(|e| ScenarioLoadError::Parse(e.to_string()))?;

        if scenario.zone_sets.is_empty() {
            return Err(ScenarioLoadError::NoZoneSet);
        }
        let active_zone_set = 0;
        let active_indices = scenario.zone_set_active_bsps(active_zone_set);
        if active_indices.is_empty() {
            return Err(ScenarioLoadError::NoActiveBsps);
        }

        // Pre-load the top-level scenario_lightmap (sLdT) once. Some
        // scenarios bundle per-BSP Lbsp tag refs through this layer
        // rather than via the convention `<sbsp_stem>.scenario_lightmap_bsp_data`.
        // Per-pixel mode is the common path; per-vertex is a fallback
        // some maps use for low-detail BSPs.
        let scenario_lightmap = if !scenario.new_lightmaps.is_empty() {
            let sldt_path = resolve_tag_path(&tags_root, &scenario.new_lightmaps, "scenario_lightmap");
            TagFile::read(&sldt_path).ok().and_then(|t|
                ScenarioLightmap::from_tag(&t).ok()
            )
        } else {
            None
        };

        let mut active_bsps = Vec::with_capacity(active_indices.len());
        for i in active_indices {
            let bsp_ref = &scenario.structure_bsps[i];
            let sbsp_path = resolve_tag_path(&tags_root, &bsp_ref.structure_bsp, "scenario_structure_bsp");

            let sbsp_tag = TagFile::read(&sbsp_path).map_err(|e| ScenarioLoadError::Read {
                path: sbsp_path.clone(),
                source: e,
            })?;
            let sbsp_root = sbsp_tag.root();
            let sbsp = StructureBsp::from_tag(&sbsp_tag)
                .map_err(|e| ScenarioLoadError::Parse(e.to_string()))?;

            // Per the H3 sbsp partitioning rule (verified by the H3
            // Blender Toolset's `_mesh_decoder.py` and our own
            // `AssFile::from_scenario_structure_bsp`):
            //
            // - Cluster meshes (mesh_idx >= compression_info.len()):
            //   POSITIONS already in world units → identity pos bounds.
            //   But TEXCOORDS still come from `compression_info[0]` —
            //   `pos_compressed` and `uv_compressed` are independent
            //   flags (geometry.rs:50-52).
            // - Instance def meshes: looked up via the def's
            //   `compression_index` field — one def per unique mesh,
            //   compression bounds chosen explicitly per def.
            //   We build a `mesh_idx -> compression_index` lookup from
            //   the resource_interface defs block.
            //
            // sbsp ALWAYS uses triangle list (not strip), even when the
            // schema enum says strip — `extract_sbsp_render_geometry_meshes`
            // forces list interpretation.
            let ci_count = sbsp_root
                .field_path("render geometry/compression info")
                .and_then(|f| f.as_block())
                .map(|b| b.len())
                .unwrap_or(0);
            // Cluster meshes have positions in world units AND texcoords
            // stored as raw `real_point_2d` (already decompressed). NO
            // bounds-decompression is needed for cluster UVs — applying
            // compression_info[0]'s bounds (which are for some specific
            // instance mesh) maps cluster UVs into a totally wrong range
            // and tiles every per-layer texture across the whole BSP.
            let cluster_bounds = CompressionBounds::identity();
            // Build mesh_idx → compression_index lookup from instance
            // definitions (each def names its compression_info entry
            // explicitly).
            let mut mesh_to_ci = std::collections::HashMap::<usize, usize>::new();
            for def in &sbsp.instance_definitions {
                if def.mesh_index >= 0 && def.compression_index >= 0 {
                    mesh_to_ci.insert(def.mesh_index as usize, def.compression_index as usize);
                }
            }
            // Per-BSP lightmap resolution. Two paths, in priority:
            //   1. Top-level sLdT chain — `scenario.new_lightmaps`
            //      → sLdT.per_pixel_bsp_data[i] (or per_vertex_bsp_data[i])
            //      → that path is the Lbsp tag.
            //   2. Convention — `<sbsp_stem>.scenario_lightmap_bsp_data`.
            // Treat missing as None (renderer falls back to sky probe).
            let (lightmap, lightmap_path) = load_per_bsp_lightmap(
                &tags_root, &sbsp_path, scenario_lightmap.as_ref(), i,
            );

            // Pick render geometry. Canonical engine path
            // (`render_mesh_part_default @ 0x18069ebc0` →
            //  `mesh_get_vertex_buffer(geometry, mesh, lightmap_uv)`):
            // when a per-bsp lightmap exists, the engine renders from
            // the lightmap tag's `imported geometry` — that block holds
            // the FULL vertex stream (positions/normals/uvs +
            // lightmap_texcoord), with chart-split duplicates that
            // give per-texel UVs in the atlas. The sbsp's
            // `render geometry` is the authoring source; its raw
            // vertex counts are SHORTER than the lightmap's wherever
            // a vertex sits on a UV chart seam, so zipping by index
            // landed wrong UVs on those vertices.
            let lbsp_tag_for_geom = lightmap_path.as_ref().and_then(|p| TagFile::read(p).ok());
            let (meshes, per_instance_lightmap_uvs) = if let Some(lbsp_tag) = &lbsp_tag_for_geom {
                let lbsp_root = lbsp_tag.root();
                let m = extract_imported_geometry_meshes(&lbsp_root, |mi| {
                    if mi >= ci_count {
                        cluster_bounds
                    } else if let Some(&ci) = mesh_to_ci.get(&mi) {
                        read_compression_bounds_at(&sbsp_root, ci)
                    } else {
                        read_compression_bounds_at(&sbsp_root, mi)
                    }
                })
                .map_err(|e| ScenarioLoadError::Parse(e.to_string()))?;
                // Diagnostic: dump pre-bake raw_vertex.lightmap_texcoord
                // for canonical zanzibar cluster meshes. Compared
                // against protomorph's post-bake USHORT2N cache stream
                // to find any transform tool.exe applied at bake.
                eprintln!(
                    "[halo-vert-probe] imported_geometry mesh_count={} ci_count={ci_count}",
                    m.len()
                );
                for &probe_mi in &[0usize, 56, 336, 337, 338] {
                    if let Some(mesh) = m.get(probe_mi) {
                        eprintln!(
                            "[halo-vert-probe] mesh={probe_mi} verts={}",
                            mesh.vertices.len()
                        );
                        let dump_n = mesh.vertices.len().min(12);
                        for vi in 0..dump_n {
                            let v = &mesh.vertices[vi];
                            eprintln!(
                                "[halo-vert] mesh={probe_mi} vi={vi:>3} \
                                 pos=({:.4},{:.4},{:.4}) lm_uv=({:.6},{:.6})",
                                v.position.x, v.position.y, v.position.z,
                                v.lightmap_texcoord.x, v.lightmap_texcoord.y,
                            );
                        }
                    } else {
                        eprintln!("[halo-vert-probe] mesh={probe_mi} — out of range");
                    }
                }
                let uvs = blam_tags::extract_per_instance_lightmap_uvs(&lbsp_root);
                (m, uvs)
            } else {
                let m = extract_sbsp_render_geometry_meshes(&sbsp_root, |mi| {
                    if mi >= ci_count {
                        cluster_bounds
                    } else if let Some(&ci) = mesh_to_ci.get(&mi) {
                        read_compression_bounds_at(&sbsp_root, ci)
                    } else {
                        read_compression_bounds_at(&sbsp_root, mi)
                    }
                })
                .map_err(|e| ScenarioLoadError::Parse(e.to_string()))?;
                (m, Vec::new())
            };

            // Log per-bsp water-mesh stats — the count of meshes
            // carrying `raw_water_data` from blam-tags' parser. Useful
            // to verify Phase A1c parsed the BSP's water surfaces; the
            // rmw render path consumes these in Phase A1d+.
            // Water-mesh stats — `RawWaterData.triangles` is already
            // resolved per the Reach `create_mesh_water_vertex_buffer`
            // walk (`reference_water_vertex_buffer_build.md`).
            let mut water_mesh_count = 0usize;
            let mut water_tri_total = 0usize;
            for m in &meshes {
                if let Some(wd) = m.water_data.as_ref() {
                    if !wd.triangles.is_empty() {
                        water_mesh_count += 1;
                        water_tri_total += wd.triangles.len();
                    }
                }
            }
            if water_mesh_count > 0 {
                eprintln!(
                    "[bsp[{}]] water surfaces: {} meshes, {} source triangles",
                    i, water_mesh_count, water_tri_total,
                );
            }

            // `structure lighting_info^` (sbsp top-level tag-ref) →
            // `.scenario_structure_lighting_info` (`stli`). H3 MCC
            // sbsps strip this ref at cache-build time, so we ALSO
            // probe the convention path: `<sbsp_stem>.scenario_structure_lighting_info`
            // (same tree-stem as the sbsp). Both Riverworld and
            // Guardian have the loose stli on disk via this convention.
            let lighting_info = sbsp_root
                .read_tag_ref_path("structure lighting_info")
                .filter(|p| !p.is_empty())
                .map(|rel_path| {
                    resolve_tag_path(
                        &tags_root,
                        &rel_path,
                        "scenario_structure_lighting_info",
                    )
                })
                .or_else(|| {
                    let mut p = sbsp_path.clone();
                    p.set_extension("scenario_structure_lighting_info");
                    if p.exists() { Some(p) } else { None }
                })
                .and_then(|stli_path| {
                    let tag = TagFile::read(&stli_path).ok()?;
                    StructureLightingInfo::from_tag(&tag).ok()
                });
            if let Some(li) = lighting_info.as_ref() {
                eprintln!(
                    "[bsp[{}]] lighting_info: {} definitions, {} instances",
                    i,
                    li.generic_light_definitions.len(),
                    li.generic_light_instances.len(),
                );
            }

            // One-shot dump of the BSP's camera_fx_palette so the
            // user can see which clusters carry authored exposure /
            // bloom overrides. Engine reads these via
            // `cluster_palette_entry` in `c_camera_fx_values::update`;
            // wrong-named field reads silently default-to-zero
            // (parser fixed 2026-05-20). Hide behind
            // `PROTOMORPH_DEBUG_EXPOSURE` to avoid noise in normal runs.
            if std::env::var_os("PROTOMORPH_DEBUG_EXPOSURE").is_some()
                && !sbsp.camera_fx_palette.is_empty()
            {
                eprintln!(
                    "[exposure-palette] bsp_idx={i} entries={}",
                    sbsp.camera_fx_palette.len(),
                );
                use blam_tags::structure_bsp::CameraFxPaletteFlags;
                for (pi, entry) in sbsp.camera_fx_palette.iter().enumerate() {
                    let force_exp = entry.flags.contains(CameraFxPaletteFlags::ForceExposure);
                    let force_auto = entry.flags.contains(CameraFxPaletteFlags::ForceAutoExposure);
                    let minmax = entry.flags.contains(CameraFxPaletteFlags::OverrideExposureBounds);
                    let bloom = entry.flags.contains(CameraFxPaletteFlags::OverrideInherentBloom);
                    eprintln!(
                        "[exposure-palette]   #{pi:02} \"{}\" flags={:?} \
                         (force_exp={force_exp} force_auto_b={force_auto} minmax={minmax} bloom={bloom}) \
                         exp={:.3} auto_b={:.3} min/max=[{:.3},{:.3}] inh={:.3} bi={:.3}",
                        entry.name, entry.flags,
                        entry.forced_exposure, entry.forced_auto_exposure_brightness,
                        entry.exposure_min, entry.exposure_max,
                        entry.inherent_bloom, entry.bloom_intensity,
                    );
                }
            }

            active_bsps.push(LoadedBsp {
                scenario_bsp_index: i,
                sbsp_path,
                sbsp,
                meshes,
                per_instance_lightmap_uvs,
                lightmap,
                lighting_info,
            });
        }

        // Wind tags — per-bsp + globals fallback. Engine:
        // `get_wind_definition @ 0x1809081E0` walks
        // `scenario.structure_bsps[bsp_index].wind` per camera position;
        // when the camera is outside any BSP3D leaf it falls back to
        // `globals.default_wind_settings`. We pre-load both at scenario
        // load and let the per-frame wind update pick from
        // `active_bsp_winds[active_bsp_index].or(default_wind)`.
        let active_bsp_winds: Vec<Option<blam_tags::wind::Wind>> = active_bsps
            .iter()
            .map(|bsp| {
                let bsp_ref = &scenario.structure_bsps[bsp.scenario_bsp_index];
                if bsp_ref.wind.is_empty() {
                    return None;
                }
                let path = resolve_tag_path(&tags_root, &bsp_ref.wind, "wind");
                TagFile::read(&path)
                    .ok()
                    .and_then(|t| blam_tags::wind::Wind::from_tag(&t).ok())
            })
            .collect();
        // Globals fallback. A proper globals (matg) walker would read
        // `globals.default_wind_settings.path`; we hardcode the engine's
        // canonical fallback path until the walker lands. Every shipping
        // Halo 3 install ships this tag.
        let default_wind = {
            let path = resolve_tag_path(&tags_root, "globals\\default_wind", "wind");
            TagFile::read(&path)
                .ok()
                .and_then(|t| blam_tags::wind::Wind::from_tag(&t).ok())
        };

        // Load atmosphere parameters (scenario.atmospheric → .sky_atm_parameters).
        // Skipped silently when the tag is missing or fails to parse;
        // shaders fall back to neutral defaults.
        let atmosphere = if !scenario.atmospheric.is_empty() {
            let atm_path = resolve_tag_path(&tags_root, &scenario.atmospheric, "sky_atm_parameters");
            TagFile::read(&atm_path).ok().and_then(|t| SkyAtmosphere::from_tag(&t).ok())
        } else {
            None
        };

        // Load camera fx settings (scenario.camera_fx_settings →
        // .camera_fx_settings). Provides the level's exposure stops
        // for `g_exposure`. Shaders fall back to 0.67 (Halo default).
        let camera_fx = if !scenario.camera_fx_settings.is_empty() {
            let cfx_path = resolve_tag_path(&tags_root, &scenario.camera_fx_settings, "camera_fx_settings");
            TagFile::read(&cfx_path).ok().and_then(|t|
                blam_tags::camera_fx_settings::CameraFxSettings::from_tag(&t).ok()
            )
        } else {
            None
        };

        // Load chocolate_mountain_new (scenario.chocolate_mountain →
        // .chocolate_mountain_new). Per-object-type minimum-luminance
        // floor table, indexed positionally by `e_object_type`. Engine:
        // `c_chocalate_moutain::apply_chocalate_mountain_lighting @
        // 0x1806f9ee0`. `None` when the scenario doesn't reference
        // one (engine returns mountain_scale = -1.0 in that case).
        let chocolate_mountain = if !scenario.chocolate_mountain.is_empty() {
            let cmt_path = resolve_tag_path(&tags_root, &scenario.chocolate_mountain, "chocolate_mountain_new");
            TagFile::read(&cmt_path).ok().and_then(|t|
                blam_tags::chocolate_mountain::ChocolateMountain::from_tag(&t).ok()
            )
        } else {
            None
        };

        // Walk the primary sky chain: scenario.skies[0].sky (`.scenery`)
        // → object/model (.model) → render_model. dllcache's
        // `get_sun_constants_from_sky @ 0x1803adcb0` does this same
        // walk every time it needs the dominant light.
        let sky_lighting = load_sky_lighting(&tags_root, &scenario);

        // Load each `.decorator_set` tag referenced by the scenario's
        // decorators[i].sets[j]. The set-definition tag carries the
        // render_model + texture + 6-shader-flavor selector + per-type
        // variation params. Tag-shipped MCC scenarios reference these
        // standalone tags at `levels/shared/decorators/<name>/<name>.decorator_set`.
        // See `reference_mcc_strips_decorator_data.md` for the full
        // data path.
        let decorator_sets: Vec<Vec<Option<DecoratorSet>>> = scenario
            .decorators
            .iter()
            .map(|block| {
                block
                    .sets
                    .iter()
                    .map(|set_entry| {
                        if set_entry.decorator_set.is_empty() {
                            return None;
                        }
                        let path = resolve_tag_path(
                            &tags_root,
                            &set_entry.decorator_set,
                            "decorator_set",
                        );
                        TagFile::read(&path)
                            .ok()
                            .and_then(|t| DecoratorSet::from_tag(&t).ok())
                    })
                    .collect()
            })
            .collect();

        // For each loaded decorator_set, also load its render_model. The
        // .render_model carries the actual instanced mesh geometry the
        // decorator system draws thousands of copies of. Missing tags
        // → None; the renderer just skips those sets.
        let (decorator_render_models, decorator_subpart_geometry): (
            Vec<Vec<Option<blam_tags::RenderModel>>>,
            Vec<Vec<Option<blam_tags::render_model::DecoratorSubpartGeometry>>>,
        ) = {
            let pairs: Vec<Vec<(
                Option<blam_tags::RenderModel>,
                Option<blam_tags::render_model::DecoratorSubpartGeometry>,
            )>> = decorator_sets
                .iter()
                .map(|block_sets| {
                    block_sets
                        .iter()
                        .map(|loaded| {
                            let Some(set) = loaded.as_ref() else { return (None, None) };
                            if set.render_model_path.is_empty() {
                                return (None, None);
                            }
                            let path = resolve_tag_path(
                                &tags_root,
                                &set.render_model_path,
                                "render_model",
                            );
                            // Open the tag ONCE; extract both the
                            // standard parsed `RenderModel` (for
                            // instance_placements lookup) AND the
                            // decorator-specific per-subpart triangle
                            // lists (for engine-faithful per-type draws).
                            let Ok(tag) = TagFile::read(&path) else { return (None, None) };
                            let rm = blam_tags::RenderModel::from_tag(&tag).ok();
                            let geom =
                                blam_tags::render_model::extract_decorator_subparts(&tag);
                            (rm, geom)
                        })
                        .collect()
                })
                .collect();
            // Split the parallel vec-of-vec-of-tuples into two parallel
            // vec-of-vec-of-options.
            let mut rms: Vec<Vec<Option<blam_tags::RenderModel>>> = Vec::with_capacity(pairs.len());
            let mut geoms: Vec<
                Vec<Option<blam_tags::render_model::DecoratorSubpartGeometry>>,
            > = Vec::with_capacity(pairs.len());
            for block in pairs {
                let mut block_rms = Vec::with_capacity(block.len());
                let mut block_geoms = Vec::with_capacity(block.len());
                for (rm, geom) in block {
                    block_rms.push(rm);
                    block_geoms.push(geom);
                }
                rms.push(block_rms);
                geoms.push(block_geoms);
            }
            (rms, geoms)
        };

        // Walk scenario lights[] + light_palette and resolve each
        // placement to a runtime light entry. Cluster lights from
        // `lighting_info` are a SEPARATE path (no light-tag indirection,
        // no shadow_casting flag, no lens flare) — this only covers
        // scenario-placed dynamic lights.
        let runtime_lights = {
            let result = crate::halo::scenario::runtime_lights::load_runtime_lights(
                &scenario,
                &tags_root,
            );
            let shadow_casters = result
                .lights
                .iter()
                .filter(|l| l.passes_shadow_scheduler_gate())
                .count();
            let lens_flares = result
                .lights
                .iter()
                .filter(|l| l.has_lens_flare())
                .count();
            eprintln!(
                "[scenario] runtime_lights: {} resolved / {} placements (palette={}, tags_loaded={}, skipped_no_palette={}, skipped_failed_load={}); shadow_casters={}, lens_flare_attached={}",
                result.lights.len(),
                result.placement_count,
                result.palette_count,
                result.tags_loaded,
                result.skipped_no_palette,
                result.skipped_failed_load,
                shadow_casters,
                lens_flares,
            );
            result.lights
        };

        // Walk scenario.decal_palette[] and load each `.decal_system`
        // tag once. Per-placement projection-mesh bake (D2) consumes
        // these. Empty palettes → empty Vec; load failures keep the
        // slot as `None` so palette indices remain stable.
        let decal_palette = {
            let mut out = Vec::with_capacity(scenario.decal_palette.len());
            let mut loaded = 0usize;
            let mut empty = 0usize;
            let mut failed = 0usize;
            for entry in &scenario.decal_palette {
                if entry.decal_system.is_empty() {
                    empty += 1;
                    out.push(None);
                    continue;
                }
                let path = resolve_tag_path(&tags_root, &entry.decal_system, "decal_system");
                let parsed = TagFile::read(&path)
                    .ok()
                    .and_then(|t| blam_tags::decal_system::DecalSystem::from_tag(&t).ok());
                match &parsed {
                    Some(_) => loaded += 1,
                    None => failed += 1,
                }
                out.push(parsed);
            }
            eprintln!(
                "[scenario] decals: {} placements / {} palette entries (loaded={loaded}, empty={empty}, failed={failed})",
                scenario.decals.len(),
                scenario.decal_palette.len(),
            );
            out
        };

        // Fix #3 — engine bakes `m_bitmap_aspect = base_map.width /
        // base_map.height` at cache build time and stores it in
        // `c_decal_definition::m_bitmap_aspect`. Most palette tags ship
        // with this field = 0 (the bake didn't run / didn't persist), so
        // the runtime falls back to the bitmap's actual dimensions. We
        // mirror the fallback here: per palette entry, resolve the first
        // definition's `base_map` parameter, load the bitmap, divide
        // width by height. Cache one aspect per palette to avoid
        // re-reading the same bitmap for every placement that shares
        // the palette.
        //
        // Engine cite: `c_decal::create @ 0x18039A800` line
        // `*((float *)v17 + 11) = v25 * p_m_name[28]` — `p_m_name[28]`
        // is `c_decal_definition::m_bitmap_aspect` at definition offset
        // 112. Tag-build sets it from the base_map bitmap. v1 of this
        // port computed only when the tag's stored value is 0.
        let decal_palette_resolved_aspects: Vec<f32> = decal_palette
            .iter()
            .map(|entry| {
                let Some(ds) = entry else { return 1.0_f32 };
                let Some(def) = ds.definitions.first() else { return 1.0_f32 };
                // Already populated by cache builder — trust it.
                if def.bitmap_aspect > 0.0 {
                    return def.bitmap_aspect;
                }
                let Some(shader) = def.shader.as_ref() else { return 1.0_f32 };
                let base_map = shader
                    .parameters
                    .iter()
                    .find(|p| p.parameter_name == "base_map");
                let Some(base_map) = base_map else { return 1.0_f32 };
                if base_map.bitmap_path.is_empty() {
                    return 1.0;
                }
                let path = resolve_tag_path(&tags_root, &base_map.bitmap_path, "bitmap");
                let Ok(tag) = TagFile::read(&path) else { return 1.0_f32 };
                let Ok(bitmap) = blam_tags::bitmap::Bitmap::new(&tag) else { return 1.0_f32 };
                let Some(image) = bitmap.image(0) else { return 1.0_f32 };
                let w = image.width().max(1) as f32;
                let h = image.height().max(1) as f32;
                w / h
            })
            .collect();

        // D2a + D2b — for each preplaced decal placement:
        //   1. `c_decal_system::collide @ 0x1803996B0` casts the
        //      projection ray + finds up to 16 surface hits in a
        //      cylinder around the placement (`src/halo/decal/collide.rs`).
        //   2. `c_decal::build_mesh @ 0x18039B820` runs a BFS across
        //      the collision-surface graph from each hit, producing a
        //      folded triangle mesh that wraps around terrain curvature
        //      (`src/halo/decal/orchestrator.rs`).
        //   3. `c_decal::smooth_mesh_fragment + write_mesh_fragment`
        //      pack the folded mesh into `RasterizerVertexWorld` for
        //      the GPU decal pass (`smoother.rs` + `writer.rs`).
        let mut decal_renderer =
            crate::halo::render::render_decals::DecalRenderer::new(active_bsps.len());
        if !scenario.decals.is_empty() {
            use crate::halo::decal::collide::collide as decal_collide;
            use crate::halo::decal::mesh_builder::DecalBuildContext;
            use crate::halo::decal::orchestrator::{
                build_mesh as decal_build_mesh, DecalCollisionResult, DecalParams,
            };
            use crate::halo::decal::types::{DecalFragment, DecalMeshBuilder};
            use crate::halo::decal::writer::{write_mesh_fragment, RasterizerVertexWorld};
            use blam_tags::math::RealMatrix4x3;
            use blam_tags::math::{RealPoint3d, RealVector3d};
            use glam::{Quat, Vec3};

            let mut built = 0usize;
            let mut skipped_no_palette = 0usize;
            let mut skipped_no_bsp = 0usize;
            let mut skipped_no_collision = 0usize;
            let mut skipped_overlap = 0usize;
            let mut total_collisions = 0usize;
            let mut total_packed_bytes = 0u64;

            // Engine `c_decal_system::check_overlap @ 0x18039A050`: as
            // each decal is placed, the engine retains its primary hit
            // center + definition reference in `c_decal_system::x_data_array`
            // and rejects subsequent placements of the same definition
            // whose hit center falls within `2*runtime_max_radius -
            // overlapping_threshold` of an existing entry once
            // `max_overlapping` neighbours are matched. Default tags
            // have `max_overlapping == 0` (inert). Survey: 1 of 249 MP
            // decal_systems triggers this — `s3d_powerhouse` references
            // `s3d_p_circle_yellow_blank` with `max_overlapping = 1`.
            struct PlacedDecal {
                palette_index: i16,
                center: RealPoint3d,
            }
            let mut placed_pool: Vec<PlacedDecal> =
                Vec::with_capacity(scenario.decals.len());

            // PROTOMORPH_DEBUG_DECAL_DUMP=N — dump first N built decals'
            // per-placement diagnostics: palette name, sizes, collision
            // count, fragment count, packed-vertex UV range. The UV
            // range is the key signal — if `min..max` doesn't span
            // close to [0, 1], the BFS walker isn't reaching the
            // bitmap edges (where alpha=0 in a feathered mask), so the
            // decal samples only the opaque center → "shrinkwrapped"
            // visual symptom. `=1` dumps 1 decal, etc.
            let mut decal_debug_remaining: usize = std::env::var("PROTOMORPH_DEBUG_DECAL_DUMP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            // Heap-allocated mesh builder reused across placements
            // (51 KB of working buffers; engine stack-allocates the
            // equivalent per call to build_mesh).
            let mut mesh_builder = Box::new(DecalMeshBuilder::new());
            let mut fragments: Vec<DecalFragment> = Vec::with_capacity(16);

            // s3d_powerhouse-class hang debug: every Nth placement prints
            // progress so a freeze tells us exactly which placement is
            // hot. PROTOMORPH_DEBUG_DECAL_PROGRESS=N sets the stride
            // (default 50; set 1 for per-placement).
            let progress_stride: usize = std::env::var("PROTOMORPH_DEBUG_DECAL_PROGRESS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(50);
            let total_placements = scenario.decals.len();

            for (placement_idx, placement) in scenario.decals.iter().enumerate() {
                if progress_stride > 0 && placement_idx % progress_stride == 0 {
                    eprintln!(
                        "[scenario] decal progress: {}/{} (palette={}, built={built}, no_collision={skipped_no_collision})",
                        placement_idx, total_placements, placement.palette_index,
                    );
                }
                let placement = placement;
                let bsp_slot = if (placement.editing_bound_to_bsp as i32) >= 0
                    && (placement.editing_bound_to_bsp as usize) < active_bsps.len()
                {
                    placement.editing_bound_to_bsp as usize
                } else {
                    0
                };
                let bsp = match active_bsps.get(bsp_slot) {
                    Some(b) => b,
                    None => {
                        skipped_no_bsp += 1;
                        continue;
                    }
                };
                let collision_bsp = match bsp.sbsp.collision_bsp.as_ref() {
                    Some(c) => c,
                    None => {
                        skipped_no_bsp += 1;
                        continue;
                    }
                };

                let decal_system = match decal_palette
                    .get(placement.palette_index as usize)
                    .and_then(|opt| opt.as_ref())
                {
                    Some(ds) => ds,
                    None => {
                        skipped_no_palette += 1;
                        continue;
                    }
                };
                let decal_def = match decal_system.definitions.first() {
                    Some(d) => d,
                    None => {
                        skipped_no_palette += 1;
                        continue;
                    }
                };

                // Engine `c_decal_system::place @ 0x180398610` builds
                // projection_forward = -q*Z, projection_up = -q*X. The
                // engine then writes `m_projection.basis[0] =
                // projection_forward`, `m_projection.basis[2] =
                // projection_up`, `m_projection.basis[1] = basis[0] ×
                // basis[2]` (`c_decal_system::create_at_index @
                // 0x180398730` lines ~120-160).
                let q = Quat::from_xyzw(
                    placement.rotation.i,
                    placement.rotation.j,
                    placement.rotation.k,
                    placement.rotation.w,
                );
                // Engine `c_decal_system::create_at_index @ 0x180398730:120-160`
                // builds m_projection in this order:
                //   basis[0] (forward) = projection_forward
                //   basis[2] (up)      = projection_up
                //   basis[1] (left)    = up × forward
                //   basis[2] (up)      = basis[1] × basis[0] (renormalized)
                //
                // IDA verification — basis[1].x = up.y*forward.z -
                // forward.y*up.z (lines 0x180398730:142+). That's
                // (up × forward).x, NOT (forward × up).x — confirmed
                // by visual A/B (forward × up renders text mirrored).
                //
                // c_decal_system::place @ 0x180398610 supplies:
                //   projection_forward = down     = -orientation.up    = -q·Z
                //   projection_up      = backward = -orientation.fwd   = -q·X
                //
                // Engine `c_decal_system::create_at_index @ 0x180398730`
                // builds the orthonormal basis:
                //   basis[0] = forward = normalize(down)
                //   basis[2] = up_input = normalize(backward)        [INITIAL]
                //   basis[1] = left = normalize(basis[2] × basis[0]) = backward × down
                //   basis[2] = basis[0] × basis[1] = forward × left  [RE-ORTHOGONALIZED, RH]
                //
                // **2026-05-17 second-pass fix:** an earlier attempt used
                // `left × forward` (LH convention) which flipped the
                // projection's up axis vs engine — same magnitude but
                // opposite sign. That mirrored every decal vertically,
                // so the bitmap region sampled at each mesh fragment
                // landed on different bitmap content (typically white
                // pixels for industrial signs/snow patches), producing
                // visible bright-white polygons. Engine's
                // `create_at_index @ 0x180398730` decompile shows the
                // initial setup is unconditionally right-handed —
                // `basis[2].x = basis[0].y * basis[1].z - basis[1].y * basis[0].z`
                // which is `(basis[0] × basis[1]).x = (forward × left).x`.
                let forward = (-q.mul_vec3(Vec3::Z)).normalize_or_zero();
                let up_initial = (-q.mul_vec3(Vec3::X)).normalize_or_zero();
                let left = up_initial.cross(forward).normalize_or_zero();
                let up = forward.cross(left).normalize_or_zero();
                if forward.length_squared() < 0.5
                    || up.length_squared() < 0.5
                    || left.length_squared() < 0.5
                {
                    skipped_no_palette += 1;
                    continue;
                }

                let world_origin = RealPoint3d {
                    x: placement.position.x,
                    y: placement.position.y,
                    z: placement.position.z,
                };
                let world_velocity = RealVector3d {
                    i: forward.x,
                    j: forward.y,
                    k: forward.z,
                };
                // Mutable because `collide` rewrites `position` to the
                // primary hit point (engine `set_center @ 0x82a0c5c0`).
                let mut world_projection = RealMatrix4x3 {
                    scale: 1.0,
                    forward: RealVector3d {
                        i: forward.x,
                        j: forward.y,
                        k: forward.z,
                    },
                    left: RealVector3d {
                        i: left.x,
                        j: left.y,
                        k: left.z,
                    },
                    up: RealVector3d {
                        i: up.x,
                        j: up.y,
                        k: up.z,
                    },
                    position: world_origin,
                };

                // ---- D2a: collide ----
                let mut collisions: Vec<DecalCollisionResult> = Vec::with_capacity(16);
                // Per-step trace for the hang debug (only when stride=1):
                if progress_stride == 1 {
                    eprintln!("[decal-step {placement_idx}] entering collide");
                }
                let ok = decal_collide(
                    decal_system,
                    world_origin,
                    world_velocity,
                    -1, // ignore_object_index (no objects in protomorph yet)
                    collision_bsp,
                    &bsp.sbsp.instanced_geometry_instances,
                    &bsp.sbsp.instance_definitions,
                    bsp_slot as i32,
                    &mut world_projection,
                    &mut collisions,
                );
                if progress_stride == 1 {
                    eprintln!(
                        "[decal-step {placement_idx}] collide returned ok={ok} collisions={}",
                        collisions.len(),
                    );
                }
                if !ok || collisions.is_empty() {
                    // Engine logs "Couldn't get a collision for
                    // preplaced decal '%s'" here.
                    skipped_no_collision += 1;
                    continue;
                }

                // Engine `c_decal_system::check_overlap @ 0x18039A050`,
                // called from inside `collide` right after the primary
                // `test_vector` succeeds. The check passes the primary
                // hit point (`m_collision_result.point`, which our
                // `decal_collide` writes into `world_projection.position`
                // via the `set_center` port at collide.rs:Phase 3) and
                // bails the whole collide on rejection. Same effect as
                // dropping the placement here.
                if decal_system.max_overlapping > 0 {
                    let inner_r = decal_system.runtime_max_radius
                        - decal_system.overlapping_threshold;
                    let threshold = inner_r + decal_system.runtime_max_radius;
                    let threshold_sq = threshold * threshold;
                    let center = world_projection.position;
                    let mut overlap_count: i32 = 0;
                    for placed in &placed_pool {
                        if placed.palette_index != placement.palette_index {
                            continue;
                        }
                        let dx = placed.center.x - center.x;
                        let dy = placed.center.y - center.y;
                        let dz = placed.center.z - center.z;
                        if dx * dx + dy * dy + dz * dz <= threshold_sq {
                            overlap_count += 1;
                            if overlap_count >= decal_system.max_overlapping {
                                break;
                            }
                        }
                    }
                    if overlap_count >= decal_system.max_overlapping {
                        skipped_overlap += 1;
                        continue;
                    }
                }
                total_collisions += collisions.len();

                // ---- D2b: build_mesh ----
                //
                // Single call processes all seeds; the orchestrator
                // detects instance seeds via `collision.instance_definition_index >= 0`
                // and inverse-transforms each into instance-local space
                // internally (matches engine `c_decal::build_mesh_fragment
                // @ 0x18039CCC0:184-218`). The per-fragment
                // `instance_local_to_world` is stored on the fragment
                // for the writer to use below.
                //
                // Engine `c_decal::create @ 0x18039A800` writes
                // m_texture_scale per c_decal:
                //   v25 = random(radius_min, radius_max)
                //   *((float *)v17 + 12) = v25                            // m_texture_scale.y
                //   *((float *)v17 + 11) = v25 * definition.m_bitmap_aspect // m_texture_scale.x
                //
                // The texcoord is then `position.{y,z} / (2 *
                // texture_scale.{x,y}) + 0.5`
                // (`build_mesh_fragment_recursive @ 0x18039C2D0`). Pixel
                // shader `decal_fx.hlsl:465` discards UVs outside [0,1].
                //
                // We use `radius_max` deterministically (no per-decal
                // randomization at scenario load) — engine's per-c_decal
                // randomization happens later, but for preplaced layout
                // the upper bound is the right default. `placement.scale`
                // is NOT used here — engine's `c_decal_system::place @
                // 0x180398610` passes only translation + rotation; the
                // tag's scale field is vestigial (or applied at
                // tag-build, not runtime).
                let radius = decal_def.radius.1.max(0.0001);
                // Engine `c_decal::create` uses `definition.m_bitmap_aspect`
                // directly (no fallback) — the cache builder bakes it
                // from base_map width/height. Tags that ship `0` for
                // this field get the same value resolved at scenario
                // load time (Fix #3, see palette_resolved_aspects above).
                let bitmap_aspect = if decal_def.bitmap_aspect > 0.0 {
                    decal_def.bitmap_aspect
                } else {
                    decal_palette_resolved_aspects
                        .get(placement.palette_index as usize)
                        .copied()
                        .unwrap_or(1.0)
                };
                let ctx = DecalBuildContext {
                    cull_angle_radians: decal_def.cull_angle_degrees.to_radians(),
                    clamp_angle_radians: decal_def.clamp_angle_degrees.to_radians(),
                    texture_scale_x: (radius * bitmap_aspect).max(0.0001),
                    texture_scale_y: radius,
                    system_definition_flags: decal_system.flags.clone(),
                    // Engine: left-handed projection when exactly one of the
                    // two random-mirror bits is set (`(>>1 ^ >>2) & 1`).
                    left_handed: decal_system
                        .flags
                        .contains(blam_tags::decal_system::DecalSystemFlags::RandomUMirror)
                        ^ decal_system
                            .flags
                            .contains(blam_tags::decal_system::DecalSystemFlags::RandomVMirror),
                    decal_definition_flags: decal_def.flags,
                };
                let params = DecalParams {
                    local_to_world: world_projection,
                    context: ctx.clone(),
                };
                fragments.clear();
                fragments.resize(collisions.len(), DecalFragment::default());
                if progress_stride == 1 {
                    eprintln!(
                        "[decal-step {placement_idx}] entering build_mesh ({} fragments)",
                        fragments.len(),
                    );
                }
                let ok_build = decal_build_mesh(
                    collision_bsp,
                    &bsp.sbsp.collision_materials,
                    &bsp.sbsp.instanced_geometry_instances,
                    &bsp.sbsp.instance_definitions,
                    &params,
                    &mut mesh_builder,
                    &collisions,
                    &mut fragments,
                );
                if progress_stride == 1 {
                    eprintln!(
                        "[decal-step {placement_idx}] build_mesh returned ok={ok_build} indices={}",
                        mesh_builder.output_index_count,
                    );
                }
                // Accept the placement if EITHER the BFS produced
                // indexed geometry OR any fragment chose the floating-
                // quad fast path (engine `c_decal::m_floating_vertices`
                // is the only output in that case — `output_index_count`
                // stays at 0 because the BFS contribution was rewound).
                let has_floating_quad =
                    fragments.iter().any(|f| f.floating_quad.is_some());
                if !ok_build
                    || (mesh_builder.output_index_count == 0 && !has_floating_quad)
                {
                    skipped_no_collision += 1;
                    continue;
                }

                // ---- D2c: pack output, per-fragment writer dispatch ----
                //
                // Each fragment's working/output vertices may index
                // into the MAIN BSP (main-bsp seed) or an INSTANCE
                // BSP (instance-geom seed). Resolve the right BSP +
                // transform per fragment by walking the collision
                // array (aligned 1:1 with fragments).
                let mut packed_v = vec![
                    RasterizerVertexWorld {
                        position: [0.0; 3],
                        texcoord: [0.0; 2],
                        normal: [0; 4],
                        tangent: [0; 4],
                        binormal: [0; 4],
                    };
                    mesh_builder.output_vertex_count as usize
                ];
                let mut packed_i = vec![0u16; mesh_builder.output_index_count as usize];
                for (fi, fragment) in fragments.iter().enumerate() {
                    if fragment.output_intervals.vertex_count == 0 {
                        continue;
                    }
                    let collision = &collisions[fi];
                    let (frag_bsp, frag_xform): (
                        &blam_tags::structure_bsp::Bsp3d,
                        Option<RealMatrix4x3>,
                    ) = if collision.instance_definition_index < 0 {
                        (collision_bsp, None)
                    } else {
                        let inst_idx = collision.instance_definition_index as usize;
                        let inst = match bsp.sbsp.instanced_geometry_instances.get(inst_idx) {
                            Some(i) => i,
                            None => continue,
                        };
                        let inst_def = match bsp
                            .sbsp
                            .instance_definitions
                            .get(inst.definition_index as usize)
                        {
                            Some(d) => d,
                            None => continue,
                        };
                        let inst_bsp = match inst_def.bsp.as_ref() {
                            Some(b) => b,
                            None => continue,
                        };
                        (inst_bsp, fragment.instance_local_to_world)
                    };
                    write_mesh_fragment(
                        frag_bsp,
                        &mesh_builder,
                        fragment,
                        frag_xform.as_ref(),
                        decal_def.flags,
                        &mut packed_v,
                        &mut packed_i,
                    );
                }

                // Floating-quad emit — mirrors engine `c_decal::render`'s
                // `draw_primitive_up(triangle_strip, 2, m_floating_vertices,
                // 0x2Cu)` path. For each fragment that chose the quad fast
                // path in the orchestrator, append the 4 vertices + 4
                // triangle-strip indices to the per-placement packed buffer.
                // BFS output for these fragments is zero-length (rewound by
                // `build_mesh_fragment`), so this is the only geometry they
                // contribute.
                for fragment in &fragments {
                    if let Some(quad) = fragment.floating_quad {
                        let vbase = packed_v.len() as u16;
                        packed_v.extend_from_slice(&quad);
                        packed_i.extend_from_slice(&[
                            vbase,
                            vbase + 1,
                            vbase + 2,
                            vbase + 3,
                        ]);
                    }
                }

                total_packed_bytes += (packed_v.len()
                    * std::mem::size_of::<RasterizerVertexWorld>()
                    + packed_i.len() * std::mem::size_of::<u16>())
                    as u64;

                if decal_debug_remaining > 0 && !packed_v.is_empty() {
                    decal_debug_remaining -= 1;
                    let dump_idx = built;
                    let palette_name = scenario
                        .decal_palette
                        .get(placement.palette_index as usize)
                        .map(|p| p.decal_system.rsplit(['/', '\\']).next().unwrap_or(""))
                        .unwrap_or("?");
                    let base_map_path = decal_def
                        .shader
                        .as_ref()
                        .and_then(|s| {
                            s.parameters
                                .iter()
                                .find(|p| p.parameter_name == "base_map")
                                .map(|p| p.bitmap_path.as_str())
                        })
                        .unwrap_or("?");
                    let (mut u_min, mut u_max) = (f32::INFINITY, f32::NEG_INFINITY);
                    let (mut v_min, mut v_max) = (f32::INFINITY, f32::NEG_INFINITY);
                    for v in &packed_v {
                        let uv = v.texcoord; // copy out of packed struct
                        u_min = u_min.min(uv[0]);
                        u_max = u_max.max(uv[0]);
                        v_min = v_min.min(uv[1]);
                        v_max = v_max.max(uv[1]);
                    }
                    // Engine `c_decal::render @ 0x18039B100` branches on
                    // `def_flags` bits 0 (specular_modulate) and 1
                    // (bump_modulate) — bit 1 drives the second-pass RT1
                    // write. Surface them here so we can spot when an rmd
                    // option (blend_mode / bump_mapping) doesn't line up
                    // with the per-palette flags the engine actually uses.
                    eprintln!(
                        "[decal-dump #{dump_idx}] palette[{}]={} base_map={}",
                        placement.palette_index, palette_name, base_map_path
                    );
                    eprintln!(
                        "  pos=({:.2},{:.2},{:.2}) radius_range=({:.3},{:.3}) used={:.3} \
                         bitmap_aspect={:.3} texture_scale=({:.3},{:.3}) \
                         cull_angle_deg={:.3} clamp_angle_deg={:.3} \
                         cull_cos={:.6} clamp_cos={:.6}",
                        placement.position.x,
                        placement.position.y,
                        placement.position.z,
                        decal_def.radius.0,
                        decal_def.radius.1,
                        radius,
                        bitmap_aspect,
                        ctx.texture_scale_x,
                        ctx.texture_scale_y,
                        decal_def.cull_angle_degrees,
                        decal_def.clamp_angle_degrees,
                        ctx.cull_angle_radians.cos(),
                        ctx.clamp_angle_radians.cos(),
                    );
                    let n_frags_nonempty = fragments
                        .iter()
                        .filter(|f| f.output_intervals.vertex_count > 0)
                        .count();
                    eprintln!(
                        "  collisions={} fragments_nonempty={}/{} verts={} indices={}",
                        collisions.len(),
                        n_frags_nonempty,
                        fragments.len(),
                        packed_v.len(),
                        packed_i.len(),
                    );
                    eprintln!(
                        "  UV range u=[{u_min:.3} .. {u_max:.3}] v=[{v_min:.3} .. {v_max:.3}]"
                    );
                    // First 4 vertices — exact positions + texcoords.
                    for (i, v) in packed_v.iter().take(4).enumerate() {
                        let pos = v.position; // copy out of packed struct
                        let uv = v.texcoord;
                        eprintln!(
                            "    v[{i}] pos=({:.2},{:.2},{:.2}) uv=({:.3},{:.3})",
                            pos[0], pos[1], pos[2], uv[0], uv[1],
                        );
                    }
                }

                if let Some(bsp_decals) = decal_renderer.per_bsp.get_mut(bsp_slot) {
                    bsp_decals.append(
                        &packed_v[..],
                        &packed_i[..],
                        placement.palette_index as u16,
                    );
                    built += 1;
                    // Retain for `check_overlap` against subsequent
                    // placements. Engine equivalent: append to
                    // `c_decal_system::x_data_array` after `create`
                    // succeeds. We use the (rewritten) projection
                    // position so the pool tracks the actual hit
                    // center, matching the engine's read in
                    // `check_overlap`.
                    placed_pool.push(PlacedDecal {
                        palette_index: placement.palette_index,
                        center: world_projection.position,
                    });
                }
            }
            eprintln!(
                "[scenario] decal placements: built={built} no_palette={skipped_no_palette} no_bsp={skipped_no_bsp} no_collision={skipped_no_collision} overlap_rejected={skipped_overlap} | total_collisions={total_collisions} | packed {} KiB",
                total_packed_bytes / 1024,
            );
            for (i, bsp) in decal_renderer.per_bsp.iter().enumerate() {
                if bsp.decals.is_empty() {
                    continue;
                }
                eprintln!(
                    "[scenario] decal bsp[{i}]: {} decals, {} verts, {} indices",
                    bsp.decals.len(),
                    bsp.vertices.len(),
                    bsp.indices.len(),
                );
            }
        }

        // Publish the loaded scenario as the engine `global_scenario`
        // pointer — accessed by code that needs scenario-scoped data
        // (e.g. `object_compute_function_value` case `compass` reads
        // `local_north`). Set BEFORE populate_from_scenario so any
        // datum-init paths that need scenario context can read it.
        crate::halo::scenario::globals::set_global_scenario(
            std::sync::Arc::new(scenario.clone()),
        );

        // Initialize the runtime tag cache. populate_from_scenario will
        // resolve each placement's palette ObjectDefinition + chase the
        // model ref through `tag_get`.
        crate::halo::tags::tag_init(tags_root.clone());

        // Populate the global `object_header_data` table (engine-faithful
        // equivalent — see `objects::object_header_data`). Walked by
        // `object_type_compute_function_value` to route per-mesh
        // animation function-value queries to the right
        // `<type>_compute_function_value`. Phase 1 of the
        // [[project_render_method_function_port_plan_2026_05_20]] port.
        crate::halo::objects::object_header_data::populate_from_scenario(&scenario, &tags_root);

        Ok(Self {
            scenario_path: scenario_path.to_path_buf(),
            tags_root,
            scenario,
            active_zone_set,
            active_bsps,
            active_bsp_winds,
            default_wind,
            atmosphere,
            camera_fx,
            chocolate_mountain,
            sky_lighting,
            decorator_sets,
            decorator_render_models,
            decorator_subpart_geometry,
            runtime_lights,
            scenario_lightmap,
            decal_palette,
            decal_renderer,
        })
    }

    pub fn zone_set(&self) -> &ZoneSet {
        &self.scenario.zone_sets[self.active_zone_set]
    }

    pub fn skies(&self) -> &[SkyReference] {
        &self.scenario.skies
    }

    /// Authoring-time decorator palettes / sets / placements (the
    /// "new decorator block" in scenario tags). MCC tag-ships these
    /// arrays populated but the `sbsp.decorator sets` runtime arrays
    /// are stripped — see `reference_mcc_strips_decorator_data.md`.
    /// Each placement carries `runtime_bsp_index = -1` /
    /// `cluster_index = -1` and must be re-bound at load via
    /// point-in-cluster tests.
    pub fn decorators(&self) -> &[blam_tags::scenario::ScenarioDecoratorBlock] {
        &self.scenario.decorators
    }

    pub fn bsp_reference(&self, bsp: &LoadedBsp) -> &StructureBspReference {
        &self.scenario.structure_bsps[bsp.scenario_bsp_index]
    }

    /// Iterate every (palette, placements) pair across all object types
    /// the scenario carries. Convenient for reporting / Phase E1
    /// instantiation.
    pub fn all_placements(&self) -> Vec<(&'static str, &[TagReferencePalette], &[ObjectPlacement])> {
        let s = &self.scenario;
        vec![
            ("scenery", &s.scenery_palette, &s.scenery),
            ("biped", &s.biped_palette, &s.bipeds),
            ("vehicle", &s.vehicle_palette, &s.vehicles),
            ("equipment", &s.equipment_palette, &s.equipment),
            ("weapon", &s.weapon_palette, &s.weapons),
            ("machine", &s.machine_palette, &s.machines),
            ("control", &s.control_palette, &s.controls),
            ("sound_scenery", &s.sound_scenery_palette, &s.sound_scenery),
            ("crate", &s.crate_palette, &s.crates),
            ("light", &s.light_palette, &s.lights),
        ]
    }
}

/// Walk `scenario.skies[0].sky` → `.scenery → .model → .render_model`
/// and decode the SH probe + extract a dominant light from it.
/// Mirrors dllcache's `setup_default_lighting` path:
/// `calculate_and_set_ravi_constants(probe, scale=1.0)` +
/// `calculate_dominant_light_from_lightprobe(probe → dir, intensity)`
/// from `c_lighting_interface`.
///
/// NOT the same as `get_sun_constants_from_sky @ 0x1803adcb0` —
/// that's a separate code path (sky_lights[count-1] direct read).
/// dllcache's static-default lighting derives the dominant from the
/// baked SH probe so its units stay consistent with the SH evaluation.
fn load_sky_lighting(
    tags_root: &Path,
    scenario: &blam_tags::scenario::Scenario,
) -> Option<SkyLighting> {
    let sky_ref = scenario.skies.first()?;
    if sky_ref.sky.is_empty() { return None; }

    // sky scenery → object → model
    let scenery_path = resolve_tag_path(tags_root, &sky_ref.sky, "scenery");
    let scenery_tag = TagFile::read(&scenery_path).ok()?;
    let scenery_root = scenery_tag.root();
    let model_rel = blam_tags::paths::tag_ref_path(&scenery_root, "object/model")?;

    // model → render_model
    let model_path = resolve_tag_path(tags_root, &model_rel, "model");
    let model_tag = TagFile::read(&model_path).ok()?;
    let render_model_rel = blam_tags::paths::tag_ref_path(&model_tag.root(), "render model")?;

    // render_model → default_lightprobe (SH3 R/G/B 9-floats each)
    let render_model_path = resolve_tag_path(tags_root, &render_model_rel, "render_model");
    let render_model_tag = TagFile::read(&render_model_path).ok()?;
    let render_model = blam_tags::render_model::RenderModel::from_tag(&render_model_tag).ok()?;
    // `lightgen_lights[count-1]` — engine's atmosphere-sun source when
    // the atmosphere setting's flag bit 1 is unset (deadlock, riverworld,
    // most maps). `c_atmosphere_fog_interface::get_sun_constants_from_sky
    // @ 0x1803adcb0` reads the LAST entry (LIFO convention: dominant sun
    // is appended last) and applies `intensity × solid_angle × 0.2 ×
    // g_render_light_intensity`. We use g_render_light_intensity=1.0.
    let (lightgen_sun_intensity, lightgen_sun_direction) = render_model
        .sky_lights
        .last()
        .map(|sl| {
            (
                blam_tags::math::RealVector3d {
                    i: sl.intensity.i * sl.solid_angle * 0.2,
                    j: sl.intensity.j * sl.solid_angle * 0.2,
                    k: sl.intensity.k * sl.solid_angle * 0.2,
                },
                sl.direction,
            )
        })
        .map(|(i, d)| (Some(i), Some(d)))
        .unwrap_or((None, None));
    let probe = SkyProbe::from(&render_model.default_lightprobe?);

    // Extract dominant light from the SH probe (dllcache
    // `calculate_dominant_light_from_lightprobe`).
    let (sun_direction, sun_intensity) =
        calculate_dominant_light_from_lightprobe(&probe.r, &probe.g, &probe.b);

    Some(SkyLighting {
        sun_direction,
        sun_intensity,
        probe,
        lightgen_sun_intensity,
        lightgen_sun_direction,
    })
}

/// dllcache `find_dominant_light_direction`. Computes direction TO the
/// dominant light from SH3 L1 coefficients using ITU-R BT.709 luminance
/// weighting (0.213 R + 0.715 G + 0.072 B).
///
/// The L1 components of the SH probe encode the luminance gradient:
/// `sh[1]` carries the y-axis sensitivity, `sh[2]` z, `sh[3]` x. Halo's
/// sign convention negates x and y on extraction (so the returned
/// direction points TOWARD the source).
fn find_dominant_light_direction(probe_r: &[f32; 9], probe_g: &[f32; 9], probe_b: &[f32; 9])
    -> blam_tags::math::RealVector3d
{
    let lum = |i: usize| 0.21265601 * probe_r[i] + 0.71515799 * probe_g[i] + 0.072185598 * probe_b[i];
    let mut x = -lum(3);
    let mut y = -lum(1);
    let mut z = lum(2);
    let len = (x*x + y*y + z*z).sqrt();
    if len > 1e-6 { x /= len; y /= len; z /= len; }
    blam_tags::math::RealVector3d { i: x, j: y, k: z }
}

/// dllcache `sh_eval_direction(out, order=3, dir)`. Evaluates the SH3
/// basis functions at a given direction. Halo's sign convention
/// negates the m≠0 components of the L1/L2 bands.
fn sh_eval_direction_order3(dir: &blam_tags::math::RealVector3d) -> [f32; 9] {
    let x = dir.i; let y = dir.j; let z = dir.k;
    let inv_sqrt_pi = 1.0 / std::f32::consts::PI.sqrt();
    let sqrt3 = 3.0_f32.sqrt();
    let sqrt5 = 5.0_f32.sqrt();
    let sqrt15 = 15.0_f32.sqrt();
    let l1 = sqrt3 * inv_sqrt_pi;
    let l2 = sqrt15 * inv_sqrt_pi;
    let mz = sqrt5 * inv_sqrt_pi;
    [
        inv_sqrt_pi * 0.5,                       // 1/(2√π)
        l1 * y * -0.5,                           // -y √(3/π)/2
        l1 * z *  0.5,                           //  z √(3/π)/2
        l1 * x * -0.5,                           // -x √(3/π)/2
        l2 * (y * x) *  0.5,                     //  xy √(15/π)/2
        l2 * (y * z) * -0.5,                     // -yz √(15/π)/2
        mz * (3.0 * z * z - 1.0) * 0.25,         // (3z²-1) √(5/π)/4
        l2 * (z * x) * -0.5,                     // -xz √(15/π)/2
        l2 * (x*x - y*y) * 0.25,                 // (x²-y²) √(15/π)/4
    ]
}

/// dllcache `calculate_dominant_light_from_lightprobe`. Extracts a
/// dominant light direction + intensity from an SH3 probe by:
/// 1) finding the brightest direction (luminance-weighted L1),
/// 2) evaluating the SH3 basis at that direction,
/// 3) projecting each channel's probe onto the basis (dot/self_dot).
///
/// Returns `(direction, intensity_rgb)` in the same units as the input
/// probe — so for a probe with sh[0]≈1.2 the intensity is also ~1ish.
fn calculate_dominant_light_from_lightprobe(
    probe_r: &[f32; 9], probe_g: &[f32; 9], probe_b: &[f32; 9],
) -> (blam_tags::math::RealVector3d, blam_tags::math::RealVector3d) {
    let dir = find_dominant_light_direction(probe_r, probe_g, probe_b);
    let basis = sh_eval_direction_order3(&dir);

    let mut self_dot = 0.0f32;
    for v in basis.iter() { self_dot += v * v; }
    let denom = if self_dot <= 1e-4 { 1e-4 } else { self_dot };

    let project = |probe: &[f32; 9]| -> f32 {
        let mut s = 0.0f32;
        for i in 0..9 { s += probe[i] * basis[i]; }
        if s < 0.0 { 0.0 } else { s / denom }
    };

    let intensity = blam_tags::math::RealVector3d {
        i: project(probe_r),
        j: project(probe_g),
        k: project(probe_b),
    };
    (dir, intensity)
}

/// Load the per-BSP lightmap and report the resolved tag path so the
/// caller can re-read the same tag for additional walks (e.g. pulling
/// `imported geometry` for the lightmap UV stream).
fn load_per_bsp_lightmap(
    tags_root: &Path,
    sbsp_path: &Path,
    scenario_lightmap: Option<&ScenarioLightmap>,
    bsp_index: usize,
) -> (Option<LightmapBspData>, Option<std::path::PathBuf>) {
    // Path 1: sLdT chain. Pick per_pixel first (the canonical path),
    // fall back to per_vertex if the index isn't in per_pixel.
    if let Some(sldt) = scenario_lightmap {
        let lbsp_rel = sldt
            .per_pixel_bsp_data
            .get(bsp_index)
            .filter(|s| !s.is_empty())
            .or_else(|| sldt.per_vertex_bsp_data.get(bsp_index).filter(|s| !s.is_empty()));
        if let Some(rel) = lbsp_rel {
            let p = resolve_tag_path(tags_root, rel, "scenario_lightmap_bsp_data");
            if p.exists() {
                if let Ok(tag) = TagFile::read(&p) {
                    return (LightmapBspData::from_tag(&tag).ok(), Some(p));
                }
                return (None, Some(p));
            }
        }
    }

    // Path 2: convention — same stem as sbsp, swap extension.
    let mut p = sbsp_path.to_path_buf();
    p.set_extension("scenario_lightmap_bsp_data");
    if !p.exists() {
        return (None, None);
    }
    let Ok(tag) = TagFile::read(&p) else { return (None, Some(p)); };
    (LightmapBspData::from_tag(&tag).ok(), Some(p))
}
