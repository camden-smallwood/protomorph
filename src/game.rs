use crate::halo::camera::FlyingCamera;
use crate::halo::geometry::ModelData;
use crate::halo::objects::{ObjectIndex, ObjectStore};
use crate::halo::render::Renderer;
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use std::path::Path;
use winit::keyboard::KeyCode;

const MOVEMENT_SPEED: f32 = 10.0;

// ---------------------------------------------------------------------------
// FPS counter
// ---------------------------------------------------------------------------

pub struct FpsCounter {
    frame_count: u32,
    elapsed: f32,
    pub display_fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self { frame_count: 0, elapsed: 0.0, display_fps: 0.0 }
    }

    pub fn update(&mut self, dt: f32) {
        self.frame_count += 1;
        self.elapsed += dt;
        if self.elapsed >= 0.5 {
            self.display_fps = self.frame_count as f32 / self.elapsed;
            self.frame_count = 0;
            self.elapsed = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Game state — Phase 1 Halo viewer (flycam-only, no collision/physics)
// ---------------------------------------------------------------------------

/// A scenario placement parented to another object by name + marker,
/// resolved after all placements load ([`GameState::resolve_pending_attachments`]).
/// Engine equivalent: the `object_attach_to_marker` deferred until the
/// parent (which may be the sky or a later-loaded placement) exists.
struct PendingAttachment {
    /// `object_index` (header) of the child to reposition.
    child_header_index: u32,
    /// `parent id.parent object` — index into `scenario.object_names`.
    parent_name_index: i16,
    /// `parent id.parent marker` — marker on the parent to snap to.
    parent_marker: String,
}

/// Engine-faithful weather lifecycle (`c_atmosphere_fog_interface::change_pvs`).
/// Holds the active BSP's atmosphere settings + cluster geometry so each frame
/// it can resolve which atmosphere settings are active in the camera's cluster
/// sphere (PVS-wide) and keep one weather effect per active setting alive,
/// spawning on activation and destroying on deactivation — a per-setting
/// `weather_effect_indices` table keyed by setting index.
struct WeatherDriver {
    atmosphere: blam_tags::sky_atmosphere::SkyAtmosphere,
    clusters: Vec<blam_tags::structure_bsp::BspCluster>,
    portals: Vec<blam_tags::structure_bsp::BspClusterPortal>,
    planes: Vec<blam_tags::math::RealPlane3d>,
    palette: Vec<blam_tags::structure_bsp::BspAtmospherePaletteEntry>,
    tags_root: std::path::PathBuf,
    /// Per atmosphere-setting → live `effects.instances` index (engine
    /// `g_rasterizer_game_states->weather_effect_indices[setting]`). `None` =
    /// no live weather effect for that setting.
    effect_indices: Vec<Option<usize>>,
}

pub struct GameState {
    pub objects: ObjectStore,
    /// Effects attached to placed objects (scenery effe attachments).
    /// Phase 1: loaded + instanced per placement; simulated/rendered in
    /// later phases. See [`crate::halo::effects`].
    pub effects: crate::halo::effects::EffectStore,
    /// Atmosphere WEATHER lifecycle driver (engine `c_atmosphere_fog_interface
    /// ::change_pvs @0x1803AF550`): a per-setting weather-effect table that
    /// spawns one camera-relative effect per atmosphere setting active in the
    /// camera's cluster sphere, and destroys it when that setting leaves.
    /// `None` until a scenario with atmosphere settings loads.
    weather: Option<WeatherDriver>,
    /// Collected during placement load; resolved into datum world transforms
    /// by [`Self::resolve_pending_attachments`] once every object exists.
    pending_attachments: Vec<PendingAttachment>,
    pub camera: FlyingCamera,
    pub model_data: Vec<ModelData>,
    pub fps_counter: FpsCounter,
    /// `scenario.skies[0]` loaded as an object — the sky model
    /// (`.scenery → .model → .render_model`). Per Ares' `render_sky`
    /// chain, this object's node matrices get translated by the
    /// camera position each frame so the sky stays infinitely far.
    pub sky_object: Option<ObjectIndex>,
    /// Tracks how many placements have been registered against each
    /// renderer `model_index`. Used to assign each new placement an
    /// `instance_within_model` slot index (Path B per-instance cbuffer
    /// pool — see [[project_render_method_function_port_plan_2026_05_20]]).
    /// Accumulates across multiple `load_visible_placements` calls so
    /// different scenario placement types (scenery / weapons / crates)
    /// don't collide on slot indices.
    pub placements_per_model: std::collections::HashMap<usize, u32>,

    pub enable_specular_occlusion: bool,
    pub enable_vignette: bool,
    pub total_time: f32,
    /// Real per-frame `dt` written by `update`, read by the renderer.
    /// Mirrors engine `c_patchy_fog::ms_dt` (set by the main game tick,
    /// consumed by `render_patchy_fog` to scale `wind_direction`).
    pub delta_time: f32,
}

impl GameState {
    pub fn new<P: AsRef<Path>>(renderer: &mut Renderer, width: u32, height: u32, scenario_path: P) -> Self {
        let mut camera = FlyingCamera::new();
        camera.handle_resize(width, height);
        // Cyberdyne — UNSC TRAINING B FACILITY wall sign reference view
        // matching sapien Camera (X=-12.85, Y=-10.27, Z=4.13). F5/F6/F7
        // jump to other decal reference views.
        camera.position = Vec3::new(-12.85, -10.27, 4.13);
        camera.rotation = glam::Vec2::new(90.0, 0.0);
        // PROTOMORPH_CAMERA="x,y,z" / "x,y,z,yaw,pitch" override —
        // lets us jump straight to a specific decal placement for
        // visual debugging without flying.
        if let Ok(s) = std::env::var("PROTOMORPH_CAMERA") {
            let parts: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            if parts.len() >= 3 {
                camera.position = Vec3::new(parts[0], parts[1], parts[2]);
            }
            if parts.len() >= 5 {
                camera.rotation = glam::Vec2::new(parts[3], parts[4]);
            }
        }
        camera.update();

        let mut state = Self {
            objects: ObjectStore::new(),
            effects: crate::halo::effects::EffectStore::new(),
            weather: None,
            pending_attachments: Vec::new(),
            camera,
            model_data: Vec::new(),
            fps_counter: FpsCounter::new(),
            sky_object: None,
            placements_per_model: std::collections::HashMap::new(),
            enable_specular_occlusion: true,
            enable_vignette: true,
            total_time: 0.0,
            delta_time: 1.0 / 60.0,
        };

        state.load_scene(renderer, scenario_path);
        state
    }

    fn load_scene<P: AsRef<Path>>(&mut self, renderer: &mut Renderer, scenario_path: P) {
        // Phase D1 smoke test: load riverworld.scenario if it's there
        // and report what came back. This doesn't render anything yet —
        // Phase D2/D3 wires sbsp meshes through the pipeline.
        let scenario_path = scenario_path.as_ref();

        if scenario_path.exists() {
            match renderer.load_scenario(scenario_path) {
                Ok(loaded) => {
                    eprintln!(
                        "[scenario] loaded {} — zone_set[{}] '{}' (bsps active: {}, pvs={})",
                        loaded.scenario_path.display(),
                        loaded.active_zone_set,
                        loaded.zone_set().name,
                        loaded.active_bsps.len(),
                        loaded.zone_set().pvs_index,
                    );
                    for bsp in &loaded.active_bsps {
                        let r = loaded.bsp_reference(bsp);
                        let total_verts: usize =
                            bsp.meshes.iter().map(|m| m.vertices.len()).sum();
                        let total_tris: usize =
                            bsp.meshes.iter().map(|m| m.indices.len() / 3).sum();
                        eprintln!(
                            "[scenario]   bsp[{}] {} — {} clusters, {} instances, {} materials, {} meshes ({} verts, {} tris), lightmap={}",
                            bsp.scenario_bsp_index,
                            r.structure_bsp,
                            bsp.sbsp.clusters.len(),
                            bsp.sbsp.instanced_geometry_instances.len(),
                            bsp.sbsp.materials.len(),
                            bsp.meshes.len(),
                            total_verts,
                            total_tris,
                            bsp.lightmap.is_some(),
                        );

                        // Upload the BSP through the dedicated structure
                        // renderer path — separate from the per-character
                        // ModelData/ObjectData pipeline. Mirrors Ares'
                        // `c_structure_renderer` (per the deep trace in
                        // `reference_h3_structure_render_pipeline.md` in
                        // auto-memory).
                        let bsp_idx = renderer.upload_bsp(
                            bsp,
                            &loaded.tags_root,
                            &loaded.scenario.cubemaps,
                        );
                        eprintln!(
                            "[scenario]   uploaded BSP[{}] via structure renderer",
                            bsp_idx,
                        );
                    }
                    for (kind, palette, placements) in loaded.all_placements() {
                        if !placements.is_empty() {
                            eprintln!(
                                "[scenario]   {kind}: {} unique × {} placements",
                                palette.len(),
                                placements.len(),
                            );
                        }
                    }

                    // Decorator (foliage) authoring data — MCC tag-ships
                    // the per-set placement arrays but NOT the per-cluster
                    // runtime structures (those live in sbsp's stripped
                    // `decorator sets` block). Runtime cluster assignment
                    // happens at load via point-in-cluster tests.
                    for (idx, dec) in loaded.decorators().iter().enumerate() {
                        let total: usize = dec.sets.iter().map(|s| s.placements.len()).sum();
                        eprintln!(
                            "[scenario]   decorator block[{}]: {} palettes, {} sets, {} placements (count_field={})",
                            idx,
                            dec.palettes.len(),
                            dec.sets.len(),
                            total,
                            dec.decorator_count,
                        );
                        for (si, set) in dec.sets.iter().enumerate() {
                            if !set.placements.is_empty() {
                                let loaded_set = loaded
                                    .decorator_sets
                                    .get(idx)
                                    .and_then(|sets| sets.get(si))
                                    .and_then(|t| t.as_ref());
                                let shader = loaded_set
                                    .map(|t| format!("{:?}", t.render_shader))
                                    .unwrap_or_else(|| "(unloaded)".to_string());
                                let types = loaded_set
                                    .map(|t| t.decorator_types.len())
                                    .unwrap_or(0);
                                let rm = loaded
                                    .decorator_render_models
                                    .get(idx)
                                    .and_then(|rms| rms.get(si))
                                    .and_then(|t| t.as_ref());
                                let (verts, tris) = rm
                                    .map(|m| {
                                        let v: u32 = m
                                            .render_geometry
                                            .per_mesh_temporary
                                            .iter()
                                            .map(|x| x.raw_vertices.len() as u32)
                                            .sum();
                                        let t: u32 = m
                                            .render_geometry
                                            .per_mesh_temporary
                                            .iter()
                                            .map(|x| (x.raw_indices.len() / 3) as u32)
                                            .sum();
                                        (v, t)
                                    })
                                    .unwrap_or((0, 0));
                                let meshes = rm.map(|m| m.render_geometry.meshes.len()).unwrap_or(0);
                                eprintln!(
                                    "[scenario]     set[{}] {} — {} placements, shader={}, {} types, mesh_blocks={}, {} verts, {} tris",
                                    si,
                                    set.decorator_set,
                                    set.placements.len(),
                                    shader,
                                    types,
                                    meshes,
                                    verts,
                                    tris,
                                );
                            }
                        }
                    }

                    // Sky load — Ares' `c_object_renderer::submit_and_render_sky`
                    // chain. `scenario.skies[i].sky` is a `.scenery` tag-ref
                    // pointing at the sky's `_object_definition`; the
                    // existing object loader walks .scenery → .model →
                    // .render_model. We register it as a regular object
                    // and override its position to the camera per frame
                    // (Ares' `render_sky_modify_node_matrices` applies an
                    // offset = camera_position to keep the sky dome
                    // centered on the viewer).
                    //
                    // For v1 we just take skies[0]; per-cluster sky
                    // selection (`bsp.cluster.scenario_sky_index`) lands
                    // alongside PVS in Phase H.
                    // PROTOMORPH_MINIMAL_LOAD=1 — skip ALL object placements +
                    // sky object + weather (renderer-side sky/decorators/decals
                    // are gated in load_scenario). Bisection for the huge-BSP
                    // first-frame freeze; keeps BSP + lightmap + rasg only.
                    let minimal_load = std::env::var_os("PROTOMORPH_MINIMAL_LOAD").is_some();
                    if !minimal_load {
                        self.load_sky(renderer, &loaded);
                    }

                    self.pick_spawn(&loaded);

                    // Diagnostic: locate placed objects by tag name. Set
                    // `PROTOMORPH_FIND_OBJECT=<substr>` to log the world
                    // position of every placement whose palette tag path
                    // contains <substr> — used to aim the camera at a
                    // specific object (e.g. a shader carried by scenery).
                    if let Ok(target) = std::env::var("PROTOMORPH_FIND_OBJECT") {
                        let t = target.to_ascii_lowercase();
                        for (kind, palette, placements) in loaded.all_placements() {
                            for p in placements {
                                if p.palette_index < 0 { continue; }
                                let Some(pe) = palette.get(p.palette_index as usize) else { continue; };
                                if pe.tag_path.to_ascii_lowercase().contains(&t) {
                                    let pos = p.object_data.position;
                                    eprintln!(
                                        "[find_object] {kind} '{}' pos=({:.2}, {:.2}, {:.2})",
                                        pe.tag_path, pos.x, pos.y, pos.z,
                                    );
                                }
                            }
                        }
                    }

                    // PROTOMORPH_CAMERA override applies LAST so it wins
                    // over respawn-point auto-positioning. Format:
                    // "x,y,z" or "x,y,z,yaw,pitch".
                    if let Ok(s) = std::env::var("PROTOMORPH_CAMERA") {
                        let parts: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                        if parts.len() >= 3 {
                            self.camera.position = Vec3::new(parts[0], parts[1], parts[2]);
                        }
                        if parts.len() >= 5 {
                            self.camera.rotation = glam::Vec2::new(parts[3], parts[4]);
                        }
                        self.camera.update();
                        eprintln!(
                            "[scenario]   PROTOMORPH_CAMERA override -> pos=({:.2}, {:.2}, {:.2}) yaw={:.1}° pitch={:.1}°",
                            self.camera.position.x, self.camera.position.y, self.camera.position.z,
                            self.camera.rotation.x, self.camera.rotation.y,
                        );
                    }

                    // Halo's runtime renders every visible object placement
                    // (per `c_object_renderer::submit_objects`). Load
                    // scenery / crates / weapons / equipment by walking
                    // each placement array and the matching palette.
                    // Caching is per-palette so repeated tag references
                    // upload once.
                    // Scenery placements pass their per-placement
                    // engine_lighting offsets (baked from sLdT.scenery_probes).
                    // Other placement types have no analogous baked probe block
                    // — engine resolves them via airprobes / device probes / sky
                    // fallback at runtime; for v1 we pass None and inherit the
                    // frame default sky probe.
                    let scenery_offsets = renderer.scenery_lighting_offsets.clone();
                    let crate_offsets = renderer.crate_lighting_offsets.clone();
                    let vehicle_offsets = renderer.vehicle_lighting_offsets.clone();
                    let weapon_offsets = renderer.weapon_lighting_offsets.clone();
                    let equipment_offsets = renderer.equipment_lighting_offsets.clone();
                    let machine_offsets = renderer.machine_lighting_offsets.clone();
                    let control_offsets = renderer.control_lighting_offsets.clone();
                    use crate::halo::objects::object_type::ObjectType;
                    if !minimal_load {
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::Scenery,
                        &loaded.scenario.scenery,
                        &loaded.scenario.scenery_palette,
                        &loaded.tags_root,
                        "scenery",
                        |tp| tp.contains("spawn_point") || tp.contains("respawn_zone"),
                        Some(&scenery_offsets),
                    );
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::Crate,
                        &loaded.scenario.crates,
                        &loaded.scenario.crate_palette,
                        &loaded.tags_root,
                        "crate",
                        |_| false,
                        Some(&crate_offsets),
                    );
                    // Vehicles — placed like any other object. Loading them
                    // also registers each into the object header table by name
                    // so they participate as parents/children in
                    // `resolve_pending_attachments` (e.g. a turret/scenery
                    // child parented to a vehicle marker, or a vehicle parented
                    // to a moving host). Vehicle object-definition effe
                    // attachments are instanced via the shared palette path.
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::Vehicle,
                        &loaded.scenario.vehicles,
                        &loaded.scenario.vehicle_palette,
                        &loaded.tags_root,
                        "vehicle",
                        |_| false,
                        Some(&vehicle_offsets),
                    );
                    // Effect scenery (`csfe`) — objects that ARE a continuously
                    // playing effect (guardian lift columns / man cannon). Their
                    // effe attachment carries the effect; processed like scenery.
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::EffectScenery,
                        &loaded.scenario.effect_scenery,
                        &loaded.scenario.effect_scenery_palette,
                        &loaded.tags_root,
                        "effect_scenery",
                        |_| false,
                        None,
                    );
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::Weapon,
                        &loaded.scenario.weapons,
                        &loaded.scenario.weapon_palette,
                        &loaded.tags_root,
                        "weapon",
                        |_| false,
                        Some(&weapon_offsets),
                    );
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::Equipment,
                        &loaded.scenario.equipment,
                        &loaded.scenario.equipment_palette,
                        &loaded.tags_root,
                        "equipment",
                        |_| false,
                        Some(&equipment_offsets),
                    );
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::Machine,
                        &loaded.scenario.machines,
                        &loaded.scenario.machine_palette,
                        &loaded.tags_root,
                        "machine",
                        |_| false,
                        Some(&machine_offsets),
                    );
                    self.load_visible_placements(
                        renderer,
                        &loaded.scenario,
                        ObjectType::Control,
                        &loaded.scenario.controls,
                        &loaded.scenario.control_palette,
                        &loaded.tags_root,
                        "control",
                        |_| false,
                        Some(&control_offsets),
                    );

                    // All placements loaded — resolve parent-marker
                    // attachments now that every parent object (and the sky)
                    // exists.
                    self.resolve_pending_attachments(renderer, &loaded.scenario);

                    self.spawn_weather(&loaded);
                    } // end if !minimal_load

                    self.log_scenario_summary(renderer, &loaded);
                }
                Err(e) => eprintln!("[scenario] load failed: {e}"),
            }
        }

        // Vehicle placements now load via `load_visible_placements`
        // above (scenario.vehicles). Bipeds remain deferred until the
        // object-animation lifecycle (Phase E) lands — animated units
        // need the per-frame node/animation pipeline, not just a static
        // placement. Note vehicles whose shaders use unported render-
        // method options (e.g. `environment_map / dynamic`) fail loud at
        // load per the engine-faithful policy — that surfaces what to
        // port next rather than silently dropping the vehicle.
    }

    /// Sky load — Ares' `c_object_renderer::submit_and_render_sky`
    /// chain. `scenario.skies[i].sky` is a `.scenery` tag-ref
    /// pointing at the sky's `_object_definition`; the
    /// existing object loader walks .scenery → .model →
    /// .render_model. We register it as a regular object
    /// and override its position to the camera per frame
    /// (Ares' `render_sky_modify_node_matrices` applies an
    /// offset = camera_position to keep the sky dome
    /// centered on the viewer).
    ///
    /// For v1 we just take skies[0]; per-cluster sky
    /// selection (`bsp.cluster.scenario_sky_index`) lands
    /// alongside PVS in Phase H.
    fn load_sky(
        &mut self,
        renderer: &mut Renderer,
        loaded: &crate::halo::scenario::LoadedScenario,
    ) {
        if let Some(sky_ref) = loaded.skies().first() {
            if !sky_ref.sky.is_empty() {
                let sky_path = blam_tags::paths::resolve_tag_path(
                    &loaded.tags_root,
                    &sky_ref.sky,
                    "scenery",
                );
                if sky_path.exists() {
                    let (model, data) = renderer.load_object_tag(&sky_path);
                    let obj = self.objects.new_object();
                    self.model_data.push(data);
                    let slot = self.objects.get_mut(obj);
                    slot.model_index = Some(model);
                    self.sky_object = Some(obj);
                    eprintln!(
                        "[scenario]   sky: {} (model={})",
                        sky_ref.sky, model,
                    );
                } else {
                    eprintln!(
                        "[scenario]   sky tag missing: {}",
                        sky_path.display(),
                    );
                }
            }
        }
    }

    /// Pick a spawn vantage from the scenery placements that
    /// reference an MP respawn-point/zone palette entry. MP
    /// scenarios author these as `objects/multi/spawning/
    /// respawn_point.scenery` (and game-mode-specific zone
    /// variants like `slayer_respawn_zone.scenery`). Halo's
    /// runtime spawn picker is more involved (game-mode
    /// gating + respawn timer + visibility check), but for
    /// a flycam smoke test the first respawn point is fine.
    fn pick_spawn(&mut self, loaded: &crate::halo::scenario::LoadedScenario) {
        let respawns: Vec<(Vec3, f32)> = loaded
            .scenario
            .scenery
            .iter()
            .filter_map(|p| {
                let idx = p.palette_index;
                if idx < 0 { return None; }
                let palette = loaded.scenario.scenery_palette.get(idx as usize)?;
                let tp = palette.tag_path.to_ascii_lowercase();
                if tp.contains("respawn_point") || tp.contains("respawn_zone") {
                    let pos = p.object_data.position;
                    Some((
                        Vec3::new(pos.x, pos.y, pos.z),
                        p.object_data.rotation.yaw,
                    ))
                } else {
                    None
                }
            })
            .collect();
        if let Some(&(pos, yaw_rad)) = respawns.first() {
            // Eye-height offset — Halo units are world-meters,
            // biped eye height ≈ 1.7m.
            self.camera.position = pos + Vec3::new(0.0, 0.0, 1.7);
            self.camera.rotation = glam::Vec2::new(yaw_rad.to_degrees(), -5.0);
            self.camera.update();
            eprintln!(
                "[scenario]   spawning at respawn[0]: pos=({:.2}, {:.2}, {:.2}) yaw={:.1}° ({} respawn points)",
                self.camera.position.x, self.camera.position.y, self.camera.position.z,
                self.camera.rotation.x, respawns.len(),
            );
        }
    }

    /// Atmosphere WEATHER effects — each `AtmosphereSettings`
    /// entry (scenario.atmospheric → .sky_atm_parameters) can
    /// name a `weather_effect` effe (rain/snow/dust/ash). Engine
    /// `effect_new_weather @0x1802fbac0` spawns it at world
    /// ORIGIN, unattached, with two reserved location vectors:
    ///   `gravity`(string_id 180) = world-down (particles fall
    ///   along it) and `up`(187) = world-up.
    /// First cut: spawn one instance per UNIQUE weather effect
    /// (engine spawns per active atmosphere setting on PVS change
    /// + refreshes location to the active cluster — deferred; for
    /// a static flycam a single origin spawn covers the level).
    fn spawn_weather(&mut self, loaded: &crate::halo::scenario::LoadedScenario) {
        let Some(atm) = loaded.atmosphere.as_ref() else { return };
        if atm.atmosphere_settings.is_empty() {
            return;
        }
        // Cache the active BSP cluster geometry so the per-frame lifecycle
        // (`update_weather`) can resolve which atmosphere settings are active
        // in the camera's cluster sphere (engine `change_pvs` PVS walk).
        let (clusters, portals, planes, palette) = match loaded.active_bsps.first() {
            Some(bsp) => (
                bsp.sbsp.clusters.clone(),
                bsp.sbsp.cluster_portals.clone(),
                bsp.sbsp
                    .collision_bsp
                    .as_ref()
                    .map(|c| c.planes.clone())
                    .unwrap_or_default(),
                bsp.sbsp.atmosphere_palette.clone(),
            ),
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };
        let n = atm.atmosphere_settings.len();
        self.weather = Some(WeatherDriver {
            atmosphere: atm.clone(),
            clusters,
            portals,
            planes,
            palette,
            tags_root: loaded.tags_root.clone(),
            effect_indices: vec![None; n],
        });
        // The first spawn/destroy pass runs on the first `update()`.
    }

    /// Per-frame weather lifecycle (engine `c_atmosphere_fog_interface::change_pvs
    /// @0x1803AF550`): resolve the atmosphere settings active in the camera's
    /// cluster sphere, spawn a camera-relative weather effect for each newly
    /// active setting (engine `effect_new_weather` — unattached, flags=26,
    /// gravity=down/up=up), and destroy (`effect_handle_deleted_atmosphere
    /// @0x1802FDD10` → `effect_destroy`) any whose setting just left the PVS.
    fn update_weather(&mut self) {
        let Some(mut driver) = self.weather.take() else { return };
        let cam = self.camera.position;
        let n = driver.atmosphere.atmosphere_settings.len();

        // 1. Active-setting set = settings referenced by any cluster in the
        //    camera's atmosphere search sphere (PVS-wide activation).
        let mut active = vec![false; n];
        if let Some(start) = crate::halo::render::atmosphere_fog_interface::find_cluster_for_eye(
            &driver.clusters,
            cam,
        ) {
            let search_r = (driver.atmosphere.cluster_search_radius * 3.0).max(20.0);
            let gathered = if driver.clusters.is_empty() {
                vec![start]
            } else {
                crate::halo::structures::clusters_in_sphere::clusters_in_sphere(
                    &driver.clusters,
                    &driver.portals,
                    &driver.planes,
                    start,
                    [cam.x, cam.y, cam.z],
                    search_r,
                    100,
                )
            };
            for c in gathered {
                let atm_idx = driver.clusters.get(c).map(|cl| cl.atmosphere_index).unwrap_or(-1);
                if let Some(si) =
                    crate::halo::render::atmosphere_fog_interface::resolve_atmosphere_setting_index(
                        atm_idx,
                        &driver.palette,
                        n,
                    )
                {
                    if si < n {
                        active[si] = true;
                    }
                }
            }
        }

        // 2. Spawn newly-active settings; destroy newly-inactive ones.
        for si in 0..n {
            let weather_tag = driver.atmosphere.atmosphere_settings[si].weather_effect.clone();
            let want = active[si] && !weather_tag.is_empty();
            let have = driver.effect_indices[si].is_some();
            if want && !have {
                let tags_root = driver.tags_root.clone();
                if let Some(idx) = self.spawn_weather_instance(&weather_tag, &tags_root) {
                    driver.effect_indices[si] = Some(idx);
                }
            } else if !want && have {
                if let Some(idx) = driver.effect_indices[si].take() {
                    self.effects.destroy_instance(idx);
                    eprintln!("[weather] destroyed setting #{si} instance {idx}");
                }
            }
        }

        // 3. Keep every live weather effect anchored at the camera
        //    (engine `effect_refresh_location`).
        for slot in &driver.effect_indices {
            if let Some(idx) = *slot {
                self.effects.set_instance_origin(idx, cam);
            }
        }
        self.weather = Some(driver);
    }

    /// Spawn one camera-relative weather effect (engine `effect_new_weather
    /// @0x1802FBAC0`): unattached, looping (flags=26), reserved location bases
    /// `gravity`(+X=world-down) / `up`(+X=world-up). Returns the instance index.
    fn spawn_weather_instance(&mut self, weather_tag: &str, tags_root: &Path) -> Option<usize> {
        let eidx = self.effects.load_effect(weather_tag, tags_root)?;
        let loc_names = self.effects.effect_location_markers(eidx);
        let location_matrices: Vec<glam::Mat4> = loc_names
            .iter()
            .map(|nm| crate::halo::effects::reserved_location_basis(nm.as_str()))
            .collect();
        let idx = self.effects.spawn_instance(
            eidx,
            u32::MAX,             // no host → identity host @ origin
            String::new(),        // no marker
            self.camera.position, // spawn at the viewer; repositioned per frame
            location_matrices,
            String::new(), // empty primary_scale → always on
            true,          // effect_new_weather: creation flags=26 → looping
        );
        // Cluster-flagged weather → cluster effect_weight LOD path, not the
        // distance band (see EffectInstance::weather).
        self.effects.instances[idx].weather = true;
        eprintln!("[weather] spawned '{weather_tag}' (instance {idx}, locations={loc_names:?})");
        Some(idx)
    }

    /// Effects summary — what the particle subsystem will
    /// simulate/render (Phase 2+). Also builds the per-emitter
    /// render batches and caches the light-volume textures.
    fn log_scenario_summary(
        &mut self,
        renderer: &mut Renderer,
        loaded: &crate::halo::scenario::LoadedScenario,
    ) {
        eprintln!(
            "[effects] total: {} effect tags, {} instances, {} live particle systems",
            self.effects.effects.len(),
            self.effects.instances.len(),
            self.effects.live_system_count(),
        );
        // Build per-emitter render batches (each loads its own
        // base/alpha/palette bitmaps) so the billboard pass
        // shades each particle system faithfully (Phase 4).
        let descriptors = self.effects.batch_descriptors();
        renderer.register_particle_batches(&descriptors, &loaded.tags_root);
        // Track R1: cache the unique light-volume base_map textures
        // so the per-frame strip render can bind them.
        let lv_base_maps: Vec<String> = self
            .effects
            .light_volumes
            .values()
            .map(|lv| lv.render.base_map.clone())
            .filter(|b| !b.is_empty())
            .collect();
        renderer.register_light_volume_textures(&lv_base_maps, &loaded.tags_root);
        for eff in &self.effects.effects {
            let inst = self
                .effects
                .instances
                .iter()
                .filter(|i| self.effects.effects[i.effect_index].path == eff.path)
                .count();
            eprintln!(
                "[effects]   {} — {} systems, {} instances [{}]",
                eff.path,
                eff.systems.len(),
                inst,
                eff.systems
                    .iter()
                    .map(|s| {
                        s.particle_path
                            .rsplit(['\\', '/'])
                            .next()
                            .unwrap_or(&s.particle_path)
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            if std::env::var("PROTOMORPH_DIAG_INST").is_ok() {
                for i in self
                    .effects
                    .instances
                    .iter()
                    .filter(|i| self.effects.effects[i.effect_index].path == eff.path)
                {
                    eprintln!(
                        "[effects]     instance origin=({:.1},{:.1},{:.1})",
                        i.origin.x, i.origin.y, i.origin.z
                    );
                }
            }
        }
    }

    /// Walk a `[ObjectPlacement]` array + palette and create an object
    /// slot per placement. Loads each unique palette tag once (cached
    /// `palette_index → renderer_model_idx`).
    ///
    /// `skip_palette_path` returns `true` for palette entries that
    /// should be ignored (e.g. invisible spawn markers in scenery).
    /// Tag paths are passed lowercased.
    ///
    /// Per `feedback_wgsl_must_mirror_hlsl.md`: silent fallbacks hide
    /// what needs porting. Render-method dispatcher panics on
    /// unsupported shader options propagate up — no `catch_unwind`.
    /// Resolve deferred parent-marker attachments ([`PendingAttachment`])
    /// into datum world transforms. For each parented placement, the child's
    /// world = parent's marker world matrix (engine
    /// `object_attach_to_marker_immediate`): for a sky parent the sky is at
    /// the world origin so it's the sky marker's object-local matrix; for an
    /// object parent it's `parent.world_matrix × parent_marker_local`.
    ///
    /// Resolves in dependency order via a worklist — an attachment whose
    /// parent is itself still pending is deferred to a later pass, so chains
    /// (child → parent → … → sky) settle correctly. Unresolvable links
    /// (missing parent / marker, or a cycle) are logged and dropped.
    fn resolve_pending_attachments(
        &mut self,
        renderer: &Renderer,
        scenario: &blam_tags::scenario::Scenario,
    ) {
        use crate::halo::objects::object_header_data as ohd;
        if self.pending_attachments.is_empty() {
            return;
        }
        // header_index → renderer model_index, for object-parent marker lookup.
        let mut header_to_model: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (_oi, obj) in self.objects.iter() {
            if let (Some(h), Some(m)) = (obj.header_index, obj.model_index) {
                header_to_model.insert(h, m);
            }
        }
        let mut pending = std::mem::take(&mut self.pending_attachments);
        loop {
            // header indices still awaiting attachment — their world isn't final.
            let unresolved: std::collections::HashSet<u32> =
                pending.iter().map(|a| a.child_header_index).collect();
            let mut next: Vec<PendingAttachment> = Vec::new();
            let mut progressed = false;
            for a in pending.drain(..) {
                let is_sky = scenario
                    .skies
                    .iter()
                    .any(|s| s.name_index == a.parent_name_index);
                let world: Option<glam::Mat4> = if is_sky {
                    renderer.sky_marker_transform(&a.parent_marker)
                } else {
                    match ohd::object_index_from_name_index(a.parent_name_index) {
                        // Parent is itself still pending — defer to a later pass.
                        Some(ph) if unresolved.contains(&ph) => {
                            next.push(a);
                            continue;
                        }
                        Some(ph) => match (ohd::world_matrix(ph), header_to_model.get(&ph)) {
                            (Some(pw), Some(&pm)) => renderer
                                .model_marker_transform(pm, &a.parent_marker)
                                .map(|local| pw * local),
                            _ => None,
                        },
                        None => None,
                    }
                };
                match world {
                    Some(w) => {
                        ohd::set_world_transform(a.child_header_index, w);
                        progressed = true;
                    }
                    None => {
                        eprintln!(
                            "[parent] unresolved attachment: child header {} parent name idx {} \
                             marker '{}' — left at authored position",
                            a.child_header_index, a.parent_name_index, a.parent_marker
                        );
                        progressed = true; // drop; don't spin
                    }
                }
            }
            pending = next;
            if pending.is_empty() || !progressed {
                break;
            }
        }
        if !pending.is_empty() {
            eprintln!(
                "[parent] {} attachment(s) unresolved (parent cycle or missing)",
                pending.len()
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn load_visible_placements(
        &mut self,
        renderer: &mut Renderer,
        scenario: &blam_tags::scenario::Scenario,
        object_type: crate::halo::objects::object_type::ObjectType,
        placements: &[blam_tags::scenario::ObjectPlacement],
        palette: &[blam_tags::scenario::TagReferencePalette],
        tags_root: &std::path::Path,
        ext_and_label: &str,
        skip_palette_path: impl Fn(&str) -> bool,
        per_placement_lighting_offsets: Option<&[Option<u32>]>,
    ) {
        let mut palette_to_model: std::collections::HashMap<i16, Option<usize>> =
            std::collections::HashMap::new();
        // Per-palette effect attachments: `(effect_store_index, marker)`
        // resolved once per unique palette tag (alongside the model
        // load) and replayed for every placement of that palette.
        // Per palette: each effect attachment resolves to one or more instance
        // VARIANTS (the inner `Vec<Mat4>` is one variant's per-location markers;
        // a marker name that repeats → multiple variants, one per marker).
        let mut palette_to_effects: std::collections::HashMap<
            i16,
            // (effect_store_index, marker, variants, primary_scale)
            Vec<(usize, String, Vec<Vec<glam::Mat4>>, String)>,
        > = std::collections::HashMap::new();
        let mut effect_instance_count = 0usize;
        let mut loaded_count = 0usize;
        let mut skipped_marker = 0usize;
        let mut skipped_failed = 0usize;
        // Per-path cache of uploaded model-variant child objects (a
        // turret model shared by every host of the same variant — the 3
        // riverworld warthogs reuse one chaingun upload). Keyed by the
        // resolved child object tag path. `None` = load failed (skip).
        let mut attached_child_models: std::collections::HashMap<String, Option<usize>> =
            std::collections::HashMap::new();
        let mut attached_child_count = 0usize;
        for (placement_index, p) in placements.iter().enumerate() {
            let idx = p.palette_index;
            if idx < 0 { continue; }
            let Some(entry) = palette.get(idx as usize) else { continue; };
            let tp_lower = entry.tag_path.to_ascii_lowercase();
            if tp_lower.is_empty() || skip_palette_path(&tp_lower) {
                skipped_marker += 1;
                continue;
            }
            let renderer_model_idx = match palette_to_model.get(&idx).copied() {
                Some(v) => v,
                None => {
                    let path = blam_tags::paths::resolve_tag_path(
                        tags_root,
                        &entry.tag_path,
                        ext_and_label,
                    );
                    let result = if !path.exists() {
                        eprintln!(
                            "[scenario]   {ext_and_label} tag missing: {}",
                            path.display(),
                        );
                        None
                    } else {
                        // Load the model first — its markers resolve the
                        // effect LOCATION matrices (the marker rotation
                        // that orients emission, e.g. null_up "marker").
                        // No catch_unwind — unsupported shader options
                        // panic the app per the engine-faithful policy.
                        let load = crate::halo::loader::load_object(&path);
                        // Resolve any effe attachments + their per-location
                        // marker matrices (engine `attachments_new` →
                        // `effect_new_from_object`; the location marker is
                        // resolved against the host model).
                        let attachments =
                            crate::halo::loader::read_object_effect_attachments(&path);
                        if !attachments.is_empty() {
                            let model_ref = load.as_ref().ok();
                            let mut resolved = Vec::new();
                            for (effect_rel, marker, primary_scale) in attachments {
                                if let Some(eidx) =
                                    self.effects.load_effect(&effect_rel, tags_root)
                                {
                                    let loc_names =
                                        self.effects.effect_location_markers(eidx);
                                    if std::env::var("PROTOMORPH_DIAG_MARKERS").is_ok() {
                                        let avail: Vec<String> = model_ref
                                            .map(|m| {
                                                m.marker_transforms
                                                    .iter()
                                                    .map(|(n, mtx)| {
                                                        let f = mtx.transform_vector3(glam::Vec3::X);
                                                        let u = mtx.transform_vector3(glam::Vec3::Z);
                                                        format!(
                                                            "{n}[fwd=({:.2},{:.2},{:.2}) up=({:.2},{:.2},{:.2})]",
                                                            f.x, f.y, f.z, u.x, u.y, u.z,
                                                        )
                                                    })
                                                    .collect()
                                            })
                                            .unwrap_or_default();
                                        eprintln!(
                                            "[markers] host={} effect={effect_rel} loc_names={loc_names:?} avail={avail:?}",
                                            entry.tag_path,
                                        );
                                    }
                                    // Resolve each location to ALL its markers
                                    // — a marker name can repeat (the guardian
                                    // minilift has 3 `fx_minilift_pad` markers,
                                    // one per pad), and the engine emits the
                                    // effect at every match. One instance/marker.
                                    //
                                    // Reserved direction names like "up" aren't
                                    // authored marker groups, so they don't match
                                    // by name. The engine (effect_build_locations
                                    // @0x1803005f0 → string_id "up"==489) binds the
                                    // reserved direction to the host object's
                                    // primary marker. These fx hosts are the
                                    // standalone `*.effect_scenery` placements whose
                                    // model is the invisible `null_up` — a single
                                    // marker named 'marker' oriented +X→+Z (straight
                                    // up). Emission therefore rises along the host's
                                    // up axis, and the host's PLACEMENT yaw aims it
                                    // (man cannon −45°/135°, holy_light 0° → straight
                                    // up). The old `Mat4::from_rotation_z(90°)` hack
                                    // rotated +X→+Y (horizontal) instead, which fired
                                    // holy_light on its side and the man cannon left.
                                    let per_loc: Vec<Vec<glam::Mat4>> = loc_names
                                        .iter()
                                        .map(|n| {
                                            let all = model_ref
                                                .map(|m| m.marker_transforms_all(n))
                                                .unwrap_or_default();
                                            if !all.is_empty() {
                                                return all;
                                            }
                                            // Unresolved reserved direction → host's
                                            // single marker (object-local, like the
                                            // resolved-marker path above). Only when
                                            // the host has exactly one marker (the
                                            // null-object convention); else identity.
                                            let markers = model_ref
                                                .map(|m| {
                                                    m.marker_transforms
                                                        .iter()
                                                        .map(|(_, mtx)| *mtx)
                                                        .collect::<Vec<_>>()
                                                })
                                                .unwrap_or_default();
                                            let base = if markers.len() == 1 {
                                                markers
                                            } else {
                                                vec![glam::Mat4::IDENTITY]
                                            };
                                            // Spin the lean azimuth about the host's
                                            // LOCAL up axis (see
                                            // `effects::up_yaw_spin` for the rationale).
                                            crate::halo::effects::up_yaw_spin(base)
                                        })
                                        .collect();
                                    let variant_count = per_loc
                                        .iter()
                                        .map(|v| v.len())
                                        .max()
                                        .unwrap_or(1)
                                        .max(1);
                                    let variants: Vec<Vec<glam::Mat4>> = (0..variant_count)
                                        .map(|k| {
                                            per_loc
                                                .iter()
                                                .map(|markers| {
                                                    // k-th marker for this location;
                                                    // single-marker locations are shared
                                                    // across all variants (reuse last).
                                                    *markers
                                                        .get(k)
                                                        .unwrap_or_else(|| markers.last().unwrap())
                                                })
                                                .collect()
                                        })
                                        .collect();
                                    resolved.push((eidx, marker, variants, primary_scale));
                                }
                            }
                            if !resolved.is_empty() {
                                palette_to_effects.insert(idx, resolved);
                            }
                        }
                        match load {
                            Ok(model) => {
                                let r_idx = renderer.upload_model_data(&model);
                                self.model_data.push(model);
                                Some(r_idx)
                            }
                            Err(e) => {
                                eprintln!(
                                    "[scenario]   {ext_and_label} load failed {}: {}",
                                    entry.tag_path, e,
                                );
                                None
                            }
                        }
                    };
                    palette_to_model.insert(idx, result);
                    result
                }
            };
            let Some(model_idx) = renderer_model_idx else {
                skipped_failed += 1;
                continue;
            };
            let obj = self.objects.new_object();
            let pos = p.object_data.position;
            let rot = p.object_data.rotation;
            let scale = p.object_data.scale;
            // Path B: assign this placement's slot index within its
            // model's animated-material cbuffer pool. First placement
            // of a model gets slot 0; each subsequent gets +1.
            let instance_within_model = {
                let count = self.placements_per_model.entry(model_idx).or_insert(0);
                let assigned = *count;
                *count += 1;
                assigned
            };
            // Engine `object_index` for this placement — the slot in
            // `object_header_data` that `*_compute_function_value`
            // callers receive. Computed from `(object_type,
            // placement_index)` to match `populate_from_scenario`'s
            // walk order so the table entry's `object_type` matches.
            let header_index = crate::halo::objects::object_header_data
                ::header_index_for_placement(
                    scenario,
                    object_type,
                    placement_index as u32,
                );
            // Engine `c_dynamic_cubemap_sample` — nearest scenario cubemap
            // probe to this placement, for dynamic-env reflections.
            let cubemap_probe =
                renderer.nearest_object_cubemap_probe(Vec3::new(pos.x, pos.y, pos.z));
            {
                let slot = self.objects.get_mut(obj);
                slot.model_index = Some(model_idx);
                slot.instance_within_model = instance_within_model;
                slot.header_index = Some(header_index);
                slot.cubemap_probe_index = cubemap_probe;
                slot.position = Vec3::new(pos.x, pos.y, pos.z);
                slot.rotation = Vec3::new(
                    rot.yaw.to_degrees(),
                    rot.pitch.to_degrees(),
                    rot.roll.to_degrees(),
                );
                // `scale = 0.0` means "use object's authored default
                // scale" per Halo runtime convention.
                if scale > 0.0 {
                    slot.scale = Vec3::splat(scale);
                }
                // L4 — pin the per-placement engine_lighting offset so
                // per-draw bind-group rebinding picks up this object's
                // baked SH probe (instead of the frame-default sky probe).
                slot.engine_lighting_offset = per_placement_lighting_offsets
                    .and_then(|offsets| offsets.get(placement_index).copied().flatten());
                // Diag: match PROTOMORPH_DIAG_LIGHTING filter and log
                // the obj_idx ↔ lighting_offset binding so draw-time
                // logs can correlate against bake-time output.
                if let Ok(filt) = std::env::var("PROTOMORPH_DIAG_LIGHTING") {
                    let matches = filt == "all" || filt == ext_and_label || {
                        filt.strip_prefix(ext_and_label)
                            .and_then(|s| s.strip_prefix(':'))
                            .map(|rest| rest.split(',').any(|tok| tok.trim().parse::<usize>() == Ok(placement_index)))
                            .unwrap_or(false)
                    };
                    if matches {
                        eprintln!(
                            "[diag-lighting] {ext_and_label}[{placement_index}] load-time bind: obj_idx={:?} model_idx={model_idx} engine_lighting_offset={:?} header_index={header_index}",
                            obj, slot.engine_lighting_offset,
                        );
                    }
                }
            }

            // Model-variant child objects (turrets, etc.) — engine
            // `model_variant_object_block`. The warthog's `default`
            // variant attaches `chaingun.vehicle` at marker `turret`;
            // the wraith attaches `mortar` + `anti_infantry`. Spawn each
            // child snapped to its parent marker, recursing through the
            // child's own variant. Children inherit the host's baked SH
            // probe (co-located) and carry a `model_matrix_override`
            // world (no scenario placement / header of their own).
            {
                let parent_world = self.objects.get(obj).model_matrix();
                let parent_lighting = self.objects.get(obj).engine_lighting_offset;
                let parent_path =
                    blam_tags::paths::resolve_tag_path(tags_root, &entry.tag_path, ext_and_label);
                let placement_variant = p.permutation_data.variant_name.clone();
                attached_child_count += self.spawn_attached_children(
                    renderer,
                    tags_root,
                    &parent_path,
                    model_idx,
                    parent_world,
                    &placement_variant,
                    parent_lighting,
                    &mut attached_child_models,
                    0,
                );
            }

            // Instance any effe attachments this palette carries onto
            // the placed object. The host's authoritative world matrix
            // is resolved at render time from `header_index`; `pos` is
            // the standalone fallback origin (Phase 1).
            if let Some(attachments) = palette_to_effects.get(&idx) {
                for (effect_index, marker, variants, primary_scale) in attachments {
                    // One instance per marker variant (e.g. 3 minilift pads).
                    for loc_matrices in variants {
                        self.effects.spawn_instance(
                            *effect_index,
                            header_index,
                            marker.clone(),
                            Vec3::new(pos.x, pos.y, pos.z),
                            loc_matrices.clone(),
                            primary_scale.clone(),
                            // effect_new_looping (attachments_new dispatches
                            // effe attachments): creation flags |= 7 →
                            // looping instance, event durations never apply.
                            true,
                        );
                        effect_instance_count += 1;
                    }
                }
            }

            // Parent-marker attachment: defer to a post-load pass so the
            // parent object (which may load in a later placement block, or be
            // the sky) is available + already attached first. Engine path:
            // object_new → object_attach_to_marker → ...marker_immediate snaps
            // the child's frame to the parent marker's world matrix.
            if let Some(pid) = p.object_data.parent_id.as_ref() {
                if pid.parent_object_name_index >= 0 {
                    self.pending_attachments.push(PendingAttachment {
                        child_header_index: header_index,
                        parent_name_index: pid.parent_object_name_index,
                        parent_marker: pid.parent_marker.clone(),
                    });
                }
            }
            loaded_count += 1;
        }
        let unique_loaded =
            palette_to_model.values().filter(|v| v.is_some()).count();
        eprintln!(
            "[scenario]   {ext_and_label}: {} placements ({} unique tags), {} skipped, {} failed",
            loaded_count, unique_loaded, skipped_marker, skipped_failed,
        );
        if effect_instance_count > 0 {
            eprintln!(
                "[effects]    {ext_and_label}: {} effect instances spawned",
                effect_instance_count,
            );
        }
        if attached_child_count > 0 {
            eprintln!(
                "[attach]     {ext_and_label}: {} model-variant child object(s) spawned (turrets etc.)",
                attached_child_count,
            );
        }
    }

    /// Spawn the model-variant child objects ([`blam_tags::ModelVariantObject`])
    /// a host object's active variant attaches, snapping each to its parent
    /// marker and recursing through the child's own variant. Returns the
    /// number of child objects spawned (including descendants).
    ///
    /// Engine path: `object_new` resolves the model variant and, for each
    /// `model_variant_object_block` entry, spawns the child object and binds
    /// it with `object_attach_to_marker` — the child's `child marker` is made
    /// to coincide with the host's `parent marker`. With an empty `child
    /// marker` (the common case: warthog/wraith default turrets) the child
    /// origin sits at the parent marker, so `child_world = parent_world ×
    /// parent_marker_local`. A non-empty child marker (wraith `anti_air`
    /// `attach`) additionally pre-multiplies the child marker's inverse.
    ///
    /// Children are static in protomorph (no physics/turret aim), so the
    /// resolved world is baked into a `model_matrix_override`; they have no
    /// scenario placement, so no `header_index` and they inherit the host's
    /// baked SH probe.
    #[allow(clippy::too_many_arguments)]
    fn spawn_attached_children(
        &mut self,
        renderer: &mut Renderer,
        tags_root: &std::path::Path,
        parent_object_path: &std::path::Path,
        parent_model_idx: usize,
        parent_world: Mat4,
        variant_name: &str,
        lighting_offset: Option<u32>,
        child_model_cache: &mut std::collections::HashMap<String, Option<usize>>,
        depth: u32,
    ) -> usize {
        // Cycle / runaway guard (a child referencing an ancestor variant).
        if depth >= 8 {
            eprintln!(
                "[attach] child-object recursion depth limit hit at {} — stopping",
                parent_object_path.display(),
            );
            return 0;
        }
        let children =
            crate::halo::loader::read_model_variant_children(parent_object_path, variant_name);
        let mut spawned = 0usize;
        for child in children {
            if child.child_object.is_empty() {
                continue;
            }
            // Resolve the child object tag path from its group fourcc
            // (vehi → .vehicle, weap → .weapon, …).
            let ext = blam_tags::paths::group_tag_to_extension(u32::from_be_bytes(
                child.child_object_group,
            ))
            .unwrap_or("vehicle");
            let child_path =
                blam_tags::paths::resolve_tag_path(tags_root, &child.child_object, ext);
            let cache_key = child_path.to_string_lossy().to_string();

            // Load + upload the child model once; reuse for repeat hosts.
            let child_model_idx = match child_model_cache.get(&cache_key).copied() {
                Some(v) => v,
                None => {
                    let result = if !child_path.exists() {
                        eprintln!(
                            "[attach] child object tag missing: {}",
                            child_path.display()
                        );
                        None
                    } else {
                        match crate::halo::loader::load_object(&child_path) {
                            Ok(model) => {
                                let r_idx = renderer.upload_model_data(&model);
                                self.model_data.push(model);
                                Some(r_idx)
                            }
                            Err(e) => {
                                eprintln!(
                                    "[attach] child object load failed {}: {e}",
                                    child.child_object,
                                );
                                None
                            }
                        }
                    };
                    child_model_cache.insert(cache_key.clone(), result);
                    result
                }
            };
            let Some(child_model_idx) = child_model_idx else {
                continue;
            };

            // parent_marker (on the host) → its model-local matrix. Empty
            // marker = host origin. A named-but-missing marker is a real
            // mismatch worth logging; fall back to host origin so the
            // child is at least visible.
            let parent_marker_local = if child.parent_marker.is_empty() {
                Mat4::IDENTITY
            } else {
                match renderer.model_marker_transform(parent_model_idx, &child.parent_marker) {
                    Some(m) => m,
                    None => {
                        eprintln!(
                            "[attach] host {} has no marker '{}' for child {} — attaching at origin",
                            parent_object_path.display(),
                            child.parent_marker,
                            child.child_object,
                        );
                        Mat4::IDENTITY
                    }
                }
            };
            // child_marker (on the child) — make it coincide with the
            // parent marker. Empty = child origin (identity).
            let child_marker_local = if child.child_marker.is_empty() {
                Mat4::IDENTITY
            } else {
                renderer
                    .model_marker_transform(child_model_idx, &child.child_marker)
                    .unwrap_or(Mat4::IDENTITY)
            };
            let child_world =
                parent_world * parent_marker_local * child_marker_local.inverse();

            // Give the child its own object-header entry (engine: the
            // turret is a first-class `object_new`, not just geometry).
            // This is what lets its render-method function lookups
            // (`object_get_function_value`) resolve against ITS
            // `functions[]` — without a header/datum the child has
            // object_index 0xFFFFFFFF and authored functions like the
            // mortar's `mortar_illum` self-illum return 0.
            use crate::halo::objects::object_type::ObjectType;
            let child_type = ObjectType::from_group_fourcc(child.child_object_group)
                .unwrap_or(ObjectType::Vehicle);
            let child_obj_def = crate::halo::tags::tag_get(
                child_type.tag_group_fourcc(),
                &child.child_object,
            )
            .and_then(|t| t.as_object_definition());
            // Resolve the child's own variant_index (child variant name →
            // its model's variants[]) for the `variant` compute case.
            let child_variant_index = child_obj_def
                .as_ref()
                .filter(|d| !d.model.is_empty() && !child.child_variant_name.is_empty())
                .and_then(|d| crate::halo::tags::tag_get(*b"hlmt", &d.model))
                .and_then(|t| t.as_model())
                .map(|m| {
                    m.variant_names
                        .iter()
                        .position(|n| n == &child.child_variant_name)
                        .map(|i| i as i8)
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            let child_header = child_obj_def.as_ref().map(|def| {
                crate::halo::objects::object_header_data::push_object(
                    child_type,
                    child.child_object.clone(),
                    def.clone(),
                    child_variant_index,
                    child_world,
                )
            });

            // Spawn the child object slot.
            let inst = {
                let count = self.placements_per_model.entry(child_model_idx).or_insert(0);
                let assigned = *count;
                *count += 1;
                assigned
            };
            let child_cubemap_probe =
                renderer.nearest_object_cubemap_probe(child_world.w_axis.truncate());
            let obj = self.objects.new_object();
            {
                let slot = self.objects.get_mut(obj);
                slot.model_index = Some(child_model_idx);
                slot.instance_within_model = inst;
                slot.header_index = child_header;
                slot.cubemap_probe_index = child_cubemap_probe;
                // World comes from the header datum when present (render reads
                // `header_index → world_matrix`); keep the override as the
                // fallback for the no-header path.
                slot.model_matrix_override = Some(child_world);
                slot.engine_lighting_offset = lighting_offset;
                // Derive a readable position for any pos-based bookkeeping.
                slot.position = child_world.w_axis.truncate();
            }
            spawned += 1;

            // Recurse: the child's own variant may attach further objects
            // (e.g. a turret with a sub-mount). `child variant name` ("" →
            // the child object's default variant).
            spawned += self.spawn_attached_children(
                renderer,
                tags_root,
                &child_path,
                child_model_idx,
                child_world,
                &child.child_variant_name,
                lighting_offset,
                child_model_cache,
                depth + 1,
            );
        }
        spawned
    }

    pub fn update(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        self.delta_time = dt;
        self.update_movement(keys, dt);
        // Mirrors Ares' `render_sky_modify_node_matrices(offset=camera)`:
        // translating the sky model to the camera keeps its dome
        // centered on the viewer so it appears infinitely far.
        if let Some(sky) = self.sky_object {
            self.objects.get_mut(sky).position = self.camera.position;
        }
        // Particle CPU pulse — advance every effect instance's emitters,
        // refilling `effects.pending_spawns` for the renderer's GPU
        // scatter. Engine: the CPU half of `frame_advance_all_gpu`.
        // Weather effects are camera-relative — anchor their emission to the
        // viewer each frame (engine `effect_refresh_location`) so falling
        // ash/rain/dust surrounds the player instead of sitting at the fixed
        // world origin it spawned at.
        // Weather lifecycle (engine change_pvs): spawn/destroy per-setting
        // weather effects for the camera's active atmosphere settings and
        // re-anchor them at the viewer.
        self.update_weather();
        let cam_pos = self.camera.position;
        // Per-location particle LOD needs the camera position; fov_scale 1.0
        // until the real camera field_of_view_scale is threaded (engine
        // scales distance by it — wider FOV reads as farther).
        self.effects.frame_advance(dt, cam_pos, 1.0);
        self.fps_counter.update(dt);
        self.total_time += dt;
    }

    pub fn toggle_specular_occlusion(&mut self) {
        self.enable_specular_occlusion = !self.enable_specular_occlusion;
    }

    pub fn toggle_vignette(&mut self) {
        self.enable_vignette = !self.enable_vignette;
    }

    fn update_movement(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        let mut move_dir = Vec3::ZERO;
        if keys.contains(&KeyCode::KeyW) { move_dir += self.camera.forward; }
        if keys.contains(&KeyCode::KeyS) { move_dir -= self.camera.forward; }
        if keys.contains(&KeyCode::KeyA) { move_dir += self.camera.right; }
        if keys.contains(&KeyCode::KeyD) { move_dir -= self.camera.right; }
        if keys.contains(&KeyCode::KeyR) { move_dir += Vec3::Z; }
        if keys.contains(&KeyCode::KeyF) { move_dir -= Vec3::Z; }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        let mut speed = MOVEMENT_SPEED;
        if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            speed *= 2.0;
        }

        self.camera.velocity = move_dir * speed * dt;
        self.camera.update();
    }
}
