use crate::halo::animations::AnimationManager;
use crate::halo::camera::FlyingCamera;
use crate::halo::geometry::ModelData;
use crate::halo::objects::{ObjectIndex, ObjectStore};
use crate::halo::render::Renderer;
use glam::Vec3;
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

pub struct GameState {
    pub objects: ObjectStore,
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
                                        let v: u32 = m.meshes.iter().map(|x| x.vertices.len() as u32).sum();
                                        let t: u32 = m
                                            .meshes
                                            .iter()
                                            .map(|x| (x.indices.len() / 3) as u32)
                                            .sum();
                                        (v, t)
                                    })
                                    .unwrap_or((0, 0));
                                let meshes = rm.map(|m| m.meshes.len()).unwrap_or(0);
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
                                self.init_animations(obj, model);
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

                    // Pick a spawn vantage from the scenery placements that
                    // reference an MP respawn-point/zone palette entry. MP
                    // scenarios author these as `objects/multi/spawning/
                    // respawn_point.scenery` (and game-mode-specific zone
                    // variants like `slayer_respawn_zone.scenery`). Halo's
                    // runtime spawn picker is more involved (game-mode
                    // gating + respawn timer + visibility check), but for
                    // a flycam smoke test the first respawn point is fine.
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
                    let weapon_offsets = renderer.weapon_lighting_offsets.clone();
                    let equipment_offsets = renderer.equipment_lighting_offsets.clone();
                    let machine_offsets = renderer.machine_lighting_offsets.clone();
                    let control_offsets = renderer.control_lighting_offsets.clone();
                    use crate::halo::objects::object_type::ObjectType;
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
                }
                Err(e) => eprintln!("[scenario] load failed: {e}"),
            }
        }

        // Test models (grunt biped + wraith vehicle) removed — focus
        // is riverworld scenario geometry. Wraith uses
        // `environment_map / dynamic` (option 2) which we don't yet
        // port; loading it triggers the dispatcher panic. Grunt
        // animations + biped placement come back when Phase E lands
        // (object_placement_data lifecycle from scenario.bipeds).
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
        let mut loaded_count = 0usize;
        let mut skipped_marker = 0usize;
        let mut skipped_failed = 0usize;
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
                        // No catch_unwind — unsupported shader options
                        // panic the app per the engine-faithful policy.
                        match crate::halo::loader::load_object(&path) {
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
            {
                let slot = self.objects.get_mut(obj);
                slot.model_index = Some(model_idx);
                slot.instance_within_model = instance_within_model;
                slot.header_index = Some(header_index);
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
            self.init_animations(obj, model_idx);
            loaded_count += 1;
        }
        let unique_loaded =
            palette_to_model.values().filter(|v| v.is_some()).count();
        eprintln!(
            "[scenario]   {ext_and_label}: {} placements ({} unique tags), {} skipped, {} failed",
            loaded_count, unique_loaded, skipped_marker, skipped_failed,
        );
    }

    fn init_animations(&mut self, obj_index: ObjectIndex, model_index: usize) {
        let model = &self.model_data[model_index];
        // Any model with nodes needs an AnimationManager so its
        // bind-pose node matrices reach the GPU — even with zero
        // animations. Skinned vertices sample `node_matrices` and
        // would otherwise see zeros.
        if model.nodes.is_empty() {
            return;
        }

        let anim_mgr = AnimationManager::new(model);
        self.objects.get_mut(obj_index).animations = Some(anim_mgr);
    }

    pub fn update(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        self.delta_time = dt;
        self.update_movement(keys, dt);
        self.objects.update(&self.model_data, dt);
        // Mirrors Ares' `render_sky_modify_node_matrices(offset=camera)`:
        // translating the sky model to the camera keeps its dome
        // centered on the viewer so it appears infinitely far.
        if let Some(sky) = self.sky_object {
            self.objects.get_mut(sky).position = self.camera.position;
        }
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
