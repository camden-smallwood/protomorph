//! Scenario-lighting bake suite (cluster simple lights, scenery/object lighting offsets).
//!
//! Split out of `render/mod.rs` as a behavior-preserving module carve-out.
//! These are additional inherent `impl Renderer` methods; child modules of
//! `render` can see `Renderer`'s private fields, so no visibility changes
//! are required.

use super::{nearest_by, ObjectLightingCacheEntry, Renderer};

impl Renderer {
    /// L2 per-cluster simple_lights bake. For each BSP cluster, finds
    /// the up-to-8 nearest scenario lights and packs them into the
    /// next available slot in `shared.simple_lights_buffer`. Stores
    /// the slot's dynamic offset on the BSP runtime
    /// (`cluster_simple_lights_offsets`).
    ///
    /// Engine equivalent: `c_lights_view::add_simple_light_to_draw_list`
    /// called per cluster part inside `c_structure_renderer::render_cluster_mesh_part`.
    /// Our offline bake gives the same result as long as scenario_lights
    /// are static (which they are post-load — game-time effect lights
    /// would still need a runtime path).
    pub(crate) fn bake_cluster_simple_lights(
        &mut self,
        loaded: &crate::halo::scenario::LoadedScenario,
    ) {
        if self.scenario_lights.is_empty() {
            return;
        }
        let stride = crate::halo::render::shared::SIMPLE_LIGHTS_STRIDE as u64;
        let max_slots = crate::halo::render::shared::SIMPLE_LIGHTS_ENTRIES as usize;

        // Phase 1: gather per-cluster (bsp_idx, cluster_idx, payload)
        // using ONLY immutable borrows of `self.scenario_lights`.
        struct Pack {
            bsp_idx: usize,
            cluster_idx: usize,
            payload: crate::halo::lighting::GpuSimpleLights,
        }
        let mut packs: Vec<Pack> = Vec::new();
        for (bsp_idx, bsp_loaded) in loaded.active_bsps.iter().enumerate() {
            for (cluster_idx, cluster) in bsp_loaded.sbsp.clusters.iter().enumerate() {
                let center = glam::Vec3::new(
                    0.5 * (cluster.bounds_x.lower + cluster.bounds_x.upper),
                    0.5 * (cluster.bounds_y.lower + cluster.bounds_y.upper),
                    0.5 * (cluster.bounds_z.lower + cluster.bounds_z.upper),
                );
                let half = glam::Vec3::new(
                    0.5 * (cluster.bounds_x.upper - cluster.bounds_x.lower),
                    0.5 * (cluster.bounds_y.upper - cluster.bounds_y.lower),
                    0.5 * (cluster.bounds_z.upper - cluster.bounds_z.lower),
                );
                let cluster_radius = half.length();

                let mut candidates: Vec<(f32, usize)> = self
                    .scenario_lights
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| {
                        let d = (l.position - center).length();
                        if d <= l.max_dist + cluster_radius {
                            Some((d, i))
                        } else {
                            None
                        }
                    })
                    .collect();
                candidates.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                if candidates.is_empty() {
                    continue;
                }

                let mut payload = crate::halo::lighting::GpuSimpleLights::default();
                let take = candidates
                    .len()
                    .min(crate::halo::lighting::SIMPLE_LIGHTS_MAX);
                payload.count[0] = take as f32;
                for (slot_idx, (_dist, scenario_idx)) in
                    candidates.iter().take(take).enumerate()
                {
                    let l = &self.scenario_lights[*scenario_idx];
                    let mut tmp =
                        crate::halo::render::views::lights_view::SimpleLight::default();
                    crate::halo::render::views::lights_view::LightsView::initialize_simple_light(
                        &mut tmp,
                        l.position,
                        l.color,
                        l.size,
                        l.max_dist,
                        l.direction,
                        l.cone_angle,
                        l.cone_smoothness,
                        l.sphere_percentage,
                    );
                    crate::halo::lighting::pack_simple_light_at(&mut payload, slot_idx, &tmp);
                }
                packs.push(Pack { bsp_idx, cluster_idx, payload });
            }
        }

        // Phase 2: assign slot offsets and write to GPU + per-BSP runtime.
        let mut next_slot: usize = 1; // slot 0 reserved as empty
        let mut baked: usize = 0;
        for pack in &packs {
            if next_slot >= max_slots {
                eprintln!(
                    "[scenario]   simple_lights slot pool exhausted at {} slots",
                    next_slot,
                );
                break;
            }
            let offset = (next_slot as u64) * stride;
            self.shared.queue.write_buffer(
                &self.shared.simple_lights_buffer,
                offset,
                bytemuck::bytes_of(&pack.payload),
            );
            if let Some(bsp_gpu) = self.structure_renderer.bsps.get_mut(pack.bsp_idx) {
                if let Some(slot_offset) =
                    bsp_gpu.cluster_simple_lights_offsets.get_mut(pack.cluster_idx)
                {
                    *slot_offset = offset as u32;
                }
            }
            next_slot += 1;
            baked += 1;
        }
        if baked > 0 {
            eprintln!(
                "[scenario]   baked simple_lights for {} clusters across {} BSPs",
                baked,
                loaded.active_bsps.len(),
            );
        }
    }

    /// L4: pre-bake one engine_lighting slot per scenery placement that
    /// has a matching baked probe in any active BSP's
    /// `lightmap_bsp_data.scenery_probes`. Stores slot offsets in
    /// `scenery_lighting_offsets` for the per-draw bind path (L4.3).
    ///
    /// Engine equivalent: `lights_prepare_for_object_static_new @
    /// 0x1808A2930` — routes per object type:
    ///   - type 6 (scenery)        → `scenery_probes[obj.scenery_probe_index]`
    ///   - type 7 (device_machine) → `device_machine_probes[idx]`
    ///   - everything else          → `c_geometry_sampler::sample_one_air_probe`
    ///                               (nearest airprobe by world-space distance)
    ///
    /// Each branch then calls `convert_probe_to_lighting_sample` to fill
    /// the cached `s_geometry_sample`, then `c_chocalate_moutain::apply_*`
    /// for the per-type minimum-luminance floor. We bake all of this at
    /// scenario load since both placements and probes are static.
    ///
    /// Probe-to-placement correspondence:
    ///   - SCENERY: positional (`scenery_probes[i] → scenery[i]`); engine
    ///     does `c_object_identifier::is_equal` as a defensive sanity check.
    ///   - NEAREST AIRPROBE: linear scan over `airprobes[]` selecting the
    ///     entry whose `position` minimizes Euclidean distance to the
    ///     placement's `object_data.position`. Engine uses a 3-airprobe
    ///     trilinear interpolation; v1 takes the single nearest probe to
    ///     keep the bake simple. Visual delta is small for sparse
    ///     airprobe layouts (most MP maps).
    ///
    /// For multi-BSP scenarios we'd need per-placement BSP resolution;
    /// v1 matches the engine when one BSP holds all placements.
    pub(crate) fn bake_scenery_lighting_offsets(
        &mut self,
        loaded: &crate::halo::scenario::LoadedScenario,
    ) {
        use crate::halo::objects::object_type::ObjectType;
        use blam_tags::scenario_lightmap::LightmapProbe;

        // Reset the per-placement cache on every scenario load. Each
        // entry holds a slot_offset into `engine_lighting_buffer`; these
        // offsets can't be reused across scenarios because the slot
        // pool's `next_probe_slot` is reset to 1 at the top of
        // `load_scenario` (via `Renderer::reset_scene` →
        // `StructureRenderer::reset`), so a fresh bake re-allocates the
        // slots from scratch.
        self.object_lighting_cache.clear();

        // Engine reads per-bsp probes — for v1 we use BSP 0 (single-BSP
        // test scenarios). Multi-BSP support: walk `loaded.active_bsps`
        // and resolve per-placement bsp.
        let Some(bsp) = loaded.active_bsps.first() else {
            self.scenery_lighting_offsets = Vec::new();
            self.crate_lighting_offsets = Vec::new();
            self.vehicle_lighting_offsets = Vec::new();
            self.weapon_lighting_offsets = Vec::new();
            self.equipment_lighting_offsets = Vec::new();
            self.machine_lighting_offsets = Vec::new();
            self.control_lighting_offsets = Vec::new();
            return;
        };
        let Some(lbsp) = bsp.lightmap.as_ref() else {
            self.scenery_lighting_offsets = Vec::new();
            self.crate_lighting_offsets = Vec::new();
            self.vehicle_lighting_offsets = Vec::new();
            self.weapon_lighting_offsets = Vec::new();
            self.equipment_lighting_offsets = Vec::new();
            self.machine_lighting_offsets = Vec::new();
            self.control_lighting_offsets = Vec::new();
            return;
        };

        let scenery = &loaded.scenario.scenery;
        let crates = &loaded.scenario.crates;
        let vehicles = &loaded.scenario.vehicles;
        let weapons = &loaded.scenario.weapons;
        let equipment = &loaded.scenario.equipment;
        let machines = &loaded.scenario.machines;
        let controls = &loaded.scenario.controls;

        // Sky-probe fallback for non-scenery placements. Engine equivalent:
        // `lights_sky_lighing_at_point_new` → cluster's visible sky →
        // render_model.default_lightprobe → SH3. On single-sky maps
        // collapses to `skies[0]`'s probe (= `loaded.sky_lighting`). We
        // pre-compute it as a `DequantizedLightmapProbe` so the bake's
        // chmt + ravi-pack path treats it identically to an airprobe hit.
        // Without this, placements with no airprobe inherit the frame
        // default — bypassing per-type chmt boost.
        let sky_fallback: Option<blam_tags::scenario_lightmap::DequantizedLightmapProbe> =
            loaded.sky_lighting.as_ref().map(sky_probe_as_dequantized);

        // Scenery: per-placement probe at `scenery_probes[i]` (positional;
        // engine does identifier match as a defensive sanity check —
        // matched-by-index is the common case in MCC tags). NO sky
        // fallback — placements without a probe match engine-default
        // behavior (offset = None inherits the frame default).
        let scenery_offsets = self.bake_object_lighting_offsets(
            scenery.len(),
            ObjectType::Scenery,
            "scenery",
            |i, _pos| lbsp.scenery_probes.get(i).map(|p| p.probe.dequantize()),
        );

        // Non-scenery dynamic placements: engine fallback chain from
        // `lights_prepare_for_object_static_new @ 0x1808A2930`'s
        // LABEL_114 path (the "no per-object airprobe_index, no
        // per-object scenery/device probe_index" final fallback). Engine
        // order:
        //   1. `lights_distant_lighting_at_point_new` — raycast BSP +
        //      sample cluster lightprobe atlas at hit (Phase A here).
        //   2. `lights_airprobe_lighting_at_point_new` — nearest-airprobe.
        //   3. `lights_sky_lighing_at_point_new` — sky default lightprobe.
        // Order matters: on maps where airprobes are stripped (e.g.
        // deadlock has 0 airprobes), step 1 supplies dim cluster
        // lighting; without it crates inherit bright sky DC and blow
        // out. See [[project_object_lighting_fallback_2026_05_14]].
        let airprobes: &[blam_tags::scenario_lightmap::LightmapAirprobe] = loaded
            .scenario_lightmap
            .as_ref()
            .filter(|s| !s.airprobes.is_empty())
            .map(|s| s.airprobes.as_slice())
            .unwrap_or(lbsp.airprobes.as_slice());
        eprintln!(
            "[scenario]   airprobes available: {} (sky-probe fallback when 0)",
            airprobes.len(),
        );

        // Phase 9.1 cutover: dropped the invented Raycaster construction.
        // The new chain (`c_geometry_sampler::sample` via the
        // `render::geometry_sampler` shim) resolves BSP geometry via the
        // engine-faithful `collision_test_vector` + `geometry_test_collision_result`
        // chain. Per-BSP atlas decode is still needed (same `decode_bake_resources`
        // helper used by the decorator bake).
        let bake_resources = crate::halo::render::render_decorators::decode_bake_resources(loaded);
        let raycast_ready = !bake_resources.is_empty();
        if raycast_ready {
            eprintln!(
                "[scenario]   geometry_sampler ready: {} BSP atlases (new c_geometry_sampler::sample chain)",
                bake_resources.len(),
            );
        } else {
            eprintln!(
                "[scenario]   geometry_sampler unavailable (atlases={}) — falling through to airprobe/sky",
                bake_resources.len(),
            );
        }

        // PROTOMORPH_DIAG_OL4_PATH=<type>:<i> or "all" — prints the
        // resolution chain (radius/flags/dispatch/outcome/airprobe) for
        // matched placements. Use to diagnose "this object renders
        // dim/black" — distinguishes default-branch-hits-dim-atlas vs
        // airprobe-fallback vs sky-fallback.
        let probe_for_diag = std::env::var("PROTOMORPH_DIAG_OL4_PATH").ok();
        let probe_for = |pos: glam::Vec3, radius: f32, obj_def_flags: blam_tags::Flags<blam_tags::object::ObjectDefinitionFlags, u16>, diag_label: &str| -> Option<blam_tags::scenario_lightmap::DequantizedLightmapProbe> {
            let log = probe_for_diag.as_deref() == Some(diag_label) || probe_for_diag.as_deref() == Some("all");
            // Engine-faithful per-tag dispatch of
            // `lights_prepare_for_object_static_new @ 0x1808A2930`:
            // ```c
            // v53 = (_object_datum.flags & 0x2000) != 0;   // bit 13 runtime
            // if (_object_definition.flags & 2) v53 |= 1;   // bit 1 tag
            // ```
            // Bit 13 of `_object_datum.flags` is
            // `_object_static_lighting_raycast_sideways_bit`. Confirmed
            // setter audit (2026-05-24): the ONLY `*_new` setter is
            // `machine_new @ 0x18087ce28` gated on `_machine_definition.flags
            // & 0x04`. All other `*_new` (crate/scenery/weapon/equipment/
            // biped/vehicle/creature) leave bit 13 = 0 at scenario load.
            // Runtime setters (`effect_start`, `area_of_effect_cause_damage_
            // to_object`, `object_scripting_set_shield_effect`,
            // `vehicle_surge_update`, `object_update_decode`) fire on
            // gameplay events, not scenario load.
            //
            // So at scenario load, for any non-machine placement: bit 13
            // is 0. Engine MCC's dispatch is identical to ours when
            // `_object_definition.flags & 2 == 0` too: it uses the
            // default 1-ray branch. If our default branch produces
            // visibly black objects where MCC renders them correctly,
            // the divergence is in DATA (atlas bytes, per-vertex SH
            // availability, default-fill semantic), NOT dispatch.
            //
            // Diag: with `PROTOMORPH_DIAG_GEOMETRY_SAMPLER="x,y,z,r"`
            // matching the cast start, `sample()` dumps priority-chain
            // decision + per-vertex / atlas SH values. Use this to
            // discriminate the three hypotheses in
            // [[feedback_ol4_dispatch_blocked_by_bit13]].
            if log {
                eprintln!(
                    "[diag-ol4] {diag_label} pos=({:.3},{:.3},{:.3}) radius={radius:.4} obj_def_flags={obj_def_flags:?} (searches_lightmaps_on_failure={})",
                    pos.x, pos.y, pos.z,
                    obj_def_flags.contains(blam_tags::object::ObjectDefinitionFlags::SearchCardinalDirectionLightmapsOnFailure),
                );
                eprintln!(
                    "[diag-ol4]   raycast_ready={raycast_ready} bake_resources.len()={}",
                    bake_resources.len(),
                );
            }
            if raycast_ready {
                if let Some(ba) = bake_resources.first() {
                    // Engine-verbatim `s_geometry_sample` (504 B, 11 fields).
                    use crate::halo::geometry::geometry_sampling::{
                        lights_distant_lighting_at_point_new, lights_distant_lighting_flags,
                        GeometrySample, GeometrySamplerOutcome,
                    };
                    use crate::halo::math::globals::GLOBAL_UP_3D;
                    use blam_tags::math::RealPoint3d;

                    let use_sideways = obj_def_flags
                        .contains(blam_tags::object::ObjectDefinitionFlags::SearchCardinalDirectionLightmapsOnFailure);
                    let flags = if use_sideways { lights_distant_lighting_flags::SIDEWAYS } else { 0 };
                    if log {
                        if use_sideways {
                            eprintln!("[diag-ol4]   dispatch: SIDEWAYS (9-ray)");
                        } else {
                            let v28 = if radius > 0.05 { radius.max(0.4) } else { 0.4 };
                            let v29 = radius.max(0.05);
                            eprintln!(
                                "[diag-ol4]   dispatch: DEFAULT (1-ray) v28={v28:.3} v29={v29:.3} ray_start=(*,*,+{v29:.3}) ray_dir=(*,*,-{:.2})",
                                10.0 * v28,
                            );
                        }
                    }
                    let mut gs = GeometrySample::default();
                    // Engine reads object_class (n6), object_def_flags, object_radius,
                    // and `up` from object_header_data + object_get_orientation. For
                    // statically-placed scenery at scenario load we lack an object
                    // index; pass a sentinel class (255) that falls into the
                    // 1-ray default branch, world-up. The runtime datum
                    // `object_def_flags` is unavailable here, and its only branch
                    // (damaged biped) is gated on object class 0, so 0 is exact
                    // under the sentinel class.
                    let outcome = lights_distant_lighting_at_point_new(
                        flags,
                        /*object_class*/ 255,
                        /*object_def_flags*/ 0,
                        /*object_radius*/ radius,
                        /*up*/ GLOBAL_UP_3D,
                        /*ignore_object_index*/ -1,
                        /*need_valid_sample*/ false,
                        RealPoint3d { x: pos.x, y: pos.y, z: pos.z },
                        &mut gs,
                        /*override_direction*/ None,
                        /*is_flying*/ false,
                        loaded,
                        Some(&ba.sh),
                        ba.intensity.as_ref(),
                        // H3: lightprobe atlas is paired-pixel (8 layers),
                        // intensity atlas is paired-pixel (2 layers). These
                        // are the `bitmap_data[8]` bytes the engine reads.
                        /*lightprobe_pixels_per_probe*/ 8,
                        /*dominant_pixels_per_probe*/ 2,
                        /*scenario_flag_0x200*/ false,
                    );
                    if outcome == GeometrySamplerOutcome::Success {
                        if log { eprintln!("[diag-ol4]   sample returned Success — using BSP probe"); }
                        let mut red_terms = [0.0_f32; 9];
                        let mut green_terms = [0.0_f32; 9];
                        let mut blue_terms = [0.0_f32; 9];
                        red_terms.copy_from_slice(&gs.light_probe_r[..9]);
                        green_terms.copy_from_slice(&gs.light_probe_g[..9]);
                        blue_terms.copy_from_slice(&gs.light_probe_b[..9]);
                        return Some(blam_tags::scenario_lightmap::DequantizedLightmapProbe {
                            dominant_light_direction: [
                                gs.dominant_light_dir.i,
                                gs.dominant_light_dir.j,
                                gs.dominant_light_dir.k,
                            ],
                            dominant_light_intensity: [
                                gs.dominant_light_intensity.red,
                                gs.dominant_light_intensity.green,
                                gs.dominant_light_intensity.blue,
                            ],
                            red_terms,
                            green_terms,
                            blue_terms,
                        });
                    }
                    if log { eprintln!("[diag-ol4]   sample returned NoHit — falling through to airprobe"); }
                }
            }
            // Step 2 — `lights_airprobe_lighting_at_point_new`.
            let from_airprobe = nearest_airprobe(airprobes, pos)
                .map(|a| a.probe.dequantize());
            if log {
                if let Some(p) = &from_airprobe {
                    eprintln!(
                        "[diag-ol4]   AIRPROBE fallback: DC=({:.4},{:.4},{:.4}) — this is the source of the bake",
                        p.red_terms[0], p.green_terms[0], p.blue_terms[0],
                    );
                } else {
                    eprintln!("[diag-ol4]   no airprobe match → SKY fallback");
                }
            }
            from_airprobe
                // Step 3 — `lights_sky_lighing_at_point_new`.
                .or_else(|| sky_fallback.clone())
        };

        // Engine-faithful per-placement (radius, obj_def_flags) lookup
        // via the OBJECT_HEADER_DATA table populated by
        // `populate_from_scenario`. Each bake closure resolves its
        // engine `object_index` from `(object_type, i)` and reads the
        // pair from the global header table. See OL-1 + OL-4 in
        // [[reference_object_lighting_full_2026_05_24]].
        //
        // Maps the scenario object-type byte to `object_type::ObjectType`
        // (the single canonical enum, shared with the object header table).
        let scenario_ref = &loaded.scenario;
        let obj_meta_for = |object_type: ObjectType, i: u32| -> (f32, blam_tags::Flags<blam_tags::object::ObjectDefinitionFlags, u16>) {
            use crate::halo::objects::object_type::ObjectType as OT;
            let header_type = match object_type {
                ObjectType::Biped         => OT::Biped,
                ObjectType::Vehicle       => OT::Vehicle,
                ObjectType::Weapon        => OT::Weapon,
                ObjectType::Equipment     => OT::Equipment,
                ObjectType::Terminal      => OT::Terminal,
                ObjectType::Projectile    => OT::Projectile,
                ObjectType::Scenery       => OT::Scenery,
                ObjectType::Machine       => OT::Machine,
                ObjectType::Control       => OT::Control,
                ObjectType::SoundScenery  => OT::SoundScenery,
                ObjectType::Crate         => OT::Crate,
                ObjectType::Creature      => OT::Creature,
                ObjectType::Giant         => OT::Giant,
                ObjectType::EffectScenery => OT::EffectScenery,
            };
            let obj_idx = crate::halo::objects::object_header_data::header_index_for_placement(
                scenario_ref,
                header_type,
                i,
            );
            let r = crate::halo::objects::object_header_data::bounding_sphere_radius(obj_idx);
            let f = crate::halo::objects::object_header_data::object_definition_flags(obj_idx);
            (r, f)
        };

        let crate_offsets = self.bake_object_lighting_offsets(
            crates.len(),
            ObjectType::Crate,
            "crate",
            |i, _| {
                let (r, f) = obj_meta_for(ObjectType::Crate, i as u32);
                let label = format!("crate:{i}");
                probe_for(placement_position(&crates[i]), r, f, &label)
            },
        );
        // Vehicles share the non-scenery dynamic fallback chain (no
        // per-object scenery/device probe) — same as weapon/equipment:
        // distant cluster lightprobe → airprobe → sky.
        let vehicle_offsets = self.bake_object_lighting_offsets(
            vehicles.len(),
            ObjectType::Vehicle,
            "vehicle",
            |i, _| {
                let (r, f) = obj_meta_for(ObjectType::Vehicle, i as u32);
                let label = format!("vehicle:{i}");
                probe_for(placement_position(&vehicles[i]), r, f, &label)
            },
        );
        let weapon_offsets = self.bake_object_lighting_offsets(
            weapons.len(),
            ObjectType::Weapon,
            "weapon",
            |i, _| {
                let (r, f) = obj_meta_for(ObjectType::Weapon, i as u32);
                let label = format!("weapon:{i}");
                probe_for(placement_position(&weapons[i]), r, f, &label)
            },
        );
        let equipment_offsets = self.bake_object_lighting_offsets(
            equipment.len(),
            ObjectType::Equipment,
            "equipment",
            |i, _| {
                let (r, f) = obj_meta_for(ObjectType::Equipment, i as u32);
                let label = format!("equipment:{i}");
                probe_for(placement_position(&equipment[i]), r, f, &label)
            },
        );
        // Phase D — engine step 4 for type 7 (Machine):
        // `lights_prepare_for_object_static_new @ 0x1808A2930` checks
        // `obj.scenery_or_device_probe_index` and resolves
        // `device_machine_probes[idx]` for type==7. Tool.exe bakes the
        // matching positional, so machine[i] ↔ device_machine_probes[i]
        // is the engine-correct mapping. Each container holds multiple
        // per-position probes inside the machine's bbox; we take the
        // first (machine-local centroid) — the engine's per-point
        // interpolation across the container is a refinement for
        // animated machine parts that scenario-load bake can skip.
        // Lbsp vs sLdT preference matches the airprobe pattern: sLdT
        // when populated, fall back to Lbsp.
        let device_machine_probes: &[blam_tags::scenario_lightmap::LightmapDeviceMachineProbeData] = loaded
            .scenario_lightmap
            .as_ref()
            .filter(|s| !s.device_machine_probes.is_empty())
            .map(|s| s.device_machine_probes.as_slice())
            .unwrap_or(lbsp.device_machine_probes.as_slice());
        eprintln!(
            "[scenario]   device_machine_probes available: {} (LABEL_114 fallback when machine[i] has no probe)",
            device_machine_probes.len(),
        );
        let machine_offsets = self.bake_object_lighting_offsets(
            machines.len(),
            ObjectType::Machine,
            "machine",
            |i, pos| {
                device_machine_probes
                    .get(i)
                    .and_then(|c| c.probes.first())
                    .map(|p| p.probe.dequantize())
                    .or_else(|| {
                        let (r, f) = obj_meta_for(ObjectType::Machine, i as u32);
                        let label = format!("machine:{i}");
                        probe_for(pos, r, f, &label)
                    })
            },
        );
        let control_offsets = self.bake_object_lighting_offsets(
            controls.len(),
            ObjectType::Control,
            "control",
            |i, _| {
                let (r, f) = obj_meta_for(ObjectType::Control, i as u32);
                let label = format!("control:{i}");
                probe_for(placement_position(&controls[i]), r, f, &label)
            },
        );

        self.scenery_lighting_offsets = scenery_offsets;
        self.crate_lighting_offsets = crate_offsets;
        self.vehicle_lighting_offsets = vehicle_offsets;
        self.weapon_lighting_offsets = weapon_offsets;
        self.equipment_lighting_offsets = equipment_offsets;
        self.machine_lighting_offsets = machine_offsets;
        self.control_lighting_offsets = control_offsets;

        // OL-4 dispatch summary: count placements by sample-branch
        // choice. `_object_searches_lightmaps_on_failure_bit` (bit 1 of
        // `_object_definition.flags`) gates sideways (9 rays) vs default
        // (1 ray with offset, radius-scaled). Engine cites:
        // `lights_prepare_for_object_static_new @ 0x1808A2930:376` for
        // the dispatch decision. See OL-4 in
        // [[reference_object_lighting_full_2026_05_24]].
        let mut side_count = 0usize;
        let mut deft_count = 0usize;
        let count_dispatch =
            |ot: ObjectType, placements_len: usize, side: &mut usize, deft: &mut usize| {
                use crate::halo::objects::object_type::ObjectType as OT;
                let header_type = match ot {
                    ObjectType::Scenery => OT::Scenery,
                    ObjectType::Crate => OT::Crate,
                    ObjectType::Vehicle => OT::Vehicle,
                    ObjectType::Weapon => OT::Weapon,
                    ObjectType::Equipment => OT::Equipment,
                    ObjectType::Machine => OT::Machine,
                    ObjectType::Control => OT::Control,
                    _ => return,
                };
                for i in 0..placements_len {
                    let obj_idx = crate::halo::objects::object_header_data::header_index_for_placement(
                        scenario_ref, header_type, i as u32,
                    );
                    let f = crate::halo::objects::object_header_data::object_definition_flags(obj_idx);
                    if f.contains(blam_tags::object::ObjectDefinitionFlags::SearchCardinalDirectionLightmapsOnFailure) {
                        *side += 1;
                    } else {
                        *deft += 1;
                    }
                }
            };
        count_dispatch(ObjectType::Crate, crates.len(), &mut side_count, &mut deft_count);
        count_dispatch(ObjectType::Vehicle, vehicles.len(), &mut side_count, &mut deft_count);
        count_dispatch(ObjectType::Weapon, weapons.len(), &mut side_count, &mut deft_count);
        count_dispatch(ObjectType::Equipment, equipment.len(), &mut side_count, &mut deft_count);
        count_dispatch(ObjectType::Machine, machines.len(), &mut side_count, &mut deft_count);
        count_dispatch(ObjectType::Control, controls.len(), &mut side_count, &mut deft_count);
        eprintln!(
            "[scenario]   OL-4 dispatch: {side_count} via sideways (9-ray) + {deft_count} via default (1-ray-with-offset)"
        );

        // Help the borrow checker — re-import here so the closures
        // above don't shadow this name's import.
        let _: Option<&LightmapProbe> = None;
    }

    /// Bake per-placement engine_lighting cbuffer entries for one object
    /// type. `probe_lookup(i, position)` returns the SH probe to use for
    /// placement `i` (or None to skip and inherit the sky default).
    ///
    /// The resolved RAW SH (no chmt boost) is cached in
    /// `self.object_lighting_cache`. The per-frame chmt boost is applied
    /// later by [`Self::refresh_engine_lighting_for_frame`]. Dominant
    /// light direction + intensity are written to
    /// `engine_dominant_light_buffer` here at bake time — they don't get
    /// the chmt boost and never change per frame (until OL-5 lands).
    ///
    /// Returns `Vec<Option<u32>>` keyed by placement index (byte offset
    /// into the lighting buffer).
    fn bake_object_lighting_offsets<F>(
        &mut self,
        placement_count: usize,
        object_type: crate::halo::objects::object_type::ObjectType,
        type_label: &str,
        probe_lookup: F,
    ) -> Vec<Option<u32>>
    where
        F: Fn(usize, glam::Vec3) -> Option<blam_tags::scenario_lightmap::DequantizedLightmapProbe>,
    {
        use crate::halo::lighting::GpuEngineDominantLight;
        use crate::halo::render::shared::{ENGINE_LIGHTING_ENTRIES, ENGINE_LIGHTING_STRIDE};

        let mut offsets: Vec<Option<u32>> = vec![None; placement_count];
        if placement_count == 0 {
            return offsets;
        }

        let mut baked: usize = 0;
        // PROTOMORPH_DIAG_LIGHTING filter formats (matched against type_label[i]):
        //   "all"              — every placement of every type
        //   "<type>"           — every placement of one type, e.g. "crate"
        //   "<type>:<i>"       — single placement by index, e.g. "crate:41"
        //   "<type>:<a>,<b>,…" — list of indices, e.g. "crate:41,43,12"
        let diag_filter = std::env::var("PROTOMORPH_DIAG_LIGHTING").ok();
        let diag_match = |i: usize| -> bool {
            let Some(f) = diag_filter.as_deref() else { return false };
            if f == "all" || f == type_label { return true; }
            if let Some(rest) = f.strip_prefix(type_label).and_then(|s| s.strip_prefix(':')) {
                return rest.split(',').any(|tok| tok.trim().parse::<usize>() == Ok(i));
            }
            false
        };
        for i in 0..placement_count {
            let Some(dq) = probe_lookup(i, glam::Vec3::ZERO) else {
                if diag_match(i) {
                    eprintln!(
                        "[diag-lighting] {type_label}[{i}] probe_lookup returned None — \
                         placement will inherit frame-default (likely sky probe)",
                    );
                }
                continue;
            };
            if diag_match(i) {
                let r_dc = dq.red_terms[0];
                let g_dc = dq.green_terms[0];
                let b_dc = dq.blue_terms[0];
                let dom_i = dq.dominant_light_intensity;
                let dom_d = dq.dominant_light_direction;
                let sh_lum = crate::halo::render::chocalate_mountain::sh_luminance_estimate(
                    &dq.red_terms, &dq.green_terms, &dq.blue_terms,
                );
                eprintln!(
                    "[diag-lighting] {type_label}[{i}] probe_for:\n  \
                     SH_R={:?}\n  SH_G={:?}\n  SH_B={:?}\n  \
                     DC=({r_dc:.4},{g_dc:.4},{b_dc:.4}) sh_lum={sh_lum:.4}\n  \
                     dom_dir=({:.3},{:.3},{:.3}) dom_intensity=({:.4},{:.4},{:.4})",
                    dq.red_terms, dq.green_terms, dq.blue_terms,
                    dom_d[0], dom_d[1], dom_d[2],
                    dom_i[0], dom_i[1], dom_i[2],
                );
            }
            if self.structure_renderer.next_probe_slot >= ENGINE_LIGHTING_ENTRIES {
                eprintln!(
                    "[scenario]   WARNING: engine_lighting slot pool exhausted at {}[{}]",
                    type_label, i,
                );
                break;
            }

            // Diag override: PROTOMORPH_FORCE_BRIGHT_SH=<type>:<idx>[,<idx>...]
            // Replaces the baked SH for matching placements with a known-
            // bright probe (DC=1.0 on all channels, no L1/L2). Discriminates
            // "shader chain is correct, probe is dim" from "shader chain
            // is broken, probe doesn't matter". If the placement renders
            // visibly lit (~half-gray) with this override, the chain works.
            // If it stays dark, the bug is downstream of the cbuffer.
            let dq_cached = {
                let env = std::env::var("PROTOMORPH_FORCE_BRIGHT_SH").ok();
                let matches = env.as_deref().and_then(|f| {
                    f.strip_prefix(type_label).and_then(|s| s.strip_prefix(':'))
                        .map(|rest| rest.split(',').any(|tok| tok.trim().parse::<usize>() == Ok(i)))
                }).unwrap_or(false);
                if matches {
                    let mut forced = dq.clone();
                    // DC ≈ 1.0/√π × 1.0 → in our convention DC slot holds the
                    // unnormalized value. 1.0 on each channel produces a
                    // bright, neutrally-lit probe regardless of normal.
                    forced.red_terms = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    forced.green_terms = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    forced.blue_terms = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    forced.dominant_light_intensity = [1.0, 1.0, 1.0];
                    forced.dominant_light_direction = [0.0, 0.0, 1.0];
                    eprintln!(
                        "[diag-force-bright] {type_label}[{i}] overriding SH to DC=1.0 — \
                         shader-chain discriminator test",
                    );
                    forced
                } else {
                    dq.clone()
                }
            };

            let sh_lum = crate::halo::render::chocalate_mountain::sh_luminance_estimate(
                &dq_cached.red_terms, &dq_cached.green_terms, &dq_cached.blue_terms,
            );
            let dom_payload = GpuEngineDominantLight {
                direction: [
                    dq_cached.dominant_light_direction[0],
                    dq_cached.dominant_light_direction[1],
                    dq_cached.dominant_light_direction[2],
                    1.0,
                ],
                intensity: [
                    dq_cached.dominant_light_intensity[0],
                    dq_cached.dominant_light_intensity[1],
                    dq_cached.dominant_light_intensity[2],
                    0.0,
                ],
            };
            let offset = self.structure_renderer.next_probe_slot * ENGINE_LIGHTING_STRIDE;
            if diag_match(i) {
                let slot = self.structure_renderer.next_probe_slot;
                eprintln!(
                    "[diag-lighting] {type_label}[{i}] bake-out:\n  \
                     sh_lum={sh_lum:.4} (chmt boost applied per-frame, see refresh_engine_lighting_for_frame)\n  \
                     engine_lighting slot={slot} offset=0x{offset:x} ({offset} bytes)\n  \
                     dom_payload dir={:?} intensity={:?}",
                    dom_payload.direction, dom_payload.intensity,
                );
            }
            // Write dominant-light at bake — it doesn't carry a chmt
            // boost and (until OL-5 lands per-frame refresh) doesn't
            // change. Ravi gets written every frame by
            // refresh_engine_lighting_for_frame.
            self.shared.queue.write_buffer(
                &self.shared.engine_dominant_light_buffer,
                offset as u64,
                bytemuck::bytes_of(&dom_payload),
            );
            self.object_lighting_cache.push(ObjectLightingCacheEntry {
                sh_r: dq_cached.red_terms,
                sh_g: dq_cached.green_terms,
                sh_b: dq_cached.blue_terms,
                sh_lum,
                object_type,
                slot_offset: offset,
                dominant_dir: dq_cached.dominant_light_direction,
            });
            offsets[i] = Some(offset);
            self.structure_renderer.next_probe_slot += 1;
            baked += 1;
        }

        if baked > 0 {
            eprintln!(
                "[scenario]   cached {}_lighting for {}/{} placements (chmt boost applied per-frame)",
                type_label, baked, placement_count,
            );
        }
        offsets
    }
}
/// Build a [`blam_tags::scenario_lightmap::DequantizedLightmapProbe`]
/// from the scenario's sky lighting. Engine equivalent of
/// `lights_sky_lighing_at_point_new`'s happy path: copy the sky's
/// `default_lightprobe` SH3 into the per-object lighting sample, with
/// dominant direction + intensity from
/// `calculate_dominant_light_from_lightprobe`. Protomorph already
/// computed those into `sky_lighting.sun_direction` / `sun_intensity`
/// at load time — reuse them.
fn sky_probe_as_dequantized(
    sky: &crate::halo::scenario::loader::SkyLighting,
) -> blam_tags::scenario_lightmap::DequantizedLightmapProbe {
    blam_tags::scenario_lightmap::DequantizedLightmapProbe {
        dominant_light_direction: [
            sky.sun_direction.i,
            sky.sun_direction.j,
            sky.sun_direction.k,
        ],
        dominant_light_intensity: [
            sky.sun_intensity.i,
            sky.sun_intensity.j,
            sky.sun_intensity.k,
        ],
        red_terms: sky.probe.r,
        green_terms: sky.probe.g,
        blue_terms: sky.probe.b,
    }
}

/// Find the airprobe whose world-space `position` is closest to `pos`.
/// Engine `c_geometry_sampler::sample_one_air_probe` does a 3-probe
/// trilinear blend; v1 returns the single nearest probe — visually close
/// for sparse airprobe layouts (most MP maps), but sharper transitions
/// at probe boundaries than the engine. Refine to a 3-probe weighted blend
/// if visual seams show up on dense layouts.
fn nearest_airprobe<'a>(
    airprobes: &'a [blam_tags::scenario_lightmap::LightmapAirprobe],
    pos: glam::Vec3,
) -> Option<&'a blam_tags::scenario_lightmap::LightmapAirprobe> {
    nearest_by(airprobes, pos, |probe| {
        glam::Vec3::new(probe.position.x, probe.position.y, probe.position.z)
    })
    .map(|(_, p)| p)
}

/// Extract the world-space `(x, y, z)` of an [`ObjectPlacement`].
fn placement_position(placement: &blam_tags::scenario::ObjectPlacement) -> glam::Vec3 {
    glam::Vec3::new(
        placement.object_data.position.x,
        placement.object_data.position.y,
        placement.object_data.position.z,
    )
}
