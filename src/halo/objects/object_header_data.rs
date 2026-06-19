//! Object header table — the engine-side `object_header_data` global
//! (the master object-handle table). Indexed by the lower 16 bits of
//! an `object_index` handle; each entry carries the object's
//! discriminant type + (later) the data-pool pointer for its per-type
//! struct.
//!
//! Engine reference: every `*_compute_function_value` reads
//! `(char*)object_header_data->data + (object_index & 0xFFFF) * size`
//! to get the per-object datum, then accesses byte offsets (e.g.
//! offset 154 = `e_object_type` discriminant).
//!
//! **Phase 1 (this file) — minimum viable:**
//! - Per-entry `ObjectHeaderDatum` carries just the `object_type`
//!   (enough to unblock `object_type_compute_function_value`'s routing).
//! - Global `OBJECT_HEADER_DATA: RwLock<Vec<ObjectHeaderDatum>>`
//!   matches the engine's `object_header_data` global. Populated by
//!   `populate_from_loaded_scenario` at scenario load. Cleared
//!   between scenarios via `clear`.
//!
//! **Future phases:** extend `ObjectHeaderDatum` with a pointer/index
//! into per-type `*_datum` pools (Phase 5+). The base struct currently
//! has only what `object_type_compute_function_value` needs.

use crate::halo::objects::object_datum::ObjectDatum;
use crate::halo::objects::object_type::ObjectType;
use blam_tags::object::ObjectDefinition;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Per-object metadata — one entry per spawned object. Engine
/// equivalent: `s_object_header_datum` (the `object_header_data`
/// table entry).
#[derive(Debug, Clone)]
pub struct ObjectHeaderDatum {
    /// `e_object_type` discriminant — the engine field at byte offset
    /// +0 of each header entry (LSB of the type/flags packed word).
    pub object_type: ObjectType,
    /// Tag path the placement loaded from (e.g.
    /// `objects/weapons/melee/gravity_hammer/gravity_hammer.weapon`).
    /// Surfaced in `*_compute_function_value` warnings so the dev
    /// knows which authored tag asked for an unported function name.
    /// Empty when the placement's palette_index was -1 or the palette
    /// entry was empty.
    pub tag_path: String,
    /// Parsed `object_definition` from the placement's tag (or empty
    /// when the tag couldn't be loaded / had no palette entry). Shared
    /// via `Arc` so multiple placements of the same palette entry
    /// (e.g. 20 marinebeacons on shrine) refer to one parsed copy.
    ///
    /// Drives the engine `object_get_function_value @ 0x1807DBA60`
    /// fallback walk — when a `<type>_compute_function_value` returns
    /// false for some `import_name`, the walker searches THIS struct's
    /// `functions[]` for an `export_name` match and evaluates the
    /// authored curve chain.
    pub object_definition: Arc<ObjectDefinition>,
    /// Per-placement runtime state — engine `object_datum`. Engine
    /// layout: `(char*)object_header_data->data + idx*size →
    /// header_datum.datum_ptr → object_datum`. Protomorph collapses
    /// the indirection by inlining the datum on each header entry
    /// (one allocation per placement; engine pools by type).
    ///
    /// Wrapped in `RwLock` so future damage-simulation can mutate
    /// state. `Arc` lets multiple readers share without lifetime
    /// pain. For static placements with no simulation, the datum is
    /// initialized at scenario load and never mutated.
    pub object_datum: Arc<RwLock<ObjectDatum>>,
}

/// Global object header table. Engine equivalent: `object_header_data`
/// global pointer at `dllcache 0x18112E?? something` (per
/// `object_get_function_value`'s reads).
///
/// Indexed by `object_index` (lower 16 bits — engine convention).
/// Populated by scenario load; cleared between scenarios.
static OBJECT_HEADER_DATA: RwLock<Vec<ObjectHeaderDatum>> = RwLock::new(Vec::new());

/// Runtime `object name index → object index` map. Engine equivalent:
/// the `g_object_name_list` the engine fills via
/// `object_set_object_index_for_name_index` (and reads via
/// `object_index_from_name_index`). Indexed by a scenario `object names`
/// block index; value is the object's header index (lower-16 handle), or
/// `-1` for an unbound name. Populated at scenario load from each
/// placement's `name_index` and from `create_sky_object`'s sky-name
/// registration. Used to resolve a placement's `parent_id.
/// parent_object_name_index` to its parent object.
static OBJECT_NAME_MAP: RwLock<Vec<i32>> = RwLock::new(Vec::new());

/// Resolve `name_index → object_index` (engine
/// `object_index_from_name_index`). `None` for an out-of-range or
/// unbound name.
pub fn object_index_from_name_index(name_index: i16) -> Option<u32> {
    if name_index < 0 {
        return None;
    }
    let guard = OBJECT_NAME_MAP.read().unwrap();
    match guard.get(name_index as usize).copied() {
        Some(v) if v >= 0 => Some(v as u32),
        _ => None,
    }
}

/// Replace the global table with `entries`. Called by the renderer's
/// scenario-load path AFTER all placement Vecs have been resolved.
pub fn replace(entries: Vec<ObjectHeaderDatum>) {
    let mut guard = OBJECT_HEADER_DATA.write().unwrap();
    *guard = entries;
}

/// Append a header entry for a runtime-created object that has no
/// scenario placement — currently model-variant child objects (turrets
/// attached via `model_variant_object_block`). Returns the new object's
/// header index (`= table length before the push`). The entry is
/// appended AFTER every scenario placement (which already occupy
/// `[0, placement_count)` from [`populate_from_scenario`]), so it never
/// collides with a [`header_index_for_placement`] result.
///
/// Engine equivalent: `object_new` allocates a fresh `object_data` slot
/// for the spawned child and `c_object::initialize` fills its datum —
/// the child is a first-class object with its own handle, which is what
/// lets its render-method function lookups (`object_get_function_value`)
/// resolve against ITS `object_definition.functions[]` instead of
/// returning 0.
///
/// `world` is decomposed into the datum frame exactly as
/// [`set_world_transform`] does (engine `from_cols(forward,left,up,pos)`).
/// `variant_index` selects the child's own model variant (drives the
/// `variant` compute case); `-1`/`0` when unresolved.
pub fn push_object(
    object_type: ObjectType,
    tag_path: String,
    object_definition: Arc<ObjectDefinition>,
    variant_index: i8,
    world: glam::Mat4,
) -> u32 {
    use blam_tags::math::{RealPoint3d, RealVector3d};

    let mut datum = ObjectDatum::new(object_type, -1);
    let (c0, c2, p) = (world.x_axis, world.z_axis, world.w_axis);
    let fwd = glam::Vec3::new(c0.x, c0.y, c0.z);
    let s = fwd.length();
    let fwd_n = if s > 1e-6 { fwd / s } else { glam::Vec3::X };
    let up_n = glam::Vec3::new(c2.x, c2.y, c2.z).normalize_or_zero();
    datum.object.forward = RealVector3d { i: fwd_n.x, j: fwd_n.y, k: fwd_n.z };
    datum.object.up = RealVector3d { i: up_n.x, j: up_n.y, k: up_n.z };
    datum.object.relative_position = RealPoint3d { x: p.x, y: p.y, z: p.z };
    datum.object.scale = if s > 1e-6 { s } else { 1.0 };
    datum.object.bounding_sphere_center = RealPoint3d { x: p.x, y: p.y, z: p.z };
    datum.object.bounding_sphere_radius = object_definition.bounding_radius;
    datum.object.variant_index = variant_index;
    // Runtime-created: no scenario placement / no name binding.
    datum.object.scenario_datum_index = -1;
    datum.object.name_index = -1;

    let mut guard = OBJECT_HEADER_DATA.write().unwrap();
    let new_index = guard.len() as u32;
    datum.object.object_identifier.unique_id = new_index as i32;
    guard.push(ObjectHeaderDatum {
        object_type,
        tag_path,
        object_definition,
        object_datum: Arc::new(RwLock::new(datum)),
    });
    new_index
}

/// Engine `(object_header_data->data + index * size)->object_type`
/// lookup. Returns the `ObjectType` for `object_index`.
///
/// Panics if `object_index` is out of range — engine assertion path
/// (`handle_slim_assert` in `object_get_function_value:91`) does the
/// same when `object_index >= object_header_data->count`.
pub fn get_object_type(object_index: u32) -> ObjectType {
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize; // engine masks to lower 16 bits
    match guard.get(idx) {
        Some(entry) => entry.object_type,
        None => panic!(
            "[object_header_data::get_object_type] object_index={object_index} \
             (idx={idx}) out of range — header table has {} entries. \
             Engine assertion: `object_header_data->valid && \
             (unsigned __int16)object_index < object_header_data->count` \
             (object_get_function_value:91). \
             Either the scenario hasn't been loaded yet, or the caller \
             is passing a handle from a different / cleared scenario.",
            guard.len()
        ),
    }
}

/// World matrix of `object_index` from its datum (the instance) — port of
/// `object_get_world_matrix @0x1807d9650`. Rendering reads this instead of
/// the per-object `ObjectData.model_matrix`, making the datum the single
/// transform authority. `None` if the index is out of range. See
/// [`ObjectDatum::world_matrix`].
pub fn world_matrix(object_index: u32) -> Option<glam::Mat4> {
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize;
    let entry = guard.get(idx)?;
    let datum = entry.object_datum.read().unwrap();
    Some(datum.world_matrix())
}

/// Overwrite `object_index`'s datum frame so its [`world_matrix`] equals
/// `world`. Decomposes `world` (engine `from_cols(forward,left,up,pos)`)
/// into the datum's `forward`/`up`/`relative_position`/`scale` — the same
/// thing the engine's `object_attach_to_marker_immediate` does via
/// `object_set_position_internal` when snapping a child to a parent marker.
/// Used by parent-marker attachment (e.g. construct's waterfall scenery →
/// the sky's `"waterfall"` marker). No-op if the index is out of range.
pub fn set_world_transform(object_index: u32, world: glam::Mat4) {
    use blam_tags::math::{RealPoint3d, RealVector3d};
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize;
    let Some(entry) = guard.get(idx) else { return };
    let mut d = entry.object_datum.write().unwrap();
    let (c0, c2, p) = (world.x_axis, world.z_axis, world.w_axis);
    let fwd = glam::Vec3::new(c0.x, c0.y, c0.z);
    let s = fwd.length();
    let fwd_n = if s > 1e-6 { fwd / s } else { glam::Vec3::X };
    let up_n = glam::Vec3::new(c2.x, c2.y, c2.z).normalize_or_zero();
    d.object.forward = RealVector3d { i: fwd_n.x, j: fwd_n.y, k: fwd_n.z };
    d.object.up = RealVector3d { i: up_n.x, j: up_n.y, k: up_n.z };
    d.object.relative_position = RealPoint3d { x: p.x, y: p.y, z: p.z };
    d.object.scale = if s > 1e-6 { s } else { 1.0 };
}

/// Compute the engine `object_index` for `(object_type,
/// placement_index_within_type)`. Must match the walk order used by
/// [`populate_from_scenario`] so that
/// `OBJECT_HEADER_DATA[header_index].object_type == object_type`.
///
/// Walk order:
///   Biped, Vehicle, Weapon, Equipment, Scenery, Machine, Control,
///   SoundScenery, Crate.
/// Other types (Terminal, Projectile, Creature, Giant, EffectScenery)
/// don't have dedicated scenario placement Vecs in blam-tags today —
/// passing those panics with a clear message.
pub fn header_index_for_placement(
    scenario: &blam_tags::scenario::Scenario,
    object_type: ObjectType,
    placement_index_within_type: u32,
) -> u32 {
    let mut base = 0u32;
    if object_type == ObjectType::Biped {
        return base + placement_index_within_type;
    }
    base += scenario.bipeds.len() as u32;
    if object_type == ObjectType::Vehicle {
        return base + placement_index_within_type;
    }
    base += scenario.vehicles.len() as u32;
    if object_type == ObjectType::Weapon {
        return base + placement_index_within_type;
    }
    base += scenario.weapons.len() as u32;
    if object_type == ObjectType::Equipment {
        return base + placement_index_within_type;
    }
    base += scenario.equipment.len() as u32;
    if object_type == ObjectType::Scenery {
        return base + placement_index_within_type;
    }
    base += scenario.scenery.len() as u32;
    if object_type == ObjectType::Machine {
        return base + placement_index_within_type;
    }
    base += scenario.machines.len() as u32;
    if object_type == ObjectType::Control {
        return base + placement_index_within_type;
    }
    base += scenario.controls.len() as u32;
    if object_type == ObjectType::SoundScenery {
        return base + placement_index_within_type;
    }
    base += scenario.sound_scenery.len() as u32;
    if object_type == ObjectType::Crate {
        return base + placement_index_within_type;
    }
    base += scenario.crates.len() as u32;
    if object_type == ObjectType::EffectScenery {
        return base + placement_index_within_type;
    }
    panic!(
        "[object_header_data::header_index_for_placement] object_type {object_type:?} \
         has no dedicated scenario placement Vec in blam-tags. \
         Add the placement Vec to `blam_tags::scenario::Scenario` and \
         update `populate_from_scenario` + this function in matching \
         walk order. \
         See [[project_render_method_function_port_plan_2026_05_20]]."
    );
}

/// Populate the global header table from a loaded `Scenario`'s
/// per-type placement Vecs. The walk order defines `object_index`
/// assignment — must be consistent across runs so cached object
/// references stay valid.
///
/// Engine equivalent: at scenario load, every spawned object gets a
/// slot in `object_header_data` via `object_new` / `object_create` —
/// here we synthesize that table directly from the placement arrays
/// since protomorph doesn't have a runtime object-spawn pipeline yet.
///
/// Walk order (chosen to match the Ares object-type enum discriminant
/// order so manual lookup is consistent — biped=0, vehicle=1, weapon=2,
/// ...):
///   biped, vehicle, weapon, equipment, terminal, projectile,
///   scenery, machine, control, sound_scenery, crate, creature,
///   giant, effect_scenery.
///
/// Unsupported types (terminal, projectile, creature, giant,
/// effect_scenery) are skipped — blam-tags doesn't currently surface
/// dedicated placement Vecs for them (and they're rare in MP maps).
pub fn populate_from_scenario(
    scenario: &blam_tags::scenario::Scenario,
    _tags_root: &Path,
) {
    use blam_tags::scenario::{ObjectPlacement, TagReferencePalette};
    use crate::halo::tags::tag_get;

    let empty: Arc<ObjectDefinition> = Arc::new(ObjectDefinition::default());

    /// Resolve `(object_type, tag_path)` to an `Arc<ObjectDefinition>`
    /// via the global tag cache. Falls back to an empty definition on
    /// missing/failed loads (the cache logs the failure once).
    fn load_definition(
        empty: &Arc<ObjectDefinition>,
        object_type: ObjectType,
        tag_path: &str,
    ) -> Arc<ObjectDefinition> {
        if tag_path.is_empty() {
            return empty.clone();
        }
        tag_get(object_type.tag_group_fourcc(), tag_path)
            .and_then(|t| t.as_object_definition())
            .unwrap_or_else(|| empty.clone())
    }

    /// Chase an `ObjectDefinition`'s `model` ref through the tag cache
    /// and return the variant_names list. Empty when the model ref is
    /// empty or the hlmt tag failed to load.
    fn load_variant_names(obj_def: &ObjectDefinition) -> Vec<String> {
        if obj_def.model.is_empty() {
            return Vec::new();
        }
        tag_get(*b"hlmt", &obj_def.model)
            .and_then(|t| t.as_model())
            .map(|m| m.variant_names.clone())
            .unwrap_or_default()
    }

    fn push_block(
        out: &mut Vec<ObjectHeaderDatum>,
        empty: &Arc<ObjectDefinition>,
        object_type: ObjectType,
        placements: &[ObjectPlacement],
        palette: &[TagReferencePalette],
        out_base_header_index: u32,
    ) {
        for (i, placement) in placements.iter().enumerate() {
            let tag_path = if placement.palette_index >= 0 {
                palette
                    .get(placement.palette_index as usize)
                    .map(|p| p.tag_path.clone())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            let object_definition = load_definition(empty, object_type, &tag_path);
            let variant_names = load_variant_names(&object_definition);
            // Engine `object_new` populates the per-placement datum
            // from the placement transform + identifiers via
            // `object_placement_data_new_from_scenario_object` then
            // `c_object::initialize`. Protomorph mirrors via
            // `ObjectDatum::from_placement` which copies forward / up
            // / scale / position / identifier fields into the datum
            // and resolves variant_name → variant_index against the
            // model's variants[].
            // Header index = position in the global header table
            // (base + i so it matches `header_index_for_placement`).
            let header_index = out_base_header_index + i as u32;
            let object_datum = Arc::new(RwLock::new(
                crate::halo::objects::object_datum::ObjectDatum::from_placement(
                    object_type,
                    placement,
                    &variant_names,
                    i as u32,
                    header_index,
                    object_definition.bounding_radius,
                ),
            ));
            out.push(ObjectHeaderDatum {
                object_type,
                tag_path,
                object_definition,
                object_datum,
            });
        }
    }

    let mut entries = Vec::new();
    // `base` = the running header_index for the next placement. Must
    // match the walk order in `header_index_for_placement` so that
    // `OBJECT_HEADER_DATA[header_index].object_datum.object_identifier
    // .unique_id == header_index`.
    let mut base = 0u32;
    push_block(&mut entries, &empty, ObjectType::Biped,        &scenario.bipeds,        &scenario.biped_palette,        base);
    base += scenario.bipeds.len() as u32;
    push_block(&mut entries, &empty, ObjectType::Vehicle,      &scenario.vehicles,      &scenario.vehicle_palette,      base);
    base += scenario.vehicles.len() as u32;
    push_block(&mut entries, &empty, ObjectType::Weapon,       &scenario.weapons,       &scenario.weapon_palette,       base);
    base += scenario.weapons.len() as u32;
    push_block(&mut entries, &empty, ObjectType::Equipment,    &scenario.equipment,     &scenario.equipment_palette,    base);
    base += scenario.equipment.len() as u32;
    // terminal — no dedicated placement Vec in blam-tags scenario today
    // projectile — no dedicated placement Vec
    push_block(&mut entries, &empty, ObjectType::Scenery,      &scenario.scenery,       &scenario.scenery_palette,      base);
    base += scenario.scenery.len() as u32;
    push_block(&mut entries, &empty, ObjectType::Machine,      &scenario.machines,      &scenario.machine_palette,      base);
    base += scenario.machines.len() as u32;
    push_block(&mut entries, &empty, ObjectType::Control,      &scenario.controls,      &scenario.control_palette,      base);
    base += scenario.controls.len() as u32;
    push_block(&mut entries, &empty, ObjectType::SoundScenery, &scenario.sound_scenery, &scenario.sound_scenery_palette, base);
    base += scenario.sound_scenery.len() as u32;
    push_block(&mut entries, &empty, ObjectType::Crate,        &scenario.crates,        &scenario.crate_palette,        base);
    base += scenario.crates.len() as u32;
    push_block(&mut entries, &empty, ObjectType::EffectScenery, &scenario.effect_scenery, &scenario.effect_scenery_palette, base);
    let _ = base; // for symmetry; unused after last block
    // creature, giant — no dedicated placement Vec

    let total_functions: usize = entries.iter().map(|e| e.object_definition.functions.len()).sum();
    let mut nonzero_variant = 0usize;
    let mut authored_variant_name = 0usize;
    for e in &entries {
        let d = e.object_datum.read().unwrap();
        if d.object.variant_index != 0 { nonzero_variant += 1; }
    }
    // Also count placements that AUTHORED a non-empty variant_name —
    // distinguishes "no variant authored" from "variant authored but
    // didn't match any model entry".
    let count_authored_variants = |placements: &[blam_tags::scenario::ObjectPlacement]| -> usize {
        placements.iter().filter(|p| !p.permutation_data.variant_name.is_empty()).count()
    };
    authored_variant_name += count_authored_variants(&scenario.bipeds);
    authored_variant_name += count_authored_variants(&scenario.vehicles);
    authored_variant_name += count_authored_variants(&scenario.weapons);
    authored_variant_name += count_authored_variants(&scenario.equipment);
    authored_variant_name += count_authored_variants(&scenario.scenery);
    authored_variant_name += count_authored_variants(&scenario.machines);
    authored_variant_name += count_authored_variants(&scenario.controls);
    authored_variant_name += count_authored_variants(&scenario.sound_scenery);
    authored_variant_name += count_authored_variants(&scenario.crates);
    eprintln!(
        "[object_header_data] populated {} entries — biped={} vehicle={} weapon={} \
         equipment={} scenery={} machine={} control={} sound_scenery={} crate={} \
         ({} total functions[] entries; {} placements authored a variant_name, \
         {} resolved to a non-zero variant_index)",
        entries.len(),
        scenario.bipeds.len(),
        scenario.vehicles.len(),
        scenario.weapons.len(),
        scenario.equipment.len(),
        scenario.scenery.len(),
        scenario.machines.len(),
        scenario.controls.len(),
        scenario.sound_scenery.len(),
        scenario.crates.len(),
        total_functions,
        authored_variant_name,
        nonzero_variant,
    );

    // Bind every named placement into the name→object map (engine
    // `object_set_object_index_for_name_index`, called per object as it
    // spawns). The sky object registers separately in `create_sky_object`.
    {
        let mut name_map = OBJECT_NAME_MAP.write().unwrap();
        name_map.clear();
        name_map.resize(scenario.object_names.len(), -1);
        for (header_index, e) in entries.iter().enumerate() {
            let ni = e.object_datum.read().unwrap().object.name_index;
            if ni >= 0 && (ni as usize) < name_map.len() {
                name_map[ni as usize] = header_index as i32;
            }
        }
    }

    replace(entries);
}

/// Read accessor for the per-placement `ObjectDefinition`. Used by
/// `object_get_function_value` when walking the chain of authored
/// functions to resolve an unknown function-name input.
///
/// Returns an empty default-filled `Arc<ObjectDefinition>` when
/// `object_index` is out of range or the placement had no parsed
/// definition — diagnostic surface only.
pub fn get_object_definition(object_index: u32) -> Arc<ObjectDefinition> {
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize;
    guard
        .get(idx)
        .map(|entry| entry.object_definition.clone())
        .unwrap_or_else(|| Arc::new(ObjectDefinition::default()))
}

/// Engine-equivalent of dereferencing
/// `object_header_data->data[idx].datum` — returns the
/// `Arc<RwLock<ObjectDatum>>` for the per-placement runtime state.
/// Used by every `<type>_compute_function_value` to read damage /
/// vitality / phantom-scale / etc. state.
///
/// Returns a new default `Arc<RwLock<ObjectDatum>>` when
/// `object_index` is out of range — diagnostic surface that won't
/// panic in the compute chain (which fires for every animated draw).
pub fn get_object_datum(
    object_index: u32,
) -> Arc<RwLock<crate::halo::objects::object_datum::ObjectDatum>> {
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize;
    if let Some(entry) = guard.get(idx) {
        entry.object_datum.clone()
    } else {
        // No entry — synthesize a default for the diagnostic path.
        // Object type is unknown here; default to Scenery (most
        // numerous placement type, harmless for state-read accessors).
        Arc::new(RwLock::new(
            crate::halo::objects::object_datum::ObjectDatum::new(ObjectType::Scenery, -1),
        ))
    }
}

/// Engine-equivalent lookup of the per-placement source tag path for
/// surfacing in unported-function warnings. Returns the empty string
/// when `object_index` is out of range or the placement had no palette
/// entry. Never panics — diagnostic surface only.
pub fn get_tag_path(object_index: u32) -> String {
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize;
    guard
        .get(idx)
        .map(|entry| entry.tag_path.clone())
        .unwrap_or_default()
}

/// Engine `_object_datum.bounding_sphere_radius` lookup. Read by
/// `lights_distant_lighting_at_point_new @ 0x1808A3220` at
/// `*(float *)(object_datum + 0x2C)` — drives the cast-ray scale
/// derivation (`v28 = max(r, 0.4)` if `r > 0.05` else `0.4`;
/// `v29 = max(r, 0.05)`). Returns `0.0` when `object_index` is out of
/// range — engine's `v28` formula handles that case via the 0.4
/// default. Never panics.
///
/// Companion to [`header_index_for_placement`] for callers that have
/// `(object_type, placement_index_within_type)` rather than the global
/// `object_index`.
pub fn bounding_sphere_radius(object_index: u32) -> f32 {
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize;
    guard
        .get(idx)
        .map(|entry| entry.object_datum.read().unwrap().object.bounding_sphere_radius)
        .unwrap_or(0.0)
}

/// `_object_definition.flags` (u16) lookup from the object's tag.
/// Engine `lights_prepare_for_object_static_new @ 0x1808A2930:376` tests
/// `(v71->class_index & 2) != 0` to decide sideways vs default — but
/// v71 is mistyped by IDA as `s_cache_file_sound_definition*`; the real
/// type is `_object_definition*`, and `class_index` (byte +2 of the
/// mistyped struct) maps to the LOW BYTE of `_object_definition.flags`
/// at offset +2. Bit 1 of that u16 = `_object_searches_lightmaps_on_failure_bit`
/// per `e_object_definition_flags`.
///
/// Returns 0 when `object_index` is out of range or the placement has
/// no resolved tag (empty palette entry) — bit 1 unset by default, so
/// default-branch dispatch is the safe fallback (`v28 = 0.4` formula
/// floor handles missing radius).
pub fn object_definition_flags(
    object_index: u32,
) -> blam_tags::Flags<blam_tags::object::ObjectDefinitionFlags, u16> {
    let guard = OBJECT_HEADER_DATA.read().unwrap();
    let idx = (object_index & 0xFFFF) as usize;
    guard
        .get(idx)
        .map(|entry| entry.object_definition.flags.clone())
        .unwrap_or_default()
}
