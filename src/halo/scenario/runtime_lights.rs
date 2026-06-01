//! Runtime light resolver — walks scenario `lights[]` placements +
//! `light_palette[]`, loads each unique `light` tag, and produces a
//! flat list of `RuntimeLight` entries that downstream systems can
//! consume.
//!
//! Engine paths this unblocks:
//!
//! - **`c_lights_view::submit_visibility_and_render @ 0x1806C6930`** —
//!   per-light shadow scheduler. Filters `casts_shadows() &&
//!   is_frustum() && fov_radians < π`.
//! - **`light_submit_lens_flares @ 0x18086A850`** — for each light with
//!   a non-empty `Lens Flare` reference, submits a `s_lens_flare` entry
//!   to `g_lens_flare_globals.lens_flare_array` (256-slot hash table).
//!
//! `scenario_structure_lighting_info` (`stli`) is a SEPARATE thing
//! entirely — `generic_light_instances` + `material_info.emissive_*`
//! feed tool.exe's offline lightmap bake (see `surface_light_create
//! @ 0x140239C90` in tool.exe), where they get rasterized into the
//! lightmap atlas as baked surface-light contributions. At runtime the
//! engine reads those from the atlas via normal per-pixel SH; nothing
//! from stli ever reaches the runtime `simple_lights` cbuffer.
//! This resolver covers ONLY the scenario-placed `light` tag
//! placements that go through the runtime `add_simple_light_callback`.

use std::collections::HashMap;
use std::path::Path;

use blam_tags::file::TagFile;
use blam_tags::light::LightDefinition;
use blam_tags::paths::resolve_tag_path;
use blam_tags::scenario::{ObjectPlacement, Scenario, TagReferencePalette};
use glam::{EulerRot, Quat, Vec3};

/// One scenario-placed light, fully resolved.
#[derive(Debug, Clone)]
pub struct RuntimeLight {
    /// Index into `scenario.lights[]` — useful for callbacks that key
    /// on the placement (e.g. lens-flare hash table dedups on
    /// `(attachment_type, attachment_index, attachment_subindex)`).
    pub placement_index: usize,
    /// World-space position from the placement's `object_data.position`.
    pub position: Vec3,
    /// Cone direction. Halo's canonical forward axis is +X; this is
    /// `placement_rotation × (1, 0, 0)` (placement Euler is `(yaw=Z,
    /// pitch=Y, roll=X)` per `ObjectData::model_matrix`).
    pub forward: Vec3,
    /// Up axis from the same rotation — useful for frustum lights that
    /// need a full basis (the gel orientation rotates with this).
    pub up: Vec3,
    /// Resolved tag-ref path for diagnostics + dedup. Empty when the
    /// palette entry was malformed (such placements are dropped before
    /// reaching this list).
    pub definition_path: String,
    /// The walked `light` tag.
    pub definition: LightDefinition,
}

impl RuntimeLight {
    /// Convenience — the per-shadow-scheduler gate verbatim from
    /// `c_lights_view::submit_visibility_and_render`.
    pub fn passes_shadow_scheduler_gate(&self) -> bool {
        self.definition.casts_shadows()
            && self.definition.is_frustum()
            && self.definition.frustum_field_of_view.to_radians() < std::f32::consts::PI
    }

    /// Convenience — the `light_submit_lens_flares` gate. A non-empty
    /// `Lens Flare` reference is the only filter at this level (the
    /// engine submits regardless of distance / camera; per-flare
    /// occlusion / opacity culling happens later).
    pub fn has_lens_flare(&self) -> bool {
        self.definition.has_lens_flare()
    }
}

/// Walk `scenario.lights` + `scenario.light_palette` and build the
/// runtime light list. Each unique palette entry is loaded ONCE
/// (cached by index); placements with `palette_index < 0` or with a
/// missing tag are skipped (and counted in the returned summary).
///
/// The returned summary is intended for a one-line log message at
/// load time; it carries the totals so callers can flag empty-result
/// scenarios without inspecting every entry.
pub struct RuntimeLightLoadResult {
    pub lights: Vec<RuntimeLight>,
    pub placement_count: usize,
    pub palette_count: usize,
    /// Number of placements skipped because `palette_index < 0`.
    pub skipped_no_palette: usize,
    /// Number of placements whose palette tag failed to load.
    pub skipped_failed_load: usize,
    /// Number of unique tags loaded.
    pub tags_loaded: usize,
}

pub fn load_runtime_lights(
    scenario: &Scenario,
    tags_root: &Path,
) -> RuntimeLightLoadResult {
    let placements: &[ObjectPlacement] = &scenario.lights;
    let palette: &[TagReferencePalette] = &scenario.light_palette;

    let mut cache: HashMap<usize, Option<LightDefinition>> = HashMap::new();
    let mut lights = Vec::new();
    let mut skipped_no_palette = 0;
    let mut skipped_failed_load = 0;

    for (placement_index, placement) in placements.iter().enumerate() {
        let pal_idx = placement.palette_index;
        if pal_idx < 0 || (pal_idx as usize) >= palette.len() {
            skipped_no_palette += 1;
            continue;
        }
        let pal_idx_usize = pal_idx as usize;
        let tag_path = palette[pal_idx_usize].tag_path.clone();
        if tag_path.is_empty() {
            skipped_no_palette += 1;
            continue;
        }

        let definition = cache
            .entry(pal_idx_usize)
            .or_insert_with(|| load_light_tag(tags_root, &tag_path))
            .clone();

        let Some(definition) = definition else {
            skipped_failed_load += 1;
            continue;
        };

        let pos = placement.object_data.position;
        let rot = placement.object_data.rotation;
        // Halo rotation order: yaw (Z), pitch (Y), roll (X). On-disk
        // `RealEulerAngles3d` is raw f32 radians (no degree conversion
        // in `read_euler3d`). `ObjectData::model_matrix` calls
        // `to_radians()` because IT carries the rotation as degrees in
        // a `Vec3`; here we have the canonical radians directly.
        let q = Quat::from_euler(EulerRot::ZYX, rot.yaw, rot.pitch, rot.roll);
        let position = Vec3::new(pos.x, pos.y, pos.z);
        let forward = q * Vec3::X;
        let up = q * Vec3::Z;

        lights.push(RuntimeLight {
            placement_index,
            position,
            forward,
            up,
            definition_path: tag_path,
            definition,
        });
    }

    let tags_loaded = cache.values().filter(|v| v.is_some()).count();

    RuntimeLightLoadResult {
        lights,
        placement_count: placements.len(),
        palette_count: palette.len(),
        skipped_no_palette,
        skipped_failed_load,
        tags_loaded,
    }
}

fn load_light_tag(tags_root: &Path, tag_path: &str) -> Option<LightDefinition> {
    let path = resolve_tag_path(tags_root, tag_path, "light");
    let tag = TagFile::read(&path).ok()?;
    LightDefinition::from_tag(&tag).ok()
}
