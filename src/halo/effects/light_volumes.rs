//! Light-volume (`ltvl` / `rmlv`) effect parts — protomorph's runtime side
//! of the engine `light_volumes` subsystem (Ares `light_volumes.cpp`).
//!
//! An effect's `event.parts[]` can name a `light_volume_system` (`ltvl`)
//! tag; this module resolves it (the shader's blend/fog/base_map + the 8
//! authored profile curves) and carries the GPU strip render types the
//! per-frame build produces. The per-frame profile BUILD itself lives in
//! [`super::EffectStore::frame_advance`] (Track R1); only the standalone
//! types + the loader live here.

use std::path::Path;
use std::str::FromStr;

use blam_tags::effect::EffectPartType;
use blam_tags::render_method::AlphaBlendMode;
use blam_tags::paths::resolve_tag_path;
use blam_tags::TagFile;

use super::effect_definitions::LoadedEffect;
use super::effects::EffectInstance;
use super::particle_emitter;
use super::EffectStore;

/// Render-method state a light volume (`rmlv`) needs to shade — resolved
/// by name from the embedded `c_render_method_shader_light_volume`. Far
/// simpler than the particle path: light volumes only have blend_mode, fog,
/// and a single `diffuse_only` albedo `base_map` (engine `light_volume_fx
/// .hlsl` PS samples just `base_map`).
#[derive(Debug, Clone, Default)]
pub struct LightVolumeRenderInfo {
    pub blend_mode: AlphaBlendMode,
    pub fog: bool,
    pub base_map: String,
}

/// A resolved `light_volume_system` (`ltvl`) tag — the first definition's
/// shader + the 8 authored profile curves. The strip render
/// (Track R1 increment 2) consumes this: it stacks `profile_density`
/// camera-facing-but-roll-locked quads along `origin + direction·(offset +
/// i·profile_distance)`, each profile's thickness/color/alpha/intensity
/// evaluated from the curves at its percentile (engine `light_volume_fx
/// .hlsl`). blam-tags decodes the whole 380B definition.
#[derive(Debug, Clone)]
pub struct LoadedLightVolume {
    /// The first `light_volumes[]` definition (the common single-def case).
    pub definition: blam_tags::light_volume_system::LightVolumeDefinition,
    pub render: LightVolumeRenderInfo,
    /// `base_map` bitmap tag-paths referenced by the rmlv shader.
    pub bitmap_paths: Vec<String>,
}

/// One light-volume profile (a cross-section quad) as the GPU strip render
/// consumes it. Per-profile fields (position/color/intensity/thickness/alpha)
/// are evaluated CPU-side from the curves at the profile's percentile; the
/// per-volume fields (direction / profile_length / profile_distance /
/// num_profiles / brightness_ratio) are denormalized into every profile so a
/// whole strip draws as one instanced call. Mirrors `s_profile_state` +
/// `g_all_state` from `light_volume_fx.hlsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightVolumeProfileGpu {
    /// xyz = world position (origin + dir·(offset + i·profile_distance)); w unused.
    pub pos: [f32; 4],
    /// rgb = profile color, a = intensity.
    pub color_intensity: [f32; 4],
    /// xyz = shaft direction (normalized), w = profile thickness.
    pub dir_thickness: [f32; 4],
    /// profile_length, profile_distance, num_profiles, brightness_ratio.
    pub params: [f32; 4],
    /// x = profile alpha; yzw pad.
    pub alpha: [f32; 4],
}

/// A contiguous run of [`LightVolumeProfileGpu`] sharing a base_map + blend —
/// one draw of the strip pipeline (4 verts × `count` profile instances).
#[derive(Debug, Clone)]
pub struct LightVolumeDraw {
    /// `base_map` bitmap path (the draw's texture; also the group key).
    pub base_map: String,
    pub blend_mode: AlphaBlendMode,
    pub fog: bool,
    /// `[first .. first+count)` into [`super::EffectStore::light_volume_profiles`].
    pub first: u32,
    pub count: u32,
}

impl EffectStore {
    /// Read a `light_volume_system` (`ltvl`) tag and resolve its first
    /// definition + render state (the `rmlv` shader's blend_mode / fog /
    /// base_map, name-resolved). Returns `None` if the tag can't be
    /// read/parsed. Mirrors [`EffectStore::load_particle`].
    pub(super) fn load_light_volume_def(
        ltvl_rel: &str,
        tags_root: &Path,
    ) -> Option<LoadedLightVolume> {
        let abs = resolve_tag_path(tags_root, ltvl_rel, "light_volume_system");
        let tag = match TagFile::read(&abs) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[effects]   failed to read ltvl {}: {e}", abs.display());
                return None;
            }
        };
        let system = match blam_tags::light_volume_system::LightVolumeSystem::from_tag(&tag) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[effects]   {ltvl_rel}: bad light_volume_system: {e}");
                return None;
            }
        };
        // The common case is a single definition; take the first.
        let definition = system.definitions.into_iter().next()?;
        let render = definition
            .shader
            .as_ref()
            .map(|rm| {
                let choices = Self::load_rm_choices(rm, tags_root);
                // Light shafts are additive by default (the engine's typical
                // rmlv blend) when the category is absent/unknown.
                let blend_mode = match choices.get("blend_mode") {
                    Some(name) => AlphaBlendMode::from_str(name).unwrap_or(AlphaBlendMode::Additive),
                    None => AlphaBlendMode::Additive,
                };
                let base_map = rm
                    .parameters
                    .iter()
                    .find(|p| p.parameter_name == "base_map" && !p.bitmap_path.is_empty())
                    .map(|p| p.bitmap_path.clone())
                    .unwrap_or_default();
                LightVolumeRenderInfo {
                    blend_mode,
                    fog: choices.get_or("fog", "off") != "off",
                    base_map,
                }
            })
            .unwrap_or_default();
        let bitmap_paths = definition
            .shader
            .as_ref()
            .map(|rm| {
                rm.parameters
                    .iter()
                    .filter(|p| !p.bitmap_path.is_empty())
                    .map(|p| p.bitmap_path.clone())
                    .collect()
            })
            .unwrap_or_default();
        Some(LoadedLightVolume { definition, render, bitmap_paths })
    }
}

/// Track R1 light-volume strip build for a single effect instance.
///
/// For each `ltvl` part on this effect, evaluate the strip profiles into the
/// per-frame light-volume buffers. Light volumes are stateless (engine
/// re-evaluates every profile each frame), so we build them CPU-side here,
/// gated by the same host attachment function (`emit_scale`) as particle
/// emission. Profiles stack along `origin + dir·(offset + i·profile_distance)`;
/// per-profile color/thickness/alpha/intensity come from the 8 curves at the
/// profile percentile (engine `light_volume_fx.hlsl` + the GPU re-eval). Each
/// strip is one [`LightVolumeDraw`].
///
/// Extracted verbatim from `EffectStore::frame_advance` (behaviour-identical):
/// the parameters are exactly the per-instance/shared state the original block
/// read or wrote.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_instance_light_volumes(
    effect: &LoadedEffect,
    inst: &EffectInstance,
    host_matrix: glam::Mat4,
    emit_scale: f32,
    light_volumes: &std::collections::HashMap<String, LoadedLightVolume>,
    light_volume_profiles: &mut Vec<LightVolumeProfileGpu>,
    light_volume_draws: &mut Vec<LightVolumeDraw>,
) {
    if emit_scale > 1e-4 {
        for part in &effect.parts {
            if part.part_type != EffectPartType::LightVolume {
                continue;
            }
            let Some(lv) = light_volumes.get(&part.type_tag_path) else {
                continue;
            };
            let loc_idx = part.location.max(0) as usize;
            let location_matrix = match inst.location_matrices.get(loc_idx) {
                Some(m) => host_matrix * *m,
                None => host_matrix,
            };
            let origin =
                location_matrix.transform_point3(glam::Vec3::from(part.relative_offset));
            let direction = location_matrix
                .transform_vector3(glam::Vec3::X)
                .normalize_or_zero();
            if direction.length_squared() < 1e-6 {
                continue;
            }
            let def = &lv.definition;
            let eval = |p: &blam_tags::effects_properties::EditableProperty, input: f32| {
                particle_emitter::diag_eval(p, input, 0.0)
            };
            // Diagnostic: PROTOMORPH_LV_FORCE=<scale> forces
            // brightness_ratio=1 (visible head-on) and ×scale thickness
            // — to confirm the strip render rasterizes, independent of
            // the (often tiny / side-only) authored values.
            let force = std::env::var("PROTOMORPH_LV_FORCE")
                .ok()
                .and_then(|s| s.parse::<f32>().ok());
            // Profile count — engine `c_light_volume_gpu::render
            // @0x1806D29A0`: `count = (int)(m_length · m_profile_density)`
            // (TRUNCATED, not rounded; no min-1 floor), rendered only if
            // `> 0`. The old `round(profile_density).clamp(1,128)` ignored
            // length entirely → wrong count + spurious profiles for
            // zero-length / sub-1 volumes. Cap at the GPU strip buffer
            // max (8 rows × 16 = 128).
            let length = eval(&def.length, 0.0);
            let profile_density = eval(&def.profile_density, 0.0);
            let count_i = (length * profile_density) as i32;
            if count_i <= 0 {
                continue;
            }
            let num = (count_i as u32).min(128);
            let offset = eval(&def.offset, 0.0);
            let profile_length = eval(&def.profile_length, 0.0);
            let profile_distance = if num > 1 {
                length / (num as f32 - 1.0)
            } else {
                0.0
            };
            let first = light_volume_profiles.len() as u32;
            for i in 0..num {
                let pct = if num > 1 {
                    i as f32 / (num as f32 - 1.0)
                } else {
                    0.0
                };
                let pos = origin + direction * (offset + i as f32 * profile_distance);
                let thickness = eval(&def.profile_thickness, pct) * force.unwrap_or(1.0);
                let alpha = eval(&def.profile_alpha, pct);
                let intensity = eval(&def.profile_intensity, pct);
                let color = def
                    .profile_color
                    .function
                    .as_ref()
                    .map(|f| {
                        let c = f.evaluate_color(pct, 0.0);
                        [c.red, c.green, c.blue]
                    })
                    .unwrap_or([1.0, 1.0, 1.0]);
                light_volume_profiles.push(LightVolumeProfileGpu {
                    pos: [pos.x, pos.y, pos.z, 0.0],
                    color_intensity: [color[0], color[1], color[2], intensity],
                    dir_thickness: [direction.x, direction.y, direction.z, thickness],
                    params: [
                        profile_length,
                        profile_distance,
                        num as f32,
                        if force.is_some() { 1.0 } else { def.brightness_ratio },
                    ],
                    alpha: [alpha, 0.0, 0.0, 0.0],
                });
            }
            light_volume_draws.push(LightVolumeDraw {
                base_map: lv.render.base_map.clone(),
                blend_mode: lv.render.blend_mode,
                fog: lv.render.fog,
                first,
                count: num,
            });
        }
    }
}
