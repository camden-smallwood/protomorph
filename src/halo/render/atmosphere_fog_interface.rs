//! Engine-faithful port of `c_atmosphere_fog_interface`.
//!
//! Owns the per-frame `s_weighted_atmosphere_parameters` (the accumulated
//! weighted blend of all atmosphere settings that contribute to the
//! current camera position) and the lifecycle bookkeeping engine uses
//! to avoid redundant cbuffer writes.
//!
//! This module is the runtime carrier — the parsed-tag data lives in
//! `blam_tags::sky_atmosphere`. Phase 3 of the atmosphere port plan
//! ([[project_atmosphere_engine_faithful_port_plan_2026_05_19]]) will
//! port `compute_cluster_weights` and Phase 4 will populate
//! `default_parameters` via `accumulate_atmosphere_settings`. Phase 5
//! folds the cbuffer write path (`set_constant`) onto this struct.

use bytemuck::{Pod, Zeroable};

use blam_tags::math::{RealPlane3d, RealRgbColor, RealVector3d};
use blam_tags::sky_atmosphere::{AtmosphereFlags, SkyAtmosphere};
use blam_tags::structure_bsp::{BspAtmospherePaletteEntry, BspCluster, BspClusterPortal};

use crate::halo::structures::clusters::ClusterReference;
use crate::halo::structures::clusters_in_sphere;

/// `s_weighted_atmosphere_parameters` (124 B). Engine layout from IDA
/// `get_type_info` — every offset matches.
///
/// `c_atmosphere_fog_interface::accumulate_atmosphere_settings @
/// 0x1803AFD90` produces this as a weighted sum across cluster-resident
/// `c_atmosphere_setting` instances. The `weight` argument is per-setting
/// `m_weight` (computed by `compute_cluster_weights @ 0x1803ADE40` from
/// `cluster_search_radius` / `falloff_start_distance` /
/// `distance_falloff_power`).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct WeightedAtmosphereParameters {
    /// +0x000 — `bool`, padded to 4 bytes by the engine struct.
    pub atmosphere_enabled: u32,
    /// +0x004 — sun_intensity × weight, accumulated.
    pub sun_intensity: RealRgbColor,
    /// +0x010 — sun_direction × weight, accumulated (then normalised
    /// inside `set_constant` via `normalize3d_with_default(global_down3d)`).
    pub sun_direction: RealVector3d,
    /// +0x01C
    pub distance_bias: f32,
    /// +0x020
    pub max_fog_thickness: f32,
    /// +0x024
    pub beta_m: RealVector3d,
    /// +0x030
    pub beta_p: RealVector3d,
    /// +0x03C
    pub beta_m_angular: RealVector3d,
    /// +0x048
    pub beta_p_angular: RealVector3d,
    /// +0x054 — Henyey-Greenstein `g`. `slot2.w = g + 1`, `slot6 =
    /// broadcast(2g)`, `slot5.xyz *= (1 - g²)` per `set_constant`.
    pub heyey_greenstein: f32,
    /// +0x058 — engine `reference_datum_plane`. SAME field as H3 MCC
    /// tag's "Sea Level"; see [[reference_h3_sea_level_is_reference_datum_plane]].
    pub reference_datum_plane: f32,
    /// +0x05C — `slot4.w` in the cbuffer.
    pub mie_height: f32,
    /// +0x060 — `slot5.w`.
    pub rayleigh_height: f32,
    /// +0x064
    pub patchy_fog_density: f32,
    /// +0x068
    pub full_intensity_height: f32,
    /// +0x06C
    pub half_intensity_height: f32,
    /// +0x070 — world-space wind direction; consumed by `c_patchy_fog`
    /// per-frame to advance sheet UV offsets.
    pub wind_direction: RealVector3d,
}

impl Default for WeightedAtmosphereParameters {
    fn default() -> Self {
        // Engine's all-zero starting state. Consumers should treat
        // `atmosphere_enabled = 0` as "atmosphere off, skip the cbuffer
        // build" — matches the `if (parameters->atmosphere_enabled)`
        // gate inside `set_constant @ 0x1803AE530`.
        Self {
            atmosphere_enabled: 0,
            sun_intensity: RealRgbColor::default(),
            sun_direction: RealVector3d::default(),
            distance_bias: 0.0,
            max_fog_thickness: 0.0,
            beta_m: RealVector3d::default(),
            beta_p: RealVector3d::default(),
            beta_m_angular: RealVector3d::default(),
            beta_p_angular: RealVector3d::default(),
            heyey_greenstein: 0.0,
            reference_datum_plane: 0.0,
            mie_height: 0.0,
            rayleigh_height: 0.0,
            patchy_fog_density: 0.0,
            full_intensity_height: 0.0,
            half_intensity_height: 0.0,
            wind_direction: RealVector3d::default(),
        }
    }
}

impl WeightedAtmosphereParameters {
    /// `c_atmosphere_fog_interface::compute_scattering @ 0x1803AF020` —
    /// verbatim CPU per-fragment fog evaluator. Returns
    /// `(extinction, inscatter)` linear-RGB triples for a view→fragment ray.
    ///
    /// This is the engine's CPU mirror of the atmosphere shader, used for
    /// non-shader fog queries (e.g. sprite/decal tint, gameplay visibility).
    /// protomorph has no CPU fog consumer yet, so this is currently an
    /// unwired but faithful API-surface port (with unit tests pinning it to
    /// the engine arithmetic). `global_down3d = (0,0,-1)`.
    ///
    /// `atmosphere_enabled == 0` ⇒ engine's disabled return:
    /// `extinction = (1,1,1)`, `inscatter = (0,0,0)`.
    pub fn compute_scattering(
        &self,
        view_point: glam::Vec3,
        fragment_point: glam::Vec3,
        distance_bias: f32,
    ) -> ([f32; 3], [f32; 3]) {
        if self.atmosphere_enabled == 0 {
            return ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
        }
        // View → fragment direction + path length.
        let mut dx = view_point.x - fragment_point.x;
        let mut dy = view_point.y - fragment_point.y;
        let mut dz = view_point.z - fragment_point.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist.abs() >= 1.0e-4 {
            let inv = 1.0 / dist;
            dx *= inv;
            dy *= inv;
            dz *= inv;
        }
        // normalize3d_with_default(sun_direction, global_down3d).
        let sun = {
            let s = self.sun_direction;
            let l2 = s.i * s.i + s.j * s.j + s.k * s.k;
            if l2 > 1.0e-12 {
                let inv = l2.sqrt().recip();
                [s.i * inv, s.j * inv, s.k * inv]
            } else {
                [0.0, 0.0, -1.0]
            }
        };
        // Path length, biased + clamped to [0, max_fog_thickness].
        let mut path = (dist + self.distance_bias) + distance_bias;
        // v44 = -(dir · sun_dir).
        let neg_cos = -((dx * sun[0]) + (dy * sun[1]) + (dz * sun[2]));
        if path <= 0.0 {
            path = 0.0;
        }
        if path >= self.max_fog_thickness {
            path = self.max_fog_thickness;
        }
        // Height integration relative to reference_datum_plane.
        let refp = self.reference_datum_plane;
        let hv = (view_point.z - refp).max(0.0); // v18
        let hf = (fragment_point.z - refp).max(0.0); // v19
        let dh = hv - hf; // v20
        let (mie_density, ray_density);
        if dh * dh <= 0.001 {
            // Flat slab.
            mie_density = (-(hv / self.mie_height)).exp() * path;
            ray_density = (-(hv / self.rayleigh_height)).exp() * path;
        } else {
            let a = 1.0 / self.mie_height;
            let b = 1.0 / self.rayleigh_height;
            mie_density =
                -(((-(a * hv)).exp() - (-(a * hf)).exp()) * path * self.mie_height * (1.0 / dh));
            ray_density = -(((-(b * hv)).exp() - (-(b * hf)).exp())
                * path
                * self.rayleigh_height
                * (1.0 / dh));
        }
        // Extinction = exp(-(ray·β_m + mie·β_p)) per channel.
        let ext = [
            (-((ray_density * self.beta_m.i) + (mie_density * self.beta_p.i))).exp(),
            (-((ray_density * self.beta_m.j) + (mie_density * self.beta_p.j))).exp(),
            (-((ray_density * self.beta_m.k) + (mie_density * self.beta_p.k))).exp(),
        ];
        // Inscatter — Henyey-Greenstein phase + angular betas.
        let g = self.heyey_greenstein;
        let v30 = neg_cos * neg_cos + 1.0;
        let denom = (g + 1.0) - ((g * 2.0) * neg_cos); // v31
        let v32 = v30 * self.beta_m_angular.i;
        let v33 = v30 * self.beta_m_angular.j;
        let v34 = v30 * self.beta_m_angular.k;
        let inv_pow = {
            let s = denom.sqrt();
            let r = 1.0 / s;
            (1.0 - g * g) * (r * r * r)
        }; // v36
        let v37 = inv_pow * self.beta_p_angular.j + v33;
        let isc_r = (inv_pow * self.beta_p_angular.i + v32) * (1.0 - ext[0]) * self.sun_intensity.red;
        let isc_b =
            (inv_pow * self.beta_p_angular.k + v34) * (1.0 - ext[2]) * self.sun_intensity.blue;
        let isc_g = (v37 * (1.0 - ext[1])) * self.sun_intensity.green;
        let sm_r = self.beta_m.i + self.beta_p.i;
        let sm_g = self.beta_p.j + self.beta_m.j;
        let sm_b = self.beta_p.k + self.beta_m.k;
        let inscatter = if (sm_g * sm_g + sm_r * sm_r + sm_b * sm_b) < 1.0e-4 {
            [0.0, 0.0, 0.0]
        } else {
            [isc_r / sm_r, isc_g / sm_g, isc_b / sm_b]
        };
        (ext, inscatter)
    }
}

/// `c_atmosphere_fog_interface` — singleton-shape struct mirroring
/// the engine's static globals at runtime.
///
/// Engine references:
/// - `initialize @ 0x1801C6960` — no-op.
/// - `initialize_for_new_map @ 0x1803AF4A0` — clears `weather_effect_indices` (Phase 8).
/// - `initialize_for_new_structure_bsp @ 0x1803AF4E0` — swaps cluster PVS.
/// - `set_default_atmosphere_parameters @ 0x18068E5E0` — copies + applies cbuffer.
/// - `restore_default_atmosphere_constants @ 0x180681570` — re-applies if dirty.
/// - `invalidate_atmosphere_constants @ 0x18068E500` — marks dirty.
#[derive(Debug, Default)]
pub struct AtmosphereFogInterface {
    /// Engine `m_default_parameters` — the accumulated weighted blend
    /// applied to the atmosphere cbuffer each frame.
    pub default_parameters: WeightedAtmosphereParameters,
    /// Engine `m_last_custom_index`. Tracks the most recently-applied
    /// cbuffer source for redundant-write avoidance:
    /// - `-3` ⇒ disabled (atmosphere off; `disable_atmosphere` was last call).
    /// - `-2` ⇒ default_parameters applied.
    /// - `-1` ⇒ invalidated; next consumer must re-apply.
    /// - `≥0` ⇒ per-cluster custom override index (unused so far).
    pub last_custom_index: i32,
    /// Engine `m_use_local_pvs`. Driven by `change_pvs`; affects which
    /// PVS the cluster walker reads. Defaults to false (global PVS).
    pub use_local_pvs: bool,
}

impl AtmosphereFogInterface {
    /// `c_atmosphere_fog_interface::initialize_for_new_map @ 0x1803AF4A0`.
    /// Engine clears `g_rasterizer_game_states->weather_effect_indices`
    /// to all `-1`. Phase 8 plumbing — no-op for us until weather effects
    /// land. Called by `Renderer` at scenario load.
    pub fn initialize_for_new_map(&mut self) {
        // Future: clear weather_effect_indices when Phase 8 lands.
        // Also reset accumulated params so a stale prior-scenario blend
        // doesn't leak into the new map.
        self.default_parameters = WeightedAtmosphereParameters::default();
        self.last_custom_index = -1;
    }

    /// `c_atmosphere_fog_interface::accumulate_atmosphere_settings @ 0x1803AFD90`.
    /// Adds one setting's contribution (scaled by its per-frame `weight`)
    /// into the running [`WeightedAtmosphereParameters`]. Engine verbatim
    /// — every field is a `accum.f += weight * src.f` accumulator.
    ///
    /// Skips entirely when `setting.flags & 1 == 0` (Atmosphere disabled).
    /// Patchy-fog fields are gated on `setting.flags & 4` (Patchy Fog
    /// flag) per engine.
    ///
    /// Sun parameters resolve via [`get_sun_parameters`],
    /// which mirrors engine `get_sun_parameters @ 0x1803AF990`. The
    /// override branch (flags & 2) reads the setting's
    /// `m_dominant_light_*` (chromaticity-preserving xyY→RGB convert);
    /// the non-override branch needs a scenario-sky sun pulled from
    /// `get_sun_constants_from_sky` (callers pass via `sky_sun`).
    ///
    /// Beta coefficients are derived per-setting from
    /// `rayleigh_multiplier` / `mie_multiplier` / `desaturation` via
    /// [`precompute_betas`] —
    /// engine `precompute_scattering_coefficients @ 0x1803AE9C0`
    /// validated as byte-faithful per
    /// [[reference_precompute_betas_validation_2026_05_19]].
    pub fn accumulate_atmosphere_settings(
        &self,
        setting: &blam_tags::sky_atmosphere::AtmosphereSettings,
        accum: &mut WeightedAtmosphereParameters,
        weight: f32,
        sky_sun: Option<SkySun>,
    ) {
        if !setting.flags.contains(AtmosphereFlags::EnableAtmosphere) {
            return;
        }
        if weight > 0.0 {
            accum.atmosphere_enabled = 1;
        }

        // Sun. Engine takes (intensity, direction) and accumulates each
        // component separately weighted.
        let (sun_intensity, sun_direction) =
            get_sun_parameters(setting, sky_sun);
        accum.sun_intensity.red += weight * sun_intensity[0];
        accum.sun_intensity.green += weight * sun_intensity[1];
        accum.sun_intensity.blue += weight * sun_intensity[2];
        accum.sun_direction.i += weight * sun_direction[0];
        accum.sun_direction.j += weight * sun_direction[1];
        accum.sun_direction.k += weight * sun_direction[2];

        // Scalar fields, weighted.
        accum.distance_bias += weight * setting.distance_bias;
        accum.max_fog_thickness += weight * setting.max_fog_thickness;
        accum.heyey_greenstein += weight * setting.sun_phase_function;
        accum.reference_datum_plane += weight * setting.sea_level;
        accum.mie_height += weight * setting.mie_height_scale;
        accum.rayleigh_height += weight * setting.rayleigh_height_scale;

        // Beta coefficients — per-setting precomputed.
        let ((beta_m, beta_p), (beta_m_ang, beta_p_ang)) =
            precompute_betas(
                setting.rayleigh_multiplier,
                setting.mie_multiplier,
                setting.desaturation,
            );
        accum.beta_m.i += weight * beta_m[0];
        accum.beta_m.j += weight * beta_m[1];
        accum.beta_m.k += weight * beta_m[2];
        accum.beta_p.i += weight * beta_p[0];
        accum.beta_p.j += weight * beta_p[1];
        accum.beta_p.k += weight * beta_p[2];
        accum.beta_m_angular.i += weight * beta_m_ang[0];
        accum.beta_m_angular.j += weight * beta_m_ang[1];
        accum.beta_m_angular.k += weight * beta_m_ang[2];
        accum.beta_p_angular.i += weight * beta_p_ang[0];
        accum.beta_p_angular.j += weight * beta_p_ang[1];
        accum.beta_p_angular.k += weight * beta_p_ang[2];

        // Patchy fog — gated on flag bit 2.
        if setting.flags.contains(AtmosphereFlags::PatchyFog) {
            accum.patchy_fog_density += weight * setting.patchy_fog_density;
            accum.full_intensity_height += weight * setting.full_intensity_height;
            accum.half_intensity_height += weight * setting.half_intensity_height;
            accum.wind_direction.i += weight * setting.wind_direction.i;
            accum.wind_direction.j += weight * setting.wind_direction.j;
            accum.wind_direction.k += weight * setting.wind_direction.k;
        }
    }

    /// `c_atmosphere_fog_interface::populate_atmosphere_parameters @ 0x1803AE400`.
    /// Iterates every authored `AtmosphereSettings`, calls
    /// [`accumulate_atmosphere_settings`](Self::accumulate_atmosphere_settings)
    /// with its `weight` (set this frame by
    /// [`compute_cluster_weights`](Self::compute_cluster_weights)).
    ///
    /// Caller is responsible for zeroing `accum` before the first call;
    /// engine reuses the same buffer frame-to-frame and overwrites the
    /// accumulator in-place, but our consumer pattern clears + populates
    /// each frame.
    pub fn populate_atmosphere_parameters(
        &self,
        sky_atm: &blam_tags::sky_atmosphere::SkyAtmosphere,
        accum: &mut WeightedAtmosphereParameters,
        sky_sun: Option<SkySun>,
    ) {
        for setting in &sky_atm.atmosphere_settings {
            self.accumulate_atmosphere_settings(setting, accum, setting.weight, sky_sun);
        }
    }

    /// `c_atmosphere_fog_interface::compute_cluster_weights @ 0x1803ADE40`.
    /// Computes per-`AtmosphereSettings` weights for the current frame's
    /// camera position. Engine consumers (Phase 4
    /// [`populate_atmosphere_parameters`]) iterate
    /// [`SkyAtmosphere::atmosphere_settings`] reading
    /// [`AtmosphereSettings::weight`] to produce the weighted
    /// `WeightedAtmosphereParameters` blend.
    ///
    /// Engine branch:
    /// ```text
    /// clear all settings' (weight, effect_weight)
    /// if (game_is_multiplayer() && atmosphere_palette.count <= 1):
    ///     starting setting.weight = effect_weight = 1.0; return       // fast path
    /// else:                                                            // SP blend
    ///     sphere_r = max(cluster_search_radius*3, 20)
    ///     refs  = structure_clusters_in_sphere(start, eye, sphere_r, 100)
    ///     dists = structure_compute_cluster_distances(start, eye, refs)
    ///     total = Σ single_cluster_weight(dist)
    ///     for each cluster: s = get_atmosphere_setting(cluster)
    ///         s.weight += single_cluster_weight(dist) / total
    ///         if (sky_atm_flags & LockEffectsToNearestCluster):
    ///             if cluster == start: s.effect_weight = 1.0           // binary
    ///         else:
    ///             s.effect_weight += single_cluster_weight(dist) / total
    /// ```
    /// We gate the fast path on `atmosphere_palette.len() <= 1` rather than a
    /// game-mode flag: with a single palette entry every cluster resolves to
    /// the same setting, so the SP blend collapses to `weight = 1.0` on that
    /// setting — output-identical to the engine's MP fast path in every case
    /// (and to the engine's SP path on single-palette campaign maps). The SP
    /// blend runs the full `structure_clusters_in_sphere` →
    /// `compute_cluster_distances` → falloff machinery from
    /// [`crate::halo::structures::clusters_in_sphere`].
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cluster_weights(
        &mut self,
        sky_atm: &mut SkyAtmosphere,
        atmosphere_palette: &[BspAtmospherePaletteEntry],
        clusters: &[BspCluster],
        portals: &[BspClusterPortal],
        planes: &[RealPlane3d],
        starting_cluster: ClusterReference,
        starting_cluster_atmosphere_index: i8,
        eye_point: glam::Vec3,
    ) {
        // Engine clears BOTH weight and effect_weight (one `_QWORD` store).
        for setting in &mut sky_atm.atmosphere_settings {
            setting.weight = 0.0;
            setting.effect_weight = 0.0;
        }
        if !starting_cluster.is_valid() {
            return;
        }
        let setting_count = sky_atm.atmosphere_settings.len();

        // Resolve the starting cluster's setting (also the fast-path target +
        // the SP fallback when geometry is unavailable).
        let set_starting_only = |sky_atm: &mut SkyAtmosphere| {
            if let Some(idx) = resolve_atmosphere_setting_index(
                starting_cluster_atmosphere_index,
                atmosphere_palette,
                setting_count,
            ) {
                if let Some(s) = sky_atm.atmosphere_settings.get_mut(idx) {
                    s.weight = 1.0;
                    s.effect_weight = 1.0;
                }
            }
        };

        let start = starting_cluster.cluster_index;
        // Fast path (output-identical to the engine — see doc comment) or any
        // case where we lack the geometry to run the sphere walk.
        if atmosphere_palette.len() <= 1
            || clusters.is_empty()
            || start < 0
            || start as usize >= clusters.len()
        {
            set_starting_only(sky_atm);
            return;
        }
        let start = start as usize;

        // --- SP multi-cluster blend (full structure_clusters_in_sphere path) ---
        let search_radius = (sky_atm.cluster_search_radius * 3.0).max(20.0);
        let eye = [eye_point.x, eye_point.y, eye_point.z];
        let gathered = clusters_in_sphere::clusters_in_sphere(
            clusters, portals, planes, start, eye, search_radius, 100,
        );
        let dists = clusters_in_sphere::compute_cluster_distances(
            clusters, portals, planes, start, &gathered, eye,
        );

        // `compute_single_cluster_weight @ 0x1803B0530`. v16=cluster_search_radius
        // (the falloff outer distance), v17=falloff_start_distance (inner),
        // v18=distance_falloff_power.
        let csr = sky_atm.cluster_search_radius;
        let fsd = sky_atm.falloff_start_distance;
        let pow = sky_atm.distance_falloff_power;
        let single = |dist: f32| -> f32 {
            if csr <= fsd {
                1.0
            } else {
                let t = ((csr - dist) / (csr - fsd)).clamp(0.0, 1.0);
                t.powf(pow)
            }
        };

        let total: f32 = dists.iter().map(|&d| single(d)).sum();
        if total <= 0.0 {
            // Degenerate (all clusters beyond the falloff) — fall back to the
            // starting setting so atmosphere doesn't blank out.
            set_starting_only(sky_atm);
            return;
        }

        let lock = sky_atm.lock_effects_to_nearest_cluster();
        for (&c, &dist) in gathered.iter().zip(&dists) {
            let cluster_atm_idx = clusters.get(c).map(|cl| cl.atmosphere_index).unwrap_or(-1);
            let Some(idx) =
                resolve_atmosphere_setting_index(cluster_atm_idx, atmosphere_palette, setting_count)
            else {
                continue;
            };
            let w = single(dist) / total;
            if let Some(s) = sky_atm.atmosphere_settings.get_mut(idx) {
                s.weight += w;
                if lock {
                    // Binary on the starting cluster's setting.
                    if c == start {
                        s.effect_weight = 1.0;
                    }
                } else {
                    s.effect_weight += w;
                }
            }
        }
    }
}

/// `c_atmosphere_fog_interface::get_atmosphere_setting @ 0x1803AFBA0`
/// resolved to the destination index in `sky_atm.atmosphere_settings[]`.
///
/// Engine chain:
/// 1. `cluster.atmosphere_index` (i8) — `-1` ⇒ use
///    `atmosphere_settings[0]` (the global default).
/// 2. Otherwise index into `bsp.atmosphere_palette`. Out-of-range ⇒
///    `atmosphere_settings[0]`.
/// 3. The palette entry's `atmosphere_setting_index` (i16) is the
///    destination. `-1` or OOB ⇒ `atmosphere_settings[0]`.
///
/// Returns `None` only when `sky_atm.atmosphere_settings` is empty.
pub(crate) fn resolve_atmosphere_setting_index(
    cluster_atmosphere_index: i8,
    atmosphere_palette: &[BspAtmospherePaletteEntry],
    setting_count: usize,
) -> Option<usize> {
    if setting_count == 0 {
        return None;
    }
    if cluster_atmosphere_index < 0 {
        return Some(0);
    }
    let palette_idx = cluster_atmosphere_index as usize;
    let Some(palette_entry) = atmosphere_palette.get(palette_idx) else {
        eprintln!(
            "[atmosphere] cluster atmosphere palette index {palette_idx} out of range \
             ({} entries) — falling back to setting 0",
            atmosphere_palette.len(),
        );
        return Some(0);
    };
    if palette_entry.atmosphere_setting_index < 0 {
        return Some(0);
    }
    let setting_idx = palette_entry.atmosphere_setting_index as usize;
    if setting_idx >= setting_count {
        eprintln!(
            "[atmosphere] palette setting index {setting_idx} out of range \
             ({setting_count} settings) — falling back to setting 0",
        );
        Some(0)
    } else {
        Some(setting_idx)
    }
}

// ---------------------------------------------------------------------------
// Atmosphere / spectral helpers (formerly `env_probe_pass.rs`)
// ---------------------------------------------------------------------------
//
// `GpuAtmosphereData` (the runtime cbuffer mirroring Halo's
// `c_atmosphere_fog_interface::set_constant` packing), `GpuSkyParams`, and
// `precompute_betas` (spectral integration of Rayleigh + Mie scattering
// coefficients). These originally lived alongside the `EnvProbePass`
// cubemap renderer (a protomorph-only fabrication, no engine analog). The
// pass was removed in the 2026-05-09 audit; these helpers are kept because
// patchy fog and the per-frame sky uniform pipeline depend on them.

/// GPU atmosphere cbuffer — mirrors Halo's
/// `c_atmosphere_fog_interface::set_constant` 7-vec4 packing
/// (cbuffer slots 0x29 VS / 0x3F PS) with extras for our screen-space
/// sky shader (sun disc, horizon fade, etc.).
///
/// All `beta_*` fields are the RUNTIME-DERIVED scattering coefficients.
/// In Halo, `c_atmosphere_fog_interface::precompute_scattering_coefficients
/// @ 0x1803aea00` does spectral integration over 300-800nm using
/// `k_spectrum` (solar irradiance) + `k_n2_1Amplitudes` (Mie aerosol)
/// at the channel reference wavelengths (R=650nm, G=570nm, B=475nm)
/// then multiplies by `multiplier × 1000`. We approximate this with
/// 1/λ⁴ for Rayleigh and broadband for Mie — visual approximation,
/// not bit-exact.
/// Each slot is a packed [f32; 4] so vec3+scalar combinations don't
/// hit Rust vs WGSL alignment mismatches. WGSL side declares as
/// `vec4<f32>` and unpacks into vec3+scalar accessors. 11 slots × 16 = 176B.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuAtmosphereData {
    // Halo cbuffer slot 0: (sun_dir.xyz, distance_bias)
    // Per atmosphere_fx.hlsl line 13/20: SUN_DIR + DIST_BIAS share slot 0.
    pub slot0_sun_dir_dist_bias: [f32; 4],
    // Halo cbuffer slot 1: (sun_intensity_normalized.xyz, max_fog_thickness)
    // slot1.w doubles as the ATMOSPHERE_ENABLE flag (negative = disabled,
    // line 21-22 of the HLSL). Caller stores -1 to disable.
    //   sun_intensity_normalized = sun_intensity / (β_m + β_p) per channel
    //   — Halo's SUN_INTENSITY_OVER_TR_PLUS_TM normalization.
    pub slot1_sun_int_norm_thickness: [f32; 4],
    // Halo cbuffer slot 2: (β_m × log2(e) per channel, mie_g + 1)
    pub slot2_beta_m_log2e_g1: [f32; 4],
    // Halo cbuffer slot 3: (β_p × log2(e) per channel, reference_height)
    pub slot3_beta_p_log2e_refh: [f32; 4],
    // Halo cbuffer slot 4: (β_m_angular per channel, mie_height_scale)
    pub slot4_beta_m_angular_mieh: [f32; 4],
    // Halo cbuffer slot 5: ((1-g²) × β_p_angular per channel, rayleigh_height_scale)
    pub slot5_beta_p_angular_rayh: [f32; 4],
    // Halo cbuffer slot 6: (2g, padding × 3)
    pub slot6_g2: [f32; 4],
    // Sky-pass extras (sun disc + horizon — for the screen sky shader).
    pub slot7_sun_disc: [f32; 4],     // (luminance, angular_radius, edge_softness, disc_intensity)
    pub slot8_sun_glow: [f32; 4],     // (inner_glow, air_mass_scale, zenith_factor, pad)
    pub slot9_sun_tint_horizon: [f32; 4], // (tint.xyz, horizon_fade_start)
    pub slot10_horizon_pad: [f32; 4],     // (horizon_fade_end, pad, pad, pad)
}

/// Faithful port of dllcache `c_atmosphere_fog_interface::precompute_scattering_coefficients`
/// @ 0x1803ae9c0. Computes Rayleigh+Mie extinction and angular prefix
/// coefficients per RGB channel by sampling spectral curves at the
/// channel reference wavelengths (650/570/475 nm).
///
/// The spectral sources (`K_SPECTRUM`, `K_N2_1AMPLITUDES`) are
/// extracted from `halo3_dllcache_play.dll` — see
/// [`crate::halo::render::spectral_constants`]. Returns
/// `((β_m, β_p), (β_m_angular_prefix, β_p_angular_prefix))` — the
/// four 3-vector coefficient sets Halo stores on the in-memory
/// `c_atmosphere_setting` and uploads to the engine cbuffer.
///
/// The dllcache function builds 5nm-step curves from 300nm to 800nm
/// (resampling the n2/solar tables via the Catmull-Rom
/// `c_spectral_curve::get_value`), then samples them at 650/570/475nm.
/// We evaluate the formulas directly at those three wavelengths and use
/// the same Catmull-Rom interpolation ([`catmull_curve`]) for the n2/solar
/// lookups. 650/570 land on the 10nm n2 grid and all three land on the 1nm
/// solar grid (cubic → control point exactly); only the **blue 475nm n2
/// term** is a genuine cubic interpolation. Byte-faithful to the engine.
/// Sample a uniformly-spaced curve exactly like engine
/// `c_spectral_curve::get_value @ 0x180517410`: a 0.5-weighted **Catmull-Rom
/// cubic** for interior samples, falling back to **linear** only at the curve
/// ends (and clamping past the endpoints). `step_nm` is the wavelength step;
/// the curve covers 300..800nm.
///
/// Used by [`precompute_betas`] for the n2/solar lookups. The engine resamples
/// those tables onto a 5nm grid via this cubic before sampling at
/// 650/570/475nm; of those, only the **blue 475nm n2 term** is off the 10nm
/// K_N2 grid, so cubic-vs-linear changes only that one lookup (everything else
/// lands on a knot where Catmull-Rom returns the control point exactly).
pub(crate) fn catmull_curve(table: &[f32], step_nm: f32, lambda_nm: f32) -> f32 {
    let n = table.len();
    // Engine end clamps (`wave <= p[0].x` / `wave >= p[n-1].x`).
    if lambda_nm <= 300.0 {
        return table[0];
    }
    let last_x = 300.0 + step_nm * (n as f32 - 1.0);
    if lambda_nm >= last_x {
        return table[n - 1];
    }
    let idx = (lambda_nm - 300.0) / step_nm;
    let i = idx.floor() as usize; // p[i].x <= lambda < p[i+1].x
    let t = idx - i as f32;
    // Engine: linear at the ends (`i <= 0 || i >= n-2`), cubic interior.
    if i == 0 || i >= n - 2 {
        return table[i] * (1.0 - t) + table[i + 1] * t;
    }
    let (m1, p0, p1, m2) = (table[i - 1], table[i], table[i + 1], table[i + 2]);
    let t2 = t * t;
    let t3 = t2 * t;
    // Catmull-Rom basis, exactly as decoded from the get_value asm.
    let w_m1 = 0.5 * (-t3 + 2.0 * t2 - t);
    let w_p0 = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0);
    let w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t);
    let w_m2 = 0.5 * (t3 - t2);
    m1 * w_m1 + p0 * w_p0 + p1 * w_p1 + m2 * w_m2
}

pub(crate) fn precompute_betas(rayleigh_multiplier: f32, mie_multiplier: f32, desaturation: f32)
    -> (([f32; 3], [f32; 3]), ([f32; 3], [f32; 3]))
{
    use crate::halo::render::spectral_constants::{K_N2_1AMPLITUDES, K_SPECTRUM};

    /// Evaluate the four β-curve formulas at one wavelength. Mirrors
    /// the loop body of `precompute_scattering_coefficients` exactly.
    fn beta_at(lambda_nm: f32) -> [f32; 4] {
        let lambda_m = lambda_nm * 1.0e-9;
        // K_SPECTRUM is sampled in 1nm steps (501 points / 500nm range).
        // K_N2_1AMPLITUDES is sampled in 10nm steps (51 points).
        let n2 = catmull_curve(&K_N2_1AMPLITUDES, 10.0, lambda_nm);
        let solar = catmull_curve(&K_SPECTRUM, 1.0, lambda_nm);

        // Rayleigh denominator: 3 × N × λ⁴ where N = 2.545e25 (number
        // density of air molecules at STP, per dllcache constant).
        let lambda4 = lambda_m * lambda_m * lambda_m * lambda_m;
        let inv_ray = 1.0 / (3.0 * 2.545e25_f32 * lambda4);

        // Mie wavenumber-squared: (2π/λ)².
        let two_pi = 2.0 * std::f32::consts::PI;
        let k = two_pi / lambda_m;
        let k_squared = k * k;

        // Constants from the dllcache decompile:
        //   248.05023, 1.0599999 — Rayleigh extinction prefactor
        //   8.9694953e-17        — Mie extinction prefactor (solar-weighted)
        //   1.4275395e-17        — Mie angular prefactor
        //   19.73921, 1.0599999, 0.7629  — Rayleigh angular prefactor
        //   0.01                  — common Mie scale
        let beta_m       = (n2 * 248.05023 * 1.0599999) * inv_ray;
        let beta_p       = (k_squared * 8.9694953e-17 * solar) * 0.01;
        let beta_p_ang   = (k_squared * 1.4275395e-17) * 0.01;
        let beta_m_ang   = (n2 * 19.73921 * 1.0599999 * 0.7629) * inv_ray;

        [beta_m, beta_p, beta_m_ang, beta_p_ang]
    }

    let r = beta_at(650.0);
    let g = beta_at(570.0);
    let b = beta_at(475.0);

    // Multiply by user multipliers × 1000 (per dllcache: each output is
    // `curve_value * multiplier * 1000.0`).
    let mut beta_m         = [r[0]*rayleigh_multiplier*1000.0, g[0]*rayleigh_multiplier*1000.0, b[0]*rayleigh_multiplier*1000.0];
    let mut beta_p         = [r[1]*mie_multiplier*1000.0,      g[1]*mie_multiplier*1000.0,      b[1]*mie_multiplier*1000.0];
    let mut beta_m_angular = [r[2]*rayleigh_multiplier*1000.0, g[2]*rayleigh_multiplier*1000.0, b[2]*rayleigh_multiplier*1000.0];
    let mut beta_p_angular = [r[3]*mie_multiplier*1000.0,      g[3]*mie_multiplier*1000.0,      b[3]*mie_multiplier*1000.0];

    // Desaturation pass (dllcache lines 124+): blend each channel
    // toward the mean by `desaturation`. v25 = mean × desat;
    // ch_new = (1 - desat) × ch + v25.
    if desaturation > 0.0 {
        let blend = |arr: &mut [f32; 3]| {
            let mean = (arr[0] + arr[1] + arr[2]) * (1.0 / 3.0);
            let v25 = mean * desaturation;
            for v in arr.iter_mut() {
                *v = (1.0 - desaturation) * *v + v25;
            }
        };
        blend(&mut beta_m);
        blend(&mut beta_p);
        blend(&mut beta_m_angular);
        blend(&mut beta_p_angular);
    }

    ((beta_m, beta_p), (beta_m_angular, beta_p_angular))
}

/// `c_color_xyY` — CIE xyY chromaticity + luminance. Mirrors engine
/// struct used by `convert_RGB_to_xyY @ 0x1803B0780` and friends.
#[allow(non_snake_case)]
#[derive(Copy, Clone, Default)]
struct ColorXyY {
    x: f32,
    y: f32,
    Y: f32,
}

/// `convert_RGB_to_xyY @ 0x1803B0780` — verbatim port. sRGB-D65 RGB →
/// CIE XYZ → xyY (x, y from chromaticity; Y = the XYZ Y component).
#[allow(non_snake_case)]
fn convert_rgb_to_xyy(rgb: blam_tags::math::RealRgbColor) -> ColorXyY {
    let G = rgb.green;
    let B = rgb.blue;
    let v4 = (rgb.red * 0.412424)     + (G * 0.35757899)   + (B * 0.180464);
    let v5 = (rgb.red * 0.21265601)   + (G * 0.71515799)   + (B * 0.072185598);
    let v6 = (v5 + v4)
        + ((rgb.red * 0.0193324) + (G * 0.119193) + (B * 0.95044398));
    if v6 == 0.0 {
        ColorXyY { x: 0.0, y: 0.0, Y: 0.0 }
    } else {
        let v7 = 1.0 / v6;
        ColorXyY { x: v7 * v4, y: v7 * v5, Y: v5 }
    }
}

/// Scenario-sky sun pulled from `get_sun_constants_from_sky @ 0x1803ADCB0`
/// (sky model's `lightgen_lights[last]`). Threaded into the non-override
/// branch of [`get_sun_parameters`]. `intensity` is linear RGB already
/// scaled by `solid_angle × 0.2 × g_render_light_intensity` in the loader;
/// `direction` is the z-up unit vector read verbatim from the lightgen light.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SkySun {
    pub intensity: [f32; 3],
    pub direction: [f32; 3],
}

/// `c_atmosphere_fog_interface::get_sun_parameters @ 0x1803AF990` —
/// verbatim port of the `flags & 2` ("Override Real Sun Values") branch.
/// Writes `sun_intensity` (linear RGB) and `sun_direction` (z-up unit
/// vector). The non-override branch is the engine's
/// `get_sun_constants_from_sky` path (sky model's `lightgen_lights[last]`
/// computed offline) — protomorph doesn't have that data yet, so the
/// caller passes `sky_sun` as a fallback for that branch.
#[allow(non_snake_case)]
pub(crate) fn get_sun_parameters(
    parameters: &blam_tags::sky_atmosphere::AtmosphereSettings,
    sky_sun: Option<SkySun>,
) -> ([f32; 3], [f32; 3]) {
    use blam_tags::sky_atmosphere::AtmosphereFlags;
    if parameters.flags.contains(AtmosphereFlags::OverrideRealSunValues) {
        // override_color = parameters->m_dominant_light_color;
        let override_color = parameters.color;
        // convert_RGB_to_xyY(&override_color, &temp);
        let temp = convert_rgb_to_xyy(override_color);
        let mut m_dominant_light_intensity = parameters.intensity;
        let mut v10 = 0.0_f32;
        let v11;
        if temp.y == 0.0 {
            m_dominant_light_intensity = 0.0;
            v11 = 0.0;
        } else {
            v10 = (m_dominant_light_intensity / temp.y) * temp.x;
            v11 = ((1.0 - temp.x) - temp.y) * (m_dominant_light_intensity / temp.y);
        }
        // sun_intensity (linear RGB) — sRGB-D65 inverse matrix applied to (v10, intensity, v11).
        let sun_intensity = [
            (v10 *  3.240479)   - (m_dominant_light_intensity * 1.53715)   - (v11 * 0.49853501),
            (m_dominant_light_intensity *  1.875991) - (v10 * 0.96925598)  + (v11 * 0.041556001),
            (v10 *  0.055647999) - (m_dominant_light_intensity * 0.204043) + (v11 * 1.0573111),
        ];
        // sun_direction from (m_dominant_light_phi, m_dominant_light_theta).
        // Tag schema names them "Heading [0..360]" (φ) and "Pitch [0..90]" (θ).
        // 0.0055555557 ≈ 1/180; the full factor is (deg × π / 180).
        let phi_rad   = (0.0055555557_f32 * parameters.sun_heading) * std::f32::consts::PI;
        let theta_rad = (0.0055555557_f32 * parameters.sun_pitch)   * std::f32::consts::PI;
        let v12 = phi_rad.cos();
        let v13 = phi_rad.sin();
        let sun_direction = [
            v12 * theta_rad.sin(),
            v13 * theta_rad.sin(),
            theta_rad.cos(),
        ];
        (sun_intensity, sun_direction)
    } else {
        // Engine `get_sun_constants_from_sky @ 0x1803ADCB0` (non-override
        // branch): both intensity and direction come from the sky model's
        // `lightgen_lights[last]` — intensity already folded as
        // `×solid_angle ×0.2 ×g_render_light_intensity` in the loader,
        // direction read verbatim. When the sky has no lightgen lights the
        // engine defaults to a white sun pointing straight down
        // (`sun_intensity=(1,1,1)`, `*sun_direction=global_down3d=(0,0,-1)`).
        match sky_sun {
            Some(s) => (s.intensity, s.direction),
            None => {
                use std::sync::atomic::{AtomicBool, Ordering};
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[atmosphere] sky has no lightgen lights — using engine default \
                         white sun pointing down (warned once)",
                    );
                }
                ([1.0, 1.0, 1.0], [0.0, 0.0, -1.0])
            }
        }
    }
}

impl GpuAtmosphereData {
    /// Build from a `blam_tags::sky_atmosphere::SkyAtmosphere` walked
    /// from the scenario's `atmospheric` tag-ref. Picks the primary
    /// (first-enabled) atmosphere setting per
    /// `SkyAtmosphere::primary_setting`. Mirrors Halo's
    /// `populate_atmosphere_parameters` + `set_constant` chain:
    /// derives β_* via `precompute_betas` (spectral approximation),
    /// then packs the 7-vec4 cbuffer Halo expects (with the
    /// `sun_intensity / (β_m+β_p)` normalization).
    pub fn from_sky_atmosphere_with_exposure(
        atm: &blam_tags::sky_atmosphere::SkyAtmosphere,
        view_exposure: f32,
        sky_sun: Option<SkySun>,
    ) -> Self {
        match atm.primary_setting() {
            None => Self::neutral(),
            Some(s) => Self::from_atmosphere_setting(s, view_exposure, sky_sun),
        }
    }

}

/// Find the BSP cluster containing `eye`, by AABB containment test.
/// Falls back to nearest-cluster-by-center when no AABB contains the
/// point. Mirrors the protomorph decorator path's `find_cluster`. Engine
/// uses portal-aware BFS for the full multi-cluster blend; for the MP
/// fast-path the starting cluster is sufficient.
pub(crate) fn find_cluster_for_eye(
    clusters: &[blam_tags::structure_bsp::BspCluster],
    eye: glam::Vec3,
) -> Option<usize> {
    if clusters.is_empty() {
        return None;
    }
    for (i, c) in clusters.iter().enumerate() {
        if c.bounds_x.contains(eye.x)
            && c.bounds_y.contains(eye.y)
            && c.bounds_z.contains(eye.z)
        {
            return Some(i);
        }
    }
    let mut best = (f32::INFINITY, 0usize);
    for (i, c) in clusters.iter().enumerate() {
        let cx = (c.bounds_x.lower + c.bounds_x.upper) * 0.5;
        let cy = (c.bounds_y.lower + c.bounds_y.upper) * 0.5;
        let cz = (c.bounds_z.lower + c.bounds_z.upper) * 0.5;
        let d2 = (eye - glam::Vec3::new(cx, cy, cz)).length_squared();
        if d2 < best.0 {
            best = (d2, i);
        }
    }
    Some(best.1)
}

impl GpuAtmosphereData {

    /// Build atmosphere cbuffer data from a single `AtmosphereSettings`
    /// element. Used both for primary-setting selection and per-setting
    /// diagnostic dumps.
    ///
    /// `sky_sun` is the engine's `get_sun_constants_from_sky @
    /// 0x1803adcb0` result for the active scenario sky — i.e.,
    /// `lightgen_lights[last].intensity × solid_angle × 0.2 ×
    /// g_render_light_intensity`. The engine consumes this WHEN the
    /// atmosphere setting's flag bit 1 ("Override Real Sun Values") is
    /// UNSET. When bit 1 IS set, the engine uses the atmosphere's own
    /// `dominant_light_color × dominant_light_intensity` (with an
    /// xyY→XYZ→linear-RGB chromaticity-preserving conversion that we
    /// approximate as direct multiply for v1).
    pub fn from_atmosphere_setting(
        s: &blam_tags::sky_atmosphere::AtmosphereSettings,
        view_exposure: f32,
        sky_sun: Option<SkySun>,
    ) -> Self {
        let ((beta_m, beta_p), (beta_m_angular, beta_p_angular)) =
            precompute_betas(s.rayleigh_multiplier, s.mie_multiplier, s.desaturation);

        let log2e = 1.442695_f32;
        // Engine `c_atmosphere_fog_interface::get_sun_parameters @
        // 0x1803AF990` produces both `sun_intensity` and `sun_direction`
        // from the setting (override-real-sun branch reads the
        // atmosphere's own `m_dominant_light_*` fields).
        let (sun_int, sun_dir) = get_sun_parameters(s, sky_sun);
        let denom = [
            (beta_m[0] + beta_p[0]).max(1e-6),
            (beta_m[1] + beta_p[1]).max(1e-6),
            (beta_m[2] + beta_p[2]).max(1e-6),
        ];
        let g = s.sun_phase_function;
        let one_minus_g2 = 1.0 - g * g;

        // Halo's slot1.w doubles as the disable flag — engine
        // `set_constant @ 0x1803AE530` writes `max(max_fog_thickness,
        // 0.1)` when atmosphere is enabled (the 0.1 floor prevents
        // authored-zero `max_fog_thickness` from being mis-read as
        // disable). When NOT enabled, engine writes -1 to disable.
        // Engine disable path (`disable_atmosphere @ 0x1803AEFD0`) writes a
        // single vec4 of ALL `-1` to slot1 — not just `.w`. Match it: when the
        // setting is disabled, slot1 = [-1; 4] (the shader reads `.w < 0` as the
        // disable flag; the other lanes are -1 for byte-fidelity).
        if !s.is_enabled() {
            return Self {
                slot1_sun_int_norm_thickness: [-1.0; 4],
                ..Self::neutral()
            };
        }
        let thickness_or_disable = s.max_fog_thickness.max(0.1);
        Self {
            slot0_sun_dir_dist_bias: [sun_dir[0], sun_dir[1], sun_dir[2], s.distance_bias],
            // sun_int_norm = sun_intensity / (β_m + β_p) — Halo's
            // `SUN_INTENSITY_OVER_TR_PLUS_TM`.
            slot1_sun_int_norm_thickness: [sun_int[0]/denom[0], sun_int[1]/denom[1], sun_int[2]/denom[2], thickness_or_disable],
            slot2_beta_m_log2e_g1: [beta_m[0]*log2e, beta_m[1]*log2e, beta_m[2]*log2e, 1.0 + g],
            slot3_beta_p_log2e_refh: [beta_p[0]*log2e, beta_p[1]*log2e, beta_p[2]*log2e, s.sea_level],
            slot4_beta_m_angular_mieh: [beta_m_angular[0], beta_m_angular[1], beta_m_angular[2], s.mie_height_scale],
            slot5_beta_p_angular_rayh: [beta_p_angular[0]*one_minus_g2, beta_p_angular[1]*one_minus_g2, beta_p_angular[2]*one_minus_g2, s.rayleigh_height_scale],
            // Engine `set_constant` broadcasts `2g` to all 4 lanes
            // (`_mm_shuffle_ps(2g, 2g, 0)`).
            slot6_g2: [2.0*g; 4],
            // slot7.x = view_exposure (Halo's `g_exposure.r`).
            // slot7.y/z/w = sun disc params (angular_radius, edge_softness, disc_intensity).
            slot7_sun_disc: [view_exposure, 0.0093, 0.001, s.intensity],
            slot8_sun_glow: [0.5, 1.0, 1.0, 0.0],
            slot9_sun_tint_horizon: [s.color.red, s.color.green, s.color.blue, -0.1],
            slot10_horizon_pad: [0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Phase 5 — build cbuffer from the engine-faithful accumulated
    /// [`WeightedAtmosphereParameters`] instead of a single picked
    /// `AtmosphereSettings`. Slots 0-6 are populated per
    /// `c_atmosphere_fog_interface::set_constant @ 0x1803AE530`:
    /// ```text
    /// slot 0 = (normalize3d_with_default(sun_dir, global_down3d), distance_bias)
    /// slot 1 = (sun_intensity / (β_m + β_p) per channel, max(thickness, 0.1))
    /// slot 2 = (β_m × log2(e), heyey_greenstein + 1)
    /// slot 3 = (β_p × log2(e), reference_datum_plane)
    /// slot 4 = (β_m_angular, mie_height)
    /// slot 5 = (β_p_angular × (1 - g²), rayleigh_height)
    /// slot 6 = (2g, 0, 0, 0)        // engine broadcasts; we pack .x
    /// ```
    /// `params.atmosphere_enabled == 0` ⇒ disable path: only slot1.w =
    /// −1 set (engine's `disable_atmosphere`). Rest neutral.
    ///
    /// Slots 7-10 (sun disc, glow, tint, horizon — protomorph sky-pass
    /// extras, not engine `set_constant` output) source from
    /// `sky_extras_source` when provided. Phase 5.3 will move these onto
    /// a dedicated sky-pass cbuffer per engine separation.
    pub fn from_weighted_parameters(
        params: &WeightedAtmosphereParameters,
        sky_extras_source: Option<&blam_tags::sky_atmosphere::AtmosphereSettings>,
        view_exposure: f32,
    ) -> Self {
        if params.atmosphere_enabled == 0 {
            return Self::neutral();
        }

        // Engine `normalize3d_with_default(sun_direction, global_down3d)`.
        let len_sq = params.sun_direction.i * params.sun_direction.i
            + params.sun_direction.j * params.sun_direction.j
            + params.sun_direction.k * params.sun_direction.k;
        let sun = if len_sq > 1e-12 {
            let inv_len = len_sq.sqrt().recip();
            [
                params.sun_direction.i * inv_len,
                params.sun_direction.j * inv_len,
                params.sun_direction.k * inv_len,
            ]
        } else {
            // `global_down3d` per engine.
            [0.0, 0.0, -1.0]
        };

        let log2e = 1.442695_f32;
        let g = params.heyey_greenstein;
        let one_minus_g2 = 1.0 - g * g;

        // Engine guards: `if (β_m.x + β_p.x <= 0.0001 || β_m.y + β_p.y
        // <= 0.0001 || β_m.z + β_p.z <= 0.0001) sun_int_norm = 0; else
        // sun_int_norm = sun_intensity / (β_m + β_p)`.
        let sum_x = params.beta_m.i + params.beta_p.i;
        let sum_y = params.beta_m.j + params.beta_p.j;
        let sum_z = params.beta_m.k + params.beta_p.k;
        let sun_int_norm = if sum_x <= 1.0e-4 || sum_y <= 1.0e-4 || sum_z <= 1.0e-4 {
            [0.0_f32, 0.0, 0.0]
        } else {
            [
                params.sun_intensity.red / sum_x,
                params.sun_intensity.green / sum_y,
                params.sun_intensity.blue / sum_z,
            ]
        };

        // Engine `slot1.w = max(max_fog_thickness, 0.1)`.
        let thickness = params.max_fog_thickness.max(0.1);

        // Sky extras — caller picks the dominant setting and passes it
        // through. Neutral fallback when no setting is available.
        let (sun_disc_intensity, sun_color) = match sky_extras_source {
            Some(s) => (s.intensity, [s.color.red, s.color.green, s.color.blue]),
            None => (1.0, [1.0, 1.0, 1.0]),
        };

        Self {
            slot0_sun_dir_dist_bias: [sun[0], sun[1], sun[2], params.distance_bias],
            slot1_sun_int_norm_thickness: [sun_int_norm[0], sun_int_norm[1], sun_int_norm[2], thickness],
            slot2_beta_m_log2e_g1: [
                params.beta_m.i * log2e,
                params.beta_m.j * log2e,
                params.beta_m.k * log2e,
                1.0 + g,
            ],
            slot3_beta_p_log2e_refh: [
                params.beta_p.i * log2e,
                params.beta_p.j * log2e,
                params.beta_p.k * log2e,
                params.reference_datum_plane,
            ],
            slot4_beta_m_angular_mieh: [
                params.beta_m_angular.i,
                params.beta_m_angular.j,
                params.beta_m_angular.k,
                params.mie_height,
            ],
            slot5_beta_p_angular_rayh: [
                params.beta_p_angular.i * one_minus_g2,
                params.beta_p_angular.j * one_minus_g2,
                params.beta_p_angular.k * one_minus_g2,
                params.rayleigh_height,
            ],
            // Engine broadcasts `2g` to all 4 lanes (`_mm_shuffle_ps(2g,2g,0)`).
            slot6_g2: [2.0 * g; 4],
            slot7_sun_disc: [view_exposure, 0.0093, 0.001, sun_disc_intensity],
            slot8_sun_glow: [0.5, 1.0, 1.0, 0.0],
            slot9_sun_tint_horizon: [sun_color[0], sun_color[1], sun_color[2], -0.1],
            slot10_horizon_pad: [0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn neutral() -> Self {
        Self {
            // Disable flag in slot1 — atmosphere off. Engine
            // `disable_atmosphere @ 0x1803AEFD0` writes vec4(-1,-1,-1,-1) to
            // slot1; the shader reads `.w < 0` as the disable flag.
            slot0_sun_dir_dist_bias: [0.0, 0.0, 1.0, 0.0],
            slot1_sun_int_norm_thickness: [-1.0; 4],
            slot2_beta_m_log2e_g1: [0.0, 0.0, 0.0, 1.0],
            slot3_beta_p_log2e_refh: [0.0, 0.0, 0.0, 0.0],
            slot4_beta_m_angular_mieh: [0.0, 0.0, 0.0, 1.0],
            slot5_beta_p_angular_rayh: [0.0, 0.0, 0.0, 1.0],
            slot6_g2: [0.0, 0.0, 0.0, 0.0],
            slot7_sun_disc: [1.0, 0.0, 0.0, 0.0],
            slot8_sun_glow: [0.0, 0.0, 1.0, 0.0],
            slot9_sun_tint_horizon: [1.0, 1.0, 1.0, 0.0],
            slot10_horizon_pad: [0.0, 0.0, 0.0, 0.0],
        }
    }

}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSkyParams {
    pub inverse_view_projection: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub _pad: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::sky_atmosphere::AtmosphereSettings;
    use blam_tags::Flags;

    fn make_sky_atm(n: usize) -> SkyAtmosphere {
        let mut s = SkyAtmosphere::default();
        for i in 0..n {
            let mut setting = AtmosphereSettings::default();
            setting.flags = Flags::from_slice(&[AtmosphereFlags::EnableAtmosphere]);
            setting.name = format!("setting_{}", i);
            s.atmosphere_settings.push(setting);
        }
        s
    }

    fn make_palette(entries: &[(i16, &str)]) -> Vec<BspAtmospherePaletteEntry> {
        entries
            .iter()
            .map(|(idx, name)| BspAtmospherePaletteEntry {
                name: (*name).to_string(),
                atmosphere_setting_index: *idx,
            })
            .collect()
    }

    #[test]
    fn resolve_negative_cluster_index_falls_back_to_setting_zero() {
        let palette = make_palette(&[(1, "haze"), (2, "skydome")]);
        let idx = resolve_atmosphere_setting_index(-1, &palette, 3);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn resolve_palette_pickup() {
        let palette = make_palette(&[(1, "haze"), (2, "skydome")]);
        // cluster.atmosphere_index = 1 → palette[1].atmosphere_setting_index = 2
        let idx = resolve_atmosphere_setting_index(1, &palette, 3);
        assert_eq!(idx, Some(2));
    }

    #[test]
    fn resolve_palette_oob_falls_back_to_zero() {
        let palette = make_palette(&[(1, "haze")]);
        let idx = resolve_atmosphere_setting_index(5, &palette, 3);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn resolve_negative_setting_index_falls_back_to_zero() {
        let palette = make_palette(&[(-1, "fallback"), (2, "skydome")]);
        let idx = resolve_atmosphere_setting_index(0, &palette, 3);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn resolve_empty_settings_returns_none() {
        let palette = make_palette(&[(0, "x")]);
        let idx = resolve_atmosphere_setting_index(0, &palette, 0);
        assert!(idx.is_none());
    }

    #[test]
    fn compute_cluster_weights_mp_path_sets_starting_to_one() {
        let mut interface = AtmosphereFogInterface::default();
        let mut sky_atm = make_sky_atm(3);
        let palette = make_palette(&[(2, "haze_skydome")]);
        interface.compute_cluster_weights(
            &mut sky_atm,
            &palette,
            &[],
            &[],
            &[],
            ClusterReference { bsp_index: 0, cluster_index: 5 },
            0, // cluster.atmosphere_index = 0 → palette[0].setting_index = 2
            glam::Vec3::ZERO,
        );
        assert_eq!(sky_atm.atmosphere_settings[0].weight, 0.0);
        assert_eq!(sky_atm.atmosphere_settings[1].weight, 0.0);
        assert_eq!(sky_atm.atmosphere_settings[2].weight, 1.0);
        assert_eq!(sky_atm.atmosphere_settings[2].effect_weight, 1.0);
    }

    #[test]
    fn compute_cluster_weights_invalid_cluster_zeroes_all() {
        let mut interface = AtmosphereFogInterface::default();
        let mut sky_atm = make_sky_atm(2);
        // Pre-seed a stale weight to verify clear.
        sky_atm.atmosphere_settings[1].weight = 0.7;
        interface.compute_cluster_weights(
            &mut sky_atm,
            &[],
            &[],
            &[],
            &[],
            ClusterReference::NONE,
            -1,
            glam::Vec3::ZERO,
        );
        assert_eq!(sky_atm.atmosphere_settings[0].weight, 0.0);
        assert_eq!(sky_atm.atmosphere_settings[1].weight, 0.0);
    }

    fn make_enabled_setting_with_betas() -> AtmosphereSettings {
        // Authored values from guardian's first atmosphere setting (per
        // [[reference_riverworld_atmosphere_settings]] family). Picked
        // for non-zero outputs across all accumulator fields.
        let mut s = AtmosphereSettings::default();
        s.flags = Flags::from_slice(&[
            AtmosphereFlags::EnableAtmosphere,
            AtmosphereFlags::OverrideRealSunValues,
            AtmosphereFlags::PatchyFog,
        ]);
        s.sun_pitch = 30.0;
        s.sun_heading = 45.0;
        s.color = RealRgbColor { red: 1.0, green: 0.9, blue: 0.8 };
        s.intensity = 1.5;
        s.sea_level = -10.0;
        s.rayleigh_height_scale = 150.0;
        s.mie_height_scale = 6.0;
        s.rayleigh_multiplier = 0.05;
        s.mie_multiplier = 0.025;
        s.sun_phase_function = 0.2;
        s.desaturation = 0.0;
        s.distance_bias = 0.5;
        s.max_fog_thickness = 1000.0;
        s.patchy_fog_density = 500.0;
        s.full_intensity_height = 18.5;
        s.half_intensity_height = 25.0;
        s.wind_direction = RealVector3d { i: 1.0, j: 0.0, k: 0.0 };
        s
    }

    #[test]
    fn accumulate_disabled_setting_is_noop() {
        let interface = AtmosphereFogInterface::default();
        let mut s = make_enabled_setting_with_betas();
        s.flags = Flags::default(); // disabled
        let mut accum = WeightedAtmosphereParameters::default();
        interface.accumulate_atmosphere_settings(&s, &mut accum, 1.0, None);
        assert_eq!(accum.atmosphere_enabled, 0);
        assert_eq!(accum.distance_bias, 0.0);
        assert_eq!(accum.max_fog_thickness, 0.0);
    }

    #[test]
    fn accumulate_zero_weight_keeps_enabled_zero() {
        let interface = AtmosphereFogInterface::default();
        let s = make_enabled_setting_with_betas();
        let mut accum = WeightedAtmosphereParameters::default();
        interface.accumulate_atmosphere_settings(&s, &mut accum, 0.0, None);
        // Engine `if (weight > 0.0) atmosphere_enabled = 1;` — weight=0
        // still runs the accumulation but doesn't flip the flag (the
        // accumulation is a no-op since weight scales every term).
        assert_eq!(accum.atmosphere_enabled, 0);
        assert_eq!(accum.distance_bias, 0.0);
    }

    #[test]
    fn accumulate_weight_one_copies_source_fields() {
        let interface = AtmosphereFogInterface::default();
        let s = make_enabled_setting_with_betas();
        let mut accum = WeightedAtmosphereParameters::default();
        interface.accumulate_atmosphere_settings(&s, &mut accum, 1.0, None);
        assert_eq!(accum.atmosphere_enabled, 1);
        assert_eq!(accum.distance_bias, s.distance_bias);
        assert_eq!(accum.max_fog_thickness, s.max_fog_thickness);
        assert_eq!(accum.heyey_greenstein, s.sun_phase_function);
        assert_eq!(accum.reference_datum_plane, s.sea_level);
        assert_eq!(accum.mie_height, s.mie_height_scale);
        assert_eq!(accum.rayleigh_height, s.rayleigh_height_scale);
        // Patchy fog fields populated since flag bit 2 set.
        assert_eq!(accum.patchy_fog_density, s.patchy_fog_density);
        assert_eq!(accum.full_intensity_height, s.full_intensity_height);
        assert_eq!(accum.half_intensity_height, s.half_intensity_height);
        assert_eq!(accum.wind_direction.i, s.wind_direction.i);
        // Sun direction should be normalized-ish but accumulated raw.
        // For weight=1 it's exactly get_sun_parameters output → non-zero.
        assert!(accum.sun_direction.k.abs() > 0.0);
        // Betas should be non-zero with positive multipliers.
        assert!(accum.beta_m.i > 0.0);
        assert!(accum.beta_p.i > 0.0);
    }

    #[test]
    fn accumulate_no_patchy_when_flag_unset() {
        let interface = AtmosphereFogInterface::default();
        let mut s = make_enabled_setting_with_betas();
        s.flags = Flags::from_slice(&[AtmosphereFlags::EnableAtmosphere]); // enable only — no patchy fog
        let mut accum = WeightedAtmosphereParameters::default();
        interface.accumulate_atmosphere_settings(&s, &mut accum, 1.0, None);
        assert_eq!(accum.patchy_fog_density, 0.0);
        assert_eq!(accum.full_intensity_height, 0.0);
        assert_eq!(accum.wind_direction.i, 0.0);
        // Non-patchy fields still accumulated.
        assert_eq!(accum.distance_bias, s.distance_bias);
    }

    #[test]
    fn accumulate_weighted_sum_two_settings() {
        let interface = AtmosphereFogInterface::default();
        let mut a = make_enabled_setting_with_betas();
        a.distance_bias = 1.0;
        a.flags = Flags::from_slice(&[AtmosphereFlags::EnableAtmosphere]); // no patchy on a
        let mut b = make_enabled_setting_with_betas();
        b.distance_bias = 3.0;
        b.flags = Flags::from_slice(&[AtmosphereFlags::EnableAtmosphere]);
        let mut accum = WeightedAtmosphereParameters::default();
        interface.accumulate_atmosphere_settings(&a, &mut accum, 0.25, None);
        interface.accumulate_atmosphere_settings(&b, &mut accum, 0.75, None);
        // 0.25 × 1.0 + 0.75 × 3.0 = 2.5
        assert!((accum.distance_bias - 2.5).abs() < 1e-5);
    }

    #[test]
    fn populate_calls_accumulate_for_each_setting() {
        let interface = AtmosphereFogInterface::default();
        let mut sky_atm = SkyAtmosphere::default();
        let mut a = make_enabled_setting_with_betas();
        a.weight = 1.0;
        a.distance_bias = 2.0;
        a.flags = Flags::from_slice(&[AtmosphereFlags::EnableAtmosphere]);
        let mut b = make_enabled_setting_with_betas();
        b.weight = 0.0; // zero-weight should contribute nothing
        b.distance_bias = 99.0;
        b.flags = Flags::from_slice(&[AtmosphereFlags::EnableAtmosphere]);
        sky_atm.atmosphere_settings.push(a);
        sky_atm.atmosphere_settings.push(b);
        let mut accum = WeightedAtmosphereParameters::default();
        interface.populate_atmosphere_parameters(&sky_atm, &mut accum, None);
        // Only `a` should contribute.
        assert!((accum.distance_bias - 2.0).abs() < 1e-5);
        assert_eq!(accum.atmosphere_enabled, 1);
    }

    #[test]
    fn compute_scattering_disabled_returns_identity() {
        let p = WeightedAtmosphereParameters::default(); // atmosphere_enabled = 0
        let (ext, isc) = p.compute_scattering(
            glam::Vec3::new(0.0, 0.0, 10.0),
            glam::Vec3::new(0.0, 0.0, 0.0),
            0.0,
        );
        assert_eq!(ext, [1.0, 1.0, 1.0]);
        assert_eq!(isc, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn compute_scattering_enabled_attenuates() {
        let mut p = WeightedAtmosphereParameters::default();
        p.atmosphere_enabled = 1;
        p.max_fog_thickness = 1000.0;
        p.mie_height = 50.0;
        p.rayleigh_height = 150.0;
        p.heyey_greenstein = 0.2;
        p.beta_m = RealVector3d { i: 0.01, j: 0.012, k: 0.02 };
        p.beta_p = RealVector3d { i: 0.005, j: 0.005, k: 0.005 };
        p.beta_m_angular = RealVector3d { i: 0.01, j: 0.012, k: 0.02 };
        p.beta_p_angular = RealVector3d { i: 0.005, j: 0.005, k: 0.005 };
        p.sun_intensity = RealRgbColor { red: 1.0, green: 1.0, blue: 1.0 };
        p.sun_direction = RealVector3d { i: 0.0, j: 0.0, k: -1.0 };
        let (ext, isc) = p.compute_scattering(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(100.0, 0.0, 5.0),
            0.0,
        );
        // Extinction in (0,1) for every channel (some fog along a 100wu ray).
        for c in ext {
            assert!(c > 0.0 && c < 1.0, "extinction channel out of range: {c}");
        }
        // Inscatter finite and non-negative.
        for c in isc {
            assert!(c.is_finite() && c >= 0.0, "bad inscatter: {c}");
        }
    }

    #[test]
    fn catmull_curve_knot_is_identity_midpoint_is_cubic() {
        // Synthetic 10nm table (like K_N2) so we control the values:
        // 300,310,...; pick a curved profile.
        let table: Vec<f32> = (0..51).map(|i| (i as f32 * 0.13).sin()).collect();
        // At an exact knot (t=0) Catmull-Rom returns the control point.
        let at_knot = catmull_curve(&table, 10.0, 350.0); // idx=5.0
        assert!((at_knot - table[5]).abs() < 1e-6);
        // At the 10nm-grid midpoint (475nm → idx 17.5) cubic must differ from
        // the plain linear average of the two bracketing knots.
        let cubic = catmull_curve(&table, 10.0, 475.0);
        let linear = 0.5 * (table[17] + table[18]);
        assert!(
            (cubic - linear).abs() > 1e-5,
            "475nm should be a genuine cubic interpolation, not linear (cubic={cubic}, linear={linear})"
        );
        // And it must equal the hand-computed Catmull-Rom at t=0.5.
        let (m1, p0, p1, m2) = (table[16], table[17], table[18], table[19]);
        let (t, t2, t3) = (0.5_f32, 0.25_f32, 0.125_f32);
        let expected = m1 * 0.5 * (-t3 + 2.0 * t2 - t)
            + p0 * 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
            + p1 * 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
            + m2 * 0.5 * (t3 - t2);
        assert!((cubic - expected).abs() < 1e-6);
    }

    #[test]
    fn cbuffer_slot6_broadcasts_2g_to_all_lanes() {
        // Engine `set_constant` does `temp_constants[6] = _mm_shuffle_ps(2g,2g,0)`
        // → all four lanes equal `2g`.
        let mut params = WeightedAtmosphereParameters::default();
        params.atmosphere_enabled = 1;
        params.heyey_greenstein = 0.3;
        // Non-zero betas so the disable/guard paths don't short-circuit.
        params.beta_m = RealVector3d { i: 1.0, j: 1.0, k: 1.0 };
        params.beta_p = RealVector3d { i: 1.0, j: 1.0, k: 1.0 };
        let cb = GpuAtmosphereData::from_weighted_parameters(&params, None, 1.0);
        assert_eq!(cb.slot6_g2, [0.6_f32; 4]);
    }

    #[test]
    fn cbuffer_disable_fills_slot1_all_negative_one() {
        // Engine `disable_atmosphere` writes vec4(-1,-1,-1,-1) to slot1.
        let n = GpuAtmosphereData::neutral();
        assert_eq!(n.slot1_sun_int_norm_thickness, [-1.0_f32; 4]);
        // A disabled single setting takes the same disable encoding.
        let mut s = make_enabled_setting_with_betas();
        s.flags = Flags::default(); // disabled
        let cb = GpuAtmosphereData::from_atmosphere_setting(&s, 1.0, None);
        assert_eq!(cb.slot1_sun_int_norm_thickness, [-1.0_f32; 4]);
    }

    #[test]
    fn compute_cluster_weights_negative_atm_index_picks_setting_zero() {
        let mut interface = AtmosphereFogInterface::default();
        let mut sky_atm = make_sky_atm(2);
        interface.compute_cluster_weights(
            &mut sky_atm,
            &[],
            &[],
            &[],
            &[],
            ClusterReference { bsp_index: 0, cluster_index: 0 },
            -1, // cluster.atmosphere_index = -1 → setting[0]
            glam::Vec3::ZERO,
        );
        assert_eq!(sky_atm.atmosphere_settings[0].weight, 1.0);
        assert_eq!(sky_atm.atmosphere_settings[1].weight, 0.0);
    }
}
