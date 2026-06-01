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

use blam_tags::math::{RealRgbColor, RealVector3d};
use blam_tags::sky_atmosphere::SkyAtmosphere;
use blam_tags::structure_bsp::BspAtmospherePaletteEntry;

use crate::halo::structures::clusters::ClusterReference;

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
    /// `c_atmosphere_fog_interface::set_default_atmosphere_parameters @ 0x18068E5E0`.
    ///
    /// Engine body: `memcpy` from the accumulated source (in 16-byte
    /// chunks, hence the `_OWORD` casts in the decompile), set
    /// `m_last_custom_index = -2`, then `set_constant(&m_default_parameters,
    /// fogged=1, vs=1)`. The cbuffer write is wired in Phase 5; for
    /// now just stash the params + mark applied.
    pub fn set_default_atmosphere_parameters(&mut self, params: &WeightedAtmosphereParameters) {
        self.default_parameters = params.clone();
        self.last_custom_index = -2;
        // Phase 5: self.set_constant(true, true);
    }

    /// `c_atmosphere_fog_interface::restore_default_atmosphere_constants @ 0x180681570`.
    /// Idempotent: skips the cbuffer write if `last_custom_index == -2`.
    pub fn restore_default_atmosphere_constants(&mut self) {
        if self.last_custom_index != -2 {
            self.last_custom_index = -2;
            // Phase 5: self.set_constant(true, true);
        }
    }

    /// `c_atmosphere_fog_interface::invalidate_atmosphere_constants @ 0x18068E500`.
    /// Marks the cbuffer source stale so the next consumer re-applies.
    pub fn invalidate_atmosphere_constants(&mut self) {
        self.last_custom_index = -1;
    }

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
    /// Sun parameters resolve via [`crate::halo::render::env_probe_pass::get_sun_parameters`],
    /// which mirrors engine `get_sun_parameters @ 0x1803AF990`. The
    /// override branch (flags & 2) reads the setting's
    /// `m_dominant_light_*` (chromaticity-preserving xyY→RGB convert);
    /// the non-override branch needs a scenario-sky sun pulled from
    /// `get_sun_constants_from_sky` (callers pass via `sky_sun`).
    ///
    /// Beta coefficients are derived per-setting from
    /// `rayleigh_multiplier` / `mie_multiplier` / `desaturation` via
    /// [`crate::halo::render::env_probe_pass::precompute_betas`] —
    /// engine `precompute_scattering_coefficients @ 0x1803AE9C0`
    /// validated as byte-faithful per
    /// [[reference_precompute_betas_validation_2026_05_19]].
    pub fn accumulate_atmosphere_settings(
        &self,
        setting: &blam_tags::sky_atmosphere::AtmosphereSettings,
        accum: &mut WeightedAtmosphereParameters,
        weight: f32,
        sky_sun: Option<[f32; 3]>,
    ) {
        const FLAG_ENABLE_ATMOSPHERE: u16 = 0x0001;
        const FLAG_PATCHY_FOG: u16 = 0x0004;
        if (setting.flags & FLAG_ENABLE_ATMOSPHERE) == 0 {
            return;
        }
        if weight > 0.0 {
            accum.atmosphere_enabled = 1;
        }

        // Sun. Engine takes (intensity, direction) and accumulates each
        // component separately weighted.
        let (sun_intensity, sun_direction) =
            crate::halo::render::env_probe_pass::get_sun_parameters(setting, sky_sun);
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
            crate::halo::render::env_probe_pass::precompute_betas(
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
        if (setting.flags & FLAG_PATCHY_FOG) != 0 {
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
        sky_sun: Option<[f32; 3]>,
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
    /// **MP fast path only — Phase 3c of the atmosphere port plan.**
    /// Engine branches at:
    /// ```text
    /// if (game_is_multiplayer() && global_scenario->atmosphere_palette.count <= 1) {
    ///     <clear all weights; set starting cluster's setting weight=1, effect_weight=1>
    ///     return;
    /// }
    /// <SP path: structure_clusters_in_sphere → structure_compute_cluster_distances →
    ///  per-cluster falloff curve into accumulated weights>
    /// ```
    /// We mirror the MP fast path verbatim and fall through to the same
    /// behavior on the SP path until Phase 3b ports
    /// `structure_compute_cluster_distances`. Every shipping H3 MCC MP
    /// map (guardian, riverworld, chill, the_pit, etc.) has
    /// `atmosphere_palette.count <= 1` so the fast path matches engine
    /// output. SP campaign maps with multi-setting palettes get
    /// starting-cluster-only fog with no inter-setting blend at
    /// boundaries; same gap the prior `resolve_atmosphere_setting_for_eye`
    /// had.
    pub fn compute_cluster_weights(
        &mut self,
        sky_atm: &mut SkyAtmosphere,
        atmosphere_palette: &[BspAtmospherePaletteEntry],
        starting_cluster: ClusterReference,
        starting_cluster_atmosphere_index: i8,
    ) {
        // Engine `for (atm_setting in atmosphere_settings) { setting->m_weight = 0; }`
        // — clears BOTH weight and effect_weight (they share an 8-byte
        // `_QWORD` store in the decompile at offset +156).
        for setting in &mut sky_atm.atmosphere_settings {
            setting.weight = 0.0;
            setting.effect_weight = 0.0;
        }

        // No active cluster → no setting to weight; engine returns
        // early via `bsp_index != -1` gate in get_atmosphere_setting.
        if !starting_cluster.is_valid() {
            return;
        }

        // Engine MP fast path always sets weight=1, effect_weight=1 on
        // the starting cluster's atmosphere_setting. Resolve via the
        // same chain as `get_atmosphere_setting @ 0x1803AFBA0`.
        let Some(setting_index) = resolve_atmosphere_setting_index(
            starting_cluster_atmosphere_index,
            atmosphere_palette,
            sky_atm.atmosphere_settings.len(),
        ) else {
            return;
        };
        let Some(setting) = sky_atm.atmosphere_settings.get_mut(setting_index) else {
            return;
        };
        setting.weight = 1.0;
        setting.effect_weight = 1.0;
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
        return Some(0);
    };
    if palette_entry.atmosphere_setting_index < 0 {
        return Some(0);
    }
    let setting_idx = palette_entry.atmosphere_setting_index as usize;
    if setting_idx >= setting_count {
        Some(0)
    } else {
        Some(setting_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::sky_atmosphere::AtmosphereSettings;

    fn make_sky_atm(n: usize) -> SkyAtmosphere {
        let mut s = SkyAtmosphere::default();
        for i in 0..n {
            let mut setting = AtmosphereSettings::default();
            setting.flags = 0x0001; // Enable Atmosphere
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
            ClusterReference { bsp_index: 0, cluster_index: 5 },
            0, // cluster.atmosphere_index = 0 → palette[0].setting_index = 2
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
            ClusterReference::NONE,
            -1,
        );
        assert_eq!(sky_atm.atmosphere_settings[0].weight, 0.0);
        assert_eq!(sky_atm.atmosphere_settings[1].weight, 0.0);
    }

    fn make_enabled_setting_with_betas() -> AtmosphereSettings {
        // Authored values from guardian's first atmosphere setting (per
        // [[reference_riverworld_atmosphere_settings]] family). Picked
        // for non-zero outputs across all accumulator fields.
        let mut s = AtmosphereSettings::default();
        s.flags = 0x0007; // enable + override + patchy fog
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
        s.flags = 0; // disabled
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
        s.flags = 0x0001; // enable only — no patchy fog
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
        a.flags = 0x0001; // no patchy on a
        let mut b = make_enabled_setting_with_betas();
        b.distance_bias = 3.0;
        b.flags = 0x0001;
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
        a.flags = 0x0001;
        let mut b = make_enabled_setting_with_betas();
        b.weight = 0.0; // zero-weight should contribute nothing
        b.distance_bias = 99.0;
        b.flags = 0x0001;
        sky_atm.atmosphere_settings.push(a);
        sky_atm.atmosphere_settings.push(b);
        let mut accum = WeightedAtmosphereParameters::default();
        interface.populate_atmosphere_parameters(&sky_atm, &mut accum, None);
        // Only `a` should contribute.
        assert!((accum.distance_bias - 2.0).abs() < 1e-5);
        assert_eq!(accum.atmosphere_enabled, 1);
    }

    #[test]
    fn compute_cluster_weights_negative_atm_index_picks_setting_zero() {
        let mut interface = AtmosphereFogInterface::default();
        let mut sky_atm = make_sky_atm(2);
        interface.compute_cluster_weights(
            &mut sky_atm,
            &[],
            ClusterReference { bsp_index: 0, cluster_index: 0 },
            -1, // cluster.atmosphere_index = -1 → setting[0]
        );
        assert_eq!(sky_atm.atmosphere_settings[0].weight, 1.0);
        assert_eq!(sky_atm.atmosphere_settings[1].weight, 0.0);
    }
}
