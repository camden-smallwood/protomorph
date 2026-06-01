//! Mirror of `c_chocalate_moutain` (Bungie typo preserved — "chocalate"
//! not "chocolate"). The walker lives in [`blam_tags::chocolate_mountain`]
//! (which uses the schema-correct "chocolate" spelling); this file
//! holds the runtime apply path.
//!
//! `c_chocalate_moutain` is Halo's HDR-floor adjuster: a per-object-type
//! minimum-luminance table that boosts dark scenes so characters never
//! go fully black under shadow probes. The engine reads the active
//! scenario's `chocolate_mountain_settings` tag-ref and computes
//! `mountain_scale = sqrt(min_luminance / current_render_exposure)`
//! per object_type — applied via
//! `apply_chocalate_mountain_lighting(object_type, sh_*, mountain_scale)`
//! inside `object_update_cached_render_lighting @ 0x180697430`.
//!
//! Note: the engine **does not** modify the SH coefficients here —
//! `sh_red/green/blue` are passed in for compatibility with an older
//! engine variant but the H3 MCC body only writes `*mountain_scale`.

pub use blam_tags::chocolate_mountain::ChocolateMountain;

/// `c_chocalate_moutain::apply_chocalate_mountain_lighting @ 0x1806f9ee0`
/// — verbatim port. Returns the `mountain_scale` the engine writes
/// into the cached `render_lighting`. Returns `-1.0` (engine
/// sentinel) when no boost applies.
pub fn apply_chocalate_mountain_lighting(
    chmt: &ChocolateMountain,
    object_type: i32,
    render_exposure: f32,
) -> f32 {
    if object_type < 0 {
        return -1.0;
    }
    chmt.compute_mountain_scale(object_type as usize, render_exposure)
}

/// SH luminance estimate consumed by the chmt boost-gate inside
/// `c_lighting_interface::setup_object_lighting_for_entry_point @
/// 0x1806A9AA0`:
///
/// ```text
/// luminance = sum(k=0..4) |sh_r[k]|*0.21266 + |sh_g[k]|*0.71516 + |sh_b[k]|*0.07219
/// ```
///
/// Sums |R/G/B| over the first 4 SH coefs (DC + L1 directional bands)
/// weighted by NTSC luminance constants. The engine clamps the result
/// to a 0.001 floor before dividing into mountain_scale, so we mirror
/// that minimum to avoid blowups.
///
/// The math purpose: `scale = max(1, mountain_scale / sh_luminance)`
/// boosts the SH only when its current ambient brightness is below the
/// chmt floor — preventing "raise the bright objects too" overcorrection.
pub fn sh_luminance_estimate(sh_r: &[f32; 9], sh_g: &[f32; 9], sh_b: &[f32; 9]) -> f32 {
    const W_R: f32 = 0.21265601;
    const W_G: f32 = 0.71515799;
    const W_B: f32 = 0.072185598;
    let mut total = 0.0_f32;
    for k in 0..4 {
        total += sh_r[k].abs() * W_R + sh_g[k].abs() * W_G + sh_b[k].abs() * W_B;
    }
    total
}
