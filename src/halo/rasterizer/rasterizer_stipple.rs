//! Mirror of `Ares/source/rasterizer/rasterizer_stipple.h`.
//!
//! Stipple-pattern fade for LOD dissolves. Halo uses an 8×8 stipple
//! pattern + a per-pixel alpha threshold to produce a dithered
//! crossfade between LODs / dead-biped fade-outs / decal occlusion.

/// `rasterizer_stipple_initialize` — sets up the stipple pattern
/// table.
pub fn rasterizer_stipple_initialize() {}

/// `rasterizer_stipple_set_fade_byte @ h prototype`. Halo: writes
/// the per-fragment alpha cutoff to a shader constant. `alpha`
/// is the stipple-fade byte (0..255); `write_heat_flag` and
/// `write_decal_occlusion_flag` toggle the auxiliary outputs;
/// `first_person` switches the FP-specific stipple pattern.
pub fn rasterizer_stipple_set_fade_byte(
    _alpha: i32,
    _write_heat_flag: i32,
    _write_decal_occlusion_flag: i32,
    _first_person: i32,
) {
}

/// `rasterizer_stipple_deactivate_fade`.
pub fn rasterizer_stipple_deactivate_fade() {}
