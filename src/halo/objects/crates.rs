//! `crate_compute_function_value @ dllcache 0x180878630`.
//! Ares source: `source/objects/crates.cpp:240`.
//!
//! Engine body is a single `return 0;` — crates have no crate-specific
//! compute_function_value cases. Chain dispatcher falls through to
//! `object_compute_function_value` for every input. Body is
//! engine-faithfully a no-op.
//!
//! `_crate_datum` body (8 bytes, 2 fields) lives at
//! [`crate::halo::objects::object_datum::CrateBody`] — surfaced when
//! consumers need its `flags` / `self_destruction_timer`.

/// No-op engine port — returns `false` so the chain dispatcher
/// proceeds to `object_compute_function_value`. Matches dllcache
/// `0x180878630` verbatim (single `return 0;`).
pub fn crate_compute_function_value(
    _crate_index: u32,
    _function: &str,
    _function_owner_definition_index: u32,
    _value: &mut f32,
    _active: &mut bool,
    _deterministic: &mut bool,
) -> bool {
    false
}
