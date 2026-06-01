//! `object_get_function_value @ dllcache 0x1807DBA60` — top-level
//! entry called by `evaluate_function` (the render-method curve eval).
//! Dispatches through the type-specific chain, then falls back to
//! the model's artist-authored `s_object_function_definition[]` array,
//! then to the LABEL_82 c_event log path.
//!
//! Ares source: `source/objects/objects.cpp` (search for
//! `object_get_function_value`).
//!
//! Engine body (verified 2026-05-20 via IDA bridge — lines 70-220 of
//! `0x1807DBA60`):
//!
//! ```c
//! if (function == 0 || function == "one")  { *out = 1.0; return true; }
//! if (function == "zero")                  { *out = 0.0; return active; }
//!
//! if (object_index != -1
//!     && !object_type_compute_function_value(object_index, function, ...))
//! {
//!     // Walk model's s_object_function_definition[] looking for
//!     // export_name == function. If found, evaluate its formula:
//!     for (def in object_definition->object_functions) {
//!         if (def.export_name == function)
//!             return object_function_get_function_value(...);
//!     }
//!
//!     // LABEL_34: recurse to parent object if one exists. Otherwise
//!     // log a c_event ("objects:function: object %s failed to find
//!     // function '%s' (used by %s)") and return value=0, active=false.
//! }
//! ```

use crate::halo::objects::object_function_get_function_value::object_function_get_function_value;
use crate::halo::objects::object_header_data::get_object_definition;
use crate::halo::objects::object_type_compute_function_value::object_type_compute_function_value;

/// Engine `object_get_function_value` body.
///
/// Implemented:
/// - `function == ""` / `function == "one"` short-circuit → 1.0.
/// - `function == "zero"` short-circuit → 0.0.
/// - `object_type_compute_function_value` dispatch (Phase 5
///   smoke-port: most cases log+return false).
/// - **Phase 5b SHIPPED 2026-05-20:** the model-level
///   `s_object_function_definition[]` walk for artist-authored named
///   functions. When the type dispatcher returns false, walks
///   `ObjectDefinition::functions[]` (cached per placement) for an
///   `export_name == function` match and evaluates via
///   `object_function_get_function_value`.
///
/// **Phase 1 deferred:** parent-object recursion. Engine block at
/// `0x1807DBA60:115-130` — if the model's object_function count is 0
/// and the object has a `parent_object_index` field set (flag bit
/// `0x4000000`), the engine recurses into `object_get_function_value`
/// for the parent. Mostly relevant for attached objects (turrets,
/// weapons in unit hands). Skipped here until the object_datum
/// hierarchy lands.
///
/// `eval_time` carries the current game time in seconds — threaded
/// from the per-placement [`ObjectBoundContext`] (the engine reads
/// it from the `game_time_get()` global inside
/// [`object_function_get_function_value`]'s periodic branch).
pub fn object_get_function_value(
    object_index: u32,
    function: &str,
    function_owner_tag_index: u32,
    eval_time: f32,
    out_function_magnitude: &mut f32,
    deterministic: &mut bool,
) -> bool {
    // Engine: `if (!v9 || v9 == 574 /*one*/) { *out = 1.0; return true; }`
    //         — empty function name OR string_id 574 ("one") → constant 1.
    if function.is_empty() || function == "one" {
        *out_function_magnitude = 1.0;
        return true;
    }

    // Engine: `if (v9 == 575 /*zero*/) { *out = 0.0; ... }` — string_id
    // 575 ("zero") → constant 0. The engine returns `function_variable`
    // (initialized to 0) here, so the return value is effectively
    // false (active=false).
    if function == "zero" {
        *out_function_magnitude = 0.0;
        return false;
    }

    // Engine: `if (object_index != -1
    //               && !object_type_compute_function_value(...))`
    //         — call the type dispatcher first. `-1` is the "no object
    //         context" sentinel; in protomorph we represent that with
    //         `u32::MAX` (matches the engine's cast behavior).
    if object_index != u32::MAX {
        let mut active = false;
        let handled = object_type_compute_function_value(
            object_index,
            function,
            function_owner_tag_index,
            out_function_magnitude,
            &mut active,
            deterministic,
        );
        if handled {
            return active;
        }

        // Engine LABEL_34: when the type dispatcher didn't handle the
        // name, walk the placement's authored `functions[]` block
        // looking for an `export_name` match. This is how authored
        // chain names (e.g. `marinebeacon.scenery` defining `bar` from
        // `foo` from `one`) get resolved.
        let object_definition = get_object_definition(object_index);
        if let Some(entry) = object_definition.find_function_by_export(function) {
            return object_function_get_function_value(
                object_index,
                entry,
                function_owner_tag_index,
                eval_time,
                out_function_magnitude,
                deterministic,
            );
        }
    }

    // LABEL_82: engine logs a c_event and writes value=0.0,
    // active=false, returns false. We mirror with an eprintln-once
    // (lifted to module scope so each function name fires at most
    // once over the program lifetime — same rate-limit pattern as
    // `c_event::query`).
    warn_failed_once(object_index, function, function_owner_tag_index);
    *out_function_magnitude = 0.0;
    false
}

/// Engine-equivalent of `c_event::generate(..., "objects:function:
/// object %s failed to find function '%s' (used by %s)", ...)`. Logs
/// each (function, owner_tag) pair exactly once over the program
/// lifetime to match the `c_event::query` rate-limit.
fn warn_failed_once(object_index: u32, function: &str, function_owner_tag_index: u32) {
    use crate::halo::objects::object_header_data::get_tag_path;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    // Key includes object tag_path so we see WHICH tag asks per function
    // (multiple placements of the same tag are deduped).
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(String, String)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let tag_path = get_tag_path(object_index);
    let mut s = seen.lock().unwrap();
    let key = (function.to_string(), tag_path.clone());
    let _ = function_owner_tag_index;
    if s.insert(key) {
        let tag_display = if tag_path.is_empty() {
            "<no-palette>".to_string()
        } else {
            tag_path
        };
        eprintln!(
            "[object_get_function_value] failed to find function '{function}' \
             on object tag {tag_display:?} (object_index={object_index}). \
             Engine LABEL_82 c_event. Returning 0."
        );
    }
}
