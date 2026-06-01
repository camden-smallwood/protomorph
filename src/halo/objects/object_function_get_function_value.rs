//! `object_function_get_function_value @ dllcache 0x1807E85B0` —
//! per-entry evaluator for one `s_object_function_definition` in the
//! object's authored `functions[]` chain. Called by
//! `object_get_function_value` after a `functions[]` search hits an
//! `export_name == requested` match.
//!
//! Verbatim port of the engine decompile (verified 2026-05-20 via IDA
//! bridge). The control flow has 7 distinct stages:
//!
//! 1. **Script lookup** — `object_scripting_get_function_variable` for
//!    the entry's `import_name`. Protomorph stub returns false (no
//!    scripting system).
//! 2. **Import resolution** — recursive `object_get_function_value`
//!    for `import_name`. Yields a magnitude + active bit.
//! 3. **Curve evaluation** — three paths depending on function-type
//!    and `flags & 2` (`additive`):
//!       - Periodic (type=3): curve evaluated at `game_time` (with
//!         optional per-object random offset via flag 8), MULTIPLIED
//!         by the import-resolved magnitude.
//!       - Active OR non-additive: optional invert (flag 1), then
//!         curve evaluated at the (possibly-inverted) magnitude. If
//!         additive bit is clear, re-activate based on curve > 0.
//!       - Else: pass through (no curve).
//! 4. **Scale_by** — if non-empty, recursive resolve, multiply
//!    magnitudes. Active iff both sources active.
//! 5. **lower_bound** — if > 0, re-active when curve output exceeds it.
//! 6. **always_active** (flag 4) — force result=1 regardless.
//! 7. **turn_off_with** — if non-empty AND we were still active,
//!    recursive resolve; AND-gate the result. With flag 0x20, also
//!    requires the turn-off source's magnitude to be non-zero.
//! 8. **Final clamp** — magnitude clamped to [0, 1]; if result==false
//!    AND flag 0x10 (`always_emit_magnitude`) is clear, force 0.

use blam_tags::object::{
    ObjectFunctionDefinition,
    FN_FLAG_INVERT,
    FN_FLAG_ADDITIVE,
    FN_FLAG_ALWAYS_ACTIVE,
    FN_FLAG_RANDOM_TIME_OFFSET,
    FN_FLAG_ALWAYS_EMIT_MAGNITUDE,
    FN_FLAG_TURN_OFF_REQUIRES_NONZERO,
};
use blam_tags::tag_function::FunctionType;

use crate::halo::objects::object_get_function_value::object_get_function_value;

/// Verbatim port of `object_function_get_function_value` body.
///
/// Returns `true` when the entry is active (engine `function_value`),
/// writes the clamped magnitude to `*out_function_magnitude`. The
/// `deterministic` out-param starts at `true` and gets ANDed with each
/// recursive call's `deterministic` — used by the engine to decide
/// whether the value can be cached across frames.
pub fn object_function_get_function_value(
    object_index: u32,
    function: &ObjectFunctionDefinition,
    function_owner_tag_index: u32,
    eval_time: f32,
    out_function_magnitude: &mut f32,
    deterministic: &mut bool,
) -> bool {
    // Stage 1 — script-bound variable lookup. Engine:
    // `if (object_scripting_get_function_variable(object_index,
    //      function->import_name, out_function_magnitude)) return 1;`.
    // Protomorph has no scripting system, so this is always false.

    let mut magnitude: f32 = 0.0;
    let function_type = function
        .function_value
        .as_ref()
        .map(|f| f.function_type())
        .unwrap_or(FunctionType::Identity);
    let is_periodic = function_type == FunctionType::Periodic;
    let additive_clear = (function.flags & FN_FLAG_ADDITIVE) == 0;

    // Stage 2 — recurse to resolve the entry's `import_name`. May land
    // back on the same `functions[]` block (e.g. marinebeacon's
    // `bar` imports `foo` imports `one`).
    let mut function_active = object_get_function_value(
        object_index,
        &function.import_name,
        function_owner_tag_index,
        eval_time,
        &mut magnitude,
        deterministic,
    );

    // Stage 3 — three-way curve evaluation.
    let mut output_magnitude: f32;
    if is_periodic {
        // Engine: time-driven periodic curve.
        //   v15 = game_ticks_to_seconds(game_time_get());
        //   *deterministic = false;  // time-dependent → not cacheable
        //   if (flags & 8) {
        //     hashed = (1664525*object_index + 1013904223) >> 16;
        //     time = v15 + hashed * (1/65535);
        //   }
        //   v20 = evaluate_scalar(function_value, time, 1.0);
        //   magnitude = v20 * magnitude;
        *deterministic = false;
        let mut time = eval_time;
        if (function.flags & FN_FLAG_RANDOM_TIME_OFFSET) != 0 {
            // Per-object random time offset — LCG hash of object_index,
            // high 16 bits scaled to [0,1). Lets a scene of identical
            // objects animate out-of-phase without authored variance.
            let hashed = (1664525u32
                .wrapping_mul(object_index)
                .wrapping_add(1013904223))
                >> 16;
            time += (hashed as f32) * (1.0 / 65535.0);
        }
        let curve = function
            .function_value
            .as_ref()
            .map(|f| f.evaluate(time, 1.0))
            .unwrap_or(0.0);
        magnitude *= curve;
        output_magnitude = magnitude;
    } else if function_active || additive_clear {
        // Engine: curve evaluated at the import-resolved magnitude.
        //   if (flags & 1) magnitude = 1 - magnitude;
        //   v19 = evaluate_scalar(function_value, magnitude, 1.0);
        //   magnitude = v19;
        //   if (additive_clear) function_value = v19 > 0;
        if (function.flags & FN_FLAG_INVERT) != 0 {
            magnitude = 1.0 - magnitude;
        }
        let curve = function
            .function_value
            .as_ref()
            .map(|f| f.evaluate(magnitude, 1.0))
            .unwrap_or(0.0);
        magnitude = curve;
        output_magnitude = curve;
        if additive_clear {
            // Re-activation gate when additive bit is clear: a curve
            // output > 0 means the entry is "active" regardless of the
            // import's active bit (e.g. the `foo`-resolved zero still
            // triggers activity if the curve maps zero → non-zero).
            function_active = curve > 0.0;
        }
    } else {
        // Additive bit SET AND import resolution returned inactive —
        // engine takes neither curve branch, just passes magnitude
        // through (which is still 0.0 from the import-failed path).
        output_magnitude = magnitude;
    }

    // Stage 4 — scale_by. Engine:
    //   if (scale_by != 0) {
    //     v26 = object_get_function_value(scale_by, &scale_mag, &scale_det);
    //     v19 = magnitude * scale_mag;
    //     magnitude *= scale_mag;
    //     deterministic = deterministic && scale_det;
    //     if (!function_value || !v26) { function_value = 0; goto LABEL_27; }
    //     function_value = 1;
    //   } else if (!function_value) goto LABEL_27;
    let mut function_value: bool = function_active;
    let mut lower_bound_eligible = true; // true → run the lower_bound gate
    if !function.scale_by.is_empty() {
        let mut scale_magnitude = 0.0f32;
        let mut scale_deterministic = false;
        let scale_active = object_get_function_value(
            object_index,
            &function.scale_by,
            function_owner_tag_index,
            eval_time,
            &mut scale_magnitude,
            &mut scale_deterministic,
        );
        magnitude *= scale_magnitude;
        output_magnitude = magnitude;
        *deterministic = *deterministic && scale_deterministic;
        if !function_active || !scale_active {
            function_value = false;
            // Skip lower_bound gate — engine jumps to LABEL_27.
            lower_bound_eligible = false;
        } else {
            function_value = true;
        }
    } else if !function_active {
        // Engine: `else if (!function_value) goto LABEL_27;` — skip the
        // lower_bound gate entirely.
        lower_bound_eligible = false;
    }

    // Stage 5 — lower_bound gate.
    if lower_bound_eligible && function.lower_bound > 0.0 {
        function_value = output_magnitude > function.lower_bound;
    }

    // Stage 6 — always_active (flag 4) overrides everything.
    let mut result = function_value;
    if (function.flags & FN_FLAG_ALWAYS_ACTIVE) != 0 {
        result = true;
    }

    // Stage 7 — turn_off_with. Engine:
    //   if (turn_off_name != 0) {
    //     if (!result) goto LABEL_41;
    //     result = object_get_function_value(turn_off_name)
    //              && ((flags & 0x20) == 0 || t_mag != 0.0);
    //     if (!*deterministic || !t_det) v25 = 0;
    //     v19 = magnitude;
    //     *deterministic = v25;
    //   }
    let mut deterministic_after_turn_off = true;
    if !function.turn_off_with_function_name.is_empty() && result {
        let mut t_mag = 0.0f32;
        let mut t_det = false;
        let t_active = object_get_function_value(
            object_index,
            &function.turn_off_with_function_name,
            function_owner_tag_index,
            eval_time,
            &mut t_mag,
            &mut t_det,
        );
        let nonzero_required =
            (function.flags & FN_FLAG_TURN_OFF_REQUIRES_NONZERO) != 0;
        result = t_active && (!nonzero_required || t_mag != 0.0);
        output_magnitude = magnitude;
        if !*deterministic || !t_det {
            deterministic_after_turn_off = false;
        }
        *deterministic = deterministic_after_turn_off;
    }

    // Stage 8 — final clamp. Engine LABEL_41/45/46:
    //   if (!result && (flags & 0x10) == 0) goto LABEL_45 (-> v19 = 0);
    //   if (v19 > 0.0) { if (v19 >= 1.0) v19 = 1.0; }
    //   else v19 = 0.0;
    //   *out = v19;
    let always_emit = (function.flags & FN_FLAG_ALWAYS_EMIT_MAGNITUDE) != 0;
    if !result && !always_emit {
        output_magnitude = 0.0;
    } else if output_magnitude > 0.0 {
        if output_magnitude >= 1.0 {
            output_magnitude = 1.0;
        }
    } else {
        output_magnitude = 0.0;
    }

    *out_function_magnitude = output_magnitude;
    result
}
