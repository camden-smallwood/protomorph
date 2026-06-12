// particle_eval.wgsl — the GPU curve-evaluation engine, prepended to the
// particle update shader. Exact port of the math in function_fx.hlsl +
// particle_common_fx.hlsl, against a CLEAN table layout (same math as
// the engine's bit-packed s_function_definition / s_property — the
// authored curves are compiled to these tables CPU-side).
//
// Tables are flat, indexed by batch (a particle's emitter): a batch owns
// EVAL_PROPS=13 properties, EVAL_FUNCS functions, and EVAL_COLORS colors.

const EVAL_PROPS: u32 = 13u;
const EVAL_FUNCS: u32 = 32u;   // per-batch function slots
const EVAL_COLORS: u32 = 8u;   // per-batch color slots
const EVAL_MAX_SUB: u32 = 4u;  // piecewise sub-functions

// Property indices (engine _index_*).
const P_EMITTER_TINT: u32 = 0u;
const P_EMITTER_ALPHA: u32 = 1u;
const P_EMITTER_SIZE: u32 = 2u;
const P_PARTICLE_COLOR: u32 = 3u;
const P_PARTICLE_INTENSITY: u32 = 4u;
const P_PARTICLE_ALPHA: u32 = 5u;
const P_PARTICLE_SCALE: u32 = 6u;
const P_PARTICLE_ROTATION: u32 = 7u;
const P_PARTICLE_FRAME: u32 = 8u;
const P_PARTICLE_BLACK_POINT: u32 = 9u;
const P_PARTICLE_ASPECT: u32 = 10u;
const P_PARTICLE_SELF_ACCEL: u32 = 11u;
const P_PARTICLE_PALETTE: u32 = 12u;

// Function types (engine _type_*).
const FT_IDENTITY: u32 = 0u;
const FT_CONSTANT: u32 = 1u;
const FT_TRANSITION: u32 = 2u;
const FT_PERIODIC: u32 = 3u;
const FT_LINEAR: u32 = 4u;
const FT_SPLINE: u32 = 7u;
const FT_EXPONENT: u32 = 9u;
const FT_SPLINE2: u32 = 10u;

// One compiled function (mirrors s_function_definition, 64B).
struct EvalFunction {
    // x=type, y=domain_max, z=range_min, w=range_max
    type_domain_range: vec4<f32>,
    // x=flags, y=exclusion_min, z=exclusion_max
    flags_exclusion: vec4<f32>,
    innards0: vec4<f32>,
    innards1: vec4<f32>,
};

// One compiled property (clean unpacked form of s_property).
struct EvalProperty {
    // x=is_constant, y=constant_value, z=fn_green, w=fn_red
    a: vec4<f32>,
    // x=input_green, y=input_red, z=modifier, w=input_modifier
    b: vec4<f32>,
    // x=color_lo, y=color_hi
    c: vec4<f32>,
};

// Particle state values needed for evaluation (the per-particle subset +
// the per-frame uniform table). Filled by the update shader.
struct EvalState {
    age: f32,
    birth_time: f32,
    random: vec4<f32>,
    random2: vec4<f32>,
    // 28-entry per-frame uniform state (system_age, LOD, game_time, ...).
    // Indexed by state slot for the non-per-particle inputs.
};

// --- bindings (the update shader provides these) ---
@group(0) @binding(3) var<storage, read> g_properties: array<EvalProperty>;  // batch*13 + i
@group(0) @binding(4) var<storage, read> g_functions: array<EvalFunction>;   // batch*EVAL_FUNCS + i
@group(0) @binding(5) var<storage, read> g_colors: array<vec4<f32>>;         // batch*8 + i
@group(0) @binding(6) var<storage, read> g_state_uniform: array<vec4<f32>>;  // batch*28 + slot (.x = value)
// Per-batch self-acceleration interpolants (WORLD): [batch*2 + 0|1].
@group(0) @binding(7) var<storage, read> g_self_accel: array<vec4<f32>>;
// Engine transition/periodic sub-function LUTs (f32 = byte/255). Layout:
// transition rows 1..=7 at [(type-1)*1024 + k]; periodic rows 1..=11 at
// [7168 + (type-1)*1024 + k]. Mirrors transition_function_evaluate
// @0x180346C60 / periodic_function_evaluate @0x180346AC0.
@group(0) @binding(9) var<storage, read> g_func_tables: array<f32>;

const TWO_PI: f32 = 6.2831855;
const ONE_MINUS_EPS: f32 = 0.9999995;

// state value: per-particle slots from the particle, rest from uniforms.
fn get_state_value(st: EvalState, batch: u32, index: u32) -> f32 {
    if (index == 0u) { return st.age; }              // particle_age
    if (index >= 4u && index <= 7u) { return st.random[index - 4u]; }
    if (index >= 21u && index <= 24u) { return st.random2[index - 21u]; }
    if (index == 10u) { return st.birth_time; }      // particle_emit_time
    return g_state_uniform[batch * 28u + index].x;
}

// transition_function_evaluate @0x180346C60 — verbatim LUT lookup.
// `t` (function_type) 0 → identity; 1..=7 index the 1024-entry rows.
fn lut_transition(t: u32, x_in: f32) -> f32 {
    let x = clamp(x_in, 0.0, 1.0);
    if (t == 0u) { return x; }
    let row = min(t - 1u, 6u);
    let base = row * 1024u;
    let scaled = x * 1023.0;
    let frac = scaled - floor(scaled);            // fmod(scaled, 1)
    let i = u32(i32((scaled - 0.1) + 0.5));        // (int)(scaled + 0.4)
    var result: f32;
    if (i >= 1023u) {
        result = g_func_tables[base + 1023u];
    } else {
        result = g_func_tables[base + i] * (1.0 - frac)
               + g_func_tables[base + i + 1u] * frac;
    }
    return clamp(result, 0.0, 1.0);
}

// periodic_function_evaluate @0x180346AC0 — verbatim LUT lookup. `time`
// is the raw input*frequency+phase (NOT pre-wrapped); the engine scales by
// 36.57143 and round-masks to the 1024 phase index. Types 6/7 (mask 0xC0)
// get the sawtooth wrap-blend across the 1→0 seam.
fn lut_periodic(t: u32, time: f32) -> f32 {
    if (t == 0u) { return 1.0; }
    let row = min(t - 1u, 10u);
    let base = 7168u + row * 1024u;
    let scaled = time * 36.57143;
    let v4 = scaled - floor(scaled);               // fmod(scaled, 1), scaled >= 0
    let i = u32(i32((scaled - v4) + 0.5)) & 0x3FFu;
    let v7 = g_func_tables[base + i];
    let v9 = g_func_tables[base + ((i + 1u) & 0x3FFu)];
    if (((1u << t) & 0xC0u) == 0u) {
        return (1.0 - v4) * v7 + v9 * v4;
    }
    var v8 = v9;
    if (v7 > 0.75 && v9 < 0.25) { v8 = v9 + 1.0; }
    let r = (1.0 - v4) * v7 + v8 * v4;
    if (r > 1.0) { return r - 1.0; }
    return r;
}

fn eval_single(batch: u32, index: u32, x: f32) -> f32 {
    let fdef = g_functions[batch * EVAL_FUNCS + index];
    let ty = u32(fdef.type_domain_range.x);
    switch (ty) {
        case 1u: { return fdef.innards0.x; }                         // constant
        case 2u: { // transition: (amp_max-amp_min)*lut(idx, x) + amp_min
            return lut_transition(u32(fdef.innards0.x), x)
                * (fdef.innards1.y - fdef.innards1.x) + fdef.innards1.x;
        }
        case 3u: { // periodic: (amp_max-amp_min)*lut(idx, x*freq+phase) + amp_min
            let time = x * fdef.innards1.x + fdef.innards1.y;
            return lut_periodic(u32(fdef.innards0.x), time)
                * (fdef.innards1.w - fdef.innards1.z) + fdef.innards1.z;
        }
        case 4u: { return fdef.innards0.x * x + fdef.innards0.y; }     // linear
        case 7u: { // spline: i*x^3 + j*x^2 + k*x + l
            let mz = x; let my = x * x; let mx = x * my;
            return dot(fdef.innards0, vec4<f32>(mx, my, mz, 1.0));
        }
        case 10u: { // spline2: remap u=(x-left_x)/width, u'=sign(u)*|u|^bias,
                    // then the inner cubic at u' (c_spline2_function_compact
                    // @0x1804FBD40). innards1 = (left_x, width, bias, _).
            let lx = fdef.innards1.x; let w = fdef.innards1.y; let bias = fdef.innards1.z;
            var u = 0.0;
            if (w != 0.0) { u = (x - lx) / w; }
            let up = sign(u) * pow(abs(u), bias);
            let mz = up; let my = up * up; let mx = up * my;
            return dot(fdef.innards0, vec4<f32>(mx, my, mz, 1.0));
        }
        case 9u: { // exponent
            return pow(x, fdef.innards0.z) * (fdef.innards0.y - fdef.innards0.x) + fdef.innards0.x;
        }
        default: { return x; } // identity
    }
}

fn modify_output(batch: u32, index: u32, v: f32) -> f32 {
    let fdef = g_functions[batch * EVAL_FUNCS + index];
    let flags = u32(fdef.flags_exclusion.x);
    var out = v;
    if ((flags & 8u) != 0u) { // exclusion (bit 3)
        if (out > fdef.flags_exclusion.y) { out = out + (fdef.flags_exclusion.z - fdef.flags_exclusion.y); }
    }
    if ((flags & 4u) != 0u) { out = saturate(out); } // clamped (bit 2)
    return out;
}

fn eval_fn(batch: u32, index: u32, input: f32) -> f32 {
    let x = saturate(input);
    var real = index;
    var found = false;
    for (var s = 0u; s < EVAL_MAX_SUB; s = s + 1u) {
        let f = g_functions[batch * EVAL_FUNCS + index + s];
        if (!found && x <= f.type_domain_range.y) { real = index + s; found = true; }
    }
    return modify_output(batch, index, eval_single(batch, real, x));
}

fn map_to_scalar_range(batch: u32, index: u32, t: f32) -> f32 {
    let fdef = g_functions[batch * EVAL_FUNCS + index];
    return mix(fdef.type_domain_range.z, fdef.type_domain_range.w, t);
}

fn eval_scalar(batch: u32, index: u32, input: f32) -> f32 {
    return map_to_scalar_range(batch, index, eval_fn(batch, index, input));
}

fn eval_scalar_ranged(batch: u32, i1: u32, i2: u32, input: f32, range_input: f32) -> f32 {
    return map_to_scalar_range(batch, i1,
        mix(eval_fn(batch, i1, input), eval_fn(batch, i2, input), range_input));
}

// Evaluate one of the 13 particle properties → scalar.
fn particle_evaluate(st: EvalState, batch: u32, type_: u32) -> f32 {
    let p = g_properties[batch * EVAL_PROPS + type_];
    if (p.a.x == 1.0) { return p.a.y; } // plain constant → constant_value
    if (p.a.x == 2.0) {
        // ranged constant → lerp(clamp_min, clamp_max, range_state). Matches
        // the engine/CPU c_function Constant{is_ranged}: a per-particle random
        // in [a.y, a.z] driven by the range-axis state (b.y). Was collapsed to
        // a.y (the low bound) when fn_red was hardcoded 0 → uniform minimum.
        let interp = get_state_value(st, batch, u32(p.b.y));
        return p.a.y + interp * (p.a.z - p.a.y);
    }
    let in_green = u32(p.b.x);
    let fn_green = u32(p.a.z);
    let fn_red = u32(p.a.w);
    let input = get_state_value(st, batch, in_green);
    var output: f32;
    if (fn_red != 0u) { // ranged (identity=0 sentinel)
        let in_red = u32(p.b.y);
        let interp = get_state_value(st, batch, in_red);
        output = eval_scalar_ranged(batch, fn_green, fn_red, input, interp);
    } else {
        output = eval_scalar(batch, fn_green, input);
    }
    let modifier = u32(p.b.z);
    if (modifier != 0u) {
        let modby = get_state_value(st, batch, u32(p.b.w));
        if (modifier == 1u) { output = output + modby; } else { output = output * modby; }
    }
    return output;
}

// Map a scalar through a property's color range (g_colors[lo..hi]).
fn find_interval(cmin: u32, cmax: u32, t: f32) -> vec3<f32> {
    let num = f32(cmax - cmin);
    let which = floor(saturate(t) * num * ONE_MINUS_EPS);
    let lo = f32(cmin) + which;
    let interp = saturate(t) * num * ONE_MINUS_EPS - which;
    return vec3<f32>(lo, lo + 1.0, interp);
}

// map_to_vector3d_range — COMPONENT-WISE LERP of the two world self-accel
// interpolants by `t`. The engine's `map_to_vector3d_range` is a plain
// per-axis lerp (bible §06/§X.3: "SLERP is a misnomer for this build" — the
// vector/point3d range map interpolates each component linearly, NOT along a
// great circle). The prior great-circle SLERP gave non-linear magnitude +
// spurious rotation for non-unit endpoints.
fn particle_self_accel(batch: u32, t: f32) -> vec3<f32> {
    let v0 = g_self_accel[batch * 2u].xyz;
    let v1 = g_self_accel[batch * 2u + 1u].xyz;
    return mix(v0, v1, saturate(t));
}

fn particle_map_to_color(batch: u32, type_: u32, scalar: f32) -> vec3<f32> {
    let p = g_properties[batch * EVAL_PROPS + type_];
    let cmin = u32(p.c.x);
    let cmax = u32(p.c.y);
    if (cmax <= cmin) { return g_colors[batch * EVAL_COLORS + cmin].rgb; }
    let iv = find_interval(cmin, cmax, scalar);
    let a = g_colors[batch * EVAL_COLORS + u32(iv.x)].rgb;
    let b = g_colors[batch * EVAL_COLORS + u32(iv.y)].rgb;
    return mix(a, b, iv.z);
}
