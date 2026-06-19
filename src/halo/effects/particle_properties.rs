//! Compile authored `EditableProperty` / `TagFunction` curves into the
//! GPU evaluation tables the per-emitter update kernel reads each frame
//! (`particle_eval.wgsl`). This is the runtime `s_gpu_data` bake that
//! tool.exe does into the (empty-in-loose-tags) `runtime m_gpu_data` —
//! reproduced from the authored curves directly.
//!
//! The 13 particle properties (engine `_index_*`) each compile to:
//!   - a constant (fast path), or
//!   - a `fn_green` index into the per-batch function table + an input
//!     state slot, optionally a ranged `fn_red`/`in_red` and a modifier.
//! Color properties (emitter_tint / particle_color) additionally publish
//! their gradient stops into the per-batch color table.
//!
//! Table layout matches `particle_eval.wgsl`: clean structs, identical
//! eval math to `function_fx.hlsl` (not the engine's bit-packed form).

use blam_tags::effect::ParticleSystemEmitter;
use blam_tags::effects_properties::EditableProperty;
use blam_tags::particle::{ParticleDefinition, ParticlePropertyScalar};
use blam_tags::tag_function::TagFunction;
use blam_tags::typed_enums::SchemaEnum;

/// The fields the compiler needs from a property, abstracted over the two
/// authored shapes: effect `EditableProperty` (emitter properties) and
/// particle `ParticlePropertyScalar` (prt3 properties). Identical layout
/// in the tag; the walkers just use different field types.
struct PropInputs<'a> {
    function: Option<&'a TagFunction>,
    input_index: u8,
    range_index: u8,
    modifier: u8,
    modifier_input: u8,
    constant_value: f32,
}

impl<'a> PropInputs<'a> {
    fn from_editable(p: &'a EditableProperty) -> Self {
        Self {
            function: p.function.as_ref(),
            input_index: p.input_index,
            range_index: p.range_input_index,
            modifier: p.output_modifier_type,
            modifier_input: p.output_modifier_input_index,
            constant_value: p.constant_value,
        }
    }
    fn from_particle(p: &'a ParticlePropertyScalar) -> Self {
        Self {
            function: p.function.as_ref(),
            input_index: p.input_variable.get().to_index() as u8,
            range_index: p.range_variable.get().to_index() as u8,
            modifier: p.output_modifier.get().to_index() as u8,
            modifier_input: p.output_modifier_input.get().to_index() as u8,
            constant_value: p.constant_value,
        }
    }
}

pub const EVAL_PROPS: usize = 13;
pub const EVAL_FUNCS: usize = 32;
pub const EVAL_COLORS: usize = 8;
pub const EVAL_STATE_SLOTS: usize = 28;
/// Max piecewise sub-functions the WGSL domain scan reads per property
/// (`particle_eval.wgsl::EVAL_MAX_SUB`) = the engine `_maximum_sub_function
/// _count`. Multi-part chains are capped here so they can't bleed past the
/// scan window into an adjacent property's function slots.
pub const EVAL_MAX_SUB: usize = 4;
/// `domain_max` for a chain's LAST sub-function. The WGSL scan picks the
/// first sub whose `domain_max >= saturate(input)`, falling back to the
/// BASE (not the last) when none match — so the final segment must always
/// match. A value ≥ 1 guarantees that for any saturated input; the
/// sentinel also mirrors the engine/blam-tags "evaluate the last part past
/// its end" semantics (the last part extrapolates rather than dropping out).
const DOMAIN_END: f32 = f32::MAX;

/// One compiled function — mirrors WGSL `EvalFunction` (64 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EvalFunction {
    /// type, domain_max, range_min, range_max
    pub type_domain_range: [f32; 4],
    /// flags, exclusion_min, exclusion_max, pad
    pub flags_exclusion: [f32; 4],
    pub innards0: [f32; 4],
    pub innards1: [f32; 4],
}

impl Default for EvalFunction {
    fn default() -> Self {
        // identity, domain_max=1, range 0..1
        Self {
            type_domain_range: [0.0, 1.0, 0.0, 1.0],
            flags_exclusion: [0.0; 4],
            innards0: [0.0; 4],
            innards1: [0.0; 4],
        }
    }
}

/// One compiled property — mirrors WGSL `EvalProperty` (48 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EvalProperty {
    /// is_constant, constant_value, fn_green, fn_red
    pub a: [f32; 4],
    /// input_green, input_red, modifier, input_modifier
    pub b: [f32; 4],
    /// color_lo, color_hi, pad, pad
    pub c: [f32; 4],
}

impl Default for EvalProperty {
    fn default() -> Self {
        // constant 1.0 (neutral for multiplied properties).
        Self { a: [1.0, 1.0, 0.0, 0.0], b: [0.0; 4], c: [0.0; 4] }
    }
}

/// All GPU eval tables for one emitter (one render batch).
#[derive(Debug, Clone)]
pub struct EmitterEvalTables {
    pub properties: [EvalProperty; EVAL_PROPS],
    pub functions: [EvalFunction; EVAL_FUNCS],
    pub colors: [[f32; 4]; EVAL_COLORS],
}

impl Default for EmitterEvalTables {
    fn default() -> Self {
        Self {
            properties: [EvalProperty::default(); EVAL_PROPS],
            functions: [EvalFunction::default(); EVAL_FUNCS],
            colors: [[0.0; 4]; EVAL_COLORS],
        }
    }
}

/// Accumulates functions + colors while compiling an emitter's 13 props.
struct Builder {
    functions: Vec<EvalFunction>,
    colors: Vec<[f32; 4]>,
}

impl Builder {
    fn new() -> Self {
        Self { functions: Vec::new(), colors: Vec::new() }
    }

    /// Allocate `n` chained function slots; returns the base index.
    fn alloc_fns(&mut self, fns: &[EvalFunction]) -> u32 {
        let base = self.functions.len() as u32;
        self.functions.extend_from_slice(fns);
        base
    }

    fn alloc_colors(&mut self, cols: &[[f32; 4]]) -> (u32, u32) {
        let lo = self.colors.len() as u32;
        self.colors.extend_from_slice(cols);
        let hi = self.colors.len().saturating_sub(1) as u32;
        (lo, hi)
    }
}

/// Compile a single `TagFunction` to one or more chained `EvalFunction`s
/// (the piecewise types emit several with ascending `domain_max`).
fn compile_function(tf: &TagFunction) -> Vec<EvalFunction> {
    let h = tf.header();
    let flags = (h.flags.0 & 0x0c) as f32; // keep clamped(2)/exclusion(3) bits
    let rmin = h.clamp_range_min;
    let rmax = h.clamp_range_max;
    let base = |ty: u32, i0: [f32; 4], i1: [f32; 4]| EvalFunction {
        type_domain_range: [ty as f32, 1.0, rmin, rmax],
        flags_exclusion: [flags, h.exclusion_min, h.exclusion_max, 0.0],
        innards0: i0,
        innards1: i1,
    };
    match tf {
        TagFunction::Identity { .. } => vec![base(0, [0.0; 4], [0.0; 4])],
        TagFunction::Constant { .. } => {
            // Constant value lives in clamp_range_min (blam-tags as_constant).
            vec![base(1, [rmin, 0.0, 0.0, 0.0], [0.0; 4])]
        }
        TagFunction::Linear { compact, .. } => {
            vec![base(4, [compact.slope, compact.offset, 0.0, 0.0], [0.0; 4])]
        }
        TagFunction::Spline { compact, .. } => {
            vec![base(7, [compact.i, compact.j, compact.k, compact.l], [0.0; 4])]
        }
        TagFunction::Spline2 { compact, .. } => {
            let s = &compact.spline;
            vec![base(
                10,
                [s.i, s.j, s.k, s.l],
                [compact.left_x, compact.width, compact.bias, 0.0],
            )]
        }
        TagFunction::Transition { compact, .. } => vec![base(
            2,
            [compact.function_index as f32, 0.0, 0.0, 0.0],
            [compact.amplitude_min, compact.amplitude_max, 0.0, 0.0],
        )],
        TagFunction::Periodic { compact, .. } => vec![base(
            3,
            [compact.function_index as f32, 0.0, 0.0, 0.0],
            [
                compact.frequency,
                compact.phase,
                compact.amplitude_min,
                compact.amplitude_max,
            ],
        )],
        TagFunction::Exponent { compact, .. } => vec![base(
            9,
            [compact.amplitude_min, compact.amplitude_max, compact.exponent, 0.0],
            [0.0; 4],
        )],
        // Multi-part spline (type 8) → emit the AUTHORED parts verbatim:
        // one EvalFunction per segment carrying its real type (linear/
        // spline/spline2) + `ending_x`→domain_max + innards. This mirrors
        // the engine GPU bake `c_multi_part_function_compact::get_gpu_data
        // @0x1804FBF30` exactly. (The old `sample_piecewise` flattened this
        // to 4 equal-width linear bins, destroying spline curvature and
        // mis-binning non-uniform knots.)
        TagFunction::MultiSpline { compact, .. } => {
            multipart_chain(&compact.parts, rmin, rmax, flags, h.exclusion_min, h.exclusion_max)
        }
        // Linear-key (types 5/6): 4 authored (x,y) control points → a chain
        // of linear segments split at the REAL knots (engine flattens
        // linear_key to a multi-part linear chain). Reproduces
        // `LinearKeyCompact::evaluate` — exact at the knots, with the
        // endpoint clamp past the last point.
        TagFunction::LinearKey { compact, .. } | TagFunction::MultiLinearKey { compact, .. } => {
            linear_key_chain(tf, compact, rmin, rmax, flags, h.exclusion_min, h.exclusion_max)
        }
        // Recognized-but-unimplemented type → identity passthrough (the
        // engine never ships these in particle tags).
        TagFunction::Unsupported { .. } => vec![base(0, [0.0; 4], [0.0; 4])],
    }
}

/// Build one chain segment `EvalFunction`. The chain's BASE (first) segment
/// is the only one whose flags + range are read by the WGSL
/// (`modify_output`/`map_to_scalar_range` index the base), but baking them
/// into every segment is harmless and keeps the constructor uniform.
#[allow(clippy::too_many_arguments)]
fn mk_seg(
    ty: u32,
    domain_max: f32,
    rmin: f32,
    rmax: f32,
    flags: f32,
    ex_min: f32,
    ex_max: f32,
    i0: [f32; 4],
    i1: [f32; 4],
) -> EvalFunction {
    EvalFunction {
        type_domain_range: [ty as f32, domain_max, rmin, rmax],
        flags_exclusion: [flags, ex_min, ex_max, 0.0],
        innards0: i0,
        innards1: i1,
    }
}

/// Compile a `MultiSpline` (multi-part) to its authored sub-functions —
/// verbatim engine bake. Each part keeps its real type + innards; the
/// non-final parts use their authored `ending_x` as `domain_max`, the final
/// part gets [`DOMAIN_END`] so it always matches (and extrapolates past its
/// own `ending_x`, matching `MultiPartCompact::evaluate`).
fn multipart_chain(
    parts: &[blam_tags::tag_function::MultiPartSegment],
    rmin: f32,
    rmax: f32,
    flags: f32,
    ex_min: f32,
    ex_max: f32,
) -> Vec<EvalFunction> {
    use blam_tags::tag_function::MultiPartSubFunction as Sub;
    let n = parts.len().min(EVAL_MAX_SUB);
    if n == 0 {
        return vec![mk_seg(0, DOMAIN_END, rmin, rmax, flags, ex_min, ex_max, [0.0; 4], [0.0; 4])];
    }
    let mut out = Vec::with_capacity(n);
    for (i, part) in parts.iter().take(n).enumerate() {
        let dmax = if i + 1 == n { DOMAIN_END } else { part.ending_x };
        let f = match &part.function {
            Sub::Linear(c) => {
                mk_seg(4, dmax, rmin, rmax, flags, ex_min, ex_max, [c.slope, c.offset, 0.0, 0.0], [0.0; 4])
            }
            Sub::Spline(c) => {
                mk_seg(7, dmax, rmin, rmax, flags, ex_min, ex_max, [c.i, c.j, c.k, c.l], [0.0; 4])
            }
            Sub::Spline2(c) => {
                let s = &c.spline;
                mk_seg(
                    10, dmax, rmin, rmax, flags, ex_min, ex_max,
                    [s.i, s.j, s.k, s.l],
                    [c.left_x, c.width, c.bias, 0.0],
                )
            }
        };
        out.push(f);
    }
    out
}

/// Compile a `LinearKey`/`MultiLinearKey` (4 control points) to a faithful
/// linear chain. Segments split at the REAL knot x-values (not 4 equal
/// bins), so the result reproduces `LinearKeyCompact::evaluate` exactly at
/// the knots. The per-knot output is taken through `tf.evaluate` and
/// re-normalized by the function's `[rmin,rmax]` range (the same trick the
/// old sampler used) so the WGSL `map_to_scalar_range` reconstructs the real
/// value. A trailing constant segment clamps to the last point's value for
/// input past the final knot (the engine `LinearKeyCompact` clamps, not
/// extrapolates). 3 linear + 1 constant = 4 ≤ [`EVAL_MAX_SUB`].
fn linear_key_chain(
    tf: &TagFunction,
    c: &blam_tags::tag_function::LinearKeyCompact,
    rmin: f32,
    rmax: f32,
    flags: f32,
    ex_min: f32,
    ex_max: f32,
) -> Vec<EvalFunction> {
    let denom = if (rmax - rmin).abs() > 1e-9 { rmax - rmin } else { 1.0 };
    // Normalized (pre-range-map) output at x — undo `tf.evaluate`'s range
    // map so the WGSL re-applies it identically.
    let norm = |x: f32| ((tf.evaluate(x, 0.0) - rmin) / denom).clamp(0.0, 1.0);
    let xs = [
        c.graph_points[0].0,
        c.graph_points[1].0,
        c.graph_points[2].0,
        c.graph_points[3].0,
    ];
    let mut out = Vec::with_capacity(4);
    for i in 0..3 {
        let (xa, xb) = (xs[i], xs[i + 1]);
        let (ya, yb) = (norm(xa), norm(xb));
        let dx = xb - xa;
        let (slope, offset) = if dx > 1e-9 {
            let s = (yb - ya) / dx;
            (s, ya - s * xa)
        } else {
            // Degenerate / padded-duplicate knot → flat at this value.
            (0.0, ya)
        };
        out.push(mk_seg(4, xb, rmin, rmax, flags, ex_min, ex_max, [slope, offset, 0.0, 0.0], [0.0; 4]));
    }
    // Endpoint clamp past the last knot (constant = last point's value).
    let y_last = norm(xs[3]);
    out.push(mk_seg(1, DOMAIN_END, rmin, rmax, flags, ex_min, ex_max, [y_last, 0.0, 0.0, 0.0], [0.0; 4]));
    out
}

/// Unpack a `TagFunction` color graph's stops to RGBA vec4s.
fn color_stops(tf: &TagFunction) -> Vec<[f32; 4]> {
    let h = tf.header();
    let n = h.color_graph_type as usize; // 0=scalar,1..4 stops
    let unpack = |c: u32| {
        [
            ((c >> 16) & 0xff) as f32 / 255.0,
            ((c >> 8) & 0xff) as f32 / 255.0,
            (c & 0xff) as f32 / 255.0,
            1.0,
        ]
    };
    if n == 0 {
        return vec![[1.0, 1.0, 1.0, 1.0]];
    }
    // Color stops live in the header's `m_colors[4]` (ARGB8). The engine
    // `c_function_definition::map_to_color_range_legacy @0x1804f9af0` maps
    // the stop count → slots NON-LINEARLY — the LAST stop is always
    // `m_colors[3]`, earlier stops are `m_colors[0..n-1]`:
    //   1 → [0]   2 → [0,3]   3 → [0,1,3]   4 → [0,1,2,3]
    // Reading `m_colors[0..n]` linearly took `m_colors[1]` (a scalar clamp
    // float, e.g. `0x3f800000` → unpacks `(0.5,0,0)` red) for 2-color
    // gradients instead of `m_colors[3]` → the s3d_turf red snow/smoke.
    let n = n.min(4);
    (0..n)
        .map(|i| {
            let slot = if n >= 2 && i + 1 == n { 3 } else { i.min(3) };
            unpack(h.colors[slot])
        })
        .collect()
}

/// Compile a scalar property (size/alpha/intensity/etc.).
fn compile_scalar(b: &mut Builder, prop: PropInputs) -> EvalProperty {
    let mut out = EvalProperty::default();
    match prop.function {
        // Constant / no curve → fast constant path.
        Some(f) if f.is_constant() => {
            let h = f.header();
            if h.flags.is_ranged() {
                // RANGED constant: the engine's `c_function_definition` (and
                // blam-tags `evaluate_legacy` Constant{is_ranged} → returns
                // `range`, then map_to_output_range) yields
                // `lerp(clamp_min, clamp_max, range_state)` — a per-particle
                // random in [min,max]. The GPU eval (particle_eval.wgsl, a.x==2)
                // applies this; previously fn_red=0 collapsed it to clamp_min
                // (the low bound) → uniform minimum (snow self-accel/size/etc.
                // all pinned to their slowest/smallest value). b.y = range_index.
                out.a = [2.0, h.clamp_range_min, h.clamp_range_max, 0.0];
                out.b = [
                    prop.input_index as f32,
                    prop.range_index as f32,
                    prop.modifier as f32,
                    prop.modifier_input as f32,
                ];
            } else {
                out.a = [1.0, f.as_constant().unwrap_or(prop.constant_value), 0.0, 0.0];
            }
        }
        None => {
            out.a = [1.0, prop.constant_value, 0.0, 0.0];
        }
        Some(f) => {
            let fns = compile_function(f);
            let fn_green = b.alloc_fns(&fns);
            out.a = [0.0, 0.0, fn_green as f32, 0.0]; // fn_red=0 (no range)
            out.b = [
                prop.input_index as f32,
                prop.range_index as f32,
                prop.modifier as f32,
                prop.modifier_input as f32,
            ];
        }
    }
    out
}

/// Compile a color property (emitter_tint / particle_color): the scalar
/// curve drives the gradient position; the color stops go to the table.
fn compile_color(b: &mut Builder, prop: PropInputs) -> EvalProperty {
    let mut out = EvalProperty::default();
    let Some(f) = prop.function else {
        // No curve → white, single stop.
        let (lo, hi) = b.alloc_colors(&[[1.0, 1.0, 1.0, 1.0]]);
        out.a = [1.0, 0.0, 0.0, 0.0];
        out.c = [lo as f32, hi as f32, 0.0, 0.0];
        return out;
    };
    let stops = color_stops(f);
    let (lo, hi) = b.alloc_colors(&stops);
    out.c = [lo as f32, hi as f32, 0.0, 0.0];
    if f.is_constant() || stops.len() <= 1 {
        // Position is irrelevant (single stop) — constant position 0.
        out.a = [1.0, 0.0, 0.0, 0.0];
    } else {
        let fns = compile_function(f);
        let fn_green = b.alloc_fns(&fns);
        out.a = [0.0, 0.0, fn_green as f32, 0.0];
        out.b = [prop.input_index as f32, prop.range_index as f32, 0.0, 0.0];
    }
    out
}

/// Compile all 13 properties for an emitter + its particle definition.
pub fn compile_emitter(
    emitter: &ParticleSystemEmitter,
    particle: &ParticleDefinition,
) -> EmitterEvalTables {
    let mut b = Builder::new();
    let mut props = [EvalProperty::default(); EVAL_PROPS];

    // Scalar particle properties come from the prt3; map its scalar
    // property structs to EditableProperty-shaped access via a shim.
    // (ParticleDefinition stores ParticlePropertyScalar; we read its
    // function the same way.) For now the particle-level color/intensity/
    // alpha/aspect/frame/palette use the particle's properties; emitter
    // tint/alpha/size/scale/rotation/black_point/self_accel use the
    // emitter's.
    let ed = PropInputs::from_editable;
    let pp = PropInputs::from_particle;
    // Emitter-level properties (effect EditableProperty).
    props[0] = compile_color(&mut b, ed(&emitter.particle_tint)); // emitter_tint
    props[1] = compile_scalar(&mut b, ed(&emitter.particle_alpha)); // emitter_alpha
    props[2] = compile_scalar(&mut b, ed(&emitter.particle_size)); // emitter_size
    props[6] = compile_scalar(&mut b, ed(&emitter.particle_scale)); // particle_scale
    props[7] = compile_scalar(&mut b, ed(&emitter.particle_rotation)); // particle_rotation
    props[9] = compile_scalar(&mut b, ed(&emitter.particle_alpha_black_point)); // black_point
    props[11] = compile_scalar(&mut b, ed(&emitter.particle_self_acceleration)); // self_accel
    // prt3-level properties (ParticlePropertyScalar — now curve-parsed).
    props[3] = compile_color(&mut b, pp(&particle.color)); // particle_color
    props[4] = compile_scalar(&mut b, pp(&particle.intensity)); // particle_intensity
    props[5] = compile_scalar(&mut b, pp(&particle.alpha)); // particle_alpha
    props[8] = compile_scalar(&mut b, pp(&particle.frame_index)); // particle_frame
    props[10] = compile_scalar(&mut b, pp(&particle.aspect_ratio)); // particle_aspect
    props[12] = compile_scalar(&mut b, pp(&particle.palette_animation)); // particle_palette

    // Fill the fixed-size tables.
    let mut functions = [EvalFunction::default(); EVAL_FUNCS];
    for (i, f) in b.functions.iter().take(EVAL_FUNCS).enumerate() {
        functions[i] = *f;
    }
    let mut colors = [[0.0f32; 4]; EVAL_COLORS];
    for (i, c) in b.colors.iter().take(EVAL_COLORS).enumerate() {
        colors[i] = *c;
    }
    EmitterEvalTables { properties: props, functions, colors }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blam_tags::tag_function::TagFunction;

    /// Build a 32-byte TagFunction header: type byte, GPU flag (no
    /// clamp/exclusion bits → modify_output passthrough), clamp_range
    /// [rmin,rmax]. compact data is appended by the caller.
    fn hdr(ftype: u8, rmin: f32, rmax: f32) -> Vec<u8> {
        let mut h = vec![0u8; 32];
        h[0] = ftype;
        h[1] = 0x20; // GPU flag; bits 2/3 (clamp/exclusion) clear
        h[4..8].copy_from_slice(&rmin.to_le_bytes());
        h[8..12].copy_from_slice(&rmax.to_le_bytes());
        h
    }

    /// Mirror `particle_eval.wgsl::eval_single` for the chain types this
    /// module emits (constant/linear/spline/spline2).
    fn eval_single(f: &EvalFunction, x: f32) -> f32 {
        let ty = f.type_domain_range[0] as u32;
        let i0 = f.innards0;
        match ty {
            1 => i0[0],                                   // constant
            4 => i0[0] * x + i0[1],                       // linear
            7 | 10 => {                                   // spline / spline2
                let (mz, my) = (x, x * x);
                let mx = x * my;
                i0[0] * mx + i0[1] * my + i0[2] * mz + i0[3]
            }
            _ => x, // identity
        }
    }

    /// Mirror the WGSL `eval_scalar`: domain scan (first sub within
    /// EVAL_MAX_SUB whose domain_max ≥ x, else base 0), modify_output with
    /// the BASE flags, then map_to_scalar_range with the BASE range.
    fn eval_chain(chain: &[EvalFunction], input: f32) -> f32 {
        let x = input.clamp(0.0, 1.0);
        let mut real = 0usize;
        let mut found = false;
        for s in 0..chain.len().min(EVAL_MAX_SUB) {
            if !found && x <= chain[s].type_domain_range[1] {
                real = s;
                found = true;
            }
        }
        let base = &chain[0];
        // flags bits 2/3 are clear in these tests → modify_output is identity.
        let t = eval_single(&chain[real], x);
        let (rmin, rmax) = (base.type_domain_range[2], base.type_domain_range[3]);
        rmin + t * (rmax - rmin)
    }

    /// A 2-part MultiSpline (linear over [0,0.5] then a cubic over (0.5,1])
    /// must compile to a chain that reproduces `tf.evaluate` — verbatim
    /// parts, NOT the old 4-equal-bin flatten.
    #[test]
    fn multispline_chain_matches_evaluate() {
        let mut data = hdr(8 /* MultiSpline */, 0.0, 1.0);
        // function_count = 2
        data.extend_from_slice(&2i32.to_le_bytes());
        // part 0: type=Linear(4), ending_x=0.5, body slope=2 offset=0 → y=2x
        data.push(4);
        data.extend_from_slice(&[0, 0, 0]);
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // part 1: type=Spline(7), ending_x=1.0, body i,j,k,l = 0,0,1,0 → y=x
        data.push(7);
        data.extend_from_slice(&[0, 0, 0]);
        data.extend_from_slice(&1.0f32.to_le_bytes());
        for v in [0.0f32, 0.0, 1.0, 0.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let tf = TagFunction::parse(&data).unwrap();
        assert!(matches!(tf, TagFunction::MultiSpline { .. }));
        let chain = compile_function(&tf);
        assert_eq!(chain.len(), 2, "two authored parts → two chain segments");
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            let got = eval_chain(&chain, x);
            let want = tf.evaluate(x, 0.0);
            assert!((got - want).abs() < 1e-5, "x={x}: chain={got} tf={want}");
        }
    }

    /// A LinearKey over 4 knots spanning [0,1] must compile to knot-exact
    /// linear segments reproducing `tf.evaluate` (the old sampler split at
    /// fixed quarters, mis-placing the 0.5/0.75 knots).
    #[test]
    fn linear_key_chain_matches_evaluate() {
        let mut data = hdr(5 /* LinearKey */, 0.0, 1.0);
        let points = [(0.0f32, 0.0f32), (0.5, 1.0), (0.75, 0.5), (1.0, 0.8)];
        for (x, y) in points {
            data.extend_from_slice(&x.to_le_bytes());
            data.extend_from_slice(&y.to_le_bytes());
        }
        data.extend_from_slice(&[0u8; 16]); // times_vector (unused by evaluate)
        data.extend_from_slice(&[0u8; 16]); // increment_vector (recomputed)
        // y_delta_vector = y[i+1]-y[i] (used by LinearKeyCompact::evaluate)
        for d in [1.0f32, -0.5, 0.3, 0.0] {
            data.extend_from_slice(&d.to_le_bytes());
        }
        let tf = TagFunction::parse(&data).unwrap();
        assert!(matches!(tf, TagFunction::LinearKey { .. }));
        let chain = compile_function(&tf);
        // 3 linear segments + 1 constant tail.
        assert_eq!(chain.len(), 4);
        for i in 0..=40 {
            let x = i as f32 / 40.0;
            let got = eval_chain(&chain, x);
            let want = tf.evaluate(x, 0.0);
            assert!((got - want).abs() < 1e-5, "x={x}: chain={got} tf={want}");
        }
    }
}
