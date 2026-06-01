//! Per-frame animated cb13 (user) cbuffer re-upload — shared between
//! BSP materials (`BspGpu::animated_materials`) and object materials
//! (`GpuModel::animated_materials`). Engine equivalent:
//! `update_constants @ 0x180685300` walks the material's overlay list
//! each frame and writes the resolved values into the per-`c_render_method`
//! cbuffer.
//!
//! Protomorph diverges in one place: engine re-resolves only the
//! animated slot byte ranges; we re-resolve the full cbuffer and
//! merge over the load-time snapshot. Functionally equivalent — static
//! slots evaluate to the same value at any T — but does extra work.
//! `blam_tags::is_cbuffer_animated` uses the same empirical t=0/t=1
//! comparison, so this update path mirrors that choice.

use crate::halo::render_methods::cbuffer::CbufferLayout;
use crate::halo::render_methods::materials::RenderMethodSource;
use crate::halo::render_methods::serializer::serialize_bytes;

/// Per-object animation eval context — implements blam-tags'
/// `RenderMethodEvalContext` by routing named inputs through the
/// engine-faithful `object_get_function_value` chain.
///
/// Engine analog: `c_render_method_data::m_context_interface` is
/// set per object_render_context to the `object_get_function_value`
/// callback bound to a specific `object_index`. Each animated
/// cbuffer evaluation reads object state for THAT object.
///
/// For the empty (`""`) / `"time"` input names, blam-tags handles
/// time directly via `eval_time` — `resolve_named` only fires for
/// named inputs like `"battery_empty"`. Those route into
/// `object_get_function_value(object_index, name, owner_tag, …)`
/// which dispatches through the `<type>_compute_function_value`
/// chain. State-driven inputs on un-ported types fire the panicking
/// stubs (Phase 5 smoke-port loop trigger).
pub struct ObjectBoundContext {
    /// Game time in seconds for time-based inputs.
    pub eval_time: f32,
    /// Engine `object_index` — slot in `object_header_data`. Passed
    /// to `object_get_function_value` as the first arg. `u32::MAX`
    /// = no object context (engine `-1`); resolver short-circuits
    /// to LABEL_82 (log + 0).
    pub object_index: u32,
    /// Owner tag index for the render_method that owns this draw.
    /// Used by LABEL_82 c_event log to attribute "function used by
    /// %s". `u32::MAX` = unknown source.
    pub owner_tag_index: u32,
}

impl blam_tags::render_method::RenderMethodEvalContext for ObjectBoundContext {
    fn eval_time(&self) -> f32 {
        self.eval_time
    }

    fn resolve_named(&self, input_name: &str) -> f32 {
        let mut value = 0.0f32;
        let mut deterministic = false;
        crate::halo::objects::object_get_function_value::object_get_function_value(
            self.object_index,
            input_name,
            self.owner_tag_index,
            self.eval_time,
            &mut value,
            &mut deterministic,
        );
        value
    }
}

/// Per-animated-material state needed to re-resolve + write the cb13
/// user cbuffer each frame at the current time. One instance per
/// `MaterialData` flagged `is_animated`; lives on the parent
/// (`BspGpu` for BSP materials, `GpuModel` for object materials).
pub struct AnimatedMaterial {
    /// rmsh + rmt2 + rmop_params snapshot for time-aware re-resolution
    /// via `blam_tags::render_method::rebuild_cbuffer_bytes_with_optional_rmt2`.
    pub source: RenderMethodSource,
    /// Cbuffer layout (member offsets) shared across every variant of
    /// this material. `serialize_bytes` writes the re-resolved bytes
    /// into a buffer of `layout.total_size` and overlays the change-
    /// color externs. Assumption: all variant entry-points (StaticSh,
    /// Albedo, StaticPrtAmbient, etc.) emit the same cb13 layout for a
    /// given (rmsh, rmt2) pair — engine ships one layout per material,
    /// not per pass.
    pub layout: CbufferLayout,
    /// Engine externs applied at serialize time. Sourced from
    /// `MaterialData::primary_change_color` / `secondary_change_color`
    /// at load. Zero on BSP materials (no unit color schemes). On
    /// objects these are per-material defaults today; per-placement
    /// customization (engine `render_object_update_change_colors`)
    /// will mutate them when wired.
    pub primary_change_color: glam::Vec3,
    pub secondary_change_color: glam::Vec3,
    /// Load-time `mat.cbuffer.bytes` snapshot (full layout — rmt2-emitted
    /// slots plus any phantom slots appended by
    /// `loader::append_param_phantom_slots` and any terrain-padding
    /// applied by `pad_terrain_cbuffer`). Per-frame
    /// `rebuild_cbuffer_bytes_with_optional_rmt2` returns ONLY the
    /// rmt2-emitted slots, so we use this snapshot to fill the trailing
    /// phantom-slot region (and any internal gaps) before the GPU upload.
    pub loaded_cbuffer_bytes: Vec<u8>,
    /// One props_buffer per variant. BSP creates one per HaloEntryPoint
    /// (StaticPerPixel/StaticSh/StaticShPerVertex/StaticPrtAmbient/Albedo
    /// = 5). Objects share a single buffer across StaticSh / Albedo /
    /// StaticPrtAmbient bind groups (`upload_model_data` constructs one
    /// `props_buffer` and points all three bind groups at it). wgpu::Buffer
    /// is Arc-internal so cloning the handle is cheap.
    pub props_buffers: Vec<wgpu::Buffer>,
}

impl AnimatedMaterial {
    /// Per-frame cbuffer rebuild for ONE slot in the per-instance pool
    /// (Path B). Used by the per-frame walker that iterates placements
    /// using each animated material.
    ///
    /// `ctx` is engine-bound per placement (`ObjectBoundContext` —
    /// `object_index` set to that placement's `header_index`). Named
    /// inputs in the animation curves resolve through
    /// `object_get_function_value` and may panic in the un-ported
    /// `<type>_compute_function_value` stubs (Phase 5 smoke-port
    /// trigger). Time-based inputs (`""` / `"time"`) resolve from
    /// `ctx.eval_time()` inside blam-tags.
    ///
    /// Engine analog: `update_constants @ 0x180685300` writes the
    /// `g_ps_parameters_data` pool's destination slots for the
    /// specific draw's `render_method_context_tag_index` (= object
    /// index). Protomorph's pool is per-material; the slot index
    /// = the placement's `instance_within_model`.
    pub fn write_to_slot(
        &self,
        queue: &wgpu::Queue,
        slot_index: u32,
        ctx: &dyn blam_tags::render_method::RenderMethodEvalContext,
    ) {
        let cb_bytes = blam_tags::render_method::rebuild_cbuffer_bytes_with_optional_rmt2(
            &self.source.rmsh,
            self.source.rmt2.as_ref(),
            &self.source.rmop_params,
            ctx,
        );
        // Merge: start from the load-time snapshot (full layout with
        // phantom slots + terrain padding intact) and overlay the
        // freshly re-resolved rmt2-emitted slot bytes on top. Without
        // the snapshot, scalars like `fresnel_curve_steepness`,
        // `normal_specular_power`, terrain blend-map defaults etc.
        // would zero out every frame on any material flagged
        // `is_animated`.
        let mut merged = self.loaded_cbuffer_bytes.clone();
        let n = cb_bytes.len().min(merged.len());
        if n > 0 {
            merged[..n].copy_from_slice(&cb_bytes[..n]);
        }
        let bytes = serialize_bytes(
            &merged,
            &self.layout,
            self.primary_change_color,
            self.secondary_change_color,
        );
        let slot_size = bytes
            .len()
            .next_multiple_of(crate::halo::render_methods::pipeline_cache::MATERIAL_SLOT_ALIGNMENT);
        let dest_offset = (slot_index as u64) * (slot_size as u64);
        for buffer in &self.props_buffers {
            queue.write_buffer(buffer, dest_offset, &bytes);
        }
    }

    /// Legacy time-only write — kept for BSP / decorator / sky paths
    /// that don't have a per-placement object context. Writes to slot
    /// 0 only. Named inputs in these paths resolve to 0.0 via
    /// `DefaultEvalContext` (engine LABEL_82 fallback equivalent).
    pub fn write_at_time(&self, queue: &wgpu::Queue, time_seconds: f32) {
        let ctx = blam_tags::render_method::DefaultEvalContext {
            eval_time: time_seconds,
        };
        self.write_to_slot(queue, 0, &ctx);
    }
}
