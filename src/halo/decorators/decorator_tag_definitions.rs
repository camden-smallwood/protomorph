//! Engine source: `Ares/source/decorators/decorator_tag_definitions.h:78-131`.
//!
//! Runtime placement + block structs. The tag-side `s_decorator_set` is
//! parsed by `blam_tags::decorator_set`; THIS module holds the RUNTIME forms
//! that the offline bake (tool.exe `light_placement @ sub_140C4AF00`)
//! produces and the renderer reads.
//!
//! ## Ares struct layout (verbatim)
//!
//! ```cpp
//! struct s_decorator_runtime_placement      // sizeof == 16
//! {
//!     unsigned short position_x;            // 0x0
//!     unsigned short position_y;            // 0x2
//!     unsigned short position_z;            // 0x4
//!     union {
//!         struct {
//!             unsigned char motion_scale;    // 0x6
//!             unsigned char subpart_index;   // 0x7
//!         };
//!         unsigned short position_w;         // 0x6  (16-bit alias)
//!     };
//!     union {
//!         struct {
//!             char Q_I;                       // 0x8
//!             char Q_J;                       // 0x9
//!             char Q_K;                       // 0xA
//!             char Q_W;                       // 0xB
//!         };
//!         unsigned long orientation;          // 0x8  (32-bit alias)
//!     };
//!     union {
//!         struct {
//!             unsigned char R;                // 0xC
//!             unsigned char G;                // 0xD
//!             unsigned char B;                // 0xE
//!             unsigned char ground_tint;      // 0xF
//!         };
//!         unsigned long RGBE_color;           // 0xC  (32-bit alias)
//!     };
//!     static unsigned long compress_quaternion_component(float);
//!     static float decompress_quaternion_component(unsigned long);
//! };
//!
//! struct s_decorator_runtime_block          // sizeof == 60
//! {
//!     unsigned short block_decorator_placement_count;  // 0x0
//!     unsigned char  bsp_decorator_set_index;          // 0x2
//!     unsigned char  bsp_instance_vertex_buffer_index; // 0x3
//!     long           instance_vertex_buffer_byte_offset; // 0x4
//!     real_vector3d  position_bounds_0;                // 0x8
//!     float          bounding_sphere_radius;           // 0x14
//!     real_vector3d  position_bounds_1;                // 0x18
//!     real_point3d   bounding_sphere_center;           // 0x24
//!     s_tag_block    model_start_index;                // 0x30
//! };
//! ```
//!
//! The runtime placement packs:
//!  - 3 × u16 position (12-bit unsigned fixed-point per axis, relative to
//!    the cluster's `position_bounds_0..1` box — engine reconstructs world
//!    space as `bounds_0 + (pos / 65535) * (bounds_1 - bounds_0)`).
//!  - 1 × u8 motion_scale (wind sway intensity OR sun-multiplier for L8
//!    anti-double-count, depending on `decorator_set.render_shader`).
//!  - 1 × u8 subpart_index (which decorator_type within the set).
//!  - 4 × i8 quaternion (sign-stored, decompressed via
//!    [`decompress_quaternion_component`]).
//!  - 4 × u8 HDR RGB+exp byte (R, G, B, ground_tint). HDR decoded by the
//!    decorator shader as `rgb × exp2(ground_tint × 63.75 − 31.75)`.
//!    `ground_tint` here is the engine's HDR EXPONENT byte; the painted
//!    ground-tint blend factor lives in the AUTHORING-side
//!    `s_decorator_placement` (consumed by the bake, not stored at runtime).

use blam_tags::math::{RealPoint3d, RealVector3d};

// =============================================================================
// s_decorator_runtime_placement (16 B)
// =============================================================================

/// Engine `s_decorator_runtime_placement` (16 B). Output of the bake; input
/// to the renderer. Layout matches engine byte-for-byte (verified via
/// `static_assert(sizeof == 16)` in Ares header).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DecoratorRuntimePlacement {
    /// Engine `position_x` @ 0x0 — u16 fixed-point in
    /// `[bounds_0.x, bounds_1.x]`.
    pub position_x: u16,
    /// Engine `position_y` @ 0x2.
    pub position_y: u16,
    /// Engine `position_z` @ 0x4.
    pub position_z: u16,
    /// Engine `motion_scale` @ 0x6 — u8 wind-sway intensity (and L8
    /// anti-double-count factor for `DominantLightOnly` variants).
    pub motion_scale: u8,
    /// Engine `subpart_index` @ 0x7 — which `decorator_type` of the set.
    pub subpart_index: u8,
    /// Engine `Q_I` @ 0x8 — i8 packed quaternion component.
    pub q_i: i8,
    /// Engine `Q_J` @ 0x9.
    pub q_j: i8,
    /// Engine `Q_K` @ 0xA.
    pub q_k: i8,
    /// Engine `Q_W` @ 0xB.
    pub q_w: i8,
    /// Engine `R` @ 0xC — HDR-encoded red channel byte.
    pub r: u8,
    /// Engine `G` @ 0xD.
    pub g: u8,
    /// Engine `B` @ 0xE.
    pub b: u8,
    /// Engine `ground_tint` @ 0xF — HDR exponent byte (despite the field
    /// name; ground-tint blend factor is consumed only at bake time).
    pub exponent: u8,
}

// Per-Ares `static_assert(sizeof(s_decorator_runtime_placement) == 16)`.
const _: () = assert!(std::mem::size_of::<DecoratorRuntimePlacement>() == 16);

impl DecoratorRuntimePlacement {
    /// Engine `position_w` union view — 16-bit alias over `motion_scale +
    /// subpart_index`. Useful when callers need the engine's wholesale 16-
    /// bit read.
    pub fn position_w(self) -> u16 {
        u16::from_le_bytes([self.motion_scale, self.subpart_index])
    }

    /// Engine `orientation` union view — 32-bit alias over the four
    /// quaternion bytes (Q_I, Q_J, Q_K, Q_W in little-endian byte order).
    pub fn orientation(self) -> u32 {
        u32::from_le_bytes([self.q_i as u8, self.q_j as u8, self.q_k as u8, self.q_w as u8])
    }

    /// Engine `RGBE_color` union view — 32-bit alias over (R, G, B, exp).
    pub fn rgbe_color(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, self.exponent])
    }
}

// =============================================================================
// s_decorator_runtime_block (60 B)
// =============================================================================

/// Engine `s_decorator_runtime_block` (60 B). One per cluster's decorator
/// chunk — groups together all the placements that share an instance vertex
/// buffer offset + a single decorator_set.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct DecoratorRuntimeBlock {
    /// Engine `block_decorator_placement_count` @ 0x0 — number of
    /// successfully-baked placements in this block. Set by tool.exe
    /// `structure_bsp_light_decorators_from_scenario @ sub_140C4BE80`
    /// AFTER `light_placement` runs (the call increments this only on
    /// successful per-placement bakes; failed bakes are compacted out).
    pub block_decorator_placement_count: u16,
    /// Engine `bsp_decorator_set_index` @ 0x2 — index into the BSP's
    /// `decorator_set_references` block.
    pub bsp_decorator_set_index: u8,
    /// Engine `bsp_instance_vertex_buffer_index` @ 0x3 — which entry in
    /// `structure_bsp.decorator_instance_buffer.meshes[]` owns this block's
    /// runtime vertex range.
    pub bsp_instance_vertex_buffer_index: u8,
    /// Engine `instance_vertex_buffer_byte_offset` @ 0x4 — byte offset into
    /// the vertex buffer where this block's per-instance data starts.
    pub instance_vertex_buffer_byte_offset: i32,
    /// Engine `position_bounds_0` @ 0x8 — min corner of the cluster's
    /// fixed-point position range. Engine reconstructs each placement's
    /// world position as
    /// `bounds_0 + (position_xyz / 65535) * (bounds_1 - bounds_0)`.
    pub position_bounds_0: RealVector3d,
    /// Engine `bounding_sphere_radius` @ 0x14.
    pub bounding_sphere_radius: f32,
    /// Engine `position_bounds_1` @ 0x18 — max corner of the position range.
    pub position_bounds_1: RealVector3d,
    /// Engine `bounding_sphere_center` @ 0x24.
    pub bounding_sphere_center: RealPoint3d,
    /// Engine `model_start_index` @ 0x30 — per-decorator-type start indices
    /// within the block, populated by
    /// `structure_bsp_build_decorators_from_scenario @ sub_140C4C150`.
    /// Each element is a u16 (start index of the next decorator_type's
    /// placements in the block). The renderer uses this to submit each
    /// type as a contiguous instance range.
    ///
    /// Stored here as an owned `Vec<u16>` for simplicity; engine's
    /// `s_tag_block` carries a pointer + count (12 B), which the renderer
    /// indexes during draw calls.
    pub model_start_index: Vec<u16>,
}

// We deliberately don't `static_assert(size == 60)` on the Rust side — `Vec`
// is heap-backed and replaces the engine's `s_tag_block` (12 B in the engine
// struct, 24 B for Rust `Vec`). The engine LAYOUT is documented above; our
// Rust struct is a logical mirror, not a binary mirror, for the
// model_start_index field.

// =============================================================================
// Quaternion component (de)compression
// =============================================================================

/// Engine `s_decorator_runtime_placement::compress_quaternion_component`
/// @ Reach `0x82ff0d88` (Reach has named symbol; tool.exe equivalent).
///
/// Verbatim port: maps `f32 in [-1, 1]` → signed byte (which is then
/// reinterpreted as `u32` for storage). Engine uses `sqrt(2)`-scaled
/// quaternions (normalized form is `q / sqrt(2)` per component so the
/// max magnitude per axis is `1/sqrt(2)` for a unit quat).
///
/// Note: The encoded value is an UNSIGNED LONG in the engine signature, but
/// the BYTE it produces is signed (-127..127). Engine stores it back into
/// the `Q_I/J/K/W` `char` fields. We return `i8` since that's what the
/// placement struct uses.
pub fn compress_quaternion_component(f: f32) -> i8 {
    // Engine: round(f * (127 / sqrt(2))) then clamp to [-127, 127].
    let scaled = f * (127.0 / std::f32::consts::SQRT_2);
    let rounded = scaled.round().clamp(-127.0, 127.0);
    rounded as i8
}

/// Engine `s_decorator_runtime_placement::decompress_quaternion_component`
/// @ Reach `0x82798e40`.
///
/// Verbatim inverse of [`compress_quaternion_component`]. Engine multiplies
/// by `sqrt(2) / 127.0` and (in the LIGHT_PLACEMENT bake context)
/// optionally subtracts `127` from the input byte for the wrap-around case.
/// Caller is responsible for the wrap-around handling; this function does
/// the raw decompress.
pub fn decompress_quaternion_component(byte: i8) -> f32 {
    (byte as f32) * (std::f32::consts::SQRT_2 / 127.0)
}
