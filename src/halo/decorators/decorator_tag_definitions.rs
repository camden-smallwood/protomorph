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
//!  - 4 × i8 quaternion (sign-stored, decompressed via the engine's
//!    `decompress_quaternion_component`, signatures above).
//!  - 4 × u8 HDR RGB+exp byte (R, G, B, ground_tint). HDR decoded by the
//!    decorator shader as `rgb × exp2(ground_tint × 63.75 − 31.75)`.
//!    `ground_tint` here is the engine's HDR EXPONENT byte; the painted
//!    ground-tint blend factor lives in the AUTHORING-side
//!    `s_decorator_placement` (consumed by the bake, not stored at runtime).

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

