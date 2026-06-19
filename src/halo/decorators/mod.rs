//! Mirror of `Ares/source/decorators/decorator_tag_definitions.h`
//! (142 lines).
//!
//! Halo's decorator system places GPU-instanced foliage clusters
//! (grass, plants, scattered rocks) across BSP cluster space. Each
//! `s_decorator_set` (the `dctr` tag) references one render_model
//! plus a texture + shader-flavor selection; the runtime instances
//! it across `s_decorator_runtime_block`s in cluster-aligned grids.
//!
//! Decorators render in the opaque pass with a dedicated shader
//! variant (one of 6 — `_render_shader_*`). The placements are
//! compressed (16B per placement: position_xyz + quaternion +
//! RGBE color) so 48,000 placements per cluster stays under
//! cache-line pressure.

pub mod bake_material_lookup;
pub mod decorator_tag_definitions;
pub mod light_placement;
