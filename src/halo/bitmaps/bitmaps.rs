//! Engine source: `Ares/source/bitmaps/bitmaps.{h,cpp}`.
//!
//! Engine-runtime bitmap helpers (sampling, mip-walk, hardware-format
//! interpretation). Currently scoped to the subset Phase 7b
//! (`sample_lightprobe_textures`) needs.
//!
//! Function name + signature from Reach (preserved symbol
//! `?bitmap_group_2d_get_real_pixel@@YA?ATreal_argb_color@@JJPEBTreal_point2d@@JJ_N@Z`);
//! body adapted to use protomorph's pre-decoded atlas instead of the
//! engine's per-call `c_durango_address_computer` walk.

use glam::Vec2;

use blam_tags::math::{RealArgbColor, RealPoint2d};

use crate::halo::decorators::atlas_decode::DecodedAtlas;
