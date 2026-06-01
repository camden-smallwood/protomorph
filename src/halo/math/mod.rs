//! Engine source: `Ares/source/math/`.
//!
//! Engine-side math helpers that mirror the standalone files in
//! `Ares/source/math/*.{h,cpp}`. Tag-format math types (e.g. `RealPoint3d`,
//! `RealVector3d`, `RealMatrix4x3`) live in [`blam_tags::math`] because they
//! cross the tag-extraction boundary; this module hosts the engine-runtime
//! math that operates on those types.

pub mod color_math;
pub mod globals;
pub mod matrix_math;
pub mod vector_math;
