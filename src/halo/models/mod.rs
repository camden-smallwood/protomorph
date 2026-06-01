//! Model subsystem — hlmt (`.model`) variant/region/permutation
//! resolution + render_model (`.render_model`) skeleton + meshes.
//! Type-only ports landed; the full hlmt walker + render_model
//! loader land alongside Phase 13's models/ rewrite.

pub mod model_constants;
pub mod model_definitions;
pub mod model_skinning;
pub mod render_model_definitions;
pub mod render_model_instances;
