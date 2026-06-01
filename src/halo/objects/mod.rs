// Halo objects — only modules with actively-used types. Stub
// mirrors (lights, model_customization, object_data,
// object_definitions, object_lights) were removed in the
// 2026-05-09 audit (zero-call-site dead skeletons).
pub mod crates;
pub mod object_compute_function_value;
pub mod object_constants;
pub mod object_datum;
pub mod object_function_get_function_value;
pub mod object_get_function_value;
pub mod object_header_data;
pub mod object_type;
pub mod object_type_compute_function_value;

/// Shared smoke-port helper used by every un-ported
/// `<type>_compute_function_value` stub. Engine-faithful behavior:
/// when a `*_compute_function_value` does not recognize `function`,
/// it returns `false` so the caller (`object_get_function_value`)
/// can fall through to the model `s_object_function_definition[]`
/// walk (Phase 5b — `ObjectDefinition::functions[]`) and ultimately
/// to LABEL_82 if even that fails.
///
/// The walker resolves authored chains (`bar` ← `foo` ← `one` on
/// `marinebeacon.scenery`), so the FALSE RETURN from this stub is
/// usually NOT a bug — it's the legitimate handoff path. Logging
/// here would double-log every authored-chain resolution, so we
/// stay silent. The genuinely-unresolved cases surface at LABEL_82
/// in `object_get_function_value::warn_failed_once`, which is the
/// engine's actual diagnostic signal and gives us the real inventory
/// of state-driven names that need a type-compute body port.
///
/// Parameters are retained on the signature so callers don't need to
/// change when a stub is replaced by an engine-faithful body — the
/// new body uses them, the stub ignores them.
#[inline]
pub fn warn_unported_compute(_type_function: &str, _object_index: u32, _requested: &str) {}

use crate::halo::animations::AnimationManager;
use crate::halo::geometry::ModelData;
use glam::{EulerRot, Mat4, Vec3};

// ---------------------------------------------------------------------------
// Object data
// ---------------------------------------------------------------------------

pub struct ObjectData {
    pub position: Vec3,
    /// Halo convention: `[0]=yaw (around Z, vertical)`, `[1]=pitch (Y)`,
    /// `[2]=roll (X)`. Stored in degrees. Matches `real_euler_angles_3d`
    /// .yaw/.pitch/.roll from scenario placements (after radians→degrees).
    pub rotation: Vec3,
    pub scale: Vec3,
    pub model_index: Option<usize>,
    /// Sequential index of this placement among all placements sharing
    /// the same `model_index`. Used as the slot index into per-model
    /// animated-material cbuffer pools (Path B per-instance cbuffer
    /// state — see [[project_render_method_function_port_plan_2026_05_20]]).
    /// First placement of a model gets `instance_within_model = 0`;
    /// each subsequent placement increments. `0` for objects without
    /// a model.
    pub instance_within_model: u32,
    /// Engine `object_index` into `object_header_data` for this
    /// placement (the unified handle used by
    /// `*_compute_function_value`). `None` for objects without a
    /// scenario placement (e.g. sky, test objects). Populated at
    /// `load_visible_placements` time via
    /// `objects::object_header_data::header_index_for_placement`.
    pub header_index: Option<u32>,
    pub animations: Option<AnimationManager>,
    /// L4 — `engine_lighting_buffer` byte offset for this object's
    /// per-placement SH probe (resolved at scenario load via
    /// `Renderer::bake_scenery_lighting_offsets` from the lightmap's
    /// `scenery_probes[]`). `None` when the object is not a scenery
    /// placement, or has no matching probe — falls back to the frame
    /// default (sky probe at offset 0).
    ///
    /// Engine equivalent: the resolved sample's slot in the per-object
    /// `cached_render_lighting`, surfaced through
    /// `c_lighting_interface::setup_object_lighting_for_entry_point @
    /// 0x1806A9AA0` for the per-draw `calculate_and_set_ravi_constants`
    /// upload. We pre-bake at load time (probes are static) and bind via
    /// dynamic offset — equivalent under the static-probe assumption.
    pub engine_lighting_offset: Option<u32>,
}

impl ObjectData {
    /// Construct the model matrix using Halo's rotation convention:
    ///   rotation_matrix = Yaw(Z) * Pitch(Y) * Roll(X)   (intrinsic ZYX)
    ///   model_matrix    = Translation * rotation_matrix * Scale
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.scale,
            glam::Quat::from_euler(
                EulerRot::ZYX,
                self.rotation.x.to_radians(), // yaw (Z)
                self.rotation.y.to_radians(), // pitch (Y)
                self.rotation.z.to_radians(), // roll (X)
            ),
            self.position,
        )
    }
}

// ---------------------------------------------------------------------------
// Handle type
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct ObjectIndex(pub usize);

// ---------------------------------------------------------------------------
// Object store
// ---------------------------------------------------------------------------

pub struct ObjectStore {
    objects: Vec<Option<ObjectData>>,
}

impl ObjectStore {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    pub fn new_object(&mut self) -> ObjectIndex {
        // Reuse first empty slot
        for (i, slot) in self.objects.iter().enumerate() {
            if slot.is_none() {
                self.objects[i] = Some(ObjectData {
                    position: Vec3::ZERO,
                    rotation: Vec3::ZERO,
                    scale: Vec3::ONE,
                    model_index: None,
                    instance_within_model: 0,
                    header_index: None,
                    animations: None,
                    engine_lighting_offset: None,
                });

                return ObjectIndex(i);
            }
        }

        // No empty slot — push new
        let index = self.objects.len();

        self.objects.push(Some(ObjectData {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            model_index: None,
            instance_within_model: 0,
            header_index: None,
            animations: None,
            engine_lighting_offset: None,
        }));

        ObjectIndex(index)
    }

    pub fn get(&self, index: ObjectIndex) -> &ObjectData {
        self.objects[index.0]
            .as_ref()
            .expect("object slot is empty")
    }

    pub fn get_mut(&mut self, index: ObjectIndex) -> &mut ObjectData {
        self.objects[index.0]
            .as_mut()
            .expect("object slot is empty")
    }

    pub fn delete(&mut self, index: ObjectIndex) {
        self.objects[index.0] = None;
    }

    pub fn iter(&self) -> impl Iterator<Item = (ObjectIndex, &ObjectData)> {
        self.objects
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|data| (ObjectIndex(i), data)))
    }

    pub fn update(&mut self, model_data_list: &[ModelData], delta_seconds: f32) {
        for slot in self.objects.iter_mut().flatten() {
            if let (Some(model_idx), Some(anim_mgr)) = (slot.model_index, slot.animations.as_mut()) {
                anim_mgr.update(&model_data_list[model_idx], delta_seconds);
            }
        }
    }
}
