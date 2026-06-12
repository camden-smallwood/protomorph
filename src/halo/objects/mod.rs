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

use glam::{Mat4, Vec3};

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
    /// When set, [`Self::model_matrix`] returns this verbatim instead of
    /// composing `position`/`rotation`/`scale`. Used for objects whose
    /// world transform comes from a parent-marker attachment
    /// (`scenario.object placement parent id` → engine
    /// `object_attach_to_marker @0x1807D7F70`), e.g. construct's waterfall
    /// scenery snapped to the sky's `"waterfall"` marker. `None` for the
    /// normal placement-transform path.
    pub model_matrix_override: Option<Mat4>,
    /// Nearest scenario dynamic-env cubemap probe index for this object's
    /// position (engine `c_dynamic_cubemap_sample`). Resolved at load via
    /// `Renderer::nearest_object_cubemap_probe`. Selects the cube bound to
    /// the material's `dynamic_environment_map_*` slots so glass/env objects
    /// reflect the real environment. `None` → rasg `DefaultDynamicCubeMap`.
    pub cubemap_probe_index: Option<u16>,
}

impl ObjectData {
    /// Construct the model matrix the **engine** way — from the
    /// `forward`/`up` basis the engine derives from the placement euler
    /// (`matrix4x3_rotation_from_angles`), assembled per
    /// `matrix4x3_from_point_and_vectors @0x1802c42b0`: rows are
    /// `forward`, `left = cross(up, forward)`, `up`, translation = position.
    /// In glam column-major that is `from_cols(forward, left, up, position)`
    /// (local X→forward, Y→left, Z→up), with uniform scale folded into the
    /// basis. This replaces the earlier glam-euler `Quat::from_euler(ZYX)`
    /// compose, which produced a *different* rotation than the engine
    /// (verified numerically: Y-sign/basis mismatch). Matching the engine
    /// convention here means the object datum's `forward`/`up` and this
    /// matrix agree, so the upcoming datum convergence is a neutral swap.
    pub fn model_matrix(&self) -> Mat4 {
        // Parent-marker attachment (e.g. sky-attached waterfall) supplies a
        // ready-made world matrix; bypass the placement compose.
        if let Some(m) = self.model_matrix_override {
            return m;
        }
        let (yaw, pitch, roll) = (
            self.rotation.x.to_radians(), // yaw (around Z)
            self.rotation.y.to_radians(), // pitch (Y)
            self.rotation.z.to_radians(), // roll (X)
        );
        let (cy, sy) = (yaw.cos(), yaw.sin());
        let (cp, sp) = (pitch.cos(), pitch.sin());
        let (cr, sr) = (roll.cos(), roll.sin());
        // Engine `matrix4x3_rotation_from_angles` rows 0 + 2.
        let forward = Vec3::new(cy * cp, sy * cr - sp * sr * cy, sp * cr * cy + sy * sr);
        let up = Vec3::new(-sp, -cp * sr, cp * cr);
        let left = up.cross(forward); // engine row1 = cross(up, forward)
        Mat4::from_cols(
            (forward * self.scale.x).extend(0.0),
            (left * self.scale.y).extend(0.0),
            (up * self.scale.z).extend(0.0),
            self.position.extend(1.0),
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
                    engine_lighting_offset: None,
                    model_matrix_override: None,
                cubemap_probe_index: None,
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
            engine_lighting_offset: None,
            model_matrix_override: None,
        cubemap_probe_index: None,
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
}
