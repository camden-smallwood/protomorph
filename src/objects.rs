use crate::{animation::AnimationManager, models::ModelData};
use glam::{EulerRot, Mat4, Vec3};

// ---------------------------------------------------------------------------
// Object data
// ---------------------------------------------------------------------------

pub struct ObjectData {
    pub position: Vec3,
    pub rotation: Vec3, // euler angles in degrees: [0]=X (yaw), [1]=Y (pitch), [2]=Z (roll)
    pub scale: Vec3,
    pub model_index: Option<usize>,
    pub animations: Option<AnimationManager>,
}

impl ObjectData {
    /// Construct the model matrix matching the C render.c rotation order:
    ///   rotation_matrix = Roll(Z) * Yaw(X) * Pitch(Y)
    ///   model_matrix = Translation * rotation_matrix * Scale
    pub fn model_matrix(&self) -> Mat4 {
        // Roll(Z) * Yaw(X) * Pitch(Y) = EulerRot::ZXY
        Mat4::from_scale_rotation_translation(
            self.scale,
            glam::Quat::from_euler(
                EulerRot::ZXY,
                self.rotation.z.to_radians(),
                self.rotation.x.to_radians(),
                self.rotation.y.to_radians(),
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
                    animations: None,
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
            animations: None,
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
