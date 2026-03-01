use bitflags::bitflags;
use glam::{Mat4, Quat, Vec3};
use crate::model::ModelData;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MAXIMUM_NUMBER_OF_MODEL_NODES: usize = 256;

// ---------------------------------------------------------------------------
// Animation data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationChannelType {
    Node,
    Mesh,
    Morph,
}

#[derive(Debug, Clone)]
pub struct PositionKey {
    pub time: f32,
    pub position: Vec3,
}

#[derive(Debug, Clone)]
pub struct RotationKey {
    pub time: f32,
    pub rotation: Quat,
}

#[derive(Debug, Clone)]
pub struct ScalingKey {
    pub time: f32,
    pub scaling: Vec3,
}

#[derive(Debug, Clone)]
pub struct MeshKey {
    pub time: f32,
    pub mesh_index: i32,
}

#[derive(Debug, Clone)]
pub struct MorphKey {
    pub time: f32,
    pub values: Vec<i32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct AnimationChannel {
    pub channel_type: AnimationChannelType,
    pub target_index: i32,
    pub position_keys: Vec<PositionKey>,
    pub rotation_keys: Vec<RotationKey>,
    pub scaling_keys: Vec<ScalingKey>,
    pub mesh_keys: Vec<MeshKey>,
    pub morph_keys: Vec<MorphKey>,
}

#[derive(Debug, Clone)]
pub struct AnimationData {
    pub name: String,
    pub duration: f32,
    pub ticks_per_second: f32,
    pub channels: Vec<AnimationChannel>,
}

// ---------------------------------------------------------------------------
// Animation state
// ---------------------------------------------------------------------------

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct AnimationStateFlags: u32 {
        const LOOPING  = 1 << 0;
        const PAUSED   = 1 << 1;
        const FADE_IN  = 1 << 2;
        const FADE_OUT = 1 << 3;
    }
}

#[derive(Debug, Clone)]
struct AnimationNodeState {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
}

impl Default for AnimationNodeState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnimationState {
    pub flags: AnimationStateFlags,
    pub time: f32,
    pub speed: f32,
    pub weight: f32,
    pub fade_in_duration: f32,
    pub fade_out_duration: f32,
    node_states: Vec<AnimationNodeState>,
}

// ---------------------------------------------------------------------------
// Animation manager
// ---------------------------------------------------------------------------

pub struct AnimationManager {
    active_animations: Vec<bool>,
    states: Vec<AnimationState>,
    pub node_matrices: Vec<Mat4>,
}

impl AnimationManager {
    pub fn new(model: &ModelData) -> Self {
        let node_count = model.nodes.len();
        let anim_count = model.animations.len();

        let states: Vec<AnimationState> = (0..anim_count)
            .map(|_| AnimationState {
                flags: AnimationStateFlags::empty(),
                time: 0.0,
                speed: 1.0,
                weight: 1.0,
                fade_in_duration: 0.0,
                fade_out_duration: 0.0,
                node_states: vec![AnimationNodeState::default(); node_count],
            })
            .collect();

        Self {
            active_animations: vec![false; anim_count],
            states,
            node_matrices: vec![Mat4::IDENTITY; node_count],
        }
    }

    pub fn get_state(&self, index: usize) -> &AnimationState {
        &self.states[index]
    }

    pub fn get_state_mut(&mut self, index: usize) -> &mut AnimationState {
        &mut self.states[index]
    }

    pub fn is_active(&self, index: usize) -> bool {
        self.active_animations[index]
    }

    pub fn set_active(&mut self, index: usize, active: bool) {
        let state = &mut self.states[index];
        state.time = 0.0;
        state.weight = 1.0;

        self.active_animations[index] = active;
    }

    pub fn is_looping(&self, index: usize) -> bool {
        self.states[index].flags.contains(AnimationStateFlags::LOOPING)
    }

    pub fn set_looping(&mut self, index: usize, looping: bool) {
        self.states[index].flags.set(AnimationStateFlags::LOOPING, looping);
    }

    pub fn is_paused(&self, index: usize) -> bool {
        self.states[index].flags.contains(AnimationStateFlags::PAUSED)
    }

    pub fn set_paused(&mut self, index: usize, paused: bool) {
        self.states[index].flags.set(AnimationStateFlags::PAUSED, paused);
    }

    pub fn set_time(&mut self, index: usize, time: f32) {
        self.states[index].time = time;
    }

    pub fn set_speed(&mut self, index: usize, speed: f32) {
        self.states[index].speed = speed;
    }

    pub fn set_fade_in_duration(&mut self, index: usize, duration: f32) {
        self.states[index].flags.insert(AnimationStateFlags::FADE_IN);
        self.states[index].fade_in_duration = duration;
    }

    pub fn set_fade_out_duration(&mut self, index: usize, duration: f32) {
        self.states[index].flags.insert(AnimationStateFlags::FADE_OUT);
        self.states[index].fade_out_duration = duration;
    }

    pub fn update(&mut self, model: &ModelData, delta_seconds: f32) {
        for animation_index in 0..model.animations.len() {
            self.update_animation(model, animation_index, delta_seconds);
        }

        if model.nodes.is_empty() {
            return;
        }

        if let Some(root) = model.get_root_node() {
            self.compute_node_matrices(model, root, Mat4::IDENTITY);
        }
    }

    // --- Private methods ---

    fn update_animation(
        &mut self,
        model: &ModelData,
        animation_index: usize,
        delta_seconds: f32,
    ) {
        let animation = &model.animations[animation_index];
        let state = &mut self.states[animation_index];

        // Compute fade weight
        if state.flags.contains(AnimationStateFlags::FADE_IN)
            && state.time <= state.fade_in_duration
        {
            state.weight = if state.fade_in_duration > 0.0 {
                state.time / state.fade_in_duration
            } else {
                1.0
            };
        } else if state.flags.contains(AnimationStateFlags::FADE_OUT)
            && state.time >= (animation.duration - state.fade_out_duration)
        {
            state.weight = if state.fade_out_duration > 0.0 {
                (animation.duration - state.time) / state.fade_out_duration
            } else {
                1.0
            };
        } else {
            state.weight = 1.0;
        }

        // Advance time if active
        if self.active_animations[animation_index] {
            let state = &mut self.states[animation_index];
            state.time += (animation.ticks_per_second * state.speed) * delta_seconds;

            if state.flags.contains(AnimationStateFlags::LOOPING) {
                if animation.duration > 0.0 {
                    state.time = state.time.rem_euclid(animation.duration);
                }
            } else if state.time >= animation.duration {
                self.states[animation_index].time = 0.0;
                self.states[animation_index].weight = 1.0;
                self.active_animations[animation_index] = false;
            }
        }

        Self::compute_node_orientations(model, animation, &mut self.states[animation_index]);
    }

    fn compute_node_orientations(
        model: &ModelData,
        animation: &AnimationData,
        state: &mut AnimationState,
    ) {
        for node_index in 0..model.nodes.len() {
            let node = &model.nodes[node_index];
            let mut animation_count = 0;

            // Accumulate interpolated values directly (no matrix intermediary)
            let mut total_position = Vec3::ZERO;
            let mut total_rotation = Quat::IDENTITY;
            let mut total_scale = Vec3::ONE;

            for channel in &animation.channels {
                if channel.channel_type != AnimationChannelType::Node {
                    continue;
                }

                if channel.target_index != node_index as i32 {
                    continue;
                }

                // Position keys
                if channel.position_keys.len() == 1 {
                    total_position = total_position + channel.position_keys[0].position;
                    animation_count += 1;
                } else if !channel.position_keys.is_empty() {
                    for i in 0..channel.position_keys.len() {
                        let key = &channel.position_keys[i];

                        let next_i = if i + 1 >= channel.position_keys.len() {
                            0
                        } else {
                            i + 1
                        };

                        let next_key = &channel.position_keys[next_i];

                        if state.time < key.time {
                            let denom = next_key.time - key.time;

                            let factor = if denom.abs() > f32::EPSILON {
                                (state.time - key.time) / denom
                            } else {
                                0.0
                            };

                            let interpolated = key.position.lerp(next_key.position, factor);
                            
                            total_position = total_position + interpolated;
                            animation_count += 1;
                            break;
                        }
                    }
                }

                // Rotation keys
                if channel.rotation_keys.len() == 1 {
                    total_rotation = total_rotation * channel.rotation_keys[0].rotation;
                    animation_count += 1;
                } else if !channel.rotation_keys.is_empty() {
                    for i in 0..channel.rotation_keys.len() {
                        let key = &channel.rotation_keys[i];

                        let next_i = if i + 1 >= channel.rotation_keys.len() {
                            0
                        } else {
                            i + 1
                        };

                        let next_key = &channel.rotation_keys[next_i];

                        if state.time < key.time {
                            let denom = next_key.time - key.time;

                            let factor = if denom.abs() > f32::EPSILON {
                                (state.time - key.time) / denom
                            } else {
                                0.0
                            };

                            let interpolated = key.rotation.slerp(next_key.rotation, factor);

                            total_rotation = total_rotation * interpolated;
                            animation_count += 1;
                            break;
                        }
                    }
                }

                // Scaling keys
                if channel.scaling_keys.len() == 1 {
                    total_scale = total_scale * channel.scaling_keys[0].scaling;
                    animation_count += 1;
                } else if !channel.scaling_keys.is_empty() {
                    for i in 0..channel.scaling_keys.len() {
                        let key = &channel.scaling_keys[i];

                        let next_i = if i + 1 >= channel.scaling_keys.len() {
                            0
                        } else {
                            i + 1
                        };

                        let next_key = &channel.scaling_keys[next_i];

                        if state.time < key.time {
                            let denom = next_key.time - key.time;

                            let factor = if denom.abs() > f32::EPSILON {
                                (state.time - key.time) / denom
                            } else {
                                0.0
                            };

                            let interpolated = key.scaling.lerp(next_key.scaling, factor);

                            total_scale = total_scale * interpolated;
                            animation_count += 1;
                            break;
                        }
                    }
                }
            }

            let node_state = &mut state.node_states[node_index];

            if animation_count == 0 {
                // No channels target this node — use cached decomposition
                let (s, r, t) = node.default_decomposed;
                node_state.position = t;
                node_state.rotation = r;
                node_state.scale = s;
            } else {
                // Store interpolated values directly (no matrix compose/decompose)
                node_state.position = total_position;
                node_state.rotation = total_rotation;
                node_state.scale = total_scale;
            }
        }
    }

    fn compute_node_matrices(
        &mut self,
        model: &ModelData,
        node_index: usize,
        parent_transform: Mat4,
    ) {
        let mut animation_count = 0;
        let mut total_position = Vec3::ZERO;
        let mut total_rotation = Quat::IDENTITY;
        let mut total_scale = Vec3::ONE;

        for animation_index in 0..model.animations.len() {
            if !self.active_animations[animation_index] {
                continue;
            }

            let state = &self.states[animation_index];

            if state.weight <= 0.0 {
                continue;
            }

            let node_state = &state.node_states[node_index];

            if animation_count == 0 {
                total_position = node_state.position;
                total_rotation = node_state.rotation;
                total_scale = node_state.scale;
            } else {
                let blend = 0.5 * state.weight;

                total_position = node_state.position.lerp(total_position, blend);
                total_rotation = node_state.rotation.slerp(total_rotation, blend);
                total_scale = node_state.scale.lerp(total_scale, blend);
            }

            animation_count += 1;
        }

        let local_transform = if animation_count == 0 {
            model.nodes[node_index].default_transform
        } else {
            let position_matrix = Mat4::from_translation(total_position);
            let rotation_matrix = Mat4::from_quat(total_rotation);
            let scaling_matrix = Mat4::from_scale(total_scale);

            position_matrix * rotation_matrix * scaling_matrix
        };

        let global_transform = parent_transform * local_transform;
        self.node_matrices[node_index] = global_transform * model.nodes[node_index].offset_matrix;

        // Recurse to children via first_child / next_sibling linked list
        let mut child_index = model.nodes[node_index].first_child_index;
        
        while child_index != -1 {
            let ci = child_index as usize;
            self.compute_node_matrices(model, ci, global_transform);
            child_index = model.nodes[ci].next_sibling_index;
        }
    }
}
