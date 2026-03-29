use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::models::ModelData;

pub const EYE_HEIGHT: f32 = 0.7;
pub const GRAVITY: f32 = 9.8;
pub const JUMP_VELOCITY: f32 = 3.43; // ~0.6 unit peak height, ~0.35s to apex
pub const PLAYER_RADIUS: f32 = 0.3;
pub const GROUND_SNAP: f32 = 0.15;

/// A wall edge projected to XY, with the Z range it spans.
pub struct WallSegment {
    pub a: Vec2,
    pub b: Vec2,
    pub z_min: f32,
    pub z_max: f32,
}

pub struct CollisionMesh {
    pub floor_triangles: Vec<(Vec3, Vec3, Vec3)>,
    pub wall_segments: Vec<WallSegment>,
}

pub struct PlayerPhysics {
    pub vertical_velocity: f32,
    pub is_grounded: bool,
}

impl PlayerPhysics {
    pub fn new() -> Self {
        Self {
            vertical_velocity: 0.0,
            is_grounded: true,
        }
    }
}

/// Extract collision geometry from a model's meshes, transformed to world space.
/// Classifies triangles by face normal into floor (for raycasting) and wall segments
/// (for horizontal collision).
pub fn build_collision_mesh(model: &ModelData, model_matrix: Mat4) -> CollisionMesh {
    let mut floor_triangles = Vec::new();
    let mut wall_segments = Vec::new();

    for mesh in &model.meshes {
        for chunk in mesh.indices.chunks(3) {
            if chunk.len() < 3 {
                continue;
            }

            let p0 = mesh.vertices[chunk[0] as usize].position;
            let p1 = mesh.vertices[chunk[1] as usize].position;
            let p2 = mesh.vertices[chunk[2] as usize].position;

            // Transform to world space
            let v0 = (model_matrix * Vec4::new(p0[0], p0[1], p0[2], 1.0)).truncate();
            let v1 = (model_matrix * Vec4::new(p1[0], p1[1], p1[2], 1.0)).truncate();
            let v2 = (model_matrix * Vec4::new(p2[0], p2[1], p2[2], 1.0)).truncate();

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(edge2);
            if normal.length_squared() < 1e-10 {
                continue; // degenerate triangle
            }
            let normal = normal.normalize();

            if normal.z.abs() > 0.7 {
                // Floor or ceiling triangle
                floor_triangles.push((v0, v1, v2));
            } else if normal.z.abs() < 0.3 {
                // Wall triangle — extract edges projected to XY with Z range
                let tri_z_min = v0.z.min(v1.z).min(v2.z);
                let tri_z_max = v0.z.max(v1.z).max(v2.z);
                let edges = [(v0, v1), (v1, v2), (v2, v0)];
                for (a, b) in edges {
                    let a2 = Vec2::new(a.x, a.y);
                    let b2 = Vec2::new(b.x, b.y);
                    // Skip near-zero-length edges in XY
                    if (b2 - a2).length_squared() > 1e-6 {
                        // Only add if not a duplicate (shared edge from adjacent triangle)
                        let is_dup = wall_segments.iter().any(|seg: &WallSegment| {
                            (seg.a.distance(a2) < 0.01 && seg.b.distance(b2) < 0.01)
                                || (seg.a.distance(b2) < 0.01 && seg.b.distance(a2) < 0.01)
                        });
                        if !is_dup {
                            wall_segments.push(WallSegment {
                                a: a2,
                                b: b2,
                                z_min: tri_z_min,
                                z_max: tri_z_max,
                            });
                        }
                    }
                }
            }
        }
    }

    CollisionMesh {
        floor_triangles,
        wall_segments,
    }
}

/// Moller-Trumbore ray-triangle intersection.
/// Returns the ray parameter t if hit (hit point = origin + dir * t).
fn ray_triangle(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = dir.cross(edge2);
    let a = edge1.dot(h);

    if a.abs() < 1e-8 {
        return None; // parallel
    }

    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(edge1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(q);
    if t > 1e-6 {
        Some(t)
    } else {
        None
    }
}

/// Cast ray downward from position. Returns the Z height of the closest floor below.
pub fn ground_raycast(position: Vec3, floor_tris: &[(Vec3, Vec3, Vec3)]) -> Option<f32> {
    let ray_origin = position;
    let ray_dir = Vec3::NEG_Z;

    let mut closest_t = f32::MAX;
    let mut hit = false;

    for &(v0, v1, v2) in floor_tris {
        if let Some(t) = ray_triangle(ray_origin, ray_dir, v0, v1, v2) {
            if t < closest_t {
                closest_t = t;
                hit = true;
            }
        }
    }

    if hit {
        Some(ray_origin.z - closest_t)
    } else {
        None
    }
}

/// Closest point on a 2D line segment to a point.
fn closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    let len_sq = ab.dot(ab);
    if len_sq < 1e-10 {
        return a;
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a + ab * t
}

/// Resolve horizontal position against wall segments.
/// Only collides with walls whose Z range overlaps the player's feet.
/// Iterates up to 3 times for corner cases.
pub fn collide_and_slide(mut pos: Vec2, feet_z: f32, wall_segments: &[WallSegment], radius: f32) -> Vec2 {
    for _ in 0..3 {
        let mut pushed = false;
        for seg in wall_segments {
            // Skip walls the player is above or below
            if feet_z >= seg.z_max || feet_z < seg.z_min {
                continue;
            }
            let closest = closest_point_on_segment(pos, seg.a, seg.b);
            let diff = pos - closest;
            let dist = diff.length();

            if dist < radius && dist > 1e-6 {
                let normal = diff / dist;
                let penetration = radius - dist;
                pos += normal * penetration;
                pushed = true;
            }
        }
        if !pushed {
            break;
        }
    }
    pos
}
