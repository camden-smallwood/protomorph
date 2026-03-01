use glam::{Mat4, Vec2, Vec3};

pub struct Camera {
    pub field_of_view: f32, // radians
    pub aspect_ratio: f32,
    pub near_clip: f32,
    pub far_clip: f32,
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Vec2, // [0]=yaw, [1]=pitch, in degrees
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3, // fixed (0, 0, 1)
    pub view: Mat4,
    pub projection: Mat4,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            field_of_view: 0.70,
            aspect_ratio: 1.0,
            near_clip: 0.01,
            far_clip: 1000.0,
            position: Vec3::new(0.0, 0.0, 0.7),
            velocity: Vec3::ZERO,
            rotation: Vec2::ZERO,
            forward: Vec3::Y,
            right: Vec3::X,
            up: Vec3::Z,
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
        }
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect_ratio = width as f32 / height as f32;
        }
    }

    pub fn update(&mut self) {
        // Clamp pitch to ±89°
        self.rotation.y = self.rotation.y.clamp(-89.0, 89.0);

        let yaw_rad = self.rotation.x.to_radians();
        let pitch_rad = self.rotation.y.to_radians();
        let pitch_cos = pitch_rad.cos();

        // Compute forward from yaw/pitch (Z-up spherical coords)
        self.forward = Vec3::new(
            yaw_rad.cos() * pitch_cos,
            yaw_rad.sin() * pitch_cos,
            pitch_rad.sin(),
        )
        .normalize();

        // right = normalize(cross(up, forward))
        self.right = self.up.cross(self.forward).normalize();

        // Apply velocity to position
        self.position += self.velocity;

        // view = look_at_rh(position, position + forward, up)
        let target = self.position + self.forward;
        self.view = Mat4::look_at_rh(self.position, target, self.up);

        // projection = perspective_rh(fov, aspect, near, far)
        self.projection =
            Mat4::perspective_rh(self.field_of_view, self.aspect_ratio, self.near_clip, self.far_clip);
    }

    pub fn rotate_towards_point(&mut self, point: Vec3, amount: f32) {
        let distance = point - self.position;
        let length = distance.length();
        self.rotation.x = (distance.y.atan2(distance.x)).to_degrees() * amount;
        self.rotation.y = (distance.z / length).asin().to_degrees() * amount;
    }
}
