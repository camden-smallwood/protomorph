// light_volume.wgsl — roll-locked billboard strip for light volumes (rmlv).
// A faithful port of light_volume_fx.hlsl (VS lines 90-178, PS 200-231).
// One instance = one profile (a cross-section quad, 4 verts). Profiles are
// stacked along the shaft (built CPU-side) and each is a camera-facing quad
// that is ROLL-LOCKED to the shaft direction, with overdraw-compensated alpha
// so the shaft reads the same brightness from any angle / profile density.
//
// Per-profile data (denormalized so a whole strip draws as one instanced
// call) mirrors `s_profile_state` + the per-system `g_all_state`.

struct Camera { view: mat4x4<f32>, projection: mat4x4<f32>, };
// x = V_ILLUM_EXPOSURE (the engine self-illum exposure for additive blends).
struct Frame { exposure: vec4<f32>, };
struct Profile {
    pos: vec4<f32>,             // xyz = world position; w unused
    color_intensity: vec4<f32>, // rgb = color, a = intensity
    dir_thickness: vec4<f32>,   // xyz = shaft direction (unit), w = thickness
    params: vec4<f32>,          // profile_length, profile_distance, num_profiles, brightness_ratio
    alpha: vec4<f32>,           // x = alpha
};
// x = is_multiply blend (1 → multiply output convention, no exposure).
struct DrawParams { flags: vec4<u32>, };

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var<storage, read> profiles: array<Profile>;
// Scene (opaque + water) depth — for hard occlusion (the engine depth-tests
// light volumes against geometry).
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var<uniform> draw: DrawParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    var out: VsOut;
    // Quad corner (engine shift {{0,0},{1,0},{1,1},{0,1}}); the ^ folds the
    // 6-index triangle list into the 4 unique corners (matches the particle VS).
    let qi = vid ^ ((vid & 2u) >> 1u);
    var shifts = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let shift = shifts[qi];

    let s = profiles[iid];
    let position = s.pos.xyz;
    let direction = s.dir_thickness.xyz;
    let thickness = s.dir_thickness.w;
    let profile_length = s.params.x;
    let profile_distance = s.params.y;
    let num_profiles = s.params.z;
    let brightness_ratio = s.params.w;

    // Camera position from the view matrix (rows of the rotation = camera
    // axes in world; same derivation as particle_render.wgsl).
    let vt = vec3<f32>(camera.view[3][0], camera.view[3][1], camera.view[3][2]);
    let cam_pos = -vec3<f32>(
        dot(camera.view[0].xyz, vt),
        dot(camera.view[1].xyz, vt),
        dot(camera.view[2].xyz, vt),
    );
    let camera_to_profile = normalize(position - cam_pos);
    // How perpendicular the shaft is to the view (0 = looking down it).
    let sin_view_angle = length(cross(direction, camera_to_profile));

    // Head-on the profile is square (thickness); side-on it stretches to
    // profile_length along the shaft projection.
    let profile_length_eff = mix(thickness, profile_length, sin_view_angle);
    let billboard_pos = (shift - 0.5) * vec2<f32>(profile_length_eff, thickness);

    // Basis: faces the camera but is rolled so the long axis follows the shaft.
    let basis1 = normalize(cross(camera_to_profile, direction));
    let basis0 = cross(camera_to_profile, basis1);
    let world_pos = position + billboard_pos.x * basis0 + billboard_pos.y * basis1;

    out.clip = camera.projection * (camera.view * vec4<f32>(world_pos, 1.0));
    out.uv = shift;

    var color = vec4<f32>(s.color_intensity.rgb * s.color_intensity.w, s.alpha.x);
    if (draw.flags.x == 0u) {
        // non-multiply → self-illum exposure (engine V_ILLUM_EXPOSURE).
        color = vec4<f32>(color.rgb * frame.exposure.x, color.a);
    }
    // Total intensity ~constant from any angle / density: divide alpha by the
    // expected overdraw. spacing guarded to avoid NaN when looking down the
    // shaft (sin→0): the ratio → huge, min() clamps overdraw to num_profiles.
    let spacing = max(profile_distance * sin_view_angle, 1e-6);
    let overdraw = min(num_profiles, profile_length_eff / spacing);
    color.a = color.a * mix(brightness_ratio, 1.0, sin_view_angle) / max(overdraw, 1e-4);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Sample first (uniform control flow) so the conditional discard below is
    // legal. `in.clip` in the FS = framebuffer position (xy pixels, z depth).
    var blended = textureSample(base_tex, samp, in.uv) * in.color;
    // Hard depth occlusion (engine depth-tests light volumes): kill fragments
    // behind opaque geometry. Reverse-Z — bigger depth = closer, so a fragment
    // with SMALLER depth than the scene is behind it.
    let scene_d = textureLoad(scene_depth, vec2<i32>(in.clip.xy), 0);
    if (scene_d > 0.0 && in.clip.z < scene_d) {
        discard;
    }
    if (draw.flags.x != 0u) {
        // multiply blend output convention (engine: lerp(1, rgb, a)).
        blended = vec4<f32>(mix(vec3<f32>(1.0, 1.0, 1.0), blended.rgb, blended.a), blended.a);
    }
    return blended;
}
