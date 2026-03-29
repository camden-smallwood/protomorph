// 3D noise texture generator for volumetric clouds
// Outputs 128x128x128 Rgba8Unorm:
//   R: Perlin-Worley hybrid (low-freq base shape)
//   G: Worley F1 (frequency 4)
//   B: Worley F2 (frequency 8)
//   A: Worley F3 (frequency 16)

@group(0) @binding(0) var output: texture_storage_3d<rgba8unorm, write>;

// --- PCG-based hash functions (no sin(), no stripe artifacts) ---

fn pcg(v: u32) -> u32 {
    var state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg3(p: vec3<u32>) -> vec3<u32> {
    var v = p * vec3(1664525u, 1013904223u, 1103515245u);
    v = v + v.yzx * v.zxy;
    v = v ^ (v >> vec3(16u));
    v = v + v.yzx * v.zxy;
    return v;
}

fn hash3(p: vec3<f32>) -> vec3<f32> {
    let ip = vec3<u32>(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z));
    let h = pcg3(ip);
    return vec3<f32>(
        f32(h.x) / 4294967295.0,
        f32(h.y) / 4294967295.0,
        f32(h.z) / 4294967295.0
    );
}

fn hash3i(p: vec3<i32>) -> vec3<f32> {
    let ip = vec3<u32>(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z));
    let h = pcg3(ip);
    return vec3<f32>(
        f32(h.x) / 4294967295.0,
        f32(h.y) / 4294967295.0,
        f32(h.z) / 4294967295.0
    );
}

fn hash1i(p: vec3<i32>) -> f32 {
    let combined = bitcast<u32>(p.x) + bitcast<u32>(p.y) * 1013u + bitcast<u32>(p.z) * 2147u;
    return f32(pcg(combined)) / 4294967295.0;
}

// --- Worley noise (cellular) ---

fn worley(p: vec3<f32>, freq: f32) -> f32 {
    let scaled = p * freq;
    let cell = vec3<i32>(floor(scaled));
    let local = fract(scaled);

    var min_dist = 1.0;

    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let offset = vec3<f32>(f32(x), f32(y), f32(z));
                let neighbor = cell + vec3(x, y, z);
                // Wrap for tiling
                let freq_i = i32(freq);
                let wrapped = ((neighbor % freq_i) + freq_i) % freq_i;
                let feature_point = offset + hash3i(wrapped) - local;
                let dist = length(feature_point);
                min_dist = min(min_dist, dist);
            }
        }
    }

    return saturate(min_dist);
}

// --- Perlin noise (gradient) ---

fn fade(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn grad(hash_val: f32, p: vec3<f32>) -> f32 {
    let h = u32(hash_val * 16.0) & 15u;
    let u_val = select(p.y, p.x, h < 8u);
    let v_val = select(select(p.z, p.x, h == 12u || h == 14u), p.y, h < 4u);
    return select(-u_val, u_val, (h & 1u) == 0u) + select(-v_val, v_val, (h & 2u) == 0u);
}

fn perlin(p: vec3<f32>, freq: f32) -> f32 {
    let scaled = p * freq;
    let cell = vec3<i32>(floor(scaled));
    let local = fract(scaled);
    let f = fade(local);
    let freq_i = i32(freq);

    // 8 corner gradients with tiling
    let c000 = ((cell) % freq_i + freq_i) % freq_i;
    let c100 = ((cell + vec3(1, 0, 0)) % freq_i + freq_i) % freq_i;
    let c010 = ((cell + vec3(0, 1, 0)) % freq_i + freq_i) % freq_i;
    let c110 = ((cell + vec3(1, 1, 0)) % freq_i + freq_i) % freq_i;
    let c001 = ((cell + vec3(0, 0, 1)) % freq_i + freq_i) % freq_i;
    let c101 = ((cell + vec3(1, 0, 1)) % freq_i + freq_i) % freq_i;
    let c011 = ((cell + vec3(0, 1, 1)) % freq_i + freq_i) % freq_i;
    let c111 = ((cell + vec3(1, 1, 1)) % freq_i + freq_i) % freq_i;

    let g000 = grad(hash1i(c000), local);
    let g100 = grad(hash1i(c100), local - vec3(1.0, 0.0, 0.0));
    let g010 = grad(hash1i(c010), local - vec3(0.0, 1.0, 0.0));
    let g110 = grad(hash1i(c110), local - vec3(1.0, 1.0, 0.0));
    let g001 = grad(hash1i(c001), local - vec3(0.0, 0.0, 1.0));
    let g101 = grad(hash1i(c101), local - vec3(1.0, 0.0, 1.0));
    let g011 = grad(hash1i(c011), local - vec3(0.0, 1.0, 1.0));
    let g111 = grad(hash1i(c111), local - vec3(1.0, 1.0, 1.0));

    let lerp_x0 = mix(g000, g100, f.x);
    let lerp_x1 = mix(g010, g110, f.x);
    let lerp_x2 = mix(g001, g101, f.x);
    let lerp_x3 = mix(g011, g111, f.x);

    let lerp_y0 = mix(lerp_x0, lerp_x1, f.y);
    let lerp_y1 = mix(lerp_x2, lerp_x3, f.y);

    return mix(lerp_y0, lerp_y1, f.z) * 0.5 + 0.5;
}

// --- Remap helper ---

fn remap(value: f32, old_min: f32, old_max: f32, new_min: f32, new_max: f32) -> f32 {
    return clamp((value - old_min) / (old_max - old_min), 0.0, 1.0) * (new_max - new_min) + new_min;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = 128u;
    if (id.x >= size || id.y >= size || id.z >= size) {
        return;
    }

    let uvw = vec3<f32>(id) / f32(size);

    // R: Perlin-Worley hybrid — smooth blobs with cellular edges
    let perlin_val = perlin(uvw, 4.0);
    let worley_low = worley(uvw, 4.0);
    let worley_inv = 1.0 - worley_low;
    let perlin_worley = remap(perlin_val, worley_inv * 0.4, 1.0, 0.0, 1.0);

    // G, B, A: Worley FBM octaves at increasing frequencies (inverted for billowy look)
    let w1 = 1.0 - worley(uvw, 4.0);
    let w2 = 1.0 - worley(uvw, 8.0);
    let w3 = 1.0 - worley(uvw, 16.0);

    textureStore(output, id, vec4(perlin_worley, w1, w2, w3));
}

// --- Detail noise: 32^3 high-frequency Worley for cloud edge erosion ---
// Reuses pcg, pcg3, hash3i, worley from above.

@compute @workgroup_size(4, 4, 4)
fn cs_detail(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = 32u;
    if (id.x >= size || id.y >= size || id.z >= size) { return; }
    let uvw = vec3<f32>(id) / f32(size);
    let w1 = 1.0 - worley(uvw, 2.0);
    let w2 = 1.0 - worley(uvw, 4.0);
    let w3 = 1.0 - worley(uvw, 8.0);
    textureStore(output, id, vec4(w1, w2, w3, 1.0));
}

