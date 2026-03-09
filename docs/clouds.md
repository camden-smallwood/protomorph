# Procedural Cloud Rendering — Research & Plan

## 5 Approaches Ranked by Quality vs Effort

| Rank | Approach | Quality | Effort | Cost/frame |
|------|----------|---------|--------|------------|
| 1 | **2D Noise Flat Layer** | 5/10 | 1-2 days | 0.1-0.3ms |
| 2 | **Cubemap-Baked Volumetric** | 7/10 | ~1 week | 0.05ms amortized |
| 3 | **Full Ray-Marched Volumetric** | 10/10 | 2-4 weeks | 0.5-2.0ms |
| 4 | **Billboard Particles** | 6/10 | 3-5 days | 0.3-1.0ms |
| 5 | **Halo 3 Dual-Layer** | 6/10 | 2-3 days | 0.2-0.5ms |

---

## Approach 1: 2D Noise Flat Layer (Cheapest, simplest)

Render 1-2 flat cloud planes at fixed altitudes inside `compute_sky()`. Use fractal Brownian motion (fBM) from scrolling 2D noise octaves for density. Add fake volumetric lighting with 4 shadow samples along the sun direction (Beer's law). ~80 lines of WGSL, zero new passes or textures.

### Core Algorithm

```wgsl
fn cloud_density(uv: vec2<f32>, time: f32) -> f32 {
    var f = 0.0;
    var scale = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 6; i++) {
        let offset = time * vec2(0.01, 0.005) * freq;
        f += scale * noise_2d((uv + offset) * freq);
        freq *= 2.02;
        scale *= 0.5;
    }
    let a = 0.4; // coverage control
    let b = 0.7;
    return clamp((f - a) / (b - a), 0.0, 1.0);
}
```

### Fake Volumetric Lighting on Flat Layer

Even a flat layer can appear volumetric with 4 shadow samples along the sun direction:

```wgsl
fn cloud_shadow(uv: vec2<f32>, sun_dir_2d: vec2<f32>, time: f32) -> f32 {
    var shadow = 0.0;
    for (var i = 1; i <= 4; i++) {
        let offset_uv = uv + sun_dir_2d * f32(i) * 0.01;
        shadow += max(cloud_density(offset_uv, time) - cloud_density(uv, time), 0.0);
    }
    return exp(-shadow * 2.0); // Beer's law approximation
}
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Trivial to implement (< 100 lines of WGSL) | Flat appearance, no parallax |
| Nearly free performance (< 0.3ms) | No volumetric self-shadowing |
| Works entirely in fragment shader | Looks bad from above or at extreme angles |
| Can use a single 2D noise texture or compute procedurally | Limited cloud type variety |

---

## Approach 2: Cubemap-Baked Volumetric (Best quality-per-ms)

Ray-march volumetric clouds into a low-res cubemap (128-256 per face), updating 1-2 faces per frame. Sample in `compute_sky()` and blend with atmosphere.

### Integration

We already have cubemap infrastructure for the environment probe (`env_probe_sky.wgsl`). The plan:
1. Render clouds into a separate low-res cubemap (128 or 256 per face)
2. Update 1-2 faces per frame (full update every 3-6 frames)
3. In `compute_sky()`, sample the cloud cubemap with the view ray direction
4. Alpha-blend cloud color over atmospheric sky color

### Update Budget

At 128x128 per face with 64-step ray march: ~16K pixels x 64 steps = ~1M texture samples per face. One face per frame is ~0.3ms.

### Pros/Cons

| Pros | Cons |
|------|------|
| Amortized cost is tiny | 3-6 frame latency for cloud updates |
| Reuses existing cubemap infrastructure | Cannot fly through clouds |
| Clouds automatically available for reflections | Low angular resolution (parallax issues) |
| Simple compositing in existing sky shader | Popping artifacts during fast camera rotation |

---

## Approach 3: Full Ray-Marched Volumetric (The Gold Standard)

The Horizon Zero Dawn / Nubis approach (Schneider, SIGGRAPH 2015/2017/2023). Industry standard for AAA cloud rendering.

### 3D Noise Textures (Two-Texture Approach)

**Base Shape Noise (128x128x128, RGBA8, ~8 MB):**
- R: Perlin-Worley noise (low-frequency billowy shapes)
- G, B, A: Worley noise at increasing frequencies (3 octaves)
- Defines macro cloud shape

**Detail Erosion Noise (32x32x32, RGBA8, ~128 KB):**
- Worley noise at high frequencies in each channel
- Erodes edges of base shape for wispy details

**Weather Map (512x512, RGBA8, ~1 MB):**
- R: Cloud coverage (0 = clear, 1 = overcast)
- G: Cloud type (0 = stratus, 1 = cumulus)
- B: Precipitation / wetness
- Can be procedurally generated from animated Perlin noise

### Density Function

```
fn sample_cloud_density(p: vec3<f32>, weather: vec4<f32>) -> f32 {
    let height_fraction = (p.z - CLOUD_MIN_HEIGHT) / (CLOUD_MAX_HEIGHT - CLOUD_MIN_HEIGHT);
    let density_gradient = height_gradient(height_fraction, weather.g);

    // Sample base shape noise
    let base_noise = textureSample(t_cloud_base, s_cloud, p * BASE_SCALE);
    let base_fbm = base_noise.g * 0.625 + base_noise.b * 0.25 + base_noise.a * 0.125;
    let base_cloud = remap(base_noise.r, base_fbm - 1.0, 1.0, 0.0, 1.0);

    // Apply weather coverage
    let coverage = weather.r;
    var density = remap(base_cloud * density_gradient, 1.0 - coverage, 1.0, 0.0, 1.0);
    density *= coverage;

    // Detail erosion (only where density > 0)
    if (density > 0.0) {
        let detail = textureSample(t_cloud_detail, s_cloud, p * DETAIL_SCALE);
        let detail_fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
        let detail_modifier = mix(detail_fbm, 1.0 - detail_fbm,
            clamp(height_fraction * 10.0, 0.0, 1.0));
        density = remap(density, detail_modifier * 0.2, 1.0, 0.0, 1.0);
    }

    return max(density, 0.0);
}
```

The `remap` function: `remap(x, lo, hi, new_lo, new_hi) = new_lo + (x - lo) / (hi - lo) * (new_hi - new_lo)`

The `height_gradient` returns altitude-dependent density profiles per cloud type. Stratus = thin flat, cumulus = tall billowy with density concentrated in the middle.

### Ray Marching Loop

```
fn ray_march_clouds(ray_origin: vec3, ray_dir: vec3) -> vec4 {
    // Intersect ray with cloud layer slab [CLOUD_MIN, CLOUD_MAX]
    let t_min = (CLOUD_MIN_HEIGHT - ray_origin.z) / ray_dir.z;
    let t_max = (CLOUD_MAX_HEIGHT - ray_origin.z) / ray_dir.z;

    var t = t_min;
    let step_size = (t_max - t_min) / f32(MAX_STEPS); // 64-128 steps
    t += blue_noise_offset * step_size; // temporal jitter

    var transmittance = 1.0;
    var light_energy = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        if (transmittance < 0.01) { break; } // early termination

        let pos = ray_origin + ray_dir * t;
        let weather = sample_weather_map(pos.xy);
        let density = sample_cloud_density(pos, weather);

        if (density > 0.0) {
            let light_transmittance = light_march(pos, sun_direction);

            // Beer-Powder combined term
            let beer = exp(-density * step_size * EXTINCTION_COEFF);
            let powder = 1.0 - exp(-density * step_size * 2.0);
            let beer_powder = beer * mix(powder, 1.0, remap(cos_theta, -1, 1, 0, 1));

            // Dual-lobe Henyey-Greenstein phase function
            let phase = hg_phase(cos_theta, 0.8) * 0.8 + hg_phase(cos_theta, -0.5) * 0.2;

            let luminance = light_transmittance * beer_powder * phase;
            light_energy += luminance * transmittance * density * step_size;
            transmittance *= beer;
        }

        t += step_size;
    }

    let alpha = 1.0 - transmittance;
    return vec4(light_energy * sun_color + ambient_sky_color * alpha, alpha);
}
```

### Lighting Model

**Beer's Law**: `T(d) = exp(-sigma_extinction * d)` where sigma_extinction ~ 0.03-0.04/m for dense clouds.

**Henyey-Greenstein Phase Function**:
```wgsl
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 / (4.0 * PI)) * (1.0 - g2) / (denom * sqrt(denom));
}
```
Dual-lobe: forward (g=0.8, weight 0.8) + back (g=-0.5, weight 0.2) for silver linings.

**Powder Effect**: `powder(d) = 1.0 - exp(-2.0 * d)`. Simulates edge darkening from in-scattering. View-angle dependent — strongest when looking away from sun.

**Light Marching (Self-Shadowing)**: 5-6 samples toward sun with exponentially increasing step sizes:
```
fn light_march(pos: vec3, sun_dir: vec3) -> f32 {
    var optical_depth = 0.0;
    var light_step = 50.0;
    var light_pos = pos;
    for (var j = 0; j < 6; j++) {
        light_pos += sun_dir * light_step;
        optical_depth += sample_cloud_density(light_pos, sample_weather(light_pos.xy));
        light_step *= 1.5;
    }
    return exp(-optical_depth * EXTINCTION_COEFF);
}
```

### Performance Optimizations

The raw ray march at full resolution is 10-30ms. These optimizations bring it to 0.5-2.0ms:

**Quarter-Resolution + Temporal Reprojection (the big win):**
- Render at 1/4 res with 4x4 checkerboard pattern (Bayer matrix offsets)
- Each frame renders 1/16th of pixels; over 16 frames every pixel updates once
- Upsample quarter-res to full-res with bilinear filtering
- Blend with reprojected previous frame (convergence ~0.75 zenith, ~0.5 horizon)
- 16x cost reduction

**Blue Noise Temporal Offset:**
```
let jitter = blue_noise(pixel_coord) + halton_offset[frame_index % 8];
ray_start += jitter * step_size;
```

**Adaptive Step Size:** Large steps (100m) in empty space, small steps (20m) inside clouds.

**Skip Detail Texture When Far:** Beyond a distance threshold, skip the detail erosion noise sample.

### Memory Budget

| Resource | Size | Notes |
|----------|------|-------|
| Base shape 3D (128^3 RGBA8) | 8 MB | Perlin-Worley + Worley octaves |
| Detail erosion 3D (32^3 RGBA8) | 128 KB | High-freq Worley |
| Weather map (512^2 RGBA8) | 1 MB | Coverage, type, precipitation |
| Blue noise (128^2 R8) | 16 KB | Shared with other effects |
| Cloud render target (quarter-res) | ~1 MB | RGBA16F |
| Temporal history (full-res) | ~16 MB | RGBA16F, double-buffered |
| **Total** | **~26 MB** | |

### Fragment-Only (No Compute Shaders)

Entirely achievable. Horizon Zero Dawn used fragment shaders on PS4:
1. **Cloud pass**: Full-screen triangle at quarter res, fragment shader ray marches, outputs to RGBA16F
2. **Upsample/composite pass**: Full-screen triangle at full res, temporal blend + bilateral upsample
3. **Integration**: Composite into lighting pass on sky pixels or as separate alpha-blend pass

### Pros/Cons

| Pros | Cons |
|------|------|
| Photorealistic results | Complex (~500-1000 lines WGSL + Rust) |
| Dynamic time-of-day lighting | Requires 3D noise texture pipeline |
| Fly-through capability | Temporal reprojection ghosting |
| Industry-proven | ~26 MB memory |

---

## Approach 4: Billboard Cloud Particles

Camera-facing quads with pre-rendered cloud puff textures (8-16 variants). Scatter hundreds of billboards, sort back-to-front, apply aerial perspective per billboard. Forward render pass with alpha blending.

| Pros | Cons |
|------|------|
| Simple geometry | Obvious "card" look at close range |
| Artist-controllable | Sorting overhead |
| Very cheap per-billboard | No dynamic lighting response |
| Low memory | Can't fly through |

---

## Approach 5: Halo 3 / Halo Reach Style

Two scrolling noise-based cloud layers at different altitudes rendered as part of the sky dome. Artist-authored cloud textures that scroll and blend. Simple directional lighting. Essentially Approach 1 with two layers for depth.

---

## Noise Generation

### Perlin-Worley Noise

The key innovation from Schneider (2015). Combines Perlin noise (smooth, connected shapes) with Worley/cellular noise (sharp, bubbly shapes):
```
perlin_worley = remap(perlin, 0.0, 1.0, worley, 1.0)
```
Creates noise with Perlin's connected structure but Worley's sharp edges — very cloud-like.

### Baking vs Runtime

**Bake offline (recommended):** Generate noise on CPU at build time, save as 3D texture files. Load as wgpu `Texture3D` with `Rgba8Unorm`. 128^3 at RGBA8 = 8 MB.

**Runtime:** Requires compute shaders. Not recommended for our fragment-only setup.

### Typical Textures

| Texture | Resolution | Format | Size | Contents |
|---------|-----------|--------|------|----------|
| Base shape | 128^3 | RGBA8 | 8 MB | R: Perlin-Worley, GBA: Worley octaves |
| Detail | 32^3 | RGBA8 | 128 KB | RGB: Worley high-freq, A: curl-like |
| Weather | 512^2 | RGBA8 | 1 MB | Coverage, type, precipitation |
| Blue noise | 128^2 | R8 | 16 KB | Spatiotemporal dithering |
| Curl noise 2D | 128^2 | RG8 | 32 KB | Distortion for animation |

---

## Integration with Our Renderer

### Current State

Our `compute_sky()` in `lighting.wgsl` already does Rayleigh/Mie scattering on `depth <= 0` pixels. Environment probe cubemap renders sky via `env_probe_sky.wgsl`. No compute shaders.

### Compositing Order

1. Render atmosphere (Rayleigh/Mie sky color) — already have this
2. Composite clouds over atmosphere with alpha blending
3. Add sun disk on top (occluded by clouds where alpha > threshold)
4. Apply atmospheric fog to geometry as usual

### Aerial Perspective on Clouds

Distant clouds fade into haze using existing `compute_scattering()`:
```wgsl
let cloud_world_pos = ray_origin + ray_dir * mean_cloud_distance;
let scattering = compute_scattering(camera_pos, cloud_world_pos);
let fogged_cloud = cloud_color * scattering[0] + scattering[1];
final = mix(sky_color, fogged_cloud, cloud_alpha);
```

### Cloud Shadows on Terrain

Modulate directional shadow factor by cloud transmittance:
```wgsl
let cloud_shadow = textureSample(t_weather, s_cloud, frag_position.xy * CLOUD_UV_SCALE);
shadow_factor *= mix(1.0, exp(-cloud_shadow.r * 3.0), cloud_shadow_strength);
```

---

## Recommended Phased Plan

**Phase 1 (immediate):** Add 2D noise clouds to `compute_sky()` in `lighting.wgsl`. Use 2-3 scrolling noise octaves with a coverage threshold. Apply simple Beer's law shadowing with 4 sun-direction samples. Zero new passes, zero new textures, zero new bind groups. ~80 lines of WGSL.

**Phase 2 (when ready):** Separate quarter-res cloud render pass with 3D noise textures and temporal reprojection. Bake noise textures offline. Composite in lighting shader on sky pixels. This gets to Nubis-level quality.

**Phase 3 (stretch):** Render clouds into environment probe cubemap for cloud reflections. Add cloud shadow modulation to CSM shadow factor.

---

## References

- [The Real-time Volumetric Cloudscapes of Horizon Zero Dawn (Schneider, SIGGRAPH 2015)](https://advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf)
- [Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine (SIGGRAPH 2017)](https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine)
- [Nubis Evolved (Guerrilla Games)](https://www.guerrilla-games.com/read/nubis-evolved)
- [Nubis Cubed (Schneider, 2023)](https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/Nubis%20Cubed.pdf)
- [A Scalable and Production Ready Sky and Atmosphere Rendering Technique (Hillaire, 2020)](https://sebh.github.io/publications/egsr2020.pdf)
- [Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite (Hillaire, SIGGRAPH 2016)](https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering)
- [Convincing Cloud Rendering - Frostbite Thesis (Hogfeldt)](https://www.cse.chalmers.se/~uffe/xjobb/RurikH%C3%B6gfeldt.pdf)
- [Optimisations for Real-Time Volumetric Cloudscapes (Toft & Bowles, 2016)](https://arxiv.org/abs/1609.05344)
- [Upsampling to Improve Volumetric Cloud Render Performance](https://www.vertexfragment.com/ramblings/volumetric-cloud-upsampling/)
- [Inigo Quilez - 2D Dynamic Clouds](https://iquilezles.org/articles/dynclouds/)
- [Sebastian Lague - Clouds (GitHub)](https://github.com/SebLague/Clouds)
- [Meteoros - Vulkan Cloud Renderer](https://github.com/AmanSachan1/Meteoros)
- [Creating the Atmospheric World of Red Dead Redemption 2 (Bauer, SIGGRAPH 2019)](https://advances.realtimerendering.com/s2019/index.htm)
- [Graphics Study: Red Dead Redemption 2](https://imgeself.github.io/posts/2020-06-19-graphics-study-rdr2/)
- [Halo Reach Effects Tech (Tchou, GDC 2011)](https://gdcvault.com/play/1014347/HALO-REACH-Effects)
- [Schneider's Publications](https://sites.google.com/view/vonschneidz/publications)
- [webgpu-sky-atmosphere (WebGPU Hillaire implementation)](https://github.com/JolifantoBambla/webgpu-sky-atmosphere)
