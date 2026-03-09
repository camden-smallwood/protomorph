# Protomorph Unified Rendering Improvement Plan

All findings from research rounds 1-7, deduplicated, grouped by feature, sorted by benefit within each section. Exclusive alternatives include pros/cons comparisons.

**Current renderer baseline:** wgpu/Rust deferred renderer, 1600x1200, Apple Silicon / Metal (TBDR). Beckmann NDF, Cook-Torrance geometry, Schlick Fresnel, Lambertian diffuse. 16-step linear SSR, 8-pass bloom, FXAA, half-res SSAO, half-res god rays, 3 CSM cascades, 2 point lights (6-face cubemap), 2 spot lights, 4-tap PCF. No clustered lighting, no TAA, no velocity buffer, no multi-scatter energy compensation.

---

## 1. G-Buffer & Rendering Architecture

### 1.1 Eliminate Position Buffer (reconstruct from depth)
- **Effort:** Medium | **Benefit:** 15+ MB/frame bandwidth saved
- Reconstruct world position from depth + `inv_view_proj` in lighting pass. Frees an entire Rgba16Float MRT slot that can be repurposed for velocity or removed entirely.

### 1.2 Fullscreen Triangle (replace fullscreen quad)
- **Effort:** Trivial | **Benefit:** Free — eliminates wasted helper pixels along quad diagonal
- 3 vertices, no vertex buffer. Vertex shader computes UVs from `vertex_index`. No reason not to do this immediately.

### 1.3 Depth Pre-Pass with Equal Compare
- **Effort:** Low | **Benefit:** Eliminates overdraw in G-buffer pass
- Render depth-only first, then G-buffer with `Equal` depth compare. Guarantees zero wasted fragment shading.

### 1.4 Reverse-Z Depth Buffer
- **Effort:** Low | **Benefit:** Dramatically better depth precision, enables infinite far plane
- Map near=1.0, far=0.0. Float32 has more precision near 0, which cancels the 1/z distribution. Sub-mm precision to ~1000m.

### 1.5 DCC Best Practices (Delta Color Compression)
- **Effort:** Trivial | **Benefit:** 20-40% effective bandwidth reduction (free from hardware)
- Always write ALL channels (partial writes disable DCC), clear to 0.0 or 1.0, avoid mutable formats. Audit existing clears and writes.

### 1.6 Normal Encoding — EXCLUSIVE CHOICES

Pick one encoding for the normal G-buffer target:

| Approach | Format | Size | Pros | Cons |
|----------|--------|------|------|------|
| **Octahedral (Rg16Float)** | Rg16Float | 4 B/px | Standard, well-understood, sufficient precision | No room for extra data |
| **RGB10A2 packed** | Rgb10a2Unorm | 4 B/px | Packs roughness (10-bit) + material flags (2-bit) into same target | Slightly less normal precision (10+10 vs 16+16 bits) |
| **Diamond encoding** | Rg16Float | 4 B/px | Preserves TBN handedness for anisotropy | More complex encode/decode |
| **Current (Rgba16Float)** | Rgba16Float | 8 B/px | Maximum precision | 2x bandwidth waste, overkill |

**Recommendation:** RGB10A2 if you want to pack material data; octahedral Rg16Float for simplicity. Either saves 4 bytes/pixel (7.68 MB/frame) over current Rgba16Float.

### 1.7 R11G11B10Float Lighting Buffer
- **Effort:** Low | **Benefit:** 7.68 MB/frame saved (halves lighting output)
- Replaces Rgba16Float. No alpha needed for lighting output. Sufficient precision for HDR lighting values.

### 1.8 Velocity Buffer (in freed position slot)
- **Effort:** Medium | **Benefit:** Enables TAA, temporal SSAO, SSR reprojection, motion blur
- Use the MRT slot freed by position buffer elimination. Prerequisite for many temporal techniques.

### 1.9 Material Palette Indexing
- **Effort:** Low | **Benefit:** 75% material target bandwidth savings
- 256-entry palette reduces material target to R8Uint. Inspired by Teardown's approach.

### 1.10 Architecture — EXCLUSIVE CHOICES (Long-term)

These are fundamental architecture alternatives. Each replaces the current deferred G-buffer:

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Traditional deferred (current)** | N/A | Simple, well-understood, good for your light count | Bandwidth-heavy MRTs, overdraw wastes shading |
| **Visibility buffer (R32Uint)** | High | Zero overdraw shading, compute material eval, scales to millions of triangles | Major rewrite, requires GPU-driven pipeline |
| **Deferred+ texturing (UV+matID)** | High | 30-50% G-buffer bandwidth, no overdraw texture fetches | Complex material fetch in compute, requires bindless |
| **Forward+ (eliminate G-buffer)** | High | Removes 28+ bytes/pixel G-buffer entirely | Requires robust light culling, harder to add screen-space effects |

**Recommendation:** Stay deferred for now. The format optimizations above (1.1, 1.6, 1.7) capture most bandwidth wins. Consider visibility buffer only when adding GPU-driven rendering.

### 1.11 Light Culling — EXCLUSIVE CHOICES

| Approach | Effort | Scales To | Pros | Cons |
|----------|--------|-----------|------|------|
| **Flat array (current)** | N/A | ~16 lights | Simplest | Doesn't scale |
| **Z-bin + bitmask** | Medium | 100+ lights | Memory-efficient, O(X*N+Y*N+Z) | Less established than clustered |
| **Clustered shading (3D grid)** | High | Hundreds | Well-proven in AAA | Higher memory, more complex |
| **Tiled Forward+** | Medium | 100+ lights | 2x faster than tiled deferred with MSAA | Less flexible for screen-space effects |

**Recommendation:** Z-bin + bitmask if you need more lights soon (medium effort, good scaling). Clustered shading for maximum future-proofing.

---

## 2. PBR / BRDF / Materials

### 2.1 Multi-Scatter Energy Compensation — EXCLUSIVE CHOICES

The single highest-impact PBR fix. Rough metals currently lose up to 60% energy.

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **LUT-free polynomial (Sforza & Pellacini)** | Low | Zero LUT overhead, ~15-20 ALU, 19/39 coefficients | Slightly less accurate than LUT at extreme values |
| **Kulla-Conty via existing DFG LUT** | Low | ~5 ALU using LUT you already have | Requires the DFG LUT texture binding |
| **Fast-MSX (SIGGRAPH Asia 2023)** | Medium | More physically motivated, better saturated colors on gold/copper | More complex math, V-groove model |

**Recommendation:** LUT-free polynomial — lowest effort, highest impact, no texture dependency. Fixes the most visible deficiency in the renderer.

### 2.2 GGX NDF + Hammon Fast Visibility
- **Effort:** Low | **Benefit:** ~8 ops/light saved, industry-standard specular
- Replace Beckmann NDF + Cook-Torrance geometry with GGX (Trowbridge-Reitz) + Hammon's fast V term. GGX has longer tails that look more natural on most materials. This is the industry standard since ~2014.

### 2.3 EON Energy-Preserving Diffuse — EXCLUSIVE CHOICES

Replace Lambertian with a roughness-aware diffuse model:

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **EON (Oren-Nayar with energy compensation)** | Low | Drop-in GLSL, adopted by OpenPBR, ~5-10 extra ALU, uses existing roughness | Slight ALU increase |
| **VMF Diffuse (EGSR 2024)** | Medium | Wider range of behaviors (dusty/lunar/porous), unified model | More complex math |
| **Lambertian (current)** | N/A | Simplest, cheapest | Inaccurate for rough surfaces, no retroreflection |

**Recommendation:** EON — lowest effort for significant quality gain on rough dielectrics (concrete, clay, fabric).

### 2.4 Geometric Specular Anti-Aliasing
- **Effort:** Low | **Benefit:** Fixes specular flicker at distance, ~4-6 ALU
- Compute normal variance from `dpdx(N)`/`dpdy(N)`, add to roughness^2. Production-ready (Unity HDRP, Square Enix). No precomputation.

### 2.5 EnvBRDFApprox (Analytical DFG)
- **Effort:** Low | **Benefit:** Frees one texture slot, ~4 ops, correct env Fresnel
- Karis's polynomial approximation replaces the 128x128 DFG LUT with ~4 ALU ops. Eliminates a dependent texture read per lit pixel.

### 2.6 GTSO Specular Occlusion via Bent Normals
- **Effort:** Low | **Benefit:** Prevents probe light leaking into occluded areas
- 1-line addition to lighting shader using bent normal from GTAO. Cone-cone intersection modulates specular probe contribution.

### 2.7 Improved Schlick Fresnel
- **Effort:** Trivial | **Benefit:** Better accuracy for metals at grazing angles
- Shape control parameter, same 6 instructions as standard Schlick.

### 2.8 White Furnace Test (CI Validation)
- **Effort:** Trivial | **Benefit:** Catches normalization/reciprocity/alpha bugs automatically
- Render white sphere against uniform environment. Energy should be conserved. Add as automated test.

### 2.9 Advanced Material Models (Future)

These add new material classes beyond the base metal/dielectric:

| Material | Effort | What it Enables |
|----------|--------|-----------------|
| **LTC Sheen (Zeltner 2022)** | Medium | Cloth, velvet, dust, peach fuzz via single LTC lobe |
| **Thin-Film Iridescence (Belcour 2017)** | Medium | Soap bubbles, oil slicks, tempered metals — modifies Fresnel only |
| **Clearcoat / Layered (Belcour 2018)** | Medium | Car paint, lacquered wood — ~5 ALU per additional layer |
| **Anisotropic IBL (Pacific Graphics 2024)** | Medium | Brushed metal, scratched plastic — uses existing split-sum cubemap |
| **Subsurface Scattering (Jimenez separable)** | Medium | Skin translucency — two-pass stencil-masked blur, sub-0.5ms |
| **Hair (Marschner via UE4 LUT)** | Medium | R/TT/TRT hair paths, correct energy conservation |
| **OpenPBR v1.1 uber-shader** | High | Full base+coat+fuzz+thin-film layered material system |

### 2.10 Parallax-Corrected Cubemaps
- **Effort:** Low | **Benefit:** ~5 ALU for spatially accurate reflections
- Ray-box intersection gives correct reflections from a single low-res cubemap. Dramatically reduces the number of probes needed.

### 2.11 FP16 Precision Management
- **Effort:** Medium | **Benefit:** 2x arithmetic throughput on compatible hardware
- FP16 is safe for dot products, SH, Lambert. Must keep FP32 for NDF power terms and GGX denominator.

---

## 3. SSR / Reflections

### 3.1 Roughness Culling (skip SSR for rough pixels)
- **Effort:** Trivial | **Benefit:** 30-60% pixel savings
- Current cutoff 0.8 is too generous. Lower to 0.4-0.5. Rough surfaces produce barely visible SSR anyway and are better served by probe fallback.

### 3.2 SSR Output UVs Only (decouple trace from resolve)
- **Effort:** Low | **Benefit:** 10-15% SSR cost + 50% output bandwidth
- Trace pass outputs hit UV + distance to Rg16Float. Separate resolve pass samples scene color. Enables roughness mip sampling, temporal reuse, and format savings.

### 3.3 Confidence-Based SSR/Probe Blending
- **Effort:** Low | **Benefit:** Eliminates hard seam between SSR and probe reflections
- Compute per-pixel confidence from hit distance, angle, edge proximity. Lerp between SSR and probe instead of hard binary switch. Essentially free (~few ALU).

### 3.4 Stride Relaxation + Distance-Based Thickness Bias
- **Effort:** Trivial | **Benefit:** 50-70% fewer false-hit artifacts on distant geometry
- Two variable changes in existing march loop. Stride grows with distance, thickness test threshold adapts to depth.

### 3.5 Jitter + Conditional Binary Refinement
- **Effort:** Trivial | **Benefit:** 15-25% SSR cost savings + eliminates banding
- Skip binary refinement for roughness > 0.3 (blur masks imprecision). Jitter initial ray by `0.5 * stepSize * blueNoise`.

### 3.6 Clip-Space Linear Interpolation
- **Effort:** Low | **Benefit:** 15-20% cheaper per SSR step
- Interpolate z in clip space, 1/w in clip space, xy in screen space. Eliminates per-step matrix-vector multiplication.

### 3.7 Previous-Frame Color Lookup
- **Effort:** Low | **Benefit:** Richer reflections including transparencies, fog, particles
- Sample previous frame's fully composited image instead of current frame's lighting buffer. Decouples SSR from render order.

### 3.8 SSR Tracing Method — EXCLUSIVE CHOICES

These are different algorithms for the core ray march loop:

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Linear march (current, 16 steps)** | N/A | Simplest | Very range-limited, wastes steps near camera |
| **Adaptive step sizing (contact-adaptive)** | Low | 2-3x ray distance, same 16 steps, zero extra memory | Slightly more ALU per step |
| **View-space proportional stride** | Low | 64-step quality from ~16 steps, uniform screen coverage | Needs per-step depth-based scaling |
| **Hi-Z tracing** | Medium | 40-60% cost reduction, 16-32 steps = 64+ linear steps | Requires Hi-Z mip chain generation pass |
| **DDA screen-space tracing (McGuire JCGT 2014)** | Low | Perspective-correct, contiguous pixel sampling, minimal divergence | Still fundamentally linear |

**Recommendation:** Start with **adaptive step sizing** (lowest effort, biggest bang). Graduate to **Hi-Z tracing** when you want maximum distance coverage. These are not exclusive — Hi-Z can use adaptive sizing within cells.

### 3.9 SSR Pixel Culling — EXCLUSIVE CHOICES

| Approach | Effort | Savings | Pros | Cons |
|----------|--------|---------|------|------|
| **Roughness cutoff (3.1 above)** | Trivial | 30-60% | Dead simple | Coarse granularity |
| **Material-mask gated (Playdead/INSIDE)** | Low | 60-80% | Artist control, flag specific surfaces | Requires material authoring |
| **Tile classification + stream compaction** | Medium | 50-70% | Dispatches only reflective pixels, best occupancy | Requires compute pre-pass, subgroup ops |

**Recommendation:** Roughness cutoff first (trivial). Add tile classification later for maximum savings.

### 3.10 SSR Temporal Accumulation
- **Effort:** Medium | **Benefit:** 50% step count reduction (8 steps + temporal = 16+ steady-state)
- Jitter ray origin per frame, reproject and blend with history. Requires velocity buffer (see 1.8). Dual blend/correction knobs for artist tuning.

### 3.11 Half-Res SSR with Bilateral Upsample
- **Effort:** Medium | **Benefit:** ~75% SSR cost
- Trace at 800x600, depth-aware bilateral upsample to full res. Combine with temporal accumulation for best results.

### 3.12 SSPR for Planar Surfaces (Water/Floors)
- **Effort:** Low-Medium | **Benefit:** 5-10x cheaper than SSR for flat surfaces
- Scatter-based projection via compute atomicMin. One dispatch, zero ray marching, ~0.3-0.4ms. Perfect for water planes and polished floors. Use stencil bit to route planar reflectors to SSPR.

### 3.13 Stochastic SSR with GGX VNDF (Future)
- **Effort:** High | **Benefit:** 50-70% ray ALU, quality of 4-8 rays from 1 ray + resolve
- Importance-sample the visible normal distribution. Requires restructuring into two passes + blue noise + temporal denoiser. Best combined with temporal accumulation.

### 3.14 Environment Probe Optimizations

| Technique | Effort | Benefit |
|-----------|--------|---------|
| **1 face/frame cycling** | Low | ~83% per-frame probe cost reduction |
| **LOD mip rendering for distant probes** | Low | ~256x fewer pixels for far probes |
| **Fast cubemap filtering (Manson & Sloan)** | Medium | 160-730us mip gen, fast enough every frame |
| **Compute IBL prefiltering (R11G11B10)** | Medium | 0.18ms on RTX 5080, avoids framebuffer switches |
| **BC6H real-time GPU compression** | Low | 8:1 memory savings, 5-15% faster probe sampling |

---

## 4. Shadows

### 4.1 CSM Stabilization (Anti-Shimmer)
- **Effort:** Low | **Benefit:** Free — eliminates shadow edge swimming
- Bounding sphere + texel snapping. CPU-side matrix modification only, zero GPU cost.

### 4.2 Shadow Pancaking
- **Effort:** Trivial | **Benefit:** Better precision, less acne without more bias
- Clamp `projPos.z = max(projPos.z, 0.0)` in shadow vertex shader. Tighter near-far range.

### 4.3 Slope-Scale Depth Bias (Hardware)
- **Effort:** Trivial | **Benefit:** Zero shader cost bias
- Use rasterizer-state `constant_bias + slope_scaled_bias` instead of shader-computed bias.

### 4.4 16-bit CSM Depth Maps
- **Effort:** Trivial | **Benefit:** Halves CSM bandwidth (48 MB to 24 MB)
- Orthographic projection = linear depth. Depth16Unorm is sufficient for CSM.

### 4.5 Far Cascade Update Skipping
- **Effort:** Low | **Benefit:** ~33% fewer CSM passes amortized
- Update far cascades every other frame. They move only a few texels per frame.

### 4.6 Shadow Filtering — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **4-tap PCF (current)** | N/A | Simple, fast | Hard edges, fixed penumbra |
| **Vogel disk + IGN** | Low | Better distribution at same tap count, integrates with TAA | Still fixed penumbra width |
| **DPCF (Dilated PCF)** | Low | Contact hardening at near-PCF cost, drop-in upgrade | Approximate penumbra |
| **EVSM/MSM (filterable maps)** | Medium | Single filtered lookup, hardware-filterable | Light bleeding (EVSM) or higher memory (MSM) |
| **Dithered + temporal supersampling** | Low | Free 4-8x effective samples via TAA convergence | Requires TAA, temporal ghosting risk |
| **Mip-chain dilation** | Low | PCF only on 10-20% boundary pixels, 60-80% filtering savings | Requires min-max mipmap |

**Recommendation:** **Vogel disk + IGN** is the lowest-effort quality upgrade. Add **DPCF** for contact-hardening. **Dithered + temporal** is essentially free if you add TAA.

### 4.7 Shadow Map Atlas
- **Effort:** Medium | **Benefit:** Eliminates per-light texture rebinds, simpler shader
- Pack all 17 shadow views (12 cubemap + 2 spot + 3 CSM) into one texture. Single shadow sampling function in lighting shader.

### 4.8 Shadow Map Caching (Static/Dynamic Split)
- **Effort:** Medium | **Benefit:** 80-90% fewer shadow triangles per frame
- Cache static geometry shadows, re-render only dynamic objects. Most shadow casters are static.

### 4.9 Point Light Shadows — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **6-face cubemap (current)** | N/A | Correct, simple | 6 passes per point light |
| **Dual paraboloid** | Medium | 2 passes instead of 6 (67% reduction) | Minor seam artifacts at hemisphere boundary |
| **Per-face visibility culling** | Low | Only render visible faces, 30-60% savings | Still up to 6 passes worst case |
| **Tetrahedron mapping** | High | 4 faces instead of 6 | More complex, less standard |

**Recommendation:** **Per-face visibility culling** first (low effort). Consider **dual paraboloid** if point light shadows are a bottleneck.

### 4.10 Screen-Space Contact Shadows
- **Effort:** Medium | **Benefit:** Fine shadow detail at contact points
- Per-pixel ray march toward light through depth buffer, 8-16 steps with IGN jitter. Catches shadow detail that shadow maps miss at geometry contacts.

### 4.11 SDSM (Sample Distribution Shadow Maps)
- **Effort:** Medium | **Benefit:** Auto-fit CSM splits to actual depth distribution
- Tighter frusta = less wasted resolution + better culling. Reads depth histogram to determine optimal cascade splits.

### 4.12 CSM Multiview Single-Pass Rendering
- **Effort:** Medium | **Benefit:** 3 cascade passes collapsed to 1
- Render all cascades in one pass via texture array layer selection. Reduces CPU-GPU sync barriers.

### 4.13 Advanced Shadow Techniques (Future)

| Technique | Effort | Benefit |
|-----------|--------|---------|
| **Proxy geometry (shadow LOD)** | Low | 50-90% shadow vertex savings for distant objects |
| **GPU-driven shadow culling** | High | 1-5ms CPU savings for large scenes |
| **Virtual shadow maps (UE5-style)** | High | 16K virtual maps with page-level caching |
| **SDF shadows** | High | Eliminate shadow map sampling for static geometry |

---

## 5. SSAO / Ambient Occlusion

### 5.1 AO Applied Only to Indirect Lighting
- **Effort:** Low | **Benefit:** Zero-cost correctness fix
- Multiply AO against indirect light only, not direct. Prevents physically incorrect darkening under direct illumination.

### 5.2 R2 Quasi-Random Sample Sequence
- **Effort:** Trivial | **Benefit:** ~25% effective sample increase from better coverage
- Replace current noise with R2 low-discrepancy sequence. 5-line change, better hemisphere coverage.

### 5.3 Normal Reconstruction from Depth
- **Effort:** Low | **Benefit:** 1 fewer texture binding + bandwidth per SSAO pixel
- Reconstruct view-space normals from 5 depth taps. Select neighbor pair with smallest depth difference for edge robustness.

### 5.4 SSAO Algorithm — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Hemisphere SSAO (current)** | N/A | Simple, well-understood | Most texture fetches, no bent normals, thin-object artifacts |
| **GTAO with visibility bitmask** | Medium | Better quality, free bent normals, cache-friendly, fixes thin-object over-darkening | New algorithm implementation |
| **Bevy VBAO (production WGSL)** | Low | Same as GTAO-VB but with production WGSL code directly portable to wgpu | Bevy-specific code style |
| **SAO (Scalable Ambient Obscurance)** | Medium | ~50% fewer texture fetches at comparable quality | Different visual character |
| **HBAO (horizon-based)** | Medium | Physically grounded, 30-40% fewer fetches | More complex than SAO |

**Recommendation:** **Bevy VBAO** — production WGSL code you can directly port, better quality than current SSAO, provides bent normals for specular occlusion (5.6).

### 5.5 SSAO Temporal Stabilization — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Spatial blur (current)** | N/A | Simple | Blurs detail, extra pass |
| **IGN noise + temporal blend** | Low | Drop from 16 to 8 samples, recover quality temporally | Needs history buffer |
| **Temporal accumulation (replace blur)** | Medium | Eliminates blur pass entirely, smoother result | Requires motion vectors for reprojection |
| **Frictional Games temporal blur** | Low | Blur-then-accumulate, converges in 3-4 frames, no motion vectors | Slight ghosting |

**Recommendation:** **IGN noise + temporal blend** for immediate gains without motion vectors. **Full temporal accumulation** once velocity buffer (1.8) is added.

### 5.6 GTSO Specular Occlusion
- **Effort:** Low | **Benefit:** Prevents probe light leaking into concavities
- 1-line addition to lighting shader. Requires bent normals from GTAO/VBAO (5.4).

### 5.7 Stenciled SSAO (Skip Non-Occluded Regions)
- **Effort:** Low | **Benefit:** 30-60% AO invocation reduction
- Low-sample pre-pass marks pixels where AO is reliably 1.0. Stencil-cull the full AO pass.

### 5.8 Compute Thread Group Z-Order Swizzling
- **Effort:** Low | **Benefit:** 10-40% cache speedup for AO compute
- Morton-order thread dispatch improves cache coherency for bandwidth-limited AO passes.

### 5.9 Depth-Aware Bilateral Upsample
- **Effort:** Low | **Benefit:** Eliminates silhouette halos from half-res AO
- Compare half-res depth at 4 bilinear neighbors to full-res depth, reject mismatched surfaces.

### 5.10 Screen-Space Indirect Bounce — EXCLUSIVE CHOICES (Future)

These extend AO to also produce one-bounce indirect diffuse GI:

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **HBIL (horizon-based indirect lighting)** | Medium | Unified AO + indirect in single pass, reuses SSAO infrastructure | Screen-space limitations |
| **SSILVB (bitmask + indirect bounce)** | Medium-High | AO + GI at ~1.5-2x GTAO cost from directional bitmask | More complex than HBIL |
| **SSRT3 horizon-based SSGI** | Medium | 25-50% faster than HDRP SSGI, 2-3ms with probe fallback | Still screen-space limited |

---

## 6. Global Illumination (Future)

All full-scene GI systems are high effort. Listed by practical applicability:

### 6.1 GI System — EXCLUSIVE CHOICES

| Approach | Effort | Cost | Pros | Cons |
|----------|--------|------|------|------|
| **DDGI (probe grid)** | High | ~1-2ms | Well-established, octahedral irradiance maps, many references | Probe placement, light leaking |
| **Surfel-based GI (EA GIBS)** | High | ~1-2ms fixed | Resolution-independent, shipping in AAA games | Complex surfel management |
| **Radiance cascades** | High | Fixed cost | Noise-free, no temporal accumulation needed | Newer technique, fewer references |
| **Voxel GI with clipmaps** | High | Variable | World-space, complements screen-space | Memory-intensive, voxelization pass |
| **Screen-space indirect (5.10)** | Medium | 2-3ms | Cheapest, reuses existing AO | Screen-space limitations, missing off-screen |

**Recommendation:** Screen-space indirect bounce (5.10) is the most practical near-term addition. Full GI is a major architectural investment — defer until needed.

### 6.2 SH / Irradiance Representation — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **L2 SH (9 coefficients, current)** | N/A | Standard, full sphere | 9 coefficient evaluations |
| **Ambient cube (6 axis colors)** | Low | 33% fewer coefficients, simpler 6-MAD eval | Slight quality reduction |
| **H-Basis (6 hemispherical coefficients)** | Low | 33% savings, hemisphere-restricted | Only captures upper hemisphere |
| **ZH3 quadratic zonal harmonics** | Low | Free quality upgrade over L2 SH, no rebake | Slightly more complex |

**Recommendation:** **ZH3** for quality improvement, or **ambient cube** for simplicity + performance.

---

## 7. Bloom / HDR Post-Processing

### 7.1 Karis Average Anti-Firefly
- **Effort:** Trivial | **Benefit:** Eliminates flickering bloom from HDR fireflies
- Weight first downsample by `1.0 / (1.0 + luma)`. ~4 ALU in one pass.

### 7.2 Correct Pipeline Ordering
- **Effort:** Low | **Benefit:** Correct bloom falloff shape, prevents flat white halos
- Scene HDR → exposure → bloom downsample/upsample (all HDR) → composite → tonemap → gamma → output. Bloom MUST precede tonemapping.

### 7.3 Energy-Conserving Bloom (Threshold-Free)
- **Effort:** Low | **Benefit:** Eliminates prefilter pass (~16 MB/frame bandwidth saved)
- Lerp blend in HDR before tonemap. HDR values naturally emphasize bright pixels without artificial threshold. Removes "threshold halo" artifacts.

### 7.4 Bilinear-Paired Taps (9→5 tap upsample)
- **Effort:** Low | **Benefit:** ~44% fewer texture fetches per upsample pass
- Pair adjacent tap weights and use bilinear sample at weighted midpoint. Imperceptible quality difference.

### 7.5 R11G11B10Float Bloom Chain
- **Effort:** Low | **Benefit:** 50% bandwidth on all bloom passes
- Bloom is bandwidth-bound. Halving format from Rgba16Float doubles effective throughput.

### 7.6 Store Luma in Alpha for FXAA
- **Effort:** Trivial | **Benefit:** ~20x ALU savings in FXAA pass
- Bloom composite writes luminance to alpha channel. FXAA reads `.a` instead of computing luminance.

### 7.7 Bloom Blur Kernel — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **9-tap tent upsample (current)** | N/A | Jimenez standard, well-tested | More taps than needed |
| **Dual Kawase (5/8-tap)** | Low | 44% fewer taps, 1.5-3x faster on bandwidth-limited HW | Slightly different falloff shape |
| **Compute bloom with LDS tile preload** | Medium | 5-10x fewer texture fetches from shared memory | More complex compute shader |
| **5-tap bilinear-paired (7.4)** | Low | Drop-in replacement for current tent | Not as bandwidth-efficient as compute |

**Recommendation:** **Bilinear-paired taps** as immediate drop-in (7.4). Consider **Dual Kawase** for maximum bandwidth savings on TBDR.

### 7.8 Single Pass Downsampler (SPD)
- **Effort:** Medium | **Benefit:** 40-60% bloom downsample time reduction
- AMD FidelityFX SPD generates up to 12 mip levels in one compute dispatch. Eliminates 7 barrier/synchronization points. WebGPU port exists (webgpu-spd).

### 7.9 Pre-Expose HDR Buffer
- **Effort:** Low | **Benefit:** Prevents NaN/inf in bloom from extreme HDR values
- Scalar multiply by previous frame's exposure before bloom chain. Prevents fp16 overflow.

### 7.10 Anamorphic / Cinematic Bloom (Optional)
- **Effort:** Low | **Benefit:** Cinematic lens effects at +0.05-0.15ms
- Horizontal-only 1D blur at quarter-res for anamorphic streaks. Chromatic aberration via per-channel UV offset at composite (free).

### 7.11 Screen-Space Lens Flare
- **Effort:** Low | **Benefit:** Near-zero additional cost reusing bloom prefilter
- Generate ghosts and halos from existing threshold buffer.

---

## 8. Tone Mapping

### 8.1 Tone Mapper — EXCLUSIVE CHOICES

| Operator | Effort | Pros | Cons |
|----------|--------|------|------|
| **ACES (current)** | N/A | Industry standard, well-known | "Six notorious colors" hue shift, overly warm highlights |
| **AgX** | Trivial | Film-like rolloff, fixes ACES hue problems, matches Blender viewport | Slightly desaturated look |
| **Khronos PBR Neutral** | Trivial | 13-line GLSL, analytically invertible, standardized by Khronos | Very neutral/clinical look |
| **TonyMcMapface** | Trivial | Natural highlight desaturation, prevents colored bloom halos, Bevy default | LUT-based (small 3D texture) |

**Recommendation:** **AgX** — fixes the most visible ACES problems (hue shift on saturated highlights), matches Blender pipeline if using Blender for content. Ready WGSL code available from Bevy and Godot. **PBR Neutral** as an alternative if you want maximum material fidelity.

### 8.2 Auto-Exposure — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Manual exposure (current)** | N/A | Full artist control | Breaks between bright/dark areas |
| **Histogram-based auto-exposure** | Low | Excludes outliers, prevents pumping, ~0.1-0.2ms | Needs two compute passes |
| **Mipmap average luminance** | Trivial | Simplest auto-exposure | Sensitive to fireflies, exposure pumping |

**Recommendation:** **Histogram-based** — the percentile exclusion prevents bloom/emissive pixels from inflating average luminance. Two compute passes with workgroup atomics, ~0.1-0.2ms.

### 8.3 Exposure Fusion (Local Tonemapping)
- **Effort:** Medium-High | **Benefit:** Detail in both shadows and highlights simultaneously
- Laplacian pyramid fusion of 3 synthetic exposures. Can share bloom downsample chain. ~0.5-1.0ms but eliminates need for manual per-area exposure tuning.

---

## 9. Anti-Aliasing

### 9.1 AA Method — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **FXAA (current)** | N/A | Cheapest, simplest | Blurs texture detail and text |
| **CMAA 2.0** | Medium | Compute-based, sharper than FXAA, 10-20% faster at 4K | More complex (3 compute kernels) |
| **SMAA T2x** | Medium | Near-TAA quality, morphological + temporal | Requires motion vectors |
| **TAA** | Medium | Superior quality, enables temporal techniques | Requires velocity buffer, ghosting risk |

**Recommendation:** **TAA** if you add a velocity buffer (1.8) — it unlocks temporal SSAO, temporal SSR, dithered shadows, and more. **CMAA 2.0** if you want better AA without temporal infrastructure.

### 9.2 Geometric Specular Anti-Aliasing
- **Effort:** Low | **Benefit:** Fixes specular shimmer/flicker, ~4-6 ALU
- Not a post-process AA but complements any AA method. Screen-space normal derivatives broaden roughness at distance.

---

## 10. Atmosphere / Sky / Fog

### 10.1 inverseSqrt for Mie Phase
- **Effort:** Trivial | **Benefit:** ~12 cycles/pixel saved
- Replace `pow(x, 1.5)` with `inverseSqrt`. Drop-in optimization.

### 10.2 Precompute Camera-Height Atmosphere on CPU
- **Effort:** Low | **Benefit:** ~32 cycles/pixel saved
- Pass `exp(-h_cam/R)` as uniforms. Camera height changes slowly.

### 10.3 Nishita 2D Optical Depth Table
- **Effort:** Low | **Benefit:** ~50x fewer transcendentals in atmosphere inner loop
- 64x64 or 128x64 LUT indexed by (mu, h). One texture fetch replaces ~50 exp() calls.

### 10.4 God Rays 32→16 Samples
- **Effort:** Trivial | **Benefit:** 50% god rays cost
- Minimal visual quality loss at half the sample count.

### 10.5 Atmosphere Pipeline — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Per-pixel integration (current)** | N/A | Accurate | 4-6ms of exp/pow per pixel |
| **Hosek-Wilkie analytical model** | Medium | Fast table-driven evaluation, eliminates transcendentals | Fixed atmosphere model, no aerial perspective |
| **Nishita 2D LUT (10.3)** | Low | 50x fewer transcendentals, simple upgrade | Still per-pixel, no multi-scatter |
| **Hillaire 4-LUT pipeline** | Medium-High | Full precomputed system with aerial perspective + multi-scatter | Complex 4-LUT pipeline to implement |
| **Bruneton precomputed scattering** | High | Most physically correct, ozone + custom density profiles | Most complex to implement |

**Recommendation:** **Nishita 2D LUT** as immediate upgrade (low effort, huge perf win). Graduate to **Hillaire 4-LUT pipeline** for full aerial perspective and multi-scatter if needed. A wgpu port exists (JolifantoBambla).

### 10.6 Fog — EXCLUSIVE CHOICES

| Approach | Effort | Pros | Cons |
|----------|--------|------|------|
| **Per-pixel ray march (current god rays)** | N/A | Simple | Expensive, banding |
| **Analytical height fog (closed-form)** | Low | Exact integral, no marching, replaces ~15 of 32 march steps | Only handles exponential density |
| **Analytical point light fog (atan formula)** | Low | ~12 ALU per light, zero texture fetches | Only for point/spot lights |
| **Froxel-based volumetric fog** | Medium | Per-pixel cost = one 3D tex fetch, handles arbitrary density | 1.1ms fixed cost, 3D volume texture |

**Recommendation:** **Analytical height fog** for immediate savings. Add **analytical point light fog** for local lights. **Froxel-based** is the long-term solution for full volumetric effects.

### 10.7 Sun Disk Limb Darkening
- **Effort:** Trivial | **Benefit:** Prevents flat-disc artifact at low sun angles
- ~3 ALU ops. Trivial quality improvement.

### 10.8 LUT Dirty-Flag Throttling
- **Effort:** Trivial | **Benefit:** 0.5-2ms saved when atmosphere parameters are static
- Skip LUT regeneration when sun/atmosphere parameters haven't changed.

---

## 11. wgpu / GPU Platform Optimization

### 11.1 Compile Out wgpu Trace Logs
- **Effort:** Trivial | **Benefit:** 5-15% CPU encoding overhead eliminated
- Add `log = { features = ["release_max_level_warn"] }` to Cargo.toml.

### 11.2 StoreOp::Discard on Transient Attachments
- **Effort:** Low | **Benefit:** 30-60% bandwidth on intermediate passes (Apple Silicon TBDR)
- For intermediate attachments consumed same frame (shadow maps, bloom intermediates), use `StoreOp::Discard` to avoid DRAM writeback.

### 11.3 Metal Command Buffer Batching
- **Effort:** Medium | **Benefit:** 5-15ms/frame on macOS
- wgpu Metal backend creates a new MTLCommandBuffer per CommandEncoder. Consolidate to 1-3 encoders per frame.

### 11.4 Pipeline-Overridable Constants
- **Effort:** Low | **Benefit:** 10-20% gains from dead-code elimination
- `override USE_FEATURE: bool = false;` in WGSL. Set at pipeline creation. Compiler eliminates dead branches.

### 11.5 Bind Group Frequency Organization
- **Effort:** Low | **Benefit:** Fewer redundant rebinds
- Group 0: per-frame, Group 1: per-material, Group 2: per-object with dynamic offsets.

### 11.6 Spatiotemporal Blue Noise
- **Effort:** Low | **Benefit:** 2-4x faster convergence for all stochastic effects
- Replace white noise / hash jitter with STBN textures. Benefits SSAO, SSR, shadows, god rays.

### 11.7 GPU Profiling (Prerequisite)
- **Effort:** Low | **Benefit:** Find real bottlenecks before optimizing
- wgpu-profiler crate with timestamp queries per pass. Tracy integration for live visualization. **Do this first.**

### 11.8 Shader Hot Reload
- **Effort:** Low | **Benefit:** Cuts shader iteration from ~30s to ~0.2s
- notify-based file watcher triggers shader recompilation on save.
