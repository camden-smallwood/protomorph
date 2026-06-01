// Verbatim port of `sky_dome_simple_hlsl.hlsl` (`@generate sky`).
//
// Engine HLSL reference (full body, lines 14-53):
//
//   //@generate sky
//
//   struct VS_INPUT  { float3 vPos:POSITION; float3 vColor:TEXCOORD0; float3 vNormal:NORMAL; };
//   struct VS_OUTPUT { float4 pos:SV_Position; float3 color:TEXCOORD0; float3 normal:TEXCOORD1; };
//
//   VS_OUTPUT default_vs(VS_INPUT input) {
//       VS_OUTPUT output;
//       float4 world_pos;
//       world_pos.xyz = transform_point(float4(input.vPos, 1.0f), Nodes[0]);
//       world_pos.w   = 1.0f;
//       output.pos    = mul(world_pos, View_Projection);
//       output.color  = input.vColor;
//       output.normal = normalize(input.vPos);
//       return output;
//   }
//
//   accum_pixel default_ps(VS_OUTPUT input) {
//       float4 out_color = float4(input.color * g_exposure.rrr, 1.0f);
//       float3 sun = pow(max(dot(p_lighting_constant_0.xyz, normalize(input.normal)), 0.0f),
//                        p_lighting_constant_0.w) * p_lighting_constant_1.rgb;
//       out_color.rgb += sun;
//       return convert_to_render_target(out_color, true, false);
//   }
//
// `p_lighting_constant_0/1` (engine cbuffer slots 0x170000 / 0x170001):
// IDA bridge dllcache 13372 confirmed the ONLY writer is
// `setup_sun_constants @ 0x1806912E0`, called exclusively from
// `c_structure_renderer::render_decorators @ 0x1806901A0`. Sky renders
// BEFORE decorators in the static-lighting pass, so when the sky shader
// runs those cbuffer slots hold either zero (first cluster) or the
// previous object's SH L0 coefficient (subsequent clusters). Treating
// them as zero here matches the engine PC's most common path: sky
// emits `vert_color * g_exposure` with no sun glare. (If we later
// observe a glare halo around the sun in MCC reference, we'll wire
// the lightgen direction/intensity through a dedicated sky cbuffer.)
//
// `convert_to_render_target(out_color, clamp_positive=true, no_log_encoding=false)`
// per `render_target_fx.hlsl`: on PC `LDR_gamma2 == HDR_gamma2 == 0`, so
// the log/gamma encoding is identity. `clamp_positive=true` applies
// `max(rgb, 0)` to the lit color. Engine writes BOTH RT0 (color) and
// RT1 (dark_color = color × DARK_COLOR_MULTIPLIER) via the
// `accum_pixel` struct. `DARK_COLOR_MULTIPLIER = g_exposure.g` per
// engine `setup_render_target_globals_with_exposure @ 0x180670AD0`.

// ---------------------------------------------------------------------------
// Engine bindings — minimal subset needed by the sky path. Slot numbers
// mirror `assets/halo_shaders/engine_bindings.wgsl` so the sky pipeline
// layout reuses the existing camera_bgl / model_bgl / node_matrices_bgl.
// ---------------------------------------------------------------------------

struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct EngineExposure {
    g_exposure: vec4<f32>,
    g_alt_exposure: vec4<f32>,
}
@group(0) @binding(9) var<uniform> engine_exposure: EngineExposure;

struct ModelUniforms { model: mat4x4<f32>, }
@group(1) @binding(0) var<uniform> model_u: ModelUniforms;

@group(3) @binding(0) var<storage, read> node_matrices: array<mat4x4<f32>>;

// `g_exposure.r` per engine `setup_render_target_globals_with_exposure`.
// HLSL `g_exposure.rrr` reads the same scalar 3 times.
fn g_exposure() -> f32 {
    return engine_exposure.g_exposure.x;
}

// ---------------------------------------------------------------------------
// Vertex / fragment IO
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal_oct: vec2<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent_sign: vec4<f32>,
    @location(4) node_indices: vec4<u32>,
    @location(5) node_weights: vec4<f32>,
    @location(6) lightmap_texcoord: vec2<f32>,
    // Per-vertex baked sky color — the engine `_vertex_buffer_usage_vert_color`
    // stream. Populated when `RenderMesh::has_vertex_color` is set,
    // bound at vertex buffer slot 1 by the sky pipeline dispatch.
    @location(12) vert_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// HLSL `transform_point(p, Nodes[0])` — applies the bone-0 transform
// to a position. `SkyGpu::new` initializes Nodes[0] to identity, so this
// is mathematically a pass-through. Keep the multiply explicit to mirror
// the HLSL line-for-line.
fn transform_point(p: vec4<f32>, node: mat4x4<f32>) -> vec3<f32> {
    return (node * p).xyz;
}

@vertex
fn default_vs(in: VertexInput) -> VertexOutput {
    // HLSL: world_pos.xyz = transform_point(float4(input.vPos, 1.0f), Nodes[0]);
    //       world_pos.w   = 1.0f;
    let nodes_0 = node_matrices[0];
    let world_pos_xyz = transform_point(vec4<f32>(in.position, 1.0), nodes_0);
    let world_pos = vec4<f32>(world_pos_xyz, 1.0);
    // HLSL: output.pos = mul(world_pos, View_Projection);
    let clip = camera.projection * camera.view * world_pos;

    var out: VertexOutput;
    out.clip_position = clip;
    // HLSL: output.color  = input.vColor;
    out.color = in.vert_color.rgb;
    // HLSL: output.normal = normalize(input.vPos);
    out.normal = normalize(in.position);
    return out;
}

// Engine `accum_pixel` (render_target_fx.hlsl:8-18):
//   struct accum_pixel { float4 color : COLOR0; float4 dark_color : COLOR1; };
struct AccumPixel {
    @location(0) color: vec4<f32>,
    @location(1) dark_color: vec4<f32>,
}

@fragment
fn default_ps(in: VertexOutput) -> AccumPixel {
    // HLSL: float4 out_color = float4(input.color * g_exposure.rrr, 1.0f);
    let exposure = g_exposure();
    let lit_rgb = in.color * vec3<f32>(exposure, exposure, exposure);

    // HLSL: float3 sun = pow(max(dot(p_lighting_constant_0.xyz,
    //                               normalize(input.normal)), 0.0f),
    //                        p_lighting_constant_0.w)
    //                  * p_lighting_constant_1.rgb;
    //       out_color.rgb += sun;
    // See file-top comment — engine slots 0x170000 / 0x170001 are
    // unwritten on the sky path, so `sun_glare` evaluates to zero
    // here. Keeping the literal `+ vec3(0.0)` so the lit-color line
    // mirrors the HLSL `out_color.rgb += sun` in shape; the compiler
    // folds it away.
    let sun_glare = vec3<f32>(0.0);
    let composed = lit_rgb + sun_glare;

    // HLSL: return convert_to_render_target(out_color, true, false);
    // From `render_target_fx.hlsl:38`:
    //   result.dark_color.rgb = color.rgb / DARK_COLOR_MULTIPLIER;
    //   #define DARK_COLOR_MULTIPLIER g_exposure.g
    // PC has LDR_gamma2 == HDR_gamma2 == 0 so the log/gamma encode is
    // identity. On PC `g_exposure.y == 1.0` (we upload
    // `hdr_target_stops=0`, so `2^0 = 1`), making dark_color == color.
    let clamped = max(composed, vec3<f32>(0.0));
    let color = vec4<f32>(clamped, 1.0);
    let dark_multiplier = engine_exposure.g_exposure.y;
    let dark_color = vec4<f32>(clamped / dark_multiplier, 1.0);
    return AccumPixel(color, dark_color);
}
