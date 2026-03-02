// FXAA 3.11 Quality — Timothy Lottes' algorithm adapted for WGSL
// Single fullscreen pass, reads tonemapped sRGB input

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_linear: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

const FXAA_EDGE_THRESHOLD: f32 = 0.125;
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0625;
const FXAA_SUBPIX_TRIM: f32 = 0.25;
const FXAA_SUBPIX_CAP: f32 = 0.75;
const FXAA_SEARCH_STEPS: i32 = 8;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords;
    let tex_size = vec2<f32>(textureDimensions(t_input));
    let rcpFrame = 1.0 / tex_size;

    // Sample center and cardinal neighbors
    let rgbM = textureSample(t_input, s_linear, uv).rgb;
    let rgbN = textureSample(t_input, s_linear, uv + vec2<f32>(0.0, -rcpFrame.y)).rgb;
    let rgbS = textureSample(t_input, s_linear, uv + vec2<f32>(0.0, rcpFrame.y)).rgb;
    let rgbE = textureSample(t_input, s_linear, uv + vec2<f32>(rcpFrame.x, 0.0)).rgb;
    let rgbW = textureSample(t_input, s_linear, uv + vec2<f32>(-rcpFrame.x, 0.0)).rgb;

    let lumM = luminance(rgbM);
    let lumN = luminance(rgbN);
    let lumS = luminance(rgbS);
    let lumE = luminance(rgbE);
    let lumW = luminance(rgbW);

    let rangeMax = max(max(lumN, lumS), max(max(lumE, lumW), lumM));
    let rangeMin = min(min(lumN, lumS), min(min(lumE, lumW), lumM));
    let range = rangeMax - rangeMin;

    // Early exit for low contrast areas
    if range < max(FXAA_EDGE_THRESHOLD_MIN, rangeMax * FXAA_EDGE_THRESHOLD) {
        return vec4<f32>(rgbM, 1.0);
    }

    // Sample corners
    let rgbNW = textureSample(t_input, s_linear, uv + vec2<f32>(-rcpFrame.x, -rcpFrame.y)).rgb;
    let rgbNE = textureSample(t_input, s_linear, uv + vec2<f32>(rcpFrame.x, -rcpFrame.y)).rgb;
    let rgbSW = textureSample(t_input, s_linear, uv + vec2<f32>(-rcpFrame.x, rcpFrame.y)).rgb;
    let rgbSE = textureSample(t_input, s_linear, uv + vec2<f32>(rcpFrame.x, rcpFrame.y)).rgb;

    let lumNW = luminance(rgbNW);
    let lumNE = luminance(rgbNE);
    let lumSW = luminance(rgbSW);
    let lumSE = luminance(rgbSE);

    // Sub-pixel aliasing test
    let lumAvg = (lumN + lumS + lumE + lumW) * 0.25;
    let subpixA = abs(lumAvg - lumM);
    let subpixB = clamp(subpixA / range, 0.0, 1.0);
    let subpixC = (-2.0 * subpixB + 3.0) * subpixB * subpixB;
    let subpixBlend = subpixC * subpixC * FXAA_SUBPIX_CAP;

    // Determine edge orientation
    let edgeH = abs((-2.0 * lumN) + lumNW + lumNE) +
                abs((-2.0 * lumM) + lumW + lumE) * 2.0 +
                abs((-2.0 * lumS) + lumSW + lumSE);
    let edgeV = abs((-2.0 * lumW) + lumNW + lumSW) +
                abs((-2.0 * lumM) + lumN + lumS) * 2.0 +
                abs((-2.0 * lumE) + lumNE + lumSE);
    let isHorizontal = edgeH >= edgeV;

    // Choose edge direction
    var stepLength: f32;
    var lumPos: f32;
    var lumNeg: f32;

    if isHorizontal {
        stepLength = rcpFrame.y;
        lumPos = lumS;
        lumNeg = lumN;
    } else {
        stepLength = rcpFrame.x;
        lumPos = lumE;
        lumNeg = lumW;
    }

    let gradientPos = abs(lumPos - lumM);
    let gradientNeg = abs(lumNeg - lumM);

    if gradientNeg > gradientPos {
        stepLength = -stepLength;
    }

    let gradientScaled = max(gradientPos, gradientNeg) * 0.25;
    let lumLocalAvg = 0.5 * (select(lumNeg, lumPos, gradientPos >= gradientNeg) + lumM);

    // Step to edge midpoint
    var edgeUV = uv;
    if isHorizontal {
        edgeUV.y += stepLength * 0.5;
    } else {
        edgeUV.x += stepLength * 0.5;
    }

    // Search along edge
    var searchStep: vec2<f32>;
    if isHorizontal {
        searchStep = vec2<f32>(rcpFrame.x, 0.0);
    } else {
        searchStep = vec2<f32>(0.0, rcpFrame.y);
    }

    var uvPos = edgeUV + searchStep;
    var uvNeg = edgeUV - searchStep;

    var lumEndPos = luminance(textureSample(t_input, s_linear, uvPos).rgb) - lumLocalAvg;
    var lumEndNeg = luminance(textureSample(t_input, s_linear, uvNeg).rgb) - lumLocalAvg;

    var reachedPos = abs(lumEndPos) >= gradientScaled;
    var reachedNeg = abs(lumEndNeg) >= gradientScaled;

    for (var i = 1; i < FXAA_SEARCH_STEPS; i++) {
        if reachedPos && reachedNeg { break; }

        if !reachedPos {
            uvPos += searchStep;
            lumEndPos = luminance(textureSample(t_input, s_linear, uvPos).rgb) - lumLocalAvg;
            reachedPos = abs(lumEndPos) >= gradientScaled;
        }
        if !reachedNeg {
            uvNeg -= searchStep;
            lumEndNeg = luminance(textureSample(t_input, s_linear, uvNeg).rgb) - lumLocalAvg;
            reachedNeg = abs(lumEndNeg) >= gradientScaled;
        }
    }

    // Compute distance to endpoints
    var distPos: f32;
    var distNeg: f32;
    if isHorizontal {
        distPos = uvPos.x - uv.x;
        distNeg = uv.x - uvNeg.x;
    } else {
        distPos = uvPos.y - uv.y;
        distNeg = uv.y - uvNeg.y;
    }

    let isCloserToNeg = distNeg < distPos;
    let lumEnd = select(lumEndPos, lumEndNeg, isCloserToNeg);

    // Reject if the endpoint luminance delta and center luminance delta
    // are in the same direction (not actually crossing the edge)
    if (lumM - lumLocalAvg < 0.0) == (lumEnd < 0.0) {
        // Sub-pixel blend only
        return vec4<f32>(textureSample(t_input, s_linear, uv).rgb, 1.0);
    }

    let spanLength = distPos + distNeg;
    let edgeBlend = 0.5 - min(distPos, distNeg) / spanLength;

    let finalBlend = max(edgeBlend, subpixBlend);

    // Apply blend along the perpendicular direction
    var finalUV = uv;
    if isHorizontal {
        finalUV.y += stepLength * finalBlend;
    } else {
        finalUV.x += stepLength * finalBlend;
    }

    return vec4<f32>(textureSample(t_input, s_linear, finalUV).rgb, 1.0);
}
