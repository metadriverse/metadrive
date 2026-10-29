/**
 *
 * RenderPipeline
 *
 * Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#pragma once

#pragma include "includes/color_spaces.inc.glsl"

float mitchell_netravali_weights(float x)
{
    // Mitchell Netravali Reconstruction Filter

    // cubic B-spline
    // const float b = 1.0, c = 0.0;

    // recommended
    // const float b = 1.0 / 3.0, c = 1.0 / 3.0;

    // Catmull-Rom spline
    const float b = 0.0, c = 1.0 / 2.0;

    float ax = abs(x);
    return (ax < 1.0) ?
    ((12.0 - 9.0 * b - 6.0 * c) * ax * ax * ax +
        (-18.0 + 12.0 * b + 6.0 * c) * ax * ax + (6.0 - 2.0 * b)) / 6.0
    : ((ax >= 1.0) && (ax < 2.0)) ?
    ((-b - 6.0 * c) * ax * ax * ax + (6.0 * b + 30.0 * c) * ax * ax +
        (-12.0 * b - 48.0 * c) * ax + (8.0 * b + 24.0 * c)) / 6.0
    : 0.0;
}

vec4 sample_color(sampler2D tex, vec2 uv) {
    return saturate(textureLod(tex, uv, 0));
}

// Based on:
// http://glslsandbox.com/e#14615.1
vec4 bicubic_filter(sampler2D tex, vec2 uv)
{

    vec2 px = 1.0 / textureSize(tex, 0).xy;
    vec2 f = fract(uv / px);
    vec2 texel = (uv / px - f + 0.5) * px;
    vec4 weights = vec4(mitchell_netravali_weights(1.0 + f.x),
            mitchell_netravali_weights(f.x),
            mitchell_netravali_weights(1.0 - f.x),
            mitchell_netravali_weights(2.0 - f.x));
    vec4 t1 =
    sample_color(tex, texel + vec2(-px.x, -px.y)) * weights.x +
    sample_color(tex, texel + vec2(0.0, -px.y)) * weights.y +
    sample_color(tex, texel + vec2(px.x, -px.y)) * weights.z +
    sample_color(tex, texel + vec2(2.0 * px.x, -px.y)) * weights.w;
    vec4 t2 =
    sample_color(tex, texel + vec2(-px.x, 0.0)) * weights.x +
    sample_color(tex, texel + vec2(0.0)) * weights.y +
    sample_color(tex, texel + vec2(px.x, 0.0)) * weights.z +
    sample_color(tex, texel + vec2(2.0 * px.x, 0.0)) * weights.w;
    vec4 t3 =
    sample_color(tex, texel + vec2(-px.x, px.y)) * weights.x +
    sample_color(tex, texel + vec2(0.0, px.y)) * weights.y +
    sample_color(tex, texel + vec2(px.x, px.y)) * weights.z +
    sample_color(tex, texel + vec2(2.0 * px.x, px.y)) * weights.w;
    vec4 t4 =
    sample_color(tex, texel + vec2(-px.x, 2.0 * px.y)) * weights.x +
    sample_color(tex, texel + vec2(0.0, 2.0 * px.y)) * weights.y +
    sample_color(tex, texel + vec2(px.x, 2.0 * px.y)) * weights.z +
    sample_color(tex, texel + vec2(2.0 * px.x, 2.0 * px.y)) * weights.w;

    return mitchell_netravali_weights(1.0 + f.y) * t1 +
    mitchell_netravali_weights(f.y) * t2 +
    mitchell_netravali_weights(1.0 - f.y) * t3 +
    mitchell_netravali_weights(2.0 - f.y) * t4;
}

// Simple bilinear filter
vec4 bilinear_filter(sampler2D tex, vec2 uv) {
    vec2 pixel_size = 1.0 / textureSize(tex, 0).xy;
    vec4 color = vec4(0);
    float radius = 0.5;
    color += sample_color(tex, uv + vec2(-radius, radius) * pixel_size);
    color += sample_color(tex, uv + vec2(radius, radius) * pixel_size);
    color += sample_color(tex, uv + vec2(-radius, -radius) * pixel_size);
    color += sample_color(tex, uv + vec2(radius, -radius) * pixel_size);
    return color * 0.25;
}

// Directional filter with a small sharpen kernel
vec4 directional_filter(sampler2D tex, vec2 uv) {
    vec2 pixel_size = 1.0 / textureSize(tex, 0).xy;
    float radius = 1.0;

    vec4 color_nw = sample_color(tex, uv + vec2(-radius, -radius) * pixel_size);
    vec4 color_ne = sample_color(tex, uv + vec2(radius, -radius) * pixel_size);
    vec4 color_sw = sample_color(tex, uv + vec2(-radius, radius) * pixel_size);
    vec4 color_se = sample_color(tex, uv + vec2(radius, radius) * pixel_size);

    vec4 color = (color_nw + color_ne + color_sw + color_se) * 0.25;

    float luma_nw = get_luminance(color_nw.xyz);
    float luma_ne = get_luminance(color_ne.xyz) + 1e-5;
    float luma_sw = get_luminance(color_sw.xyz);
    float luma_se = get_luminance(color_se.xyz);

    float center_luminance = get_luminance(color.xyz);
    float uv_offs = saturate(center_luminance) * 2 - 1;

    float diag_bl_tr = luma_sw - luma_ne;
    float diag_br_tl = luma_se - luma_nw;

    vec2 direction = normalize(vec2(diag_bl_tr + diag_br_tl, diag_bl_tr - diag_br_tl) + 1e-7);
    vec2 direction_inv = vec2(-direction.y, direction.x);
    uv_offs = 1 - ((1 - uv_offs) * (1 - uv_offs));

    vec2 uv_offset = uv + direction_inv * pixel_size * abs(uv_offs) * -0.125;
    vec4 color_n = sample_color(tex, uv_offset - direction * 0.125 * pixel_size);
    vec4 color_p = sample_color(tex, uv_offset + direction * 0.125 * pixel_size);

    const float sharpness = 0.9;
    return (color_n + color_p) * ((sharpness + 1.0) * 0.5) - color * sharpness;
}
