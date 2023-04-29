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


CONST_ARRAY vec2[] shadow_sample_offsets_8 = vec2[8](
    vec2(-0.7071, 0.7071),
    vec2(-0.0000, -0.8750),
    vec2(0.5303, 0.5303),
    vec2(-0.6250, -0.0000),
    vec2(0.3536, -0.3536),
    vec2(-0.0000, 0.3750),
    vec2(-0.1768, -0.1768),
    vec2(0.1250, 0.0000)
);

// Projects a point using the given mvp
vec3 project(mat4 mvp, vec3 p) {
    vec4 projected = mvp * vec4(p, 1);
    return fma(projected.xyz / projected.w, vec3(0.5), vec3(0.5));
}

// Given a shadow mvp matrix and the light vector, finds an appropriate filter size
vec2 find_filter_size(mat4 projection, vec3 light, float sample_radius) {

    // Find an arbitrary tangent and bitangent to the given normal
    vec3 tangent = vec3(1, 0, 0);
    vec3 binormal = vec3(0, 1, 0);

    // Project everything
    // TODO: We could actually only use the 3x3 part of the mat and save the
    // origin projection.
    vec2 proj_origin = project(projection, vec3(0)).xy;
    vec2 proj_tangent = abs(project(projection, tangent + binormal).xy - proj_origin);

    return vec2(max(proj_tangent.x, proj_tangent.y) * sample_radius) * 10.0;
}

// http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1/
// Returns the normal and light dependent bias
vec2 get_shadow_bias(vec3 n, vec3 l) {
    float cos_alpha = saturate(dot(n, l));
    float offset_scale_n = sqrt(1 - cos_alpha * cos_alpha); // sin(acos(L·N))
    float offset_scale_l = offset_scale_n / cos_alpha;    // tan(acos(L·N))
    return vec2(offset_scale_n, min(2, offset_scale_l));
}

// Offsets a position based on slope and normal
vec3 get_biased_position(vec3 pos, float slope_bias, float normal_bias, vec3 normal, vec3 light) {
    vec2 offsets = get_shadow_bias(normal, light);
    pos += normal * offsets.x * normal_bias;
    pos += light * offsets.y * slope_bias;
    return pos;
}
