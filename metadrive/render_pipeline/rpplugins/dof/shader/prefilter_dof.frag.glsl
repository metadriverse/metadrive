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

#version 430

#define USE_GBUFFER_EXTENSIONS
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"

uniform sampler2D ShadedScene;

out vec4 result;

/*

WORK IN PROGRESS

This code is code in progress and such not formatted very nicely nor commented!

*/


vec3 karis_average(vec3 color) {
    const float sharpness = 0.0;
    return color / (1 + (1 - sharpness) * get_luminance(color));
}

void main() {
    vec2 texcoord = get_texcoord();
    float mid_depth = textureLod(GBuffer.Depth, texcoord, 0).x;

    vec3 mid_color = textureLod(ShadedScene, texcoord, 0).xyz;

    vec3 accum = karis_average(mid_color.xyz) * 0;
    float weights = 1.0 * 0.0;

    const float scale = 0.1 * GET_SETTING(dof, blur_strength); // XXX: Todo, make it physically based
    const float focus_plane = GET_SETTING(dof, focal_point);
    const float focus_size = GET_SETTING(dof, focal_size);
    const float near_scale = GET_SETTING(dof, near_blur_strength) /
        max(0.0, focus_plane - focus_size - CAMERA_NEAR);
    float dist = get_linear_z_from_z(mid_depth);

    float coc = (dist - focus_plane);

    if (coc >= 0) {
        coc -= focus_size;
        coc *= scale;
    } else {
        coc += focus_size;
        coc *= -near_scale;
    }

    coc = clamp(coc, 0.0001, 1.0);

    const int kernel_size = 2;
    for (int x = -kernel_size; x <= kernel_size; ++x) {
        for (int y = -kernel_size; y <= kernel_size; ++y) {
            // skip center sample
            // if (x == 0 && y == 0) continue;
            vec2 offcoord = texcoord + vec2(x, y) / SCREEN_SIZE;
            vec3 sample_data = textureLod(ShadedScene, offcoord, 0).xyz;
            float sample_depth = textureLod(GBuffer.Depth, offcoord, 0).x;
            sample_data = karis_average(sample_data);

            float weight = 1 - saturate(abs(mid_depth - sample_depth) / 0.005);
            // float weight = 1;
            accum += sample_data * weight;
            weights += weight;

        }
    }
    accum /= max(0.001, weights);

    accum = mid_color;

    if (coc < 0.01) {
        // accum = karis_average(mid_color);
    }
        // accum = karis_average(mid_color);

    result = vec4(accum, coc);
}
