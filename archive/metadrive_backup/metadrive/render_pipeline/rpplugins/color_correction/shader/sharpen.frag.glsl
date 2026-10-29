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

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"

uniform sampler2D ShadedScene;
out vec3 result;

void main() {

    vec2 texcoord = (ivec2(gl_FragCoord.xy) + 0.5) / NATIVE_SCREEN_SIZE;
    vec3 scene_color = textureLod(ShadedScene, texcoord, 0).xyz;

    // Compute the sharpen strength for each individual channel
    const float sharpen_strength = GET_SETTING(color_correction, sharpen_strength);

    vec2 pixel_size = 1.0 / NATIVE_SCREEN_SIZE;
    const float r = 0.8;

    // Blur arround the current pixel
    vec3 blur_sum = vec3(0);
    blur_sum += textureLod(ShadedScene, texcoord + vec2(r, -r) * pixel_size, 0).rgb;
    blur_sum += textureLod(ShadedScene, texcoord + vec2(-r, -r) * pixel_size, 0).rgb;
    blur_sum += textureLod(ShadedScene, texcoord + vec2(r, r) * pixel_size, 0).rgb;
    blur_sum += textureLod(ShadedScene, texcoord + vec2(-r, r) * pixel_size, 0).rgb;

    vec3 pixel_diff = scene_color - blur_sum * 0.25;

    // Apply the sharpening
    #if !DEBUG_MODE
        scene_color += dot(pixel_diff, LUMA_COEFFS * sharpen_strength);
    #endif
    result = scene_color;
}
