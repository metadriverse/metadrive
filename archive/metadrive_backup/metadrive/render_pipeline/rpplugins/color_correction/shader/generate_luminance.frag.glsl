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

// Converts the scene color to a luminance

uniform sampler2D ShadedScene;
out float result;

float get_log_luminance(vec3 color) {
    float lum = get_luminance(color);
    return lum / (1 + lum);
}

void main() {
    ivec2 coord_screen = ivec2(gl_FragCoord.xy) * 4;
    vec2 local_coord = (coord_screen + 1.0) / SCREEN_SIZE;
    vec2 pixel_offset = 2.0 / SCREEN_SIZE;

    // Weight luminance based on distance to the borders - this is because
    // pixels in the center of the screen are more visually important
    float weight = 1.05 - 0.2 * distance(local_coord, vec2(0.5, 0.5));

    vec4 luminances = vec4(
        get_log_luminance(weight *
            textureLod(ShadedScene, local_coord, 0).xyz),
        get_log_luminance(weight *
            textureLod(ShadedScene, local_coord + vec2(pixel_offset.x, 0), 0).xyz),
        get_log_luminance(weight *
            textureLod(ShadedScene, local_coord + vec2(0, pixel_offset.y), 0).xyz),
        get_log_luminance(weight *
            textureLod(ShadedScene, local_coord + pixel_offset.xy, 0).xyz)
    );

    result = dot(luminances, vec4(0.25));
}
