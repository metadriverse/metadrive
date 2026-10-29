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

#pragma optionNV (unroll all)

layout(local_size_x = 10, local_size_y = 4, local_size_z = 1) in;

uniform writeonly image2D RESTRICT DestTex;
uniform samplerBuffer FPSValues;
uniform int index;
uniform float maxMs;

void main() {

    // TODO: Might make this an input
    const ivec2 widget_size = ivec2(250, 120);
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // Store the current pixels color
    vec4 color = vec4(0);

    int buffer_offset = coord.x + index;

    // Make sure the value does not get negative
    buffer_offset += 10 * widget_size.x;
    buffer_offset = buffer_offset % widget_size.x;

    float ms_value = texelFetch(FPSValues, buffer_offset).x;
    float prev_ms_value = texelFetch(
        FPSValues, (buffer_offset - 1 + widget_size.x) % widget_size.x).x;

    int ms_pixel = int(ms_value / maxMs * widget_size.y);
    int prev_ms_pixel = int(prev_ms_value / maxMs * widget_size.y);

    if (coord.y >= min(ms_pixel, prev_ms_pixel) &&
        coord.y <= max(ms_pixel, prev_ms_pixel)) {
        color = vec4(0, 1, 0, 1);
    }

    // Axis
    if (coord.x == 0 || coord.y == 0) {
        color = vec4(1);
    }

    imageStore(DestTex, coord, color);
}
