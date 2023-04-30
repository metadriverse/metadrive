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
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "motion_blur.inc.glsl"

uniform sampler2D TileMinMax;
out vec2 result;

// Based on:
// http://graphics.cs.williams.edu/papers/MotionBlurI3D12/

void main() {

    ivec2 tile = ivec2(gl_FragCoord.xy);
    ivec2 screen_coord = tile * tile_size;
    ivec2 max_tiles = textureSize(TileMinMax, 0) - 1;

    vec2 max_velocity = vec2(0.0);
    float largest_magnitude = -1.0;

    const int filter_size = 2;

    for (int x = -filter_size; x <= filter_size; ++x) {
        for (int y = -filter_size; y <= filter_size; ++y) {
            ivec2 neighbor_coord = clamp(tile + ivec2(x, y), ivec2(0), ivec2(max_tiles));
            vec2 vmax_neighbor = texelFetch(TileMinMax, neighbor_coord, 0).xy;

            float magnitude_neighbor = dot(vmax_neighbor, vmax_neighbor);

            if (magnitude_neighbor > largest_magnitude) {
                vec2 direction_of_velocity = vmax_neighbor;
                int displacement = abs(x) + abs(y);
                ivec2 point = ivec2(sign(vec2(x, y) * direction_of_velocity));
                float dist = point.x + point.y;
                if (abs(dist) == displacement) {
                    max_velocity = vmax_neighbor;
                    largest_magnitude = magnitude_neighbor;
                }
            }
        }
    }

    result = max_velocity;
}
