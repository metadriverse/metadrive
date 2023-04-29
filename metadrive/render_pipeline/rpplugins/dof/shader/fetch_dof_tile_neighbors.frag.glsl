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
#pragma include "dof.inc.glsl"

uniform sampler2D TileMinMax;
out vec2 result;

void main() {

    ivec2 tile = ivec2(gl_FragCoord.xy);
    ivec2 screen_coord = tile * tile_size;
    ivec2 max_tiles = textureSize(TileMinMax, 0) - 1;

    float max_depth = 0;
    float max_coc = 0;

    const int filter_size = 1;

    for (int x = -filter_size; x <= filter_size; ++x) {
        for (int y = -filter_size; y <= filter_size; ++y) {
            ivec2 neighbor_coord = clamp(tile + ivec2(x, y), ivec2(0), ivec2(max_tiles));
            vec2 neighbor_data = texelFetch(TileMinMax, neighbor_coord, 0).xy;
            max_depth = max(max_depth, neighbor_data.x);
            max_coc = max(max_coc, neighbor_data.y);
        }
    }

    result = vec2(max_depth, max_coc);
}
