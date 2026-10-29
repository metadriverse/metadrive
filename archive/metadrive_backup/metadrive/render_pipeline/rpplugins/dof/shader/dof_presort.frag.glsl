/**
 *
 * RenderPipeline
 *
 * Copyright (c) 2014-2016 tobspr <toref_val.springer1@gmail.com>
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
#pragma include "dof.inc.glsl"

uniform sampler2D PrecomputedCoC;
uniform sampler2D TileMinMax;
out vec3 result;


/*

WORK IN PROGRESS

This code is code in progress and such not formatted very nicely nor commented!

*/

void main() {

    vec2 texcoord = get_texcoord();
    ivec2 tile_coord = ivec2(ivec2(gl_FragCoord.xy) / vec2(tile_size));
    vec2 tile_data = texelFetch(TileMinMax, tile_coord, 0).xy;
    float tile_min_depth = tile_data.x;
    float tile_max_coc = tile_data.y;

    float pixel_depth = get_depth_at(texcoord);
    float pixel_coc = textureLod(PrecomputedCoC, texcoord, 0).w;

    float alpha = sample_alpha(pixel_coc); // XXX: Returns just strange values
    // float alpha = sample_alpha(tile_max_coc * 24.0); // XXX: Returns just strange values
    alpha = 1.0;
    vec2 depth_weights = compare_depth(pixel_depth, tile_min_depth);
    // Depth weights.x = Foreground
    // Depth weights.y = Background
    result = vec3(pixel_coc, depth_weights.xy * alpha);
    // result = vec3(pow(tile_min_depth, 100.0));
}
