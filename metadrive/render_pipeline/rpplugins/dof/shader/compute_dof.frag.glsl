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

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "dof.inc.glsl"
#pragma include "includes/noise.inc.glsl"

uniform sampler2D ShadedScene;
uniform sampler2D PresortResult;
uniform sampler2D PrecomputedCoC;
uniform sampler2D TileMinMax;
out vec4 result;

/*

WORK IN PROGRESS

This code is code in progress and such not formatted very nicely nor commented!

*/

float intersect_circle(vec2 d, float radius) {
    float latt = saturate(length(d * vec2(1, 1)) / radius);
    latt = pow(latt, 10.0);
    latt = 1 - latt;
    return latt;

    if (length(d) <= radius) {
        return 1.0;
    }
    return 0.0;
}

void main() {

    vec2 texcoord = get_texcoord();
    ivec2 tile_coord = ivec2(ivec2(gl_FragCoord.xy) / vec2(tile_size));
    vec2 tile_data = texelFetch(TileMinMax, tile_coord, 0).xy;
    float tile_max_depth = tile_data.x;
    float tile_max_coc = tile_data.y;

    vec3 presort_result = textureLod(PresortResult, texcoord, 0).xyz;
    float mid_coc = presort_result.x;

    if (tile_max_coc <= 1e-4) {
        result = textureLod(ShadedScene, texcoord, 0);
        result.w = 0;
    }

    vec2 max_radius = 16.0 / SCREEN_SIZE;

    vec3 jitter = rand_rgb(texcoord);

    const int num_ring_samples = 4;
    float max_length = length(max_radius * vec2(1, 1));

    int num_samples = 0;

    vec3 scene_color = textureLod(ShadedScene, texcoord, 0).xyz;

    float alpha = 0.0;
    vec4 foreground = vec4(scene_color, 1) * 1e-4;
    vec4 background = vec4(scene_color, 1) * 1e-4;

    for (int ring = 0; ring <= num_ring_samples; ++ring) {
        int n_samples = max(1, 8 * ring);
        vec2 r = max_radius * ring / float(num_ring_samples) * (tile_max_coc);
        // vec2 r = max_radius * ring / float(num_ring_samples);

        for (int i = 0; i < n_samples; ++i) {
            ++num_samples;
            float phi = i / float(n_samples) * TWO_PI;
            float x_offs = sin(phi);
            float y_offs = cos(phi);

            vec2 tcoord = texcoord + vec2(x_offs, y_offs) * r;
            // tcoord += jitter.xy * 0.1 * r;

            // XXX: Instead of manual clamping, use a near filtered texture
            tcoord = truncate_coordinate(tcoord);

            vec3 sample_data = textureLod(PresortResult, tcoord, 0).xyz;
            vec3 color_data = textureLod(PrecomputedCoC, tcoord, 0).xyz;

            float coc_weight = intersect_circle(r, sample_data.x * max_length);
            coc_weight *= 1.0 / max(1e-9, sample_data.x * sample_data.x);
            foreground += vec4(color_data, 1) * coc_weight;
            // foreground += vec4(color_data, 1) * coc_weight * sample_data.y;
            // background += vec4(color_data, 1) * coc_weight * sample_data.z;
        }
    }

    foreground.xyz /= max(1e-5, foreground.w);
    // background.xyz /= max(1e-5, background.w);

    // result = mix(background, foreground, saturate(2 * foreground.w / num_samples));
    result = foreground;
    result.w = mid_coc;
}
