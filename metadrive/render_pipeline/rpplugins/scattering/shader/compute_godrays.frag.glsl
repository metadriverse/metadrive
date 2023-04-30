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

#define USE_TIME_OF_DAY 1
#define USE_GBUFFER_EXTENSIONS
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"
#pragma include "includes/noise.inc.glsl"

uniform sampler2D ShadedScene;
out vec3 result;


void main() {
    vec2 texcoord = get_texcoord();
    Material m = unpack_material(GBuffer);

    // compute sun position on screen
    // TODO: could move to vertex shader
    vec3 sun_vector = get_sun_vector();
    vec3 sun_pos = sun_vector * 1e5;
    vec4 sun_proj = MainSceneData.view_proj_mat_no_jitter * vec4(sun_pos, 1);
    sun_proj.xyz /= sun_proj.w;
    if (sun_proj.w < 0.0) {
        result = textureLod(ShadedScene, texcoord, 0).xyz;
        return;
    }
    sun_proj.xy = sun_proj.xy * 0.5 + 0.5;

    // raymarch to sun and collect .. whatever
    float jitter = rand(texcoord) * 0.9;

    const int num_samples = 32;
    vec3 accum = vec3(0);
    for (int i = 0; i < num_samples; ++i) {
        float t = (i + jitter) / float(num_samples - 1);

        vec2 sample_coord = mix(texcoord, sun_proj.xy, pow(t, 1.0));
        vec3 sample_data = textureLod(ShadedScene, sample_coord, 0).xyz;

        // float weight = step(get_luminance(sample_data), 1.0);
        float weight = step(1.0, get_luminance(sample_data));
        // float weight = 1.0;
        accum += sample_data * weight * saturate(5 * (1 - t)) * 1;
        // accum += sample_data * weight;
    }

    accum /= num_samples;
    accum *= 0.001;

    accum += textureLod(ShadedScene, texcoord, 0).xyz;
    result = vec3(accum);

}
