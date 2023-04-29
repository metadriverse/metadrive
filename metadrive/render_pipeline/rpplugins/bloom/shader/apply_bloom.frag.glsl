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
uniform sampler2D BloomTex;
uniform sampler2D LensDirtTex;

out vec3 result;

void main() {
    vec2 texcoord = get_texcoord();

    vec3 scene_result = textureLod(ShadedScene, texcoord, 0).xyz;
    vec3 bloom_result = textureLod(BloomTex, texcoord, 0).xyz;

    // Apply dirt
    vec3 lens_dirt = textureLod(LensDirtTex, texcoord, 0).xyz;
    float base_dirt_factor = GET_SETTING(bloom, lens_dirt_factor);
    vec3 dirt_factor = pow(lens_dirt, vec3(1.0));

    bloom_result = mix(bloom_result, bloom_result * dirt_factor, base_dirt_factor);

    // Blend scene color and bloom color
    result = scene_result + bloom_result;
}
