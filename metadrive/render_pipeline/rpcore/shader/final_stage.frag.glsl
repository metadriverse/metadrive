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
#pragma include "includes/noise.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"

uniform sampler2D ShadedScene;
out vec4 result;

void main() {
    vec2 texcoord = get_texcoord();

    // Fetch the current's scene color
    vec3 scene_color = textureLod(ShadedScene, texcoord, 0).xyz;

    #if !DEBUG_MODE && !HAVE_PLUGIN(color_correction)
        // Do a simple sRGB correction
        scene_color = rgb_to_srgb(scene_color);
    #endif

    // Apply dithering to prevent banding, since we are converting from 16 bit
    // precision to 8 bit precision here
    #if !REFERENCE_MODE
        vec3 dither = (rand_rgb(texcoord) + rand_rgb(texcoord + 0.5787)) * 0.5 - 0.4;
        scene_color += dither / 128.0;
    #endif

    result = vec4(scene_color, 1);
}
