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

uniform sampler2D SourceTex;
out vec4 result;

// This shader just passes through the input texture

void main() {
    vec2 texcoord = (ivec2(gl_FragCoord.xy) + 0.5) / NATIVE_SCREEN_SIZE;
    result = vec4(textureLod(SourceTex, texcoord, 0).xyz, 1);

    #if SPECIAL_MODE_ACTIVE(LUMINANCE)

        // Luminance debug mode, too bright pixels get red, too dark pixels get blue,
        // rest stays green

        vec3 color = textureLod(SourceTex, texcoord, 0).xyz;
        float luminance = get_luminance(color);

        vec3 color_ok = vec3(0, 1, 0);
        vec3 color_too_bright = vec3(1, 0, 0);
        vec3 color_too_dark = vec3(0, 0, 1);

        const float max_brightness = 0.7;
        const float max_darkness = 0.3;

        color = mix(color_ok, color_too_bright, saturate(5.0 * (luminance - max_brightness)));
        color = mix(color, color_too_dark, saturate(5.0 * (max_darkness - luminance)));

        result.xyz = color;
    #endif
}
