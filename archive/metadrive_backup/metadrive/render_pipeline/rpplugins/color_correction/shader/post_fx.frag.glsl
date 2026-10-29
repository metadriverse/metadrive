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
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/noise.inc.glsl"

#pragma include "chromatic_aberration.inc.glsl"

uniform sampler2D ShadedScene;

out vec4 result;


void main() {

    vec2 texcoord = get_texcoord();

    #if !DEBUG_MODE


        vec2 ccord = (texcoord - 0.5) * vec2(1.0, ASPECT_RATIO);
        float vignette = 1 - saturate(length(ccord));

        // Chromatic abberation
        #if GET_SETTING(color_correction, use_chromatic_aberration)
            vec3 scene_color = do_chromatic_aberration(ShadedScene, texcoord, 1-vignette);
        #else
            vec3 scene_color = textureLod(ShadedScene, texcoord, 0).xyz;
        #endif

        // Compute film grain
        float film_grain = grain(MainSceneData.frame_time);
        vec3 blended_color = blend_soft_light(scene_color, vec3(film_grain));

        // Blend film grain
        float grain_factor = GET_SETTING(color_correction, film_grain_strength);
        scene_color = mix(scene_color, blended_color, grain_factor);

        // const float saturation = 0.00;
        // scene_color = max(vec3(0), (scene_color - saturation) / (1 - saturation));

        // Apply the vignette based on the vignette strength
        scene_color *= mix(1.0, vignette, GET_SETTING(color_correction, vignette_strength));


    #else
        vec3 scene_color = textureLod(ShadedScene, texcoord, 0).xyz;
    #endif // !DEBUG_MODE


    result = vec4(scene_color, 1);
}
