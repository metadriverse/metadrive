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

#pragma optionNV (unroll all)

#define USE_TIME_OF_DAY 1
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/lighting_pipeline.inc.glsl"
#pragma include "includes/lights.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"
#pragma include "includes/shadows.inc.glsl"
#pragma include "includes/noise.inc.glsl"
#pragma include "includes/skin_shading.inc.glsl"

out vec4 result;

uniform GBufferData GBuffer;
uniform sampler2D PrefilteredShadows;
uniform sampler2D ShadedScene;

void main() {

    vec3 sun_vector = get_sun_vector();

    // Get current scene color
    vec2 texcoord = get_texcoord();
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec4 scene_color = textureLod(ShadedScene, texcoord, 0);

    float prefiltered_shadow = textureLod(PrefilteredShadows, texcoord, 0).x;

    // Early out
    if (prefiltered_shadow < 1e-5) {
        result = scene_color;
        return;
    }

    // Get the material data
    Material m = unpack_material(GBuffer);
    vec3 transmittance = vec3(1);

    // Compute the sun lighting
    vec3 v = normalize(MainSceneData.camera_pos - m.position);
    vec3 l = sun_vector;
    vec3 sun_color = get_sun_color() * get_sun_color_scale(sun_vector);

    {
        vec3 reflected_dir = reflect(-v, m.normal);
        const float sun_angular_radius = degree_to_radians(0.54);
        const float r = sin(sun_angular_radius); // Disk radius
        const float d = cos(sun_angular_radius); // Distance to disk

        // Closest point to a disk (since the radius is small, this is
        // a good approximation)
        float DdotR = dot(sun_vector, reflected_dir);
        vec3 S = reflected_dir - DdotR * sun_vector;
        l = DdotR < d ? normalize(d * sun_vector + normalize(S) * r) : reflected_dir;
    }

    vec3 lighting_result = apply_light(m, v, l, sun_color, 1.0, prefiltered_shadow, transmittance);

    // Backface shading for foliage, not physically correct but a nice
    // approximation
    float foliage_factor = m.shading_model == SHADING_MODEL_FOLIAGE ? 1.0 : 0.0;
    lighting_result += foliage_factor * prefiltered_shadow * sun_color * m.basecolor * 0.12 * saturate(3.0 * dot(v, -sun_vector));

    #if DEBUG_MODE
        lighting_result *= 0;
    #endif

    #if MODE_ACTIVE(PSSM_SPLITS)
        lighting_result = vec3(prefiltered_shadow);
    #endif

    result = max(vec4(0), scene_color) * 1 + vec4(lighting_result, 0);
}
