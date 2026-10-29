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

layout(r11f_g11f_b10f) uniform imageCube RESTRICT DestCubemap;
uniform sampler2D DefaultSkydome;
uniform samplerCube DefaultEnvmap;

#pragma include "scattering_method.inc.glsl"

void main() {

    // Get cubemap coordinate
    int texsize = imageSize(DestCubemap).x;
    ivec2 coord = ivec2(gl_FragCoord.xy);

    // Get cubemap view_vector
    ivec2 clamped_coord; int face;
    vec3 view_vector = texcoord_to_cubemap(texsize, coord, clamped_coord, face);

    // Store horizon
    float horizon = view_vector.z;
    float sky_clip = 0.0;

    // Get inscattered light
    vec3 inscattered_light = DoScattering(view_vector * 1e10, view_vector, sky_clip)
                            * TimeOfDay.scattering.sun_intensity;

    if (horizon > 0.0) {
        inscattered_light *= 2.0; // XXX: This makes it look better, but has no physical background.
        
        // Render clouds to provide more variance for the cubemap
        vec3 cloud_color = textureLod(DefaultSkydome, get_skydome_coord(view_vector), 0).xyz;
        // inscattered_light = cloud_color * 15;
    } else {
        // Blend ambient cubemap at the bottom
        vec3 sun_vector = get_sun_vector();
        vec3 color_scale = get_sun_color_scale(sun_vector) * TimeOfDay.scattering.sun_color;
        inscattered_light = textureLod(DefaultEnvmap, fix_cubemap_coord(view_vector), 0).xyz
                            * TimeOfDay.scattering.sun_intensity / M_PI * color_scale;

        #if !HAVE_PLUGIN(color_correction)
            // inscattered_light *= 0.1;
        #endif
    }

    imageStore(DestCubemap, ivec3(clamped_coord, face), vec4(inscattered_light, 1.0));
}
