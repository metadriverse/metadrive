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
#pragma include "includes/gbuffer.inc.glsl"

uniform sampler2D ShadedScene;
uniform sampler2D DefaultSkydome;

uniform GBufferData GBuffer;

out vec4 result;

#pragma include "scattering_method.inc.glsl"

void main() {

    vec2 texcoord = get_texcoord();

    // Get material data
    Material m = unpack_material(GBuffer);
    vec3 view_vector = normalize(m.position - MainSceneData.camera_pos);
    vec3 orig_view_vector = view_vector;

    // Fetch scattering
    float sky_clip = 0.0;

    // Prevent a way too dark horizon by clamping the view vector
    if (is_skybox(m)) {
        view_vector.z = max(view_vector.z, 0.05);
    }

    vec3 inscattered_light = DoScattering(m.position, view_vector, sky_clip)
                                * TimeOfDay.scattering.sun_intensity;

    result.xyz = textureLod(ShadedScene, texcoord, 0).xyz;
    result.w = 1;

    // Cloud color
    if (is_skybox(m)) {

        inscattered_light *= M_PI; // XXX: This makes it look better, but has no physical background.

        #if !HAVE_PLUGIN(clouds)
            vec3 cloud_color = textureLod(DefaultSkydome, get_skydome_coord(orig_view_vector), 0).xyz;
            // inscattered_light = 15 * cloud_color * M_PI;
            cloud_color = mix(vec3(0.5), cloud_color, saturate(orig_view_vector.z / 0.05));
            inscattered_light *= 0.5 + 1 * cloud_color;
        #endif

        // Sun disk
        vec3 silhouette_col = vec3(TimeOfDay.scattering.sun_intensity) *
            inscattered_light * sky_clip;
        silhouette_col *= 2.0;
        float disk_factor = pow(saturate(dot(orig_view_vector, sun_vector) + 0.000069), 23.0 * 1e5);
        float upper_disk_factor = smoothstep(0, 1, (orig_view_vector.z + 0.045) * 1.0);
        inscattered_light += vec3(1, 0.3, 0.1) * disk_factor * upper_disk_factor *
            silhouette_col * 3.0 * 1e3;

    } else {
        inscattered_light *= 70.0;
        float dist = distance(m.position, MainSceneData.camera_pos);
        float extinction = saturate(dist / TimeOfDay.scattering.extinction * 10.0);
        inscattered_light *= extinction;
    }

    #if !DEBUG_MODE
        result.xyz += inscattered_light;
    #endif

    #if MODE_ACTIVE(SCATTERING)
        result.xyz = inscattered_light / (1 + inscattered_light);
    #endif

}
