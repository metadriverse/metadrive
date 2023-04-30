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
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"
#pragma include "includes/noise.inc.glsl"
#pragma include "vxgi.inc.glsl"

uniform sampler2D ShadedScene;
uniform GBufferData GBuffer;
out vec4 result;

void main() {
    // Get texture coordinate
    ivec2 coord = ivec2(gl_FragCoord.xy) * 2;
    vec2 texcoord = (coord + 0.5) / SCREEN_SIZE;
    vec3 noise_vec = vec3(0, 0, 0);

    // Get material data
    Material m = unpack_material(GBuffer, texcoord);
    vec3 voxel_coord = worldspace_to_voxelspace(m.position);

    // Get view vector
    vec3 view_vector = normalize(MainSceneData.camera_pos - m.position);
    vec3 reflected_dir = reflect(-view_vector, m.normal);

    if (out_of_unit_box(voxel_coord))
    {
        result = textureLod(ScatteringIBLDiffuse, m.normal, 0);
        return;
    }

    // Trace diffuse cones
    vec4 accum = vec4(0);

    // float jitter = 16.0 / (MainSceneData.frame_index % 512);
    float jitter = 0.0;

    for (int i = 0; i < 8; ++i) {
        vec3 direction = rand_rgb(vec2(texcoord + i * 0.001 +
            0.134 * (MainSceneData.frame_index % 1024)));

        direction = normalize(direction);
        direction = face_forward(direction, m.normal);
        float weight = max(0.0, dot(m.normal, direction)); // Guaranteed to be > 0
        // weight = 1.0;
        vec4 cone = trace_cone(
            voxel_coord,
            m.normal,
            direction,
            // GET_SETTING(vxgi, diffuse_cone_steps),
            32,
            false,
            0.05,
            jitter);
        accum.xyz += cone.xyz * weight;
        accum.w += weight;
    }
    accum /= max(1e-3, accum.w);
    accum *= 1.0;
    accum = clamp(accum, vec4(0.0), vec4(100.0));
    result = accum;
}
