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
#pragma include "vxgi.inc.glsl"

uniform sampler2D ShadedScene;
uniform GBufferData GBuffer;
out vec4 result;

void main() {

    Material m = unpack_material(GBuffer);
    vec3 voxel_coord = worldspace_to_voxelspace(m.position);

    // Get view vector
    vec3 view_vector = normalize(MainSceneData.camera_pos - m.position);
    vec3 reflected_dir = reflect(-view_vector, m.normal);

    if (out_of_unit_box(voxel_coord))
    {
        result = textureLod(ScatteringIBLSpecular, reflected_dir, 7) * 0.5;
        return;
    }

    // Trace specular cone
    vec4 specular = trace_cone(
        voxel_coord,
        m.normal,
        reflected_dir,
        GET_SETTING(vxgi, specular_cone_steps),
        true,
        m.roughness * 0.17);

    // specular *= 0.1;
    // specular.xyz = pow(specular.xyz, vec3(2.2));
    // specular.xyz *= 0.01;
    // specular.xyz = specular.xyz / (1 - specular.xyz);
    specular.xyz = max(vec3(0.0), specular.xyz);

    result = specular;
}
