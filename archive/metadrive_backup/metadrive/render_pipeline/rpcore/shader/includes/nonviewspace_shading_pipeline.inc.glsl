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

#pragma once

#pragma include "includes/shadows.inc.glsl"
#pragma include "includes/material.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"
#pragma include "includes/vertex_output.struct.glsl"

#if DONT_FETCH_DEFAULT_TEXTURES
    // Don't bind any samplers in this case, so the user can do it on his own
#else
    uniform sampler2D p3d_Texture0;
#endif

uniform Panda3DMaterial p3d_Material;

layout(location = 0) in VertexOutput vOutput;

#pragma include "includes/normal_mapping.inc.glsl"
#pragma include "includes/forward_shading.inc.glsl"

MaterialShaderOutput prepare_material(MaterialBaseInput mInput, vec2 texcoord) {
    MaterialShaderOutput m;

    #if DONT_SET_MATERIAL_PROPERTIES
        // Leave material properties unitialized, and hope the user knows
        // what he's doing.
    #else

        #if DONT_FETCH_DEFAULT_TEXTURES
            vec4 sampled_diffuse = vec4(1);
        #else
            vec4 sampled_diffuse = texture(p3d_Texture0, texcoord);
        #endif

        // XXX: Support for alpha testing
        // if (sampled_diffuse.w < 0.5) discard;

        // XXX: *maybe* support for normal mapping

        // Copy default material properties
        m.basecolor = sampled_diffuse.xyz * mInput.color;
        m.shading_model = mInput.shading_model;
        m.normal = vOutput.normal;
        m.metallic = mInput.metallic;
        m.specular_ior = mInput.specular_ior;
        m.roughness = mInput.roughness;
        m.shading_model_param0 = mInput.arbitrary0;
    #endif

    return m;
}
