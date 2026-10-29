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

%defines%

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/vertex_output.struct.glsl"
#pragma include "includes/material.inc.glsl"

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

uniform mat4 p3d_ViewProjectionMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;

#if EXPERIMENTAL_PREV_TRANSFORM
    uniform mat4 p3d_PrevModelMatrix;
#endif

uniform mat4 p3d_ModelMatrix;
uniform mat3 tpose_world_to_model;

layout(location = 0) out VertexOutput vOutput;

%includes%
%inout%

void main() {
    vOutput.texcoord = p3d_MultiTexCoord0;
    vOutput.normal = normalize(tpose_world_to_model * p3d_Normal).xyz;
    vOutput.position = (p3d_ModelMatrix * p3d_Vertex).xyz;

    %transform%

    // TODO: We have to account for skinning, we can maybe use hardware skinning for this.
    #if IN_GBUFFER_SHADER
        #if EXPERIMENTAL_PREV_TRANSFORM
            vOutput.last_proj_position = p3d_ViewProjectionMatrix *
                (p3d_PrevModelMatrix * p3d_Vertex);
        #else
            vOutput.last_proj_position = p3d_ViewProjectionMatrix *
                vec4(vOutput.position, 1);
        #endif
    #endif

    gl_Position = p3d_ViewProjectionMatrix * vec4(vOutput.position, 1);

    %post_transform%
}
