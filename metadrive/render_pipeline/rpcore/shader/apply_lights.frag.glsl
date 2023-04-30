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

// This shader performs the screen space shading of all lights

#pragma include "render_pipeline_base.inc.glsl"

// Tell the lighting pipeline we are doing this in screen space, so gl_FragCoord
// is available.
#define IS_SCREEN_SPACE 1

#pragma include "includes/light_culling.inc.glsl"
#pragma include "includes/transforms.inc.glsl"
#pragma include "includes/lighting_pipeline.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"

out vec4 result;

uniform GBufferData GBuffer;

// Used for velocity rendering mode
#if MODE_ACTIVE(VELOCITY)
    uniform sampler2D CombinedVelocity;
#endif

void main() {

    // Extract material properties
    vec2 texcoord = get_texcoord();
    Material m = unpack_material(GBuffer);
    ivec3 tile = get_lc_cell_index(ivec2(gl_FragCoord.xy),
        distance(MainSceneData.camera_pos, m.position));

    // Don't shade pixels out of the shading range
    #if !DEBUG_MODE
        if (tile.z >= LC_TILE_SLICES) {
            result = vec4(0, 0, 0, 1);
            return;
        }
    #endif

    // Apply all lights
    result = vec4(shade_material_from_tile_buffer(m, tile), 1);

    /*

    Various debugging modes for previewing materials

    */

    #if MODE_ACTIVE(DIFFUSE)
        result.xyz = vec3(m.basecolor);
    #endif

    #if MODE_ACTIVE(ROUGHNESS)
        result.xyz = vec3(m.roughness);
    #endif

    #if MODE_ACTIVE(SPECULAR)

        // Multiply specular on diffuse materials to make it visible
        result.xyz = vec3(mix(
            m.specular / 0.18,
            m.specular,
            m.metallic
            ));
    #endif

    #if MODE_ACTIVE(NORMAL)
        result.xyz = vec3(abs(m.normal));
    #endif

    #if MODE_ACTIVE(METALLIC)
        result.xyz = vec3(m.metallic);
    #endif

    #if MODE_ACTIVE(TRANSLUCENCY)
        result.xyz = vec3(m.shading_model == SHADING_MODEL_FOLIAGE ? m.shading_model_param0 : 0.0);
    #endif

    #if MODE_ACTIVE(SHADING_MODEL)
        result.xyz = vec3(0.1, 0.1, 0.1);
        switch (m.shading_model) {
            case SHADING_MODEL_FOLIAGE: result.xyz = vec3(0, 1, 0); break;
            case SHADING_MODEL_CLEARCOAT: result.xyz = vec3(0, 0, 1); break;
            case SHADING_MODEL_SKIN: result.xyz = vec3(1, 0, 0); break;
            case SHADING_MODEL_EMISSIVE: result.xyz = vec3(1, 0, 1); break;
            case SHADING_MODEL_TRANSPARENT: result.xyz = vec3(0, 1, 1); break;
        }
    #endif

    #if MODE_ACTIVE(VELOCITY)
        result.xyz = abs(textureLod(CombinedVelocity, texcoord, 0).xyz) * 20.0;
    #endif

}
