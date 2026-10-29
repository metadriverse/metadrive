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

#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/brdf.inc.glsl"
#pragma include "includes/color_spaces.inc.glsl"

layout(location = 0) out vec4 result;

uniform sampler2D SourceTex;
uniform sampler2D LastFrameColor;
uniform GBufferData GBuffer;

// Fades out a coordinate on the screen edges
float get_border_fade(vec2 coord) {
    const float border_fade = max(1e-5, GET_SETTING(ssr, border_fade));
    float fade = 1.0;
    fade *= saturate(coord.x / border_fade) * saturate(coord.y / border_fade);
    fade *= saturate((1 - coord.x) / border_fade) * saturate((1 - coord.y) / border_fade);
    return fade;
}

void main() {

    // Get bilateral sample coordinates
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 bil_start_coord = get_bilateral_coord(coord);
    vec2 texcoord = ((bil_start_coord * 2) + 0.5) / SCREEN_SIZE;

    // Get current pixel data
    float mid_depth = get_gbuffer_depth(GBuffer, coord);
    vec3 ws_normal = get_gbuffer_normal(GBuffer, coord);

    Material m = unpack_material(GBuffer, texcoord);

    // Early out on skybox
    if (is_skybox(m.position)) {
        result = vec4(0);
        return;
    }

    const float max_depth_diff = 0.002;

    float weights = 0.0;
    vec4 accum = vec4(0);

    // Controls how many other pixels should be taken into account, besides
    // of the 4 direct neighbors.
    const int search_radius = 0;

    vec3 view_vector = normalize(MainSceneData.camera_pos - m.position);
    vec4 avg_position = vec4(0);

    // Get reflection directory
    vec3 reflected_dir = get_reflection_vector(m, -view_vector);
    float roughness = get_effective_roughness(m);

    // Accumulate all samples
    for (int x = -search_radius; x < 2 + search_radius; ++x) {
        for (int y = -search_radius; y < 2 + search_radius; ++y) {

            // Get sample coordinate
            ivec2 source_coord = bil_start_coord + ivec2(x, y);
            ivec2 screen_coord = source_coord * 2;
            vec2 intersection = texelFetch(SourceTex, source_coord, 0).xy;
            float intersection_weight = intersection.x > 1e-5 ? 1 : 0.0;

            // Skip empty samples
            if (intersection_weight < 0.5) {
                weights += 1.0;
                continue;
            }

            float sample_depth = get_gbuffer_depth(GBuffer, screen_coord);

            // Find depth of intersection pixel
            float intersection_depth = get_gbuffer_depth(GBuffer, intersection);

            // Get world space position of intersection pixel
            vec3 ws_intersection = calculate_surface_pos(intersection_depth, intersection);

            // Find distance and vector to intersection based on the pixels position
            float distance_to_intersection = 1e-9 + distance(ws_intersection, m.position);
            vec3 l = (ws_intersection - m.position) / distance_to_intersection;

            // Get the halfway vector
            vec3 h = normalize(view_vector + l);
            float NxH = max(1e-5, dot(m.normal, h));

            float weight = clamp(brdf_distribution_ggx(NxH, 0.05 + sqrt(roughness)), 0.0, 1.0);
            weight *= 1 - saturate(abs(mid_depth - sample_depth) / max_depth_diff);
            vec4 color_sample = textureLod(LastFrameColor, intersection, 0);

            // Fade out samples at the screen border
            // XXX: Do we really have to this per sample?
            color_sample *= get_border_fade(intersection);

            accum += color_sample * weight;
            weights += weight;
        }
    }

    if (weights < 0.4) {
        accum = vec4(0);
    } else {
        accum /= weights;
    }

    result = accum;
}
