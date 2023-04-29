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

// Combines camera and per object velocity by using the velocity of the
// closest fragment

#pragma optionNV (unroll all)

#define USE_GBUFFER_EXTENSIONS
#pragma include "render_pipeline_base.inc.glsl"
#pragma include "includes/gbuffer.inc.glsl"

out vec2 result;

void main() {
    vec2 texcoord = get_texcoord();
    ivec2 coord = ivec2(gl_FragCoord.xy);

    vec3 closest = vec3(0, 0, 1);
    const int filter_size = 1;

    // Take velocity of closest fragment
    for (int i = -filter_size; i <= filter_size; ++i) {
        for (int j = -filter_size; j <= filter_size; ++j) {
            if ((i == 0 && j == 0) || (abs(i) == 2 && abs(j) == 2)) {
                vec2 offcoord = texcoord + vec2(i, j) / SCREEN_SIZE;
                float depth = get_depth_at(offcoord);
                if (depth < closest.z) {
                    closest = vec3(offcoord, depth);
                }
            }
        }
    }

    // Combine camera and per object velocity.
    // XXX: Most likely this is wrong. But since per-object velocity currently
    // is disabled, its not an issue.
    vec2 camera_velocity = get_camera_velocity(closest.xy);
    vec2 per_object_velocity = get_object_velocity_at(closest.xy);
    result = camera_velocity + per_object_velocity;
}
