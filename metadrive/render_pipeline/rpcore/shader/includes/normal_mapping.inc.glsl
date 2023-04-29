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

// Reconstructs the tangent with the deltas of
// the position and texcoord
void reconstruct_tangent(out vec3 tangent, out vec3 binormal) {
    vec3 pos_dx = dFdx(vOutput.position);
    vec3 pos_dy = dFdy(vOutput.position);
    float tcoord_dx = dFdx(vOutput.texcoord.y);
    float tcoord_dy = dFdy(vOutput.texcoord.y);

    // Fix issues when the texture coordinate is wrong, this happens when
    // two adjacent vertices have the same texture coordinate, as the gradient
    // is 0 then. We just assume some hard-coded tangent and binormal then
    if (abs(tcoord_dx) < 1e-24 && abs(tcoord_dy) < 1e-24) {
        vec3 base = abs(vOutput.normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0);
        tangent = normalize(cross(vOutput.normal, base));
    } else {
        tangent = normalize(pos_dx * tcoord_dy - pos_dy * tcoord_dx);
    }

    binormal = normalize(cross(tangent, vOutput.normal));
}


// Aplies a normal map with a given base normal and displace normal, weighted by
// the bump factor
vec3 apply_normal_map(vec3 base_normal, vec3 displace_normal, float bump_factor,
        vec3 tangent, vec3 binormal) {
    // Optional: Make sure the base normal is correct
    // base_normal = normalize(base_normal);
    displace_normal = mix(vec3(0, 0, 1), displace_normal, saturate(bump_factor));
    return vec3(
        tangent * displace_normal.x +
        binormal * displace_normal.y +
        base_normal * displace_normal.z
    );
}


vec3 apply_normal_map(vec3 base_normal, vec3 displace_normal, float bump_factor) {
    vec3 tangent, binormal;
    reconstruct_tangent(tangent, binormal);
    return apply_normal_map(base_normal, displace_normal, bump_factor, tangent, binormal);
}

// Parallax Mapping
vec2 get_parallax_texcoord(sampler2D displacement_map, float strength) {
    // To disable parallax mapping:
    // return vOutput.texcoord;

    const float max_dist = 50.0;
    vec3 vec_to_cam = vOutput.position - MainSceneData.camera_pos;

    float initial_height = texture(displacement_map, vOutput.texcoord).x;
    float pixel_dist = length(vec_to_cam);

    // Early out for materials without parallax mapping
    if (initial_height > 0.999 || pixel_dist > max_dist) return vOutput.texcoord;

    float NxV = max(0.0, dot(vOutput.normal, vec_to_cam / pixel_dist)); // xxx merge with pixel dist

    float raymarch_distance = 0.2 * strength;
    int num_steps = max(5, int((40 - (pixel_dist / max_dist) * 37.0) * (1 - NxV)));

    vec3 tangent, binormal;
    reconstruct_tangent(tangent, binormal);

    vec3 view_vector = normalize(MainSceneData.camera_pos - vOutput.position);

    // Project view vector to tangent space
    vec2 tex_offs = vec2(dot(-tangent, view_vector), dot(binormal, view_vector));

    // Get the ray start and direction
    vec3 current_pos = vec3(vOutput.texcoord, 1);
    raymarch_distance *= 0.5 / clamp(dot(vOutput.normal, view_vector), 0.3, 1.0);
    vec3 offs_step = vec3(tex_offs * raymarch_distance, -1.0) / float(num_steps);

    // Raymarch
    vec3 last_hit = current_pos;
    for (int i = 0; i < num_steps; ++i) {
        float sample_h = texture(displacement_map, current_pos.xy).x;
        current_pos += offs_step;
        if (sample_h <= current_pos.z) {
            last_hit = current_pos;
        }
    }

    float fade = square(square(square(pixel_dist / max_dist)));
    return mix(last_hit.xy, vOutput.texcoord, fade);
}
