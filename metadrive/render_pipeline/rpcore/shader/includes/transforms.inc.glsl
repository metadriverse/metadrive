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

uniform mat4 trans_clip_of_mainCam_to_mainRender;
uniform mat4 trans_mainRender_to_view_of_mainCam;
uniform mat4 trans_mainRender_to_clip_of_mainCam;

// Computes linear Z from a given Z value, and near and far plane
float get_linear_z_from_z(float z, float near, float far) {
    return 2.0 * near * far / (far + near - (z * 2.0 - 1) * (far - near));
}

// Computes linear Z from a given Z value
float get_linear_z_from_z(float z) {
    return get_linear_z_from_z(z, CAMERA_NEAR, CAMERA_FAR);
}

// Computes the Z component from a position in NDC space
float get_z_from_ndc(vec3 ndc_pos) {
    return get_linear_z_from_z(ndc_pos.z);
}

// Computes linear Z from a given Z value, and near and far plane, for orthographic projections
float get_linear_z_from_z_ortographic(float z, float near, float far) {
    return 2.0 / (far + near - fma(z, 2.0, -1.0) * (far - near));
}

// Computes the surface position based on a given Z, a texcoord, and the Inverse MVP matrix
vec3 calculate_surface_pos(float z, vec2 tcoord, mat4 inverse_mvp) {
    vec3 ndc_pos = fma(vec3(tcoord.xy, z), vec3(2.0), vec3(-1.0));
    float clip_w = get_z_from_ndc(ndc_pos);

    vec4 proj = inverse_mvp * vec4(ndc_pos * clip_w, clip_w);
    return proj.xyz / proj.w;
}

// Computes the surface position based on a given Z and a texcoord
vec3 calculate_surface_pos(float z, vec2 tcoord) {
    #if 0
    return calculate_surface_pos(z, tcoord, trans_clip_of_mainCam_to_mainRender);
    #else
    float linz = get_linear_z_from_z(z);
    return mix(
        mix(MainSceneData.ws_frustum_directions[0],
            MainSceneData.ws_frustum_directions[1], tcoord.x),
        mix(MainSceneData.ws_frustum_directions[2],
            MainSceneData.ws_frustum_directions[3], tcoord.x),
        tcoord.y
    ).xyz * linz + MainSceneData.camera_pos;
    #endif
}

// Computes the surface position based on a given Z and a texcoord, aswell as a
// custom near and far plane, and the inverse MVP. This is for orthographic projections
vec3 calculate_surface_pos_ortho(float z, vec2 tcoord, float near, float far, mat4 inverse_mvp) {
    vec3 ndc_pos = fma(vec3(tcoord.xy, z), vec3(2.0), vec3(-1.0));
    float clip_w = get_linear_z_from_z_ortographic(z, near, far);
    vec4 result = inverse_mvp * vec4(ndc_pos * clip_w, clip_w);
    return result.xyz / result.w;
}

// Computes the view position from a given Z value and texcoord
vec3 calculate_view_pos(float z, vec2 tcoord) {
    vec4 view_pos = MainSceneData.inv_proj_mat *
    vec4(fma(tcoord.xy, vec2(2.0), vec2(-1.0)), z, 1.0);
    return view_pos.xyz / view_pos.w;
}

// Computes the NDC position from a given view position
vec3 view_to_screen(vec3 view_pos) {
    vec4 projected = MainSceneData.proj_mat * vec4(view_pos, 1);
    projected.xyz /= projected.w;
    projected.xy = fma(projected.xy, vec2(0.5), vec2(0.5));
    return projected.xyz;
}

// Converts a view space normal to world space
vec3 view_normal_to_world(vec3 view_normal) {
    // We need to transform the coordinate system, should not be required,
    // seems to be some panda bug?
    view_normal = view_normal.xzy * vec3(1, -1, 1);
    return normalize((vec4(view_normal, 0) * trans_mainRender_to_view_of_mainCam).xyz);
}

// Converts a world space position to screen space position (NDC)
vec3 world_to_screen(vec3 world_pos) {
    vec4 proj = trans_mainRender_to_clip_of_mainCam * vec4(world_pos, 1);
    proj.xyz /= proj.w;
    proj.xyz = fma(proj.xyz, vec3(0.5), vec3(0.5));
    return proj.xyz;
}

// Converts a world space normal to view space
vec3 world_normal_to_view(vec3 world_normal) {
    vec4 proj = trans_mainRender_to_view_of_mainCam * vec4(world_normal, 0);
    proj.xyz *= vec3(1, -1, 1);
    return normalize(proj.xzy);
}
