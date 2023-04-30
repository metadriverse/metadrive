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


vec3 sun_vector = get_sun_vector();

uniform sampler3D ScatteringLUT;


// Fetches the scattering value from the LUT
vec3 get_scattering(vec3 surface_pos) {
    surface_pos = normalize(surface_pos);

    float elevation, theta, radius;
    vector_to_spherical(surface_pos, theta, elevation, radius);

    float sun_elevation = TimeOfDay.scattering.sun_altitude / 180.0 * M_PI;
    float sun_azimuth = TimeOfDay.scattering.sun_azimuth / 180.0 * M_PI;

    float night_factor = saturate(sun_elevation * 30.0);

    float max_angle_factor = 40.0 / 180.0 * M_PI;
    float elevation_factor = sun_elevation / max_angle_factor;

    float slice_index = 0.5 / 100.0 + elevation_factor;

    float factor_x = theta;
    float factor_y = elevation;

    float neg_solar_elevation = HALF_PI - sun_elevation;

    float cos_gamma = cos(elevation) * cos(neg_solar_elevation)
        + sin(elevation) * sin(neg_solar_elevation)
        * cos(theta - sun_azimuth + M_PI);

    float gamma = acos(cos_gamma);

    elevation *= 0.95;

    vec2 lut_coord = vec2(gamma / TWO_PI, 1 - (elevation / HALF_PI));
    vec3 value = textureLod(ScatteringLUT, vec3(lut_coord, slice_index), 0).xyz ;
    value *= 25.0;
    value *= night_factor;
    return vec3(value);

}

// Fetches the scattering value at a given surface position
vec3 get_scattering_at_surface(vec3 surface_pos) {
    vec3 vec_to_cam = vec3(surface_pos - MainSceneData.camera_pos);
    surface_pos.xy = vec2(vec_to_cam.xy * 0.6);
    return get_scattering(surface_pos);
}

vec3 DoScattering(vec3 surface_pos, vec3 view_dir, out float fog_factor)
{

    // Move surface pos above ocean level
    if (surface_pos.z < -0.01) {
        vec3 v2s = surface_pos - MainSceneData.camera_pos;
        float z_factor = abs(MainSceneData.camera_pos.z) / abs(v2s.z);
        surface_pos = MainSceneData.camera_pos + v2s * z_factor;
        view_dir = normalize(surface_pos - MainSceneData.camera_pos);
    }

    float path_length = distance(surface_pos, MainSceneData.camera_pos);
    vec3 inscatter = get_scattering(surface_pos);
    fog_factor = 1.0;


    // Check if the ray is finite
    if (path_length < 20000.0) {

        // Integrate scattering
        const int num_steps = 6;
        float curr_h = MainSceneData.camera_pos.z;

        curr_h *= 1.0 - saturate(path_length / 30000.0);

        float h_step = (surface_pos.z - MainSceneData.camera_pos.z) / num_steps;
        vec3 accum = vec3(0);
        for (int i = 0; i < num_steps; ++i) {
            curr_h += h_step;
            accum += get_scattering_at_surface(vec3(surface_pos.xy, curr_h));
        }

        accum /= float(num_steps);

        // Exponential fog
        float fog_ramp = TimeOfDay.scattering.fog_ramp_size;
        fog_factor = saturate(1.0 - exp(-path_length / (0.6 * fog_ramp)));

        // Exponential height fog
        fog_factor *= exp(-pow(max(0, surface_pos.z), 1.2) / (5.0 * 4000.0));
        inscatter = accum * 0.8;
        fog_factor = saturate(1.1 * fog_factor);
    }

    return inscatter;
}
