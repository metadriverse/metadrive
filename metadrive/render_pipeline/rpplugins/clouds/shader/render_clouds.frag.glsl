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

// Renders the clouds
// WORK IN PROGRESS - This shader is unfinished and not cleaned up yet.

#define USE_TIME_OF_DAY 1
#pragma include "render_pipeline_base.inc.glsl"

#define USE_GBUFFER_EXTENSIONS
#pragma include "includes/gbuffer.inc.glsl"
#pragma include "includes/light_culling.inc.glsl"
#pragma include "includes/noise.inc.glsl"

uniform sampler3D Noise1;
uniform sampler3D Noise2;
uniform sampler2D WeatherTex;

out vec4 result;

const float KM = 1000.0;
const float METER = 1.0;

const float earth_radius = 6371.0 * KM;
const vec3 earth_mid = vec3(0, 0, -earth_radius);
const float cloud_start = earth_radius + 1.3 * KM;
const float cloud_end = earth_radius + 25.0 * KM;


float GetHeightFractionForPoint(vec3 inPosition, vec2 inCloudMinMax)
{
    float height_fraction = (inPosition.z - inCloudMinMax.x) / (inCloudMinMax.y - inCloudMinMax.x);
    return saturate(height_fraction);
}

float GetDensityHeightGradientForPoint(vec3 p, vec3 weatherData) {
    // return saturate(20.0 * (1.0 - max(0, p.z / 0.3)) ); // XXX
    // return p.z > 0.1 ? 0.0 : 1.0;
    // return pow(p.z, 0.3);
    return 1.0;
}

float SampleCloudDensity(vec3 p, vec3 weather_data, float mip_level)
{

    // float3 wind_direction = float3 (1.0 , 0.0 , 0.0) ;
    // float cloud_speed = 10.0;
    // float cloud_top_offset = 500.0;
    // p += height_fraction * wind_direction * cloud_top_offset;

    vec4 low_frequency_noises = textureLod(Noise1, p * 0.6, mip_level);
    float low_freq_FBM = (low_frequency_noises.g * 0.625)
                        + (low_frequency_noises.b * 0.25)
                        + (low_frequency_noises.a * 0.125);

    float base_cloud = max(0, square(low_frequency_noises.x * low_frequency_noises.y) - 0.07) ;
    // base_cloud = pow(base_cloud, 10.0) * 10.0;

    // base_cloud *= max(0, low_frequency_noises.w * 1.4 - 0.2);
    // base_cloud *= max(0, low_frequency_noises.z * 1.5 - 0.2);

    float density_height_gradient = GetDensityHeightGradientForPoint(p, weather_data);
    base_cloud *= density_height_gradient;

    float cloud_coverage = weather_data.r;

    // base_cloud *= cloud_coverage;

    vec3 high_frequency_noises = textureLod(Noise2, p * 5.63534, mip_level).rgb;

    // float high_freq_FBM = (high_frequency_noises.r * 0.625);
    //                     + (high_frequency_noises.g * 0.25)
    //                     + (high_frequency_noises.b * 0.125);

    // base_cloud = mix(high_frequency_noises.y * base_cloud, 1, base_cloud);

    base_cloud -= high_frequency_noises.y * 0.23 * (1 - base_cloud);
    // base_cloud *= 3.0;
    base_cloud *= 125.0 * 256.0 / GET_SETTING(clouds, raymarch_steps);

    return saturate(base_cloud);
}


vec2 get_cloud_coord(vec3 pos) {
    vec2 xy_coord = pos.xy / (cloud_end - cloud_start);
    xy_coord.xy /= 1.0 + 0.1 * length(xy_coord);
    // xy_coord.xy += 0.5;
    // xy_coord *= 0.5;
    return xy_coord;
}

float HenyeyGreenstein(vec3 inLightVector, vec3 inViewVector, float inG)
{
    float cos_angle = dot(normalize(inLightVector), normalize(inViewVector));
    return ((1.0 - inG * inG) / pow((1.0 + inG * inG - 2.0 * inG * cos_angle),
        3.0 / 2.0)) / 4.0 * M_PI;
}

void main() {
    int num_samples = GET_SETTING(clouds, raymarch_steps);
    // int num_samples = 256;

    vec2 texcoord = get_half_texcoord();
    vec3 wind_offs = vec3(0.2, 0.3, 0) * 0.052 * MainSceneData.frame_time;

    vec3 pos = get_gbuffer_position(GBuffer, texcoord);
    vec3 ray_start = MainSceneData.camera_pos;
    vec3 ray_dir = normalize(pos - ray_start);

    vec3 view_vector = normalize(MainSceneData.camera_pos - pos);

    if (!is_skybox(pos) || ray_dir.z < 0.0) {
        result = vec4(0);
        return;
    }

    float t_low, t_high, tmp;

    Sphere earth_sphere;
    earth_sphere.pos = earth_mid;
    earth_sphere.radius = cloud_start;
    bool rb = ray_sphere_intersection(earth_sphere, ray_start, ray_dir, t_low, tmp);
    earth_sphere.radius = cloud_end;
    bool rt = ray_sphere_intersection(earth_sphere, ray_start, ray_dir, t_high, tmp);

    if (t_low < 0.0) t_low = 0.0;
    if (t_high < 0.0 || distance(t_high, t_low) < 0.01) {
        result = vec4(0.2, 0, 0, 0);
        return;
    }

    // Get start and end in cloud space coordinates
    vec3 trace_start = vec3(get_cloud_coord(ray_start + t_low * ray_dir), 0.0);
    vec3 trace_end = vec3(get_cloud_coord(ray_start + t_high * ray_dir), 1.0);
    trace_start += wind_offs;
    trace_end += wind_offs;

    // trace_start.xyz += (noise*2.0-1.0) * 0.004;
    vec3 trace_step = (trace_end - trace_start) / float(num_samples);
    // trace_step.xyz += (noise*2.0-1.0) * 0.015 / num_samples;

    float density = 0.0;
    float cloud_test = 0.0;
    int zero_density_sample_count = 0;
    float mip_level = 0;

    float jitter = abs(rand(ivec2(gl_FragCoord.xy)));

    vec3 p = trace_start + (1 + jitter) * trace_step;

    vec3 accum_color = vec3(0);

    vec3 sun_vector = get_sun_vector();

    vec3 weather_data = texture(WeatherTex, p.xy).xyz;
    for (int i = 0; i < num_samples - 1; ++i)
    {
        float sampled_density = SampleCloudDensity(p, weather_data, mip_level) * 0.2;
        float sampled_sun_density = 0.0;
        for (int k = 1; k < 3; ++k) {
            sampled_sun_density += SampleCloudDensity(
                p + sun_vector * 1.0 / 256.0 * k * k, weather_data, mip_level);
        }
        sampled_density *= (1 - density);
        density += sampled_density;
        accum_color += ((0.05 + 0.99 * p.z * p.z) * sampled_density *
            (1.0 - sampled_sun_density / 3.0));

        p += trace_step;
    }

    float accum_weight = density;

    float light_samples = density * 1.0;

    float powder_sugar_effect = 1.0 - exp(- light_samples * 2.0);
    float beers_law = exp(-light_samples);
    float light_energy = 2.0 * beers_law * powder_sugar_effect;

    accum_color *= light_energy * 2.0;
    accum_color *= vec3(HenyeyGreenstein(sun_vector, -view_vector, 0.2)) * 1.0;

    float sun_influence = pow(max(0, dot(ray_dir, sun_vector)), 25.0) + 0.0;
    vec3 sun_color = sun_influence * 100.0 * TimeOfDay.scattering.sun_color;

    accum_color *= 1.0 + sun_color * max(0, 1 - 0.7 * density);
    accum_color *= TimeOfDay.clouds.cloud_brightness * 20.0 * vec3(10, 10, 15);
    accum_color *= TimeOfDay.scattering.sun_intensity / 150.0;
    accum_color *= TimeOfDay.scattering.sun_color;


    // Don't render clouds at obligue angles
    float horizon = pow(saturate(ray_dir.z * 1.0), 0.1);
    accum_color *= horizon;
    accum_weight *= horizon;

    result = vec4(accum_color, accum_weight);
}
