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



// Include local scattering code
#define NO_COMPUTE_SHADER 1
#pragma include "scattering_common.glsl"
#pragma include "includes/gbuffer.inc.glsl"

uniform sampler3D InscatterSampler;
uniform sampler2D IrradianceSampler;

vec3 sun_vector = get_sun_vector();
/*


Some parts of this code are taken from the master thesis of Stefan Sperlhofer,
which unfortunately does not seem to be online anymore, so I can't link to it.


*/

const float EPSILON_ATMOSPHERE = 0.002f;
const float EPSILON_INSCATTER = 0.01f;


// input - d: view ray in world space
// output - offset: distance to atmosphere or 0 if within atmosphere
// output - max_path_length: distance traversed within atmosphere
// output - return value: intersection occurred true/false
bool intersect_atmosphere(vec3 cam_pos, vec3 d, out float offset, out float max_path_length)
{
    offset = 0.0f;
    max_path_length = 0.0f;

    // vector from ray origin to center of the sphere
    vec3 l = -cam_pos;
    float l2 = dot(l, l);
    float s = dot(l, d);

    // adjust top atmosphere boundary by small epsilon to prevent artifacts
    float r = Rt - EPSILON_ATMOSPHERE;
    float r2 = r * r;
    if(l2 <= r2)
    {
        // ray origin inside sphere, hit is ensured
        float m2 = l2 - (s * s);
        float q = sqrt(r2 - m2);
        max_path_length = s + q;
        return true;
    }
    else if(s >= 0)
    {
        // ray starts outside in front of sphere, hit is possible
        float m2 = l2 - (s * s);
        if(m2 <= r2)
        {
            // ray hits atmosphere definitely
            float q = sqrt(r2 - m2);
            offset = s - q;
            max_path_length = (s + q) - offset;
            return true;
        }
    }
    return false;
}


// Transforms a point from world-space to atmospheric-space.
// This is required because we use different coordinate systems for normal
// rendering and atmospheric scattering.
vec3 worldspace_to_atmosphere(vec3 pos) {
    pos.z += GET_SETTING(scattering, atmosphere_start);
    pos /= 1000.0;
    pos.z = max(0, pos.z);
    pos.z += Rg;
    return pos;
}

vec3 get_inscattered_light(vec3 surface_pos, vec3 view_dir, inout vec3 attenuation,
        inout float irradiance_factor)
{
    vec3 inscattered_light = vec3(0);
    float offset;
    float max_path_length;

    vec3 cam_pos = MainSceneData.camera_pos;
    // cam_pos.z = max(cam_pos.z, surface_pos.z);
    cam_pos = worldspace_to_atmosphere(cam_pos);

    if (is_skybox(surface_pos)) {
        surface_pos = worldspace_to_atmosphere(surface_pos * 1e10);
    } else {
        surface_pos = worldspace_to_atmosphere(surface_pos);
    }

    if (intersect_atmosphere(cam_pos, view_dir, offset, max_path_length)) {
        float path_length = distance(cam_pos, surface_pos);

        // check if object occludes atmosphere
        if (path_length > offset) {

            // offsetting camera
            vec3 start_pos = cam_pos + offset * view_dir;
            float start_pos_height = length(start_pos);
            path_length -= offset;

            // starting position of path is now ensured to be inside atmosphere
            // was either originally there or has been moved to top boundary
            float mustart_pos = dot(start_pos, view_dir) / start_pos_height;
            float nustart_pos = dot(view_dir, sun_vector);
            float musstart_pos = dot(start_pos, sun_vector) / start_pos_height;

            // in-scattering for infinite ray (light in-scattered when
            // no surface hit or object behind atmosphere)
            vec4 inscatter = max(texture4D(InscatterSampler, start_pos_height,
                mustart_pos, musstart_pos, nustart_pos), 0.0f);
            float surface_pos_height = length(surface_pos);
            float musEndPos = dot(surface_pos, sun_vector) / surface_pos_height;

            // check if object hit is inside atmosphere
            if (path_length < max_path_length) {

             // reduce total in-scattered light when surface hit
             // within atmosphere
             // fíx described in chapter 5.1.1
                attenuation = analyticTransmittance(start_pos_height, mustart_pos, path_length);
                float muEndPos = dot(surface_pos, view_dir) / surface_pos_height;
                vec4 inscatterSurface = texture4D(InscatterSampler, surface_pos_height,
                    muEndPos, musEndPos, nustart_pos);
                inscatter = max(inscatter - attenuation.rgbr * inscatterSurface, 0.0f);
                irradiance_factor = 1.0f;
            } else {
                // retrieve extinction factor for infinite ray
                // fíx described in chapter 5.1.1
                attenuation = analyticTransmittance(start_pos_height, mustart_pos, path_length);
            }

            // avoids imprecision problems near horizon by interpolating between
            // two points above and below horizon
            // fíx described in chapter 5.1.2

            #if 0
                float muHorizon = -sqrt(1.0 - (Rg / start_pos_height) * (Rg / start_pos_height));
                if (abs(mustart_pos - muHorizon) < EPSILON_INSCATTER)
                {
                    float mu = muHorizon - EPSILON_INSCATTER;
                    float samplePosHeight = sqrt(start_pos_height * start_pos_height
                        + path_length * path_length + 2.0f * start_pos_height *
                        path_length * mu);
                    float muSamplePos = (start_pos_height * mu + path_length) / samplePosHeight;
                    vec4 inScatter0 = texture4D(InscatterSampler, start_pos_height, mu,
                        musstart_pos, nustart_pos);
                    vec4 inScatter1 = texture4D(InscatterSampler, samplePosHeight,
                        muSamplePos, musEndPos, nustart_pos);
                    vec4 inScatterA = max(inScatter0 - attenuation.rgbr * inScatter1, 0.0);
                    mu = muHorizon + EPSILON_INSCATTER;
                    samplePosHeight = sqrt(start_pos_height * start_pos_height
                        + path_length * path_length + 2.0f *
                        start_pos_height * path_length * mu);
                    muSamplePos = (start_pos_height * mu + path_length) / samplePosHeight;
                    inScatter0 = texture4D(InscatterSampler, start_pos_height, mu,
                        musstart_pos, nustart_pos);
                    inScatter1 = texture4D(InscatterSampler, samplePosHeight, muSamplePos,
                        musEndPos, nustart_pos);
                    vec4 inScatterB = max(inScatter0 - attenuation.rgbr * inScatter1,
                        0.0f);
                    float t = ((mustart_pos - muHorizon) + EPSILON_INSCATTER) /
                        (2.0 * EPSILON_INSCATTER);
                    inscatter = mix(inScatterA, inScatterB, t);
                }
            #endif

            // avoids imprecision problems in Mie scattering when sun is below
             //horizon
            // fíx described in chapter 5.1.3
            inscatter.w *= smoothstep(0.00, 0.02, musstart_pos);
            float phaseR = phaseFunctionR(nustart_pos);
            float phaseM = phaseFunctionM(nustart_pos);

            phaseR *= GET_SETTING(scattering, rayleigh_factor);

            inscattered_light = max(inscatter.rgb * phaseR + getMie(inscatter) * phaseM, 0.0f);

        }
    }



    #if !HAVE_PLUGIN(color_correction)
        // Reduce scattering, otherwise its way too bright without automatic
        // exposure
        // inscattered_light /= 7.0;
    #endif

    return inscattered_light;
}



vec3 DoScattering(vec3 surface_pos, vec3 view_dir, out float sky_clip)
{
    vec3 attenuation = vec3(0);
    float irradiance_factor = 0.0;
    vec3 scattering = get_inscattered_light(surface_pos, view_dir, attenuation, irradiance_factor);

    sky_clip = is_skybox(surface_pos) ? 1.0 : 0.0;

    // Reduce scattering in the near to avoid artifacts due to low precision
    scattering *= saturate(distance(surface_pos, MainSceneData.camera_pos) / 500.0 - 0.0);

    return scattering;
}
