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

// Forward shading pipeline

#pragma include "includes/shadows.inc.glsl"
#pragma include "includes/brdf.inc.glsl"
#pragma include "includes/lights.inc.glsl"
#pragma include "includes/light_data.inc.glsl"
#pragma include "includes/poisson_disk.inc.glsl"

uniform samplerCube DefaultEnvmap;
uniform samplerBuffer AllLightsData;
uniform int maxLightIndex;

#if VIEWSPACE_SHADING
    uniform sampler3D PrefilteredBRDF;
    uniform sampler2D PrefilteredCoatBRDF;
#endif

// Samplers for the specular cubemaps
#if HAVE_PLUGIN(scattering)
    uniform samplerCube ScatteringIBLDiffuse;
    uniform samplerCube ScatteringIBLSpecular;
#endif


#if HAVE_PLUGIN(pssm)
    uniform sampler2DShadow PSSMSceneSunShadowMapPCF;
    uniform mat4 PSSMSceneSunShadowMVP;
#endif

#if HAVE_PLUGIN(sky_ao) 
    #pragma include "/$$rp/rpplugins/sky_ao/shader/sky_ao.inc.glsl"
#endif

#if VIEWSPACE_SHADING

    // Also include environment probes

    #if HAVE_PLUGIN(env_probes)
        #define APPLY_ENVPROBES_PASS
        #pragma include "includes/light_culling.inc.glsl"
        #pragma include "includes/envprobes.inc.glsl"
    #endif

    struct AmbientResult {
        vec3 specular;
        vec3 diffuse;
        float fresnel;
    };

    // Full forward ambient, expensive, and only available when shading in view-space
    AmbientResult get_full_forward_ambient(Material m, vec3 v) {
        vec3 reflected_dir = get_reflection_vector(m, v);
        float roughness = get_effective_roughness(m);

        // Compute angle between normal and view vector
        float NxV = clamp(-dot(m.normal, v), 1e-5, 1.0);

        float env_mipmap = get_mipmap_for_roughness(DefaultEnvmap, roughness , NxV);

        // Sample default environment map
        vec3 ibl_specular = textureLod(DefaultEnvmap, fix_cubemap_coord(reflected_dir),
            env_mipmap).xyz * DEFAULT_ENVMAP_BRIGHTNESS;

        // Get cheap irradiance by sampling low levels of the environment map
        float ibl_diffuse_mip = get_mipmap_count(DefaultEnvmap) - 3.0;
        vec3 ibl_diffuse = textureLod(DefaultEnvmap, fix_cubemap_coord(m.normal),
            ibl_diffuse_mip).xyz * DEFAULT_ENVMAP_BRIGHTNESS;

        // Scattering specific code
        #if HAVE_PLUGIN(scattering)
            float scat_mipmap = m.shading_model == SHADING_MODEL_CLEARCOAT ?
                CLEARCOAT_ROUGHNESS : get_specular_mipmap(m);

            // Sample prefiltered scattering cubemap
            // ibl_specular = textureLod(ScatteringIBLSpecular, reflected_dir, scat_mipmap).xyz;

            // Diffuse IBL
            // ibl_diffuse = textureLod(ScatteringIBLDiffuse, m.normal, 0).xyz;
        #endif

        #if HAVE_PLUGIN(env_probes)
            // vec4 probe_spec, probe_diff;
            // apply_all_probes(m, probe_spec, probe_diff);

            // ibl_diffuse = ibl_diffuse * (1 - probe_diff.w) + probe_diff.xyz;
            // ibl_specular = ibl_specular * (1 - probe_spec.w) + probe_spec.xyz;

        #endif

        vec3 env_brdf = get_brdf_from_lut(PrefilteredBRDF, NxV, sqrt(roughness), m.specular_ior);

        // Diffuse and fresnel ambient term
        vec3 diffuse_ambient = ibl_diffuse * m.basecolor * (1-m.metallic);
        vec3 fresnel = env_brdf.ggg;
        diffuse_ambient *= env_brdf.r;

        vec3 metallic_fresnel = get_metallic_fresnel_approx(m, NxV);

        // Mix between normal and metallic fresnel
        fresnel = mix(fresnel, metallic_fresnel, m.metallic);

        // XXX: Clearcoat not supported here
        vec3 specular_ambient = fresnel * ibl_specular;

        AmbientResult ret;
        ret.specular = specular_ambient;
        ret.diffuse = diffuse_ambient;
        ret.fresnel = env_brdf.g;
        return ret;
    }

#else

    // cheap forward shaded ambient
    vec3 get_forward_ambient(Material m) {
        vec3 shading_result = vec3(0);

        #if HAVE_PLUGIN(scattering)
            vec3 diff_env = textureLod(ScatteringIBLDiffuse, m.normal, 0).rgb;
        #else
            int ibl_diffuse_mip = get_mipmap_count(DefaultEnvmap) - 5;
            vec3 diff_env = textureLod(DefaultEnvmap, m.normal, ibl_diffuse_mip).rgb *
                DEFAULT_ENVMAP_BRIGHTNESS;
        #endif

        #if HAVE_PLUGIN(env_probes)
            const float sky_ambient_factor = TimeOfDay.env_probes.sky_ambient_scale;
            const float ambient_factor = TimeOfDay.env_probes.ambient_scale;    
        #else
            const float sky_ambient_factor = 0.05;
            const float ambient_factor = 0.1;
        #endif

        #if HAVE_PLUGIN(sky_ao)
            float sky_ao = compute_sky_ao(m.position, m.normal);
        #else
            float sky_ao = 1.0;
        #endif
        shading_result += (sky_ambient_factor + ambient_factor * sqrt(m.basecolor)) * sky_ao * diff_env;



        // Emission
        if (m.shading_model == SHADING_MODEL_EMISSIVE) {
            shading_result = m.basecolor * (0.005 + diff_env) * 30.0;
        }

        return shading_result;
    }

#endif


float get_sun_shadow_factor(vec3 position, vec3 normal) {
    #if !HAVE_PLUGIN(scattering) || !HAVE_PLUGIN(pssm)
        return 1.0;
    #else
        vec3 sun_vector = get_sun_vector();

        vec3 biased_position = position;

        // XXX: make this configurable
        const float slope_bias = 0.3 * 0.02;
        const float normal_bias = 0.0 * 0.005;
        const float fixed_bias = 0.5 * 0.001;
        vec3 biased_pos = get_biased_position(
            position, slope_bias, normal_bias, vOutput.normal, sun_vector);

        vec3 projected = project(PSSMSceneSunShadowMVP, biased_position);
        projected.z -= fixed_bias;

        // Fast shadow filtering
        float filter_radius = 1.0 / textureSize(PSSMSceneSunShadowMapPCF, 0).y;
        float shadow_term = 0;
        for(uint i = 0; i < 12; ++i) {
            vec3 offset = vec3(poisson_2D_12[i] * filter_radius, 0);
            shadow_term += textureLod(PSSMSceneSunShadowMapPCF, projected + offset, 0).x;
        }
        shadow_term /= 12.0;
        return shadow_term;
    #endif
}

#if VIEWSPACE_SHADING

    vec3 get_sun_shading(Material m, vec3 view_vector) {
        #if HAVE_PLUGIN(scattering)

        // Compute the sun lighting
        vec3 sun_vector = get_sun_vector();
        vec3 l = sun_vector;
        vec3 sun_color = get_sun_color() * get_sun_color_scale(sun_vector);

        {
            vec3 reflected_dir = reflect(view_vector, m.normal);
            const float sun_angular_radius = degree_to_radians(0.54);
            const float r = sin(sun_angular_radius); // Disk radius
            const float d = cos(sun_angular_radius); // Distance to disk

            // Closest point to a disk (since the radius is small, this is
            // a good approximation)
            float DdotR = dot(sun_vector, reflected_dir);
            vec3 S = reflected_dir - DdotR * sun_vector;
            l = DdotR < d ? normalize(d * sun_vector + normalize(S) * r) : reflected_dir;
        }

        // XXX: Need a shadow map. Maybe reuse pssm but thats super expensive
        // float shadow = get_sun_shadow_factor(m.position, m.normal);

        float shadow = 1.0;
        return apply_light(m, view_vector, l, sun_color, 1.0, shadow, vec3(1));


        #else
            return vec3(0);
        #endif
    }


#else

    // Applies the sun shading, and if the pssm plugin is activated, also the
    // sun shadows
    vec3 get_sun_shading(Material m) {
        #if HAVE_PLUGIN(scattering)
            float shadow = 0.0;
            vec3 shading_result = vec3(0);
            vec3 sun_vector = get_sun_vector();
            vec3 sun_color = get_sun_color() * get_sun_color_scale(sun_vector);

            if (sun_vector.z >= SUN_VECTOR_HORIZON) {
                float NxL = saturate(dot(sun_vector, m.normal));
                if (NxL > 0.0) {
                    shadow = NxL * get_sun_shadow_factor(m.position, m.normal);
                }
            }
            #if HAVE_PLUGIN(env_probes)
                const float factor = TimeOfDay.env_probes.sun_ambient_scale;
            #else
                const float factor = 1.0;
            #endif
            return shadow * sun_color * factor * m.basecolor;
        #else
            return vec3(0);
        #endif
    }
#endif



vec3 get_forward_light_shading(Material m) {

    vec3 shading_result = vec3(0);

    for (int i = 0; i <= maxLightIndex; ++i) {
        LightData light_data = read_light_data(AllLightsData, i);
        int light_type = get_light_type(light_data);

        // Skip Null-Lights
        if (light_type < 1) continue;

        vec3 light_pos = get_light_position(light_data);
        vec3 l = light_pos - m.position;
        float l_len = length(l);
        vec3 light_color = get_light_color(light_data);

        // Shade depending on light type
        switch(light_type) {
            case LT_POINT_LIGHT: {
                float radius = get_pointlight_radius(light_data);
                float inner_radius = get_pointlight_inner_radius(light_data);
                float att = attenuation_curve(dot(l, l), radius);
                float NxL = saturate(dot(m.normal, l) / l_len);
                shading_result += saturate(att) * NxL * (m.basecolor * light_color);
                break;
            }

            case LT_SPOT_LIGHT: {
                float radius = get_spotlight_radius(light_data);
                float fov = get_spotlight_fov(light_data);
                vec3 direction = get_spotlight_direction(light_data);

                float att = get_spotlight_attenuation(l / l_len, direction,
                    fov, radius, dot(l, l), -1);
                float NxL = saturate(dot(m.normal, l) / l_len);
                shading_result += saturate(att) * NxL * (m.basecolor * light_color);
                break;
            }

        }

    }

    return shading_result / M_PI;
}
