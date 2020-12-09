// Based on code from https://github.com/KhronosGroup/glTF-Sample-Viewer

#version 300 es

precision mediump float;

#ifndef MAX_LIGHTS
#define MAX_LIGHTS 8
#endif

uniform struct p3d_MaterialParameters {
    vec4 baseColor;
    vec4 emission;
    float roughness;
    float metallic;
    float refractiveIndex;
} p3d_Material;

uniform struct p3d_LightSourceParameters {
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    #ifdef ENABLE_SHADOWS
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
    #endif
} p3d_LightSource[MAX_LIGHTS];

uniform struct p3d_LightModelParameters {
    vec4 ambient;
} p3d_LightModel;

#ifdef ENABLE_FOG
uniform struct p3d_FogParameters {
    vec4 color;
    float density;
} p3d_Fog;
#endif

uniform vec4 p3d_ColorScale;

struct FunctionParamters {
    float n_dot_l;
    float n_dot_v;
    float n_dot_h;
    float l_dot_h;
    float v_dot_h;
    float roughness;
    float metallic;
    vec3 reflection0;
    vec3 diffuse_color;
    vec3 specular_color;
};

// Give texture slots names
#define p3d_TextureBaseColor p3d_Texture0
#define p3d_TextureMetalRoughness p3d_Texture1
#define p3d_TextureNormal p3d_Texture2
#define p3d_TextureEmission p3d_Texture3

uniform sampler2D p3d_TextureBaseColor;
uniform sampler2D p3d_TextureMetalRoughness;
uniform sampler2D p3d_TextureNormal;
uniform sampler2D p3d_TextureEmission;

const vec3 F0 = vec3(0.04);
const float PI = 3.141592653589793;
const float SPOTSMOOTH = 0.001;
const float LIGHT_CUTOFF = 0.001;

in vec3 v_position;
in vec4 v_color;
in vec2 v_texcoord;
in mat3 v_tbn;
#ifdef ENABLE_SHADOWS
in vec4 v_shadow_pos[MAX_LIGHTS];
#endif

out vec4 FragColor;

// Schlick's Fresnel approximation with Spherical Gaussian approximation to replace the power
vec3 specular_reflection(FunctionParamters func_params) {
    vec3 f0 = func_params.reflection0;
    float v_dot_h= func_params.v_dot_h;
    return f0 + (1.0 - f0) * pow(2.0, (-5.55473 * v_dot_h - 6.98316) * v_dot_h);
}

// Smith GGX with optional fast sqrt approximation (see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg))
float visibility_occlusion(FunctionParamters func_params) {
    float r = func_params.roughness;
    float r2 = r * r;
    float n_dot_l = func_params.n_dot_l;
    float n_dot_v = func_params.n_dot_v;
    #ifdef SMITH_SQRT_APPROX
    float ggxv = n_dot_l * (n_dot_v * (1.0 - r) + r);
    float ggxl = n_dot_v * (n_dot_l * (1.0 - r) + r);
    #else
    float ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - r2) + r2);
    float ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - r2) + r2);
    #endif

    return max(0.0, 0.5 / (ggxv + ggxl));
}

// GGX/Trowbridge-Reitz
float microfacet_distribution(FunctionParamters func_params) {
    float roughness2 = func_params.roughness * func_params.roughness;
    float f = (func_params.n_dot_h * func_params.n_dot_h) * (roughness2 - 1.0) + 1.0;
    return roughness2 / (PI * f * f);
}

// Lambert
float diffuse_function(FunctionParamters func_params) {
    return 1.0 / PI;
}

void main() {
    vec4 metal_rough = texture(p3d_TextureMetalRoughness, v_texcoord);
    float metallic = clamp(p3d_Material.metallic * metal_rough.b, 0.0, 1.0);
    float perceptual_roughness = clamp(p3d_Material.roughness * metal_rough.g, 0.0, 1.0);
    float alpha_roughness = perceptual_roughness * perceptual_roughness;
    vec4 base_color = p3d_Material.baseColor * v_color * p3d_ColorScale * texture(p3d_TextureBaseColor, v_texcoord);
    vec3 diffuse_color = (base_color.rgb * (vec3(1.0) - F0)) * (1.0 - metallic);
    vec3 spec_color = mix(F0, base_color.rgb, metallic);
    #ifdef USE_NORMAL_MAP
    vec3 n = normalize(v_tbn * (2.0 * texture(p3d_TextureNormal, v_texcoord).rgb - 1.0));
    #else
    vec3 n = normalize(v_tbn[2]);
    #endif
    vec3 v = normalize(-v_position);

    #ifdef USE_OCCLUSION_MAP
    float ambient_occlusion = metal_rough.r;
    #else
    float ambient_occlusion = 1.0;
    #endif

    #ifdef USE_EMISSION_MAP
    vec3 emission = p3d_Material.emission.rgb * texture(p3d_TextureEmission, v_texcoord).rgb;
    #else
    vec3 emission = vec3(0.0);
    #endif

    vec4 color = vec4(vec3(0.0), base_color.a);

    for (int i = 0; i < p3d_LightSource.length(); ++i) {
        vec3 lightcol = p3d_LightSource[i].diffuse.rgb;

        if (dot(lightcol, lightcol) < LIGHT_CUTOFF) {
            continue;
        }

        vec3 l = normalize(p3d_LightSource[i].position.xyz - v_position * p3d_LightSource[i].position.w);
        vec3 h = normalize(l + v);
        float spotcos = dot(normalize(p3d_LightSource[i].spotDirection), -l);
        float spotcutoff = p3d_LightSource[i].spotCosCutoff;
        float shadowSpot = smoothstep(spotcutoff-SPOTSMOOTH, spotcutoff+SPOTSMOOTH, spotcos);
        #ifdef ENABLE_SHADOWS
        float shadowCaster = shadow2DProj(p3d_LightSource[i].shadowMap, v_shadow_pos[i]).r;
        #else
        float shadowCaster = 1.0;
        #endif
        float shadow = shadowSpot * shadowCaster;

        FunctionParamters func_params;
        func_params.n_dot_l = clamp(dot(n, l), 0.0, 1.0);
        func_params.n_dot_v = clamp(abs(dot(n, v)), 0.0, 1.0);
        func_params.n_dot_h = clamp(dot(n, h), 0.0, 1.0);
        func_params.l_dot_h = clamp(dot(l, h), 0.0, 1.0);
        func_params.v_dot_h = clamp(dot(v, h), 0.0, 1.0);
        func_params.roughness = alpha_roughness;
        func_params.metallic =  metallic;
        func_params.reflection0 = spec_color;
        func_params.diffuse_color = diffuse_color;
        func_params.specular_color = spec_color;

        vec3 F = specular_reflection(func_params);
        float V = visibility_occlusion(func_params);// V = G / (4 * n_dot_l * n_dot_v)
        float D = microfacet_distribution(func_params);

        vec3 diffuse_contrib = diffuse_color * diffuse_function(func_params);
        vec3 spec_contrib = vec3(F * V * D);
        color.rgb += func_params.n_dot_l * lightcol * (diffuse_contrib + spec_contrib) * shadow;
    }

    color.rgb += diffuse_color * p3d_LightModel.ambient.rgb * ambient_occlusion;
    color.rgb += emission;

    #ifdef ENABLE_FOG
    // Exponential fog
    float fog_distance = length(v_position);
    float fog_factor = clamp(1.0 / exp(fog_distance * p3d_Fog.density), 0.0, 1.0);
    color = mix(p3d_Fog.color, color, fog_factor);
    #endif

    FragColor = color;
}
