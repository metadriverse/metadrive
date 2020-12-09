#version 300 es

precision mediump float;

#ifndef MAX_LIGHTS
#define MAX_LIGHTS 8
#endif

#ifdef ENABLE_SHADOWS
uniform struct p3d_LightSourceParameters {
    vec4 position;
    vec4 diffuse;
    vec4 specular;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[MAX_LIGHTS];
#endif

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;

in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec3 p3d_Normal;
in vec4 p3d_Tangent;
in vec2 p3d_MultiTexCoord0;


out vec3 v_position;
out vec4 v_color;
out mat3 v_tbn;
out vec2 v_texcoord;
#ifdef ENABLE_SHADOWS
out vec4 v_shadow_pos[MAX_LIGHTS];
#endif

void main() {
    vec4 vert_pos4 = p3d_ModelViewMatrix * p3d_Vertex;
    v_position = vec3(vert_pos4);
    v_color = p3d_Color;
    vec3 normal = normalize(p3d_NormalMatrix * p3d_Normal);
    v_texcoord = p3d_MultiTexCoord0;
    #ifdef ENABLE_SHADOWS
    for (int i = 0; i < p3d_LightSource.length(); ++i) {
        v_shadow_pos[i] = p3d_LightSource[i].shadowViewMatrix * vert_pos4;
    }
        #endif

    vec3 tangent = normalize(vec3(p3d_ModelViewMatrix * vec4(p3d_Tangent.xyz, 0.0)));
    vec3 bitangent = cross(normal, tangent) * p3d_Tangent.w;
    v_tbn = mat3(
    tangent,
    bitangent,
    normal
    );

    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
