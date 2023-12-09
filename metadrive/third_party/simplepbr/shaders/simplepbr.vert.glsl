#version 120

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
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[MAX_LIGHTS];
#endif

#ifdef ENABLE_SKINNING
uniform mat4 p3d_TransformTable[100];
#endif

uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_TextureMatrix;

attribute vec4 p3d_Vertex;
attribute vec4 p3d_Color;
attribute vec3 p3d_Normal;
attribute vec4 p3d_Tangent;
attribute vec2 p3d_MultiTexCoord0;
#ifdef ENABLE_SKINNING
attribute vec4 transform_weight;
attribute vec4 transform_index;
#endif


varying vec3 v_position;
varying vec4 v_color;
varying mat3 v_tbn;
varying vec2 v_texcoord;
#ifdef ENABLE_SHADOWS
varying vec4 v_shadow_pos[MAX_LIGHTS];
#endif

void main() {
#ifdef ENABLE_SKINNING
    mat4 skin_matrix = (
        p3d_TransformTable[int(transform_index.x)] * transform_weight.x +
        p3d_TransformTable[int(transform_index.y)] * transform_weight.y +
        p3d_TransformTable[int(transform_index.z)] * transform_weight.z +
        p3d_TransformTable[int(transform_index.w)] * transform_weight.w
    );
    vec4 vert_pos4 = p3d_ModelViewMatrix * skin_matrix * p3d_Vertex;
    vec3 normal = normalize(p3d_NormalMatrix * (skin_matrix * vec4(p3d_Normal.xyz, 0.0)).xyz);
#else
    vec4 vert_pos4 = p3d_ModelViewMatrix * p3d_Vertex;
    vec3 normal = normalize(p3d_NormalMatrix * p3d_Normal);
#endif
    v_position = vec3(vert_pos4);
    v_color = p3d_Color;
    v_texcoord = (p3d_TextureMatrix * vec4(p3d_MultiTexCoord0, 0, 1)).xy;
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

    gl_Position = p3d_ProjectionMatrix * vert_pos4;
}
