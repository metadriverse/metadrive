#version 330

#ifndef MAX_LIGHTS
    #define MAX_LIGHTS 8
#endif

#ifdef ENABLE_SKINNING
uniform mat4 p3d_TransformTable[100];
#endif

uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_TextureMatrix;

in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec3 p3d_Normal;
in vec4 p3d_Tangent;
in vec2 p3d_MultiTexCoord0;
#ifdef ENABLE_SKINNING
in vec4 transform_weight;
in vec4 transform_index;
#endif


out vec3 v_position;
out vec4 v_color;
out mat3 v_tbn;
out vec2 v_texcoord;
out vec4 shadow_vtx_pos;

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
    shadow_vtx_pos = p3d_ModelMatrix * p3d_Vertex.xyzw;
    v_position = vec3(vert_pos4);
    v_color = p3d_Color;
    v_texcoord = (p3d_TextureMatrix * vec4(p3d_MultiTexCoord0, 0, 1)).xy;

    vec3 tangent = normalize(vec3(p3d_ModelViewMatrix * vec4(p3d_Tangent.xyz, 0.0)));
    vec3 bitangent = cross(normal, tangent) * p3d_Tangent.w;
    v_tbn = mat3(
        tangent,
        bitangent,
        normal
    );

    gl_Position = p3d_ProjectionMatrix * vert_pos4;
}
