#version 430

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

uniform mat4 p3d_ViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 tpose_world_to_model;

out vec3 ws_position;
out vec3 ws_normal;
out vec2 ms_uv;

void main() {
    ms_uv = p3d_MultiTexCoord0;
    ws_normal = normalize(tpose_world_to_model * p3d_Normal).xyz;
    ws_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
    gl_Position = p3d_ViewProjectionMatrix * vec4(ws_position, 1);
}
