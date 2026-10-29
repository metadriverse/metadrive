#version 400

uniform mat4 p3d_ViewProjectionMatrix;
uniform mat4 trans_model_to_world;
in vec4 p3d_Vertex;
out vec2 texcoord;

uniform sampler2D displacement;
uniform sampler2D normal;


void main() {

    int idxX = gl_InstanceID % 16;
    int idxY = gl_InstanceID / 16;

    texcoord = p3d_Vertex.xy / 4.0  + 0.5;

    vec3 displace = texture(displacement, texcoord).xyz;


    vec4 worldPos = trans_model_to_world * p3d_Vertex;
    worldPos.xy += vec2(idxX-8, idxY-8) * 8.0;
    worldPos.xyz += displace * 0.5;

    gl_Position = p3d_ViewProjectionMatrix * worldPos;
}