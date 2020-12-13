#version 120

// This is just a simple vertex shader transforming the skybox

attribute vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;

varying vec3 skybox_pos;

void main() {
  skybox_pos = p3d_Vertex.xyz;
  gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
