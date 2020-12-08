#version 150

// Uniform inputs
uniform mat4 p3d_ProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform vec4 p3d_Color;

// Vertex inputs
in vec4 p3d_Vertex;

// Vertex outputs
out float distanceToCamera;

void main() {
  vec4 cs_position = p3d_ModelViewMatrix * p3d_Vertex;
  distanceToCamera = length(cs_position.xyz);
  gl_Position = p3d_ProjectionMatrix * cs_position;
}