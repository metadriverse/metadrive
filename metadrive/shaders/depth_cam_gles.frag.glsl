#version 300 es

precision mediump float;

in float distanceToCamera;
in vec4 color;

out vec4 fragColor;

void main() {
  float base=8.0;
  float b = 16.0;
  float c = log(distanceToCamera/base)/log(b);
  fragColor = vec4(c, c, c, 0);
}