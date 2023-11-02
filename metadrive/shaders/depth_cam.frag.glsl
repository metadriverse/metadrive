#version 150

in float distanceToCamera;
in vec4 color;

out vec4 fragColor;

void main() {
  float base=5;
  // distance=32
  float b = 16;
  float c = log(distanceToCamera/base)/log(b);
  fragColor = vec4(c, c, c, 0);
}