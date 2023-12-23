#version 430

// Set the number of invocations in the work group.
// In this case, we operate on the image in 16x16 pixel tiles.
layout (local_size_x = 16, local_size_y = 16) in;

// Declare the texture inputs
uniform sampler2D fromTex;
uniform writeonly image2D toTex;

void main() {
  // Acquire the coordinates to the texel we are to process.
  ivec2 texelCoords = ivec2(gl_GlobalInvocationID.xy);

  // The normalization is very important!
  vec4 depthSample = texture2D(fromTex, texelCoords/vec2(200, 100));

  // Now write the modified pixel to the second texture.
  imageStore(toTex, texelCoords, depthSample);
}