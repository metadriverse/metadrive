#version 330

// This is the default terrain vertex shader. Most of the time you can just copy
// this and reuse it, and just modify the fragment shader.

in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;

uniform struct {
  sampler2D data_texture;
  sampler2D heightfield;
  int view_index;
  int terrain_size;
  int chunk_size;
} ShaderTerrainMesh;

uniform struct {
  vec4 position;
  vec3 color;
  vec3 attenuation;
  vec3 spotDirection;
  float spotCosCutoff;
  float spotExponent;
  sampler2DShadow shadowMap;
  mat4 shadowViewMatrix;
} p3d_LightSource[1];

out vec2 terrain_uv;
out vec3 vtx_pos;
out vec4 projecteds[1];

void main() {

  // Terrain data has the layout:
  // x: x-pos, y: y-pos, z: size, w: clod
  vec4 terrain_data = texelFetch(ShaderTerrainMesh.data_texture,
    ivec2(gl_InstanceID, ShaderTerrainMesh.view_index), 0);

  // Get initial chunk position in the (0, 0, 0), (1, 1, 0) range
  vec3 chunk_position = p3d_Vertex.xyz;

  // CLOD implementation
  float clod_factor = smoothstep(0, 1, terrain_data.w);
  chunk_position.xy -= clod_factor * fract(chunk_position.xy * ShaderTerrainMesh.chunk_size / 2.0)
                          * 2.0 / ShaderTerrainMesh.chunk_size;

  // Scale the chunk
  chunk_position *= terrain_data.z * float(ShaderTerrainMesh.chunk_size)
                    / float(ShaderTerrainMesh.terrain_size);
  chunk_position.z *= ShaderTerrainMesh.chunk_size;

  // Offset the chunk, it is important that this happens after the scale
  chunk_position.xy += terrain_data.xy / float(ShaderTerrainMesh.terrain_size);

  // Compute the terrain UV coordinates
  terrain_uv = chunk_position.xy;

  // Sample the heightfield and offset the terrain - we do not need to multiply
  // the height with anything since the terrain transform is included in the
  // model view projection matrix.
  chunk_position.z += texture(ShaderTerrainMesh.heightfield, terrain_uv).x;
  gl_Position = p3d_ModelViewProjectionMatrix * vec4(chunk_position, 1);

  // Output the vertex world space position - in this case we use this to render
  // the fog.
 // Lower the terrain on the borders - this ensures the shadow map is generated
  // correctly.
  if ( min(terrain_uv.x, terrain_uv.y) < 8.0 / ShaderTerrainMesh.terrain_size ||
     max(terrain_uv.x, terrain_uv.y) > 1 - 9.0 / ShaderTerrainMesh.terrain_size) {
     chunk_position.z = 0;
  }
  vtx_pos = (p3d_ModelMatrix * vec4(chunk_position, 1)).xyz;
  projecteds[0] = p3d_LightSource[0].shadowViewMatrix * (p3d_ModelViewMatrix * vec4(chunk_position, 1));
}
