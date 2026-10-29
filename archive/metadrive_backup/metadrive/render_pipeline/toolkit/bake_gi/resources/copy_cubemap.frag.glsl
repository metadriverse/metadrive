#version 430

uniform sampler2D SourceTex;
uniform writeonly imageCube DestTex;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    int size = textureSize(SourceTex, 0).y;
    vec4 source_data = texelFetch(SourceTex, coord, 0);

    // Convert to local cubemap coordinate
    int offset = coord.x / size;
    coord.x = coord.x % size;

    imageStore(DestTex, ivec3(coord.x, coord.y, offset), source_data);
}
