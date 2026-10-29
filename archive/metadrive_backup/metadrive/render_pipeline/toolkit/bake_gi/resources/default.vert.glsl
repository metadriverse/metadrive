#version 430
in vec4 p3d_Vertex;

void main() {
    gl_Position = vec4(p3d_Vertex.xz, 0, 1);
}
