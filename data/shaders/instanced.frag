#version 460 core

layout (location = 0) out vec4 FinalFragColor;

layout (location = 1) in VS_OUT_FS_IN {
  vec4 color;
} fs_in;

void main() {
  FinalFragColor = fs_in.color;
}
