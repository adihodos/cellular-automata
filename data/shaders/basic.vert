#version 460 core

layout (location = 0) in vec3 vs_in_pos;

layout (location = 0) uniform vec4 Color;
layout (location = 1) uniform mat4 WorldViewProj;

layout (location = 0) out gl_PerVertex {
  vec4 gl_Position;
};

layout (location = 1) out VS_OUT_FS_IN {
  vec4 color;
} vs_out;

void main() {
  gl_Position = WorldViewProj * vec4(vs_in_pos, 1.0);
  vs_out.color = Color;
}
