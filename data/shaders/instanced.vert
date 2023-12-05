#version 460 core

layout (location = 0) in vec3 vs_in_pos;

const uint COLOR_DISTANCE_TO_CENTER = 0;
const uint COLOR_BY_COORDS = 1;
const uint COLOR_BY_STATE = 2;

layout (binding = 0) uniform VertexShaderParams {
  mat4 projectionViewMatrix;
  uint cellStates;
  uint coloring;
  uint worldSize;
} shaderParams;

#if defined(LIGHTING_ON)
layout (binding = 1) uniform LightingParams {
    mat4 normalsMatrix;
    vec3 directionalLights[4];
    vec3 lightAmbient;
  } lightingParams;
#endif

struct CubeInstance {
  mat4 wvp;
  vec3 center;
  uint state;
};

layout (binding = 0, std430) readonly buffer InstancesData {
  CubeInstance instances[];
} cells;

layout (binding = 0) uniform sampler1DArray GradientTextures;

layout (location = 0) out gl_PerVertex {
  vec4 gl_Position;
};

layout (location = 1) out VS_OUT_FS_IN {
  vec4 color;
} vs_out;

void main() {
  gl_Position = shaderParams.projectionViewMatrix * cells.instances[gl_BaseInstance + gl_InstanceID].wvp * vec4(vs_in_pos, 1.0);

  vec4 color;
  if (shaderParams.coloring == COLOR_BY_STATE) {
    color = texture(GradientTextures, vec2(float(cells.instances[gl_BaseInstance + gl_InstanceID].state) / shaderParams.cellStates, 0.0));
  } else if (shaderParams.coloring == COLOR_BY_COORDS) {
    const vec3 cellCoords = cells.instances[gl_BaseInstance + gl_InstanceID].center;
    color = vec4(cellCoords / vec3(float(shaderParams.worldSize)) + vec3(0.5), 1.0);
  } else {
    const vec3 cellCoords = cells.instances[gl_BaseInstance + gl_InstanceID].center;
    color = vec4(normalize(cellCoords), 1.0);
  }
  vs_out.color = color;
}
