#version 460 core

layout (location = 0) in vec3 vs_in_pos;
layout (location = 1) in vec3 vs_in_normal;

const uint COLOR_DISTANCE_TO_CENTER = 0;
const uint COLOR_BY_COORDS = 1;
const uint COLOR_BY_STATE = 2;

layout (binding = 0) uniform VertexShaderParams {
  mat4 projectionViewMatrix;
  mat4 viewMatrix;
  uint cellStates;
  uint coloring;
  uint worldSize;
} shaderParams;

#if defined(LIGHTING_ON)
layout (binding = 1) uniform LightingParams {
  vec3 lightAmbient;
  vec3 directionalLight;
  vec3 directionalLightColor;
} lightingParams;
#endif

layout (binding = 0, std430) readonly buffer PackedInstanceData {
  uint data[];
} packedCells;

layout (binding = 0) uniform sampler1DArray GradientTextures;

layout (location = 0) out gl_PerVertex {
  vec4 gl_Position;
};

layout (location = 1) out VS_OUT_FS_IN {
  vec4 color;
} vs_out;

uvec4 unpackVertexData(uint v) {
  return uvec4(v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF);
}

mat4 BuildTranslation(vec3 delta)
{
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(delta, 1.0));
}

void main() {
  const uvec4 vtx = unpackVertexData(packedCells.data[gl_BaseInstance + gl_InstanceID]);
  vec3 cubeCenter = vec3(vtx.xyz) - vec3(float(shaderParams.worldSize - 1) * 0.5);
  const mat4 model = BuildTranslation(cubeCenter);
  gl_Position = shaderParams.projectionViewMatrix * model * vec4(vs_in_pos, 1.0);

  vec4 color;
  if (shaderParams.coloring == COLOR_BY_STATE) {
    color = texture(GradientTextures, vec2(float(vtx.w) / float(shaderParams.cellStates), 0.0));
  } else if (shaderParams.coloring == COLOR_BY_COORDS) {
    color = vec4(vtx.xyz / vec3(float(shaderParams.worldSize)), 1.0);
  } else {
    color = vec4(normalize(vtx.xyz), 1.0);
  }

#if defined(LIGHTING_ON)
  // model matrix is just a translation so we can ignore it
  vec3 N = normalize(mat3(shaderParams.viewMatrix) * vs_in_normal);
  vec3 L = lightingParams.directionalLight;
  color = vec4((lightingParams.lightAmbient + max(dot(N, L), 0.0) * lightingParams.directionalLightColor) * color.xyz, 1.0);
#endif

  vs_out.color = color;
}
