#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 0, binding = 0) uniform texture2D a_texture;
layout(set = 0, binding = 1) uniform sampler sampler_nlr;

layout (location = 0) in vec4 o_color;
layout (location = 0) out vec4 uFragColor;

void main() {
    uFragColor = texture(sampler2D(a_texture, sampler_nlr), gl_FragCoord.xy / vec2(800, 600));
}