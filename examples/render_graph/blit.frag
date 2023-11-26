#version 460

layout(set = 0, binding = 0) uniform sampler2D src_texture;
layout(location = 0) in vec2 in_tex;
layout(location = 0) out vec4 out_frag;

void main() {
    out_frag = texture(src_texture, in_tex);
}
