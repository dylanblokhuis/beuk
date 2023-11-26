#version 460

layout(location = 0) out vec2 out_tex;

void main() {
    vec2 pos[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));

    out_tex = (pos[gl_VertexIndex] + 1.0) * 0.5;
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
}
