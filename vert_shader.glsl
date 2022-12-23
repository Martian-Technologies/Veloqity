#version 330
in vec4 position;
in vec4 normal;
uniform mat4 modelview;
uniform mat4 normaltransform;
uniform mat4 projection;
out vec4 world_position;
out vec4 camera_position;
out vec4 normal_out;

void main() {
    world_position = position;
    camera_position = modelview * position;
    gl_Position = projection * camera_position;
    gl_Position /= gl_Position.w;
    normal_out = normaltransform*normal;
}