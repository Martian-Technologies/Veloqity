#version 330

in vec4 world_position;
in vec4 camera_position;
in vec4 normal_out;
out vec4 color;

void main() {
    if (camera_position.z > -2 || camera_position.z < -100) {
        discard;
    }
    vec3 eye_vec = normalize(camera_position.xyz);
    vec3 light_vec = normalize(camera_position.xyz-vec3(20, 50, 50));
    float spec = dot(eye_vec, light_vec);
    float diffuse = max(0, -dot(normal_out.xyz, light_vec));
    float ambient = 1.0;
    float light1 = diffuse*0.5+ambient*0.3;
    
    light_vec = normalize(camera_position.xyz-vec3(-20, 50, 50));
    spec = dot(eye_vec, light_vec);
    diffuse = max(0, -dot(normal_out.xyz, light_vec));
    ambient = 1.0;
    float light2 = diffuse*0.5+ambient*0.3;

    color = vec4(light2, light1*0.3+light2*0.3, light1, 1);
}