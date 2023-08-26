#version 430
#extension GL_EXT_shader_explicit_arithmetic_types: require

layout(set = 0, binding = 0) buffer Lhs{
    TYPE lhs[];
};
layout(set = 0, binding = 0) buffer Rhs{
    TYPE rhs[];
};
layout(set = 0, binding = 0) buffer Dst{
    TYPE dst[];
};

void main(){
    const ivec3 pos = ivec3(gl_GlobalInvocationID);
    
    dst[pos.x] = lhs[pos.x] + rhs[pos.x];
}
