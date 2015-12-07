#pragma once

#include <string>
#include <vector>
#include <optixu/optixu_vector_types.h>
#include <cmath>

struct MyVertex{
	optix::float3 vertex;  
	optix::float3 normal;  
	optix::float2 padding; 
}; 

#define TINYOBJLOADER_IMPLEMENTATION

void LoadObjFromFile(const char* file_name, std::vector<MyVertex>& vertices);