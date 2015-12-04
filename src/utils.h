#pragma once

#include <string>
#include <vector>
#include <optixu/optixu_vector_types.h>
#include <cmath>

struct MyVertex{
	optix::float3 vertex;  
	optix::float3 normal;  
	optix::float2 padding; 
	MyVertex(float3 v = optix::make_float3(0.0f),
		float3 n = optix::make_float3(0.0f),
		float2 p = optix::make_float2(0.0f))
		: vertex(v),
		normal(n),
		padding(p){}
}; 


void LoadObjFromFile(const char* file_name, std::vector<MyVertex>& vertices);