#pragma once

#include <string>
#include <vector>
#include <optixu/optixu_vector_types.h>
#include <cmath>

struct MyVertex{
	optix::float3 vertex;  
	optix::float3 normal;  // normal at that vertex
	optix::float3 kd;
	optix::float3 material_id;
	optix::float3 texture_uv;
	float padding; // make it 64 byte
}; 


