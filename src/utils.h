#pragma once

#include <string>
#include <vector>
#include <optixu/optixu_vector_types.h>
#include <cmath>

void Load_Obj(const char* filename,
	std::vector<optix::float4> &vertices, 
	std::vector<optix::float3> &normals,
	std::vector<int> &elements);

struct MyVertex{
	optix::float3 vertex;  
	optix::float3 normal;  // normal at that vertex
	optix::float2 padding; // make it 32 byte
};

void LoadObj(const char* filename,
	std::vector<optix::float4> &vertices,
	std::vector<optix::float3> &normals,
	std::vector<int> &elements);


void ProcessObj(const char* filename, const char* output_file);

inline float Lrgb2srgb(float c){
	float result;
	
	c = pow(c, 1.0 / 2.2);
	/*if (c <= 0.0031308){
		result =  12.92 * c;
	}
	else{
		result =  1.055 * std::pow(c, 0.41666) - 0.055;
	}*/
	
	return c;
	// return result;
	// return c;
}