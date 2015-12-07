#include "utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "tiny_obj_loader.h"
using namespace optix;
using namespace std;
using namespace tinyobj;


void LoadObjFromFile(const char* file_name, vector<MyVertex>& vertices){
	vector<shape_t> shapes;
	vector<material_t> materials;
	
	string err = " ";
	bool ret = tinyobj::LoadObj(shapes, materials, err, file_name);
	if (!err.empty()){
		cerr << err << endl;
	}
	if (!ret){
		exit(-1);
	}

	printf("Number of shapes: %d\n", shapes.size());
	printf("Number of materials: %d\n", materials.size());

	// get vertex info
	for (size_t i = 0; i < shapes.size(); i++){
		const mesh_t* mesh = &shapes[i].mesh;
		
		vector<MyVertex> vertex_list;
		for (size_t v = 0; v < mesh->positions.size() / 3; v++){
			MyVertex temp;
			temp.vertex = make_float3(mesh->positions[3 * v + 0],
				mesh->positions[3 * v + 1],
				mesh->positions[3 * v + 2]);
			temp.normal = make_float3(0.0f);
			vertex_list.push_back(temp);
		}

		for (size_t f = 0; f < mesh->indices.size() / 3; f++){
			unsigned int a = mesh->indices[3 * f + 0];
			unsigned int b = mesh->indices[3 * f + 1];
			unsigned int c = mesh->indices[3 * f + 2];
			float3 v_a = vertex_list[a].vertex;
			float3 v_b = vertex_list[b].vertex;
			float3 v_c = vertex_list[c].vertex;
			
			float3 n = optix::cross(v_b - v_a, v_c - v_a);
			vertex_list[a].normal += n;
			vertex_list[b].normal += n;
			vertex_list[c].normal += n;
		}
		
		for (size_t v = 0; v < vertex_list.size(); v++){
			vertex_list[v].normal = optix::normalize(vertex_list[v].normal);
			vertices.push_back(vertex_list[v]);
		}
	}
	printf("Finish loading obj...\n");
}


