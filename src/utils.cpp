#include "utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
using namespace optix;
using namespace std;

void Load_Obj(const char* filename, 
	std::vector<float4> &vertices, 
	std::vector<float3> &normals, 
	std::vector<int> &elements){

	printf("Load obj from file %s\n", filename);
	ifstream in(filename, ios::in);
    if (!in)
    {
        cerr << "Cannot open " << filename << endl; exit(1);
    }

    string line;
    while (getline(in, line))
    {
        if (line.substr(0,2) == "v ")
        {
            istringstream s(line.substr(2));
			float4 v; 
			s >> v.x; s >> v.y; s >> v.z; v.w = 1.0f;
            vertices.push_back(v);
        }
        else if (line.substr(0,2) == "f ")
        {
            istringstream s(line.substr(2));
			int a, b, c;
            s >> a; s >> b; s >> c;
            a--; b--; c--;
            elements.push_back(a); elements.push_back(b); elements.push_back(c);
        }
        else if (line[0] == '#')
        {
            /* ignoring this line */
        }
        else
        {
            /* ignoring this line */
        }
    }

    int vertex_num = vertices.size();
    normals.resize(vertex_num, make_float3(0.0, 0.0, 0.0));

    for (int i = 0; i < elements.size(); i+=3)
    {
		int ia = elements[i];
		int ib = elements[i + 1];
		int ic = elements[i + 2];
		float3 t1 = make_float3(vertices[ib] - vertices[ia]);
		float3 t2 = make_float3(vertices[ic] - vertices[ia]);
        float3 normal = cross(t1, t2); 

        normals[ia] += normal;
        normals[ib] += normal;
        normals[ic] += normal;
    }

    for(int i = 0;i < vertex_num;i++){
        normals[i] = normalize(normals[i]);
    }
	printf("loaded...\n");
}


void LoadObj(const char* filename,
	std::vector<float4> &vertices,
	std::vector<float3> &normals,
	std::vector<int> &elements){

	printf("Load obj from file %s\n", filename);
	FILE * file = fopen(filename, "r");
	if (!file)
	{
		cerr << "Cannot open " << filename << endl; exit(1);
	}

	char lineHeader[128];
	vector<int> vertex_normal_index;
	while (true)
	{
		int res = fscanf(file, "%s", lineHeader);

		if (res == EOF){
			break;
		}

		if (strcmp(lineHeader, "v") == 0)
		{
			float4 v;
			fscanf(file, "%f %f %f\n", &v.x, &v.y, &v.z);
			v.w = 1.0f;
			vertices.push_back(v);
		}
		else if (strcmp(lineHeader, "f") == 0)
		{
			int a, b, c, d, e, f;
			int matches = fscanf(file, "%d//%d %d//%d %d//%d\n", &a, &b, &c, &d, &e, &f);
			a--; b--; c--; d--; e--; f--;
			elements.push_back(a); vertex_normal_index.push_back(b);
			elements.push_back(c); vertex_normal_index.push_back(d);
			elements.push_back(e); vertex_normal_index.push_back(f);
		}

		else if (strcmp(lineHeader, "vn") == 0)
		{
			float3 v;
			fscanf(file, "%f %f %f\n", &v.x, &v.y, &v.z);
			normals.push_back(v);
		}
		else
		{
			/* ignoring this line */
		}
	}

	/*int vertex_num = vertices.size();
	normals.resize(vertex_num, make_float3(0.0, 0.0, 0.0));
	for (int i = 0; i < elements.size(); i++){
		int normal_index = vertex_normal_index[i];
		int vertex_index = elements[i];
		normals[vertex_index] = normalize(normals[normal_index]);
	}*/

	int vertex_num = vertices.size();
	normals.resize(vertex_num, make_float3(0.0, 0.0, 0.0));

	for (int i = 0; i < elements.size(); i += 3)
	{
		int ia = elements[i];
		int ib = elements[i + 1];
		int ic = elements[i + 2];
		float3 t1 = make_float3(vertices[ib] - vertices[ia]);
		float3 t2 = make_float3(vertices[ic] - vertices[ia]);
		float3 normal = cross(t1, t2);

		normals[ia] += normal;
		normals[ib] += normal;
		normals[ic] += normal;
	}

	for (int i = 0; i < vertex_num; i++){
		normals[i] = normalize(normals[i]);
	}
	printf("loaded...\n");

}


void ProcessObj(const char* filename, const char* output_file){
	vector<float4> vertex;
	vector<int> elements;
	vector<float3> normals;
	Load_Obj(filename, vertex, normals, elements);

	FILE* file = fopen(output_file, "w");
	int a = 1;
	for (int i = 0; i < elements.size(); i = i + 3){
		fprintf(file, "f %d %d %d\n", a, a+1, a+2);
		a = a + 3;
		float4 v = vertex[elements[i]];
		fprintf(file, "v %f %f %f\n", v.x, v.y, v.z);
		v = vertex[elements[i+1]];
		fprintf(file, "v %f %f %f\n", v.x, v.y, v.z);
		v = vertex[elements[i+2]];
		fprintf(file, "v %f %f %f\n", v.x, v.y, v.z);
	}
	fclose(file);
}

