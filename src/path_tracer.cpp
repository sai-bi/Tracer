#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "random.h"
#include "path_tracer.h"
#include "helpers.h"
#include <fstream>
#include "PathTracerScene.h"
#include "ObjLoader.h"

using namespace optix;
using namespace std;


void PathTracerScene::initContext(){
	m_context->setRayTypeCount(3);
	m_context->setEntryPointCount(1);
	m_context->setStackSize(68000);

	m_context["scene_epsilon"]->setFloat(1.e-3f);
	m_context["pathtrace_ray_type"]->setUint(0u);
	m_context["pathtrace_shadow_ray_type"]->setUint(1u);
	m_context["pathtrace_bsdf_shadow_ray_type"]->setUint(2u);
	m_context["rr_begin_depth"]->setUint(m_rr_begin_depth);
	m_context["sqrt_sample_num"]->setUint(m_sqrt_num_samples);

	m_context["eye"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["U"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["V"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["W"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));

	m_context["bad_color"]->setFloat(1.0f, 0.0f, 0.0f);
	m_context["bg_color"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
}

void PathTracerScene::initScene(InitialCameraData& camera_data){

	LoadGeometry();

	// Setup output buffer
	Variable output_buffer = m_context["output_buffer"];
	Buffer buffer = createOutputBuffer( RT_FORMAT_FLOAT4, m_width, m_height );
	output_buffer->set(buffer);


	// Setup programs
	std::string ptx_path = ptxpath( "path_tracer", "path_tracer.cu" );
	Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "DirectRender");

	m_context->setRayGenerationProgram( 0, ray_gen_program );
	Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
	m_context->setExceptionProgram( 0, exception_program );
  
	// load env map
	const float3 default_color = make_float3(0.8f, 0.8f, 0.8f);
	m_context["envmap"]->setTextureSampler(loadTexture(m_context, m_env_path, default_color));
	m_context->setMissProgram(0, m_context->createProgramFromPTXFile(ptx_path, "envmap_miss"));

	m_context["frame_number"]->setUint(1);

	// Index of sampling_stategy (BSDF, light, MIS)
	m_sampling_strategy = 0;
	m_context["sampling_stategy"]->setInt(m_sampling_strategy);

	// Create scene geometry
	createGeometry();


	float max_dim = m_aabb.maxExtent();
	float3 eye = m_aabb.center();
	eye.z += 2.0f * max_dim;

	camera_data = InitialCameraData(eye,                           // eye
									m_aabb.center(),               // lookat
									make_float3(0.0f, 1.0f, 0.0f), // up
									30.0f);

	// Finalize
	m_context->validate();
	m_context->compile();
}

bool PathTracerScene::keyPressed( unsigned char key, int x, int y )
{
	return false;
}

void PathTracerScene::trace( const RayGenCameraData& camera_data )
{
	m_context["eye"]->setFloat( camera_data.eye );
	m_context["U"]->setFloat( camera_data.U );
	m_context["V"]->setFloat( camera_data.V );
	m_context["W"]->setFloat( camera_data.W );

	Buffer buffer = m_context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize( buffer_width, buffer_height );

	printf("%d %d", buffer_width, buffer_height);

	bool camera_changed = m_camera_changed;
	if( m_camera_changed ) {
		m_camera_changed = false;
		m_frame = 1;
	}

	m_context["frame_number"]->setUint( m_frame++ );

	m_context->launch( 0,
					static_cast<unsigned int>(buffer_width),
					static_cast<unsigned int>(buffer_height)
					);
	SaveFrame("bunny_with_base_direct.txt");
	exit(-1);
}

//-----------------------------------------------------------------------------

Buffer PathTracerScene::getOutputBuffer()
{
	return m_context["output_buffer"]->getBuffer();
}


void PathTracerScene::LoadGeometry(){
	printf("Load geometry...\n");
	vector<MyVertex> all_vertices;
	LoadObjFromFile(m_filename.c_str(), all_vertices);
	
	int vertex_num = all_vertices.size();

	Buffer vertex_buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	vertex_buffer->setFormat(RT_FORMAT_USER);
	vertex_buffer->setElementSize(sizeof(MyVertex));
	vertex_buffer->setSize(vertex_num);
	memcpy(vertex_buffer->map(), &all_vertices[0], vertex_num * sizeof(MyVertex));
	vertex_buffer->unmap();
	m_context["vertices"]->set(vertex_buffer);

	m_width = vertex_num;
	m_height = 1;

	printf("Load completed... \n");
}


void PathTracerScene::setMaterial( GeometryInstance& gi,
                                   Material material,
                                   const std::string& color_name,
                                   const float3& color){
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}

void PathTracerScene::createGeometry(){
    // Set up material
	Material diffuse = m_context->createMaterial();
	Program diffuse_ch = m_context->createProgramFromPTXFile(ptxpath("path_tracer", "path_tracer.cu"), "diffuse");
	Program diffuse_ah = m_context->createProgramFromPTXFile(ptxpath("path_tracer", "path_tracer.cu"), "shadow");
	diffuse->setClosestHitProgram(0, diffuse_ch);
	diffuse->setAnyHitProgram(1, diffuse_ah);
	diffuse["Kd_map"]->setTextureSampler(loadTexture(m_context, "", make_float3(1.f, 1.f, 1.f)));

	vector<GeometryInstance> gis;
	GeometryGroup geom_group = m_context->createGeometryGroup(gis.begin(), gis.end());
	geom_group->setAcceleration(m_context->createAcceleration("Lbvh", "Bvh"));
	m_context["top_object"]->set(geom_group);
	m_context["top_shadower"]->set(geom_group);

	ObjLoader* loader = new ObjLoader(m_filename.c_str(), m_context, geom_group, diffuse, true);
	m_aabb = loader->getSceneBBox();
}


void PathTracerScene::SaveFrame(const char* filename){
	RTbuffer buffer = m_context["output_buffer"]->getBuffer()->get();	
	void* imageData;
	rtBufferMap(buffer, &imageData);

	FILE * pFile;
	pFile = fopen(filename, "w");

	if (pFile == NULL){
		printf("cannot open file\n");
		exit(-1);
	}

	for (int j = m_height - 1; j >= 0; --j) {
		float* src_1 = ((float*)imageData) + (4 * m_width*j);

		for (int i = 0; i < m_width; i++) {
			fprintf(pFile, "%c ", 'c');
			for (int elem = 0; elem < 3; ++elem) {
				float c = *src_1;
				fprintf(pFile, "%f ", c);
				src_1++;
			}
			src_1++;
		}
	}
	rtBufferUnmap(buffer);
	fclose(pFile);
	exit(-1);
}


