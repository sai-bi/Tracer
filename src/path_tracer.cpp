
/*
 * Copyright (c) 2008 - 2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

//------------------------------------------------------------------------------
//
// path_tracer.cpp: render cornell box using path tracing.
//
//------------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <GLUTDisplay.h>
#include <PPMLoader.h>
#include <OptiXMesh.h>
#include <ImageLoader.h>
#include "utils.h"
// #include <sampleConfig.h>

#include "random.h"
#include "path_tracer.h"
#include "helpers.h"

#include "sampleConfig.h"
#include <ImageDisplay.h>
#include <fstream>

using namespace optix;
using namespace std;

//-----------------------------------------------------------------------------
//
// PathTracerScene
//
//-----------------------------------------------------------------------------


class PathTracerScene: public SampleScene
{
public:
  // Set the actual render parameters below in main().
  PathTracerScene()
  : m_rr_begin_depth(1u)
  , m_sqrt_num_samples( 0u )
  , m_width(512u)
  , m_height(512u)
  {}

  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();

  void   setNumSamples( unsigned int sns )                           { m_sqrt_num_samples= sns; }
  void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }
  void   setFileName(std::string file_name){
    this->m_filename = file_name;
  }
  void setEnvmapPath(std::string path){
    this->m_env_path = path;
  }

  void LoadGeometry();

  void SaveFrame(const char* file_name);

private:
  // Should return true if key was handled, false otherwise.
  virtual bool keyPressed(unsigned char key, int x, int y);
  void createGeometry();

  GeometryInstance createParallelogram( const float3& anchor,
                                        const float3& offset1,
                                        const float3& offset2);

  GeometryInstance createLightParallelogram( const float3& anchor,
                                             const float3& offset1,
                                             const float3& offset2,
                                             int lgt_instance = -1);
  void setMaterial( GeometryInstance& gi,
                    Material material,
                    const std::string& color_name,
                    const float3& color);


  Program        m_pgram_bounding_box;
  Program        m_pgram_intersection;

  unsigned int   m_rr_begin_depth;
  unsigned int   m_sqrt_num_samples;
  unsigned int   m_width;
  unsigned int   m_height;
  unsigned int   m_frame;
  unsigned int   m_sampling_strategy;
  std::string    m_filename;
  std::string  m_env_path;
  Aabb m_aabb;
};


// void PathTracerScene::initScene( InitialCameraData& camera_data )
void PathTracerScene::initScene(InitialCameraData& camera_data)
{
  printf("here ok 1\n");
  m_context->setRayTypeCount( 3 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 68000 );

  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["pathtrace_ray_type"]->setUint(0u);
  m_context["pathtrace_shadow_ray_type"]->setUint(1u);
  m_context["pathtrace_bsdf_shadow_ray_type"]->setUint(2u);
  m_context["rr_begin_depth"]->setUint(m_rr_begin_depth);
  m_context["samples_per_vertex"]->setUint(1000u);

  LoadGeometry();

  // Setup output buffer
  Variable output_buffer = m_context["output_buffer"];
  Buffer buffer = createOutputBuffer( RT_FORMAT_FLOAT4, m_width, m_height );
  //Buffer buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_width, m_height);
  output_buffer->set(buffer);
  
  Variable output_buffer_1 = m_context["output_buffer_1"];
  Buffer buffer_1 = createOutputBuffer(RT_FORMAT_FLOAT4, m_width, m_height);
  //Buffer buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_width, m_height);
  output_buffer_1->set(buffer_1);

  Variable output_buffer_2 = m_context["output_buffer_2"];
  Buffer buffer_2 = createOutputBuffer(RT_FORMAT_FLOAT4, m_width, m_height);
  //Buffer buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_width, m_height);
  output_buffer_2->set(buffer_2);

  Variable output_buffer_3 = m_context["output_buffer_3"];
  Buffer buffer_3 = createOutputBuffer(RT_FORMAT_FLOAT4, m_width, m_height);
  //Buffer buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_width, m_height);
  output_buffer_3->set(buffer_3);
  
  
  
  // Set up camera
  //camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -800.0f ), // eye
  //                                 make_float3( 278.0f, 273.0f, 0.0f ),    // lookat
  //                                 make_float3( 0.0f, 1.0f,  0.0f ),       // up
  //                                 35.0f );                                // vfov

  // Declare these so validation will pass
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  m_context["sqrt_num_samples"]->setUint( m_sqrt_num_samples );
  m_context["bad_color"]->setFloat( 1.0f, 0.0f, 0.0f );
  m_context["bg_color"]->setFloat( make_float3(0.0f, 0.0f, 0.0f) );

  // Setup programs
  std::string ptx_path = ptxpath( "path_tracer", "path_tracer.cu" );
  // Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pathtrace_camera" );
  // Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "DirectRender");
  Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "DirectRender");

  m_context->setRayGenerationProgram( 0, ray_gen_program );
  Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  
  // miss program
  // m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
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

  // eye = make_float3(0, 0, 20);
  camera_data = InitialCameraData(eye,                             // eye
          m_aabb.center(),                         // lookat
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
	std::vector<float4> vertices;
	vector<float3> normals;
	vector<int> elements;
  
	LoadObj(m_filename.c_str(), vertices, normals, elements);

	// create vertex buffer
	int vertex_num = vertices.size();
	MyVertex* mesh_vertex = new MyVertex[vertex_num];
	for (int i = 0; i < vertex_num; i++){
		MyVertex temp;
		temp.vertex = make_float3(vertices[i]);
		temp.normal = normals[i];
		temp.padding = make_float2(0.0f,0.0f);
		mesh_vertex[i] = temp;
	}

	Buffer vertex_buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	vertex_buffer->setFormat(RT_FORMAT_USER);
	vertex_buffer->setElementSize(sizeof(MyVertex));
	vertex_buffer->setSize(vertex_num);
	memcpy(vertex_buffer->map(), mesh_vertex, vertex_num * sizeof(MyVertex));
	vertex_buffer->unmap();
	m_context["vertices"]->set(vertex_buffer);

	m_width = vertex_num;
	m_height = 1;

	printf("load completed... \n");
	delete[] mesh_vertex;
}

GeometryInstance PathTracerScene::createParallelogram( const float3& anchor,
                                                       const float3& offset1,
                                                       const float3& offset2)
{
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setIntersectionProgram( m_pgram_intersection );
  parallelogram->setBoundingBoxProgram( m_pgram_bounding_box );

  float3 normal = normalize( cross( offset1, offset2 ) );
  float d = dot( normal, anchor );
  float4 plane = make_float4( normal, d );

  float3 v1 = offset1 / dot( offset1, offset1 );
  float3 v2 = offset2 / dot( offset2, offset2 );

  parallelogram["plane"]->setFloat( plane );
  parallelogram["anchor"]->setFloat( anchor );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );

  GeometryInstance gi = m_context->createGeometryInstance();
  gi->setGeometry(parallelogram);
  return gi;
}

GeometryInstance PathTracerScene::createLightParallelogram( const float3& anchor,
                                                            const float3& offset1,
                                                            const float3& offset2,
                                                            int lgt_instance)
{
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setIntersectionProgram( m_pgram_intersection );
  parallelogram->setBoundingBoxProgram( m_pgram_bounding_box );

  float3 normal = normalize( cross( offset1, offset2 ) );
  float d = dot( normal, anchor );
  float4 plane = make_float4( normal, d );

  float3 v1 = offset1 / dot( offset1, offset1 );
  float3 v2 = offset2 / dot( offset2, offset2 );

  parallelogram["plane"]->setFloat( plane );
  parallelogram["anchor"]->setFloat( anchor );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["lgt_instance"]->setInt( lgt_instance );

  GeometryInstance gi = m_context->createGeometryInstance();
  gi->setGeometry(parallelogram);
  return gi;
}

void PathTracerScene::setMaterial( GeometryInstance& gi,
                                   Material material,
                                   const std::string& color_name,
                                   const float3& color)
{
  gi->addMaterial(material);
  gi[color_name]->setFloat(color);
}

void PathTracerScene::createGeometry(){
  // Light buffer
 /* ParallelogramLight light;
  light.corner = make_float3(343.0f, 548.6f, 227.0f);
  light.v1 = make_float3(-130.0f, 0.0f, 0.0f);
  light.v2 = make_float3(0.0f, 0.0f, 105.0f);
  light.normal = normalize(cross(light.v1, light.v2));
  light.emission = make_float3(15.0f, 15.0f, 5.0f);

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(ParallelogramLight));
  light_buffer->setSize(1u);
  memcpy(light_buffer->map(), &light, sizeof(light));
  light_buffer->unmap();
  m_context["lights"]->setBuffer(light_buffer);*/

  // Set up material
  Material diffuse = m_context->createMaterial();
  // Program diffuse_ch = m_context->createProgramFromPTXFile(ptxpath("path_tracer", "path_tracer.cu"), "MirrorReflection");
  Program diffuse_ch = m_context->createProgramFromPTXFile(ptxpath("path_tracer", "path_tracer.cu"), "diffuse");
  Program diffuse_ah = m_context->createProgramFromPTXFile(ptxpath("path_tracer", "path_tracer.cu"), "shadow");
  diffuse->setClosestHitProgram(0, diffuse_ch);
  diffuse->setAnyHitProgram(1, diffuse_ah);

  /*Material diffuse_light = m_context->createMaterial();
  Program diffuse_em = m_context->createProgramFromPTXFile(ptxpath("path_tracer", "path_tracer.cu"), "diffuseEmitter");
  diffuse_light->setClosestHitProgram(0, diffuse_em);*/
  
  // Set up parallelogram programs
  std::string ptx_path = ptxpath("path_tracer", "parallelogram.cu");
  m_pgram_bounding_box = m_context->createProgramFromPTXFile(ptx_path, "bounds");
  m_pgram_intersection = m_context->createProgramFromPTXFile(ptx_path, "intersect");

  
  GeometryGroup mesh_group = m_context->createGeometryGroup();
  OptiXMesh loader(m_context, mesh_group, m_accel_desc);
  loader.loadBegin_Geometry(m_filename);

  for (size_t i = 0; i < loader.getMaterialCount(); ++i) {
    loader.setOptiXMaterial(static_cast<int>(i), diffuse);
  }
  loader.loadFinish_Materials();
  
  m_aabb = loader.getSceneBBox();
  
  Group shadow_group = m_context->createGroup();
  shadow_group->setChildCount(1);
  shadow_group->setChild(0, mesh_group);
  shadow_group->setAcceleration(m_context->createAcceleration(m_accel_desc.builder.c_str(), m_accel_desc.traverser.c_str()));
  m_context["top_shadower"]->set(shadow_group);
  
  // add light
  Group top_group = m_context->createGroup();
  /*GeometryInstance light_geo = createParallelogram(make_float3(343.0f, 548.6f, 227.0f),
    make_float3(-130.0f, 0.0f, 0.0f),
    make_float3(0.0f, 0.0f, 105.0f));
  const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);
  setMaterial(light_geo, diffuse_light, "emission_color", light_em);*/
  
  top_group->setChildCount(1);
  top_group->setChild(0, mesh_group);
  top_group->setAcceleration(m_context->createAcceleration(m_accel_desc.builder.c_str(), m_accel_desc.traverser.c_str()));
  m_context["top_object"]->set(top_group);
}

//-----------------------------------------------------------------------------
//
// main
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -n  | --sqrt_num_samples <ns>              Number of samples to perform for each frame\n"
    << "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
    << std::endl;
  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}


unsigned int getUnsignedArg(int& arg_index, int argc, char** argv)
{
  int result = -1;
  if (arg_index+1 < argc) {
    result = atoi(argv[arg_index+1]);
  } else {
    std::cerr << "Missing argument to "<<argv[arg_index]<<"\n";
    printUsageAndExit(argv[0]);
  }
  if (result < 0) {
    std::cerr << "Argument to "<<argv[arg_index]<<" must be positive.\n";
    printUsageAndExit(argv[0]);
  }
  ++arg_index;
  return static_cast<unsigned int>(result);
}


static RTresult SavePPM(const unsigned char *Pix, const char *fname, int wid, int hgt, int chan)
{
	if (Pix == NULL || chan < 1 || wid < 1 || hgt < 1) {
		fprintf(stderr, "Image is not defined. Not saving.\n");
		return RT_ERROR_UNKNOWN;
	}

	if (chan < 1 || chan > 4) {
		fprintf(stderr, "Can't save a X channel image as a PPM.\n");
		return RT_ERROR_UNKNOWN;
	}

	ofstream OutFile(fname, ios::out | ios::binary);
	if (!OutFile.is_open()) {
		fprintf(stderr, "Could not open file for SavePPM\n");
		return RT_ERROR_UNKNOWN;
	}

	bool is_float = false;
	OutFile << 'P';
	OutFile << ((chan == 1 ? (is_float ? 'Z' : '5') : (chan == 3 ? (is_float ? '7' : '6') : '8'))) << endl;
	OutFile << wid << " " << hgt << endl << 255 << endl;

	OutFile.write(reinterpret_cast<char*>(const_cast<unsigned char*>(Pix)), wid * hgt * chan * (is_float ? 4 : 1));

	OutFile.close();

	return RT_SUCCESS;
}

void PathTracerScene::SaveFrame(const char* filename){
	std::vector<unsigned char> pix(m_width * m_height * 3);

	RTbuffer buffer_1 = m_context["output_buffer_1"]->getBuffer()->get();	
	void* imageData_1;
	rtBufferMap(buffer_1, &imageData_1);

	RTbuffer buffer_2 = m_context["output_buffer_2"]->getBuffer()->get();
	void* imageData_2;
	rtBufferMap(buffer_2, &imageData_2);

	RTbuffer buffer_3 = m_context["output_buffer_3"]->getBuffer()->get();
	void* imageData_3;
	rtBufferMap(buffer_3, &imageData_3);


	FILE * pFile;
	pFile = fopen(filename, "w");

	if (pFile == NULL){
		printf("cannot open file\n");
		exit(-1);
	}

	for (int j = m_height - 1; j >= 0; --j) {
		unsigned char *dst = &pix[0] + (3 * m_width*(m_height - 1 - j));
		float* src_1 = ((float*)imageData_1) + (4 * m_width*j);
		float* src_2 = ((float*)imageData_2) + (4 * m_width*j);
		float* src_3 = ((float*)imageData_3) + (4 * m_width*j);
		for (int i = 0; i < m_width; i++) {
			    fprintf(pFile, "%c ", 'c');
				for (int elem = 0; elem < 3; ++elem) {
					float c = *src_1;
					fprintf(pFile, "%f ", c);
					src_1++;
				}
				src_1++;

				for (int elem = 0; elem < 3; ++elem) {
					float c = *src_2;
					fprintf(pFile, "%f ", c);
					src_2++;
				}
				src_2++;


			  for (int elem = 0; elem < 3; ++elem) {
				float c = *src_3;
				fprintf(pFile, "%f ", c);
				src_3++;
			  }
			  src_3++;
			  fprintf(pFile, "\n");
		}
	}
  rtBufferUnmap(buffer_1);
  rtBufferUnmap(buffer_2);
  rtBufferUnmap(buffer_3);
  exit(-1);
}



int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  unsigned int sqrt_num_samples = 10u;

  unsigned int width = 512u, height = 512u;
  float timeout = 10.0f;

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--sqrt_num_samples" || arg == "-n" ) {
      sqrt_num_samples = atoi( argv[++i] );
    } else if ( arg == "--timeout" || arg == "-t" ) {
      if(++i < argc) {
        timeout = static_cast<float>(atof(argv[i]));
      } else {
        std::cerr << "Missing argument to "<<arg<<"\n";
        printUsageAndExit(argv[0]);
      }
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    PathTracerScene scene;
    scene.setNumSamples( sqrt_num_samples );
    scene.setDimensions( width, height );
	string obj_file_name = std::string(sutilSamplesDir()) + "/simpleAnimation/bunny_with_base.obj";
	string out_obj_name = std::string(sutilSamplesDir()) + "/simpleAnimation/cube_copy.obj";
	scene.setFileName(obj_file_name.c_str());
	std::string envmap_path = std::string(sutilSamplesDir()) + "/simpleAnimation/CedarCity.hdr";
    scene.setEnvmapPath(envmap_path);
	
	/*ProcessObj(obj_file_name.c_str(), out_obj_name.c_str());
	return 0;*/

	GLUTDisplay::setProgressiveDrawingTimeout(timeout);
	GLUTDisplay::setUseSRGB(false);
	GLUTDisplay::run("Cornell Box Scene", &scene, GLUTDisplay::CDProgressive);

  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  
  return 0;
}
