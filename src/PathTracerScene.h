#ifndef PATH_TRACER_SCENE_H
#define PATH_TRACER_SCENE_H


#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include "sampleConfig.h"
#include <ImageDisplay.h>
#include <GLUTDisplay.h>
#include <PPMLoader.h>
#include <OptiXMesh.h>
#include <ImageLoader.h>

class PathTracerScene : public SampleScene
{
public:
	// Set the actual render parameters below in main().
	PathTracerScene()
		: m_rr_begin_depth(1u)
		, m_sqrt_num_samples(0u)
		, m_width(512u)
		, m_height(512u)
	{}

	virtual void   initScene(InitialCameraData& camera_data);
	virtual void   trace(const RayGenCameraData& camera_data);
	virtual optix::Buffer getOutputBuffer();

	void    setNumSamples(unsigned int sns)                           { m_sqrt_num_samples = sns; }
	void    setDimensions(const unsigned int w, const unsigned int h) { m_width = w; m_height = h; }
	void    setFileName(std::string file_name){
		this->m_filename = file_name;
	}
	void    setEnvmapPath(std::string path){
		this->m_env_path = path;
	}

	void    setOutputPath(std::string output_path){
		this->m_output_path = output_path;
	}

	void	initContext();

	void	LoadGeometry();

	void	SaveFrame(const char* file_name);

private:
	// Should return true if key was handled, false otherwise.
	virtual bool keyPressed(unsigned char key, int x, int y);
	void createGeometry();
	void setMaterial(optix::GeometryInstance& gi,
		optix::Material material,
		const std::string& color_name,
		const optix::float3& color);


	optix::Program        m_pgram_bounding_box;
	optix::Program        m_pgram_intersection;

	unsigned int   m_rr_begin_depth;
	unsigned int   m_sqrt_num_samples;
	unsigned int   m_width;
	unsigned int   m_height;
	unsigned int   m_frame;
	unsigned int   m_sampling_strategy;
	std::string    m_filename;
	std::string    m_env_path;
	optix::Aabb    m_aabb;
	std::string    m_output_path;
	
	optix::Buffer  m_rnd_seeds;
};

#endif