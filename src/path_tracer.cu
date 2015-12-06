
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "helpers.h"
#include "path_tracer.h"
#include "random.h"
#include "utils.h"

using namespace optix;

struct PerRayData_radiance{
    float3 result;
    float importance;
    int depth;
};


struct PerRayData_pathtrace_shadow{
    bool inShadow;
};

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );

// For camera
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtBuffer<float4, 2>              output_buffer;


rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  shadow_ray_type, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(PerRayData_radiance, current_prd, rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

static __device__ inline float3 powf(float3 a, float exp)
{
  return make_float3(powf(a.x, exp), powf(a.y, exp), powf(a.z, exp));
}

// For miss program
rtDeclareVariable(float3,       bg_color, , );
rtDeclareVariable(float3,        emission_color, , );

// For envirnoment map
rtTextureSampler<float4, 2> envmap;

// For shadow ray
rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

// For vertex tracer
rtBuffer<MyVertex>  vertices;



RT_PROGRAM void exception(){
  output_buffer[launch_index] = make_float4(bad_color, 0.0f);
}


RT_PROGRAM void miss(){
  current_prd.result = bg_color;
}


RT_PROGRAM void shadow(){
  current_prd_shadow.inShadow = true;
  rtTerminateRay();
}


RT_PROGRAM void envmap_miss(){
	float theta = atan2f(ray.direction.x, ray.direction.z);
	float phi = M_PIf * 0.5f - acosf(ray.direction.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	current_prd.result = make_float3(tex2D(envmap, u, v));
}


RT_PROGRAM void one_bounce_diffuse_closest_hit(){
    float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal               = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
    float2 uv                     = make_float2(texcoord);

    float3 Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));
    float3 result = make_float3(0);

    // compute indirect bounce 
    if(prd.detph < 1){
        optix::Onb onb(ffnormal);
        unsigned int seed = rot_seed(rnd_seeds[launch_index], frame);
        const float inv_sqrt_samples = 1.0f / float(sqrt_num_samples);

        int nx = sqrt_num_samples;
        int ny = sqrt_num_samples;

        while(nx--){
            while(ny--){
                float u1 = (float(nx) + rnd(seed)) * inv_sqrt_samples;
                float u2 = (float(ny) + rnd(seed)) * inv_sqrt_samples;

                float3 dir;
                optix::cosine_sample_hemisphere(u1, u2, dir);
                onb.inverse_transform(dir);

                PerRayData_radiance radiance_prd;
                radiance_prd.importance = current_prd.importance * optix::luminance(Kd);
                radiance_prd.depth = current_prd.depth + 1;

                if(radiance_prd.importance > 0.001f){
                    optix::Ray radiance_ray = optix::make_Ray(hit_point, dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
                    rtTrace(top_object, radiance_ray, radiance_prd);
                    result += radiance_prd.result;
                }
            }
        }
        result *= (Kd) / ((float)(M_PIf * sqrt_num_samples * sqrt_num_samples));
    }

    current_prd.result = result;
}



RT_PROGRAM void VertexTracer(){
    int index = launch_index.x;
    float3 vertex_pos = vertices[index].vertex;
    float3 normal = vertices[index].normal;
    normal = normalize(normal);

    unsigned int seed = tea<16>(index, 1);
    unsigned int sample_num = samples_per_vertex;
    float3 result = make_float3(0.0f);

    float3 v1, v2;
    createONB(normal, v1, v2);
    float3 direct_light = make_float3(0.0);
    sample_num = 1000;


    do{
        PerRayData_pathtrace prd;
        prd.result = make_float3(0.0f);
        prd.attenuation = make_float3(0.8f);
        prd.countEmitted = true;
        prd.done = false;
        prd.inside = false;
        prd.seed = seed;
        prd.depth = 0;

        // float z1 = (sample_num % 20 + 0.5) / 20;
        // float z2 = (sample_num / 20 + 0.5) / 20;
        float z1 = rnd(prd.seed);
        float z2 = rnd(prd.seed);

        float3 p;
        cosine_sample_hemisphere(z1, z2, p);
        float3 ray_direction = v1 * p.x  + v2 * p.y  + normal * p.z;

        ray_direction = normalize(ray_direction);
        float cos_theta = dot(ray_direction, normal);

        float3 ray_origin = vertex_pos;
        for(;;) {
            Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
            rtTrace(top_object, ray, prd);
            if(prd.done) {
                prd.result += prd.radiance * prd.attenuation;
                break;
            }

            // RR: randomly reject some rays
            if(prd.depth >= rr_begin_depth){
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            prd.depth++;
            prd.result += prd.radiance * prd.attenuation;
            ray_origin = prd.origin;
            ray_direction = prd.direction;
        }

        result += (prd.result);
        seed = prd.seed;
    } while(--sample_num);


    result = result / 1000;

    float3 pixel_color = result;
    output_buffer[launch_index] = make_float4(pixel_color, 0.0f);
}




RT_PROGRAM void InDirectRender(){
    unsigned int resolution = 10;
    int index = launch_index.x;
    float3 vertex_pos = vertices[index].vertex;
    float3 normal = vertices[index].normal;
    normal = normalize(normal);
    unsigned int seed = tea<16>(index, 1);

    float3 v1, v2;
    createONB(normal, v1, v2);
    Matrix3x3 li;
    li = 0.0 * li;

    for (unsigned int i = 0; i < resolution; i++){
        for (unsigned int j = 0; j < resolution; j++){
            float z1 = (i + 0.5) / (float)resolution;
            float z2 = (j + 0.5) / (float)resolution;
            
            PerRayData_pathtrace prd;
            prd.sh_result = 0.0f * prd.sh_result;
            prd.countEmitted = true;
            prd.done = false;
            prd.inside = false;
            prd.seed = seed;
            prd.depth = 0;
            prd.sh_coeff = 0.0 * prd.sh_coeff;

            float3 p;
            cosine_sample_hemisphere(z1, z2, p);
            float3 ray_direction = v1 * p.x + v2 * p.y + normal * p.z;
            ray_direction = normalize(ray_direction);

            float3 ray_origin = vertex_pos;

            for (;;) {
                Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
                rtTrace(top_object, ray, prd);
                if (prd.done) {
                    prd.sh_result += prd.sh_coeff;
                    break;
                }

                // RR: randomly reject some rays
                

                prd.depth++;
                prd.sh_result += prd.sh_coeff ;
                ray_origin = prd.origin;
                ray_direction = prd.direction;
				if (prd.depth == 1){
					break;
				}
            }


            li = li + prd.sh_result;
            // Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
            // rtTrace(top_object, ray, prd);
        }
    }

    li = li / (resolution * resolution);
    // li = li * (2 * M_PIf / (resolution * resolution));

    output_buffer_1[launch_index] = make_float4(li[0], li[1], li[2], 0.0f);
    output_buffer_2[launch_index] = make_float4(li[3], li[4], li[5], 0.0f);
    output_buffer_3[launch_index] = make_float4(li[6], li[7], li[8], 0.0f);
}




