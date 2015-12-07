#include "prt.h"
using namespace std;
using namespace cv;
using namespace Eigen;


/// <summary>
/// Creates a cubemap with a single pixel has color (1, 1, 1) and other pixels black.
/// </summary>
/// <param name="face_index">The index of the face of the pixel.</param>
/// <param name="x">x coordinate of pixel</param>
/// <param name="y">y coordinate of pixel</param>
/// <param name="cubemap_length">The side length of cubemaps</param>
/// <param name="cubeface">The cubemap generated.</param>
void CreateCubemap(int face_index, int x, int y, int cubemap_length, vector<Mat>& cubeface){
	cubeface.resize(6);
	for (int i = 0; i < 6; i++){
		cubeface[i].create(cubemap_length, cubemap_length, CV_32FC3);
	}
	cubeface[face_index].at<Vec3f>(y, x) = Vec3f(1, 1, 1);
}




/// <summary>
/// Subsample the pixel f.
/// </summary>
/// <param name="color">Output: the subsampled color.</param>
/// <param name="image">Input image</param>
/// <param name="x">The x coord of pixel.</param>
/// <param name="y">The y coord of pixel.</param>
/// <returns>status</returns>
int subPixelF(cv::Vec3f& color, const cv::Mat& image, float x, float y)
{
	if (x < 0 || y < 0 || x >= image.cols || y >= image.rows){
		return -1;
	}

	int ix = (int)x;
	int iy = (int)y;

	int ix_1 = ix + 1;
	int iy_1 = iy + 1;
	if (ix == image.cols - 1){
		ix_1 = ix;
	}
	if (iy == image.rows - 1){
		iy_1 = iy;
	}

	float rx = x - (float)ix;
	float ry = y - (float)iy;

	const cv::Vec3f& cxy = image.at<cv::Vec3f>(iy, ix);
	const cv::Vec3f& cx_1y = image.at<cv::Vec3f>(iy, ix_1);
	const cv::Vec3f& cx_1y_1 = image.at<cv::Vec3f>(iy_1, ix_1);
	const cv::Vec3f& cxy_1 = image.at<cv::Vec3f>(iy_1, ix);

	cv::Vec3f cx = (1.0f - ry) * cxy + ry * cxy_1;
	cv::Vec3f cx_1 = (1.0f - ry) * cx_1y + ry * cx_1y_1;

	color = (1.0f - rx) * cx + rx * cx_1;

	return 0;
}

int ConvertCubemapToLightProbe(cv::Mat& light_probe, const std::vector<cv::Mat>& cubemap, const cv::Size light_probe_size)
{
	if (cubemap.size() != 6){
		printf("incorrect cube map\n");
		return -1;
	}
	int cubSize = cubemap[0].cols;

	if (light_probe_size == cv::Size(0, 0)){
		light_probe = cv::Mat::zeros(4 * cubSize, 8 * cubSize, CV_32FC3);
	}
	else{
		light_probe = cv::Mat::zeros(light_probe_size, CV_32FC3);
	}

	for (int x = 0; x < light_probe.cols; x++)
	{
		for (int y = 0; y < light_probe.rows; y++)
		{

			float u = ((float)x + 0.5f) / (float)light_probe.cols;
			float v = ((float)y + 0.5f) / (float)light_probe.rows;

			float theta = v * CV_PI;
			float phi = u * 2.0 * CV_PI;
			float pi_4 = CV_PI / 4.0f;
			float pi_2 = CV_PI / 2.0f;

			cv::Vec3f direction;
			direction[1] = cos(theta);
			direction[0] = sin(phi) * sin(theta);
			direction[2] = -cos(phi) * sin(theta);
			int face = -1;
			float uc, vc;
			if (phi >= pi_4 && phi < pi_4 * 3){

				face = 1;
				direction /= direction[0];
			}
			else if (phi >= pi_4 * 3 && phi < pi_4 * 5){
				face = 4;
				direction /= direction[2];
			}
			else if (phi >= pi_4 * 5 && phi < pi_4 * 7){
				face = 0;
				direction /= -direction[0];
			}
			else{
				face = 5;
				direction /= -direction[2];
			}

			if (direction[1] > 1.0){
				direction /= direction[1];
				face = 2;
			}
			else if (direction[1] < -1.0){
				direction /= -direction[1];
				face = 3;
			}


			if (face == 0){
				uc = -direction[2];
				vc = -direction[1];
			}
			else if (face == 1){
				uc = direction[2];
				vc = -direction[1];
			}
			else if (face == 2){
				direction /= direction[1];
				uc = -direction[0];
				vc = direction[2];
			}
			else if (face == 3){
				direction /= -direction[1];
				uc = -direction[0];
				vc = -direction[2];
			}
			else if (face == 4){
				direction /= direction[2];
				uc = -direction[0];
				vc = -direction[1];
			}
			else if (face == 5){
				direction /= -direction[2];
				uc = direction[0];
				vc = -direction[1];
			}
			else{
				printf("wrong face\n");
				return -1;
			}

			uc = (uc + 1.f) / 2.f;
			vc = (vc + 1.f) / 2.f;

			float xc = uc * (cubSize - 1);
			float yc = vc * (cubSize - 1);

			cv::Vec3f c0;
			subPixelF(c0, cubemap[face], xc, yc);
			light_probe.at<cv::Vec3f>(y, x) = c0;
		}
	}
	return 0;
}
