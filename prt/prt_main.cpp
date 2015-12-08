#include "prt.h"

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace optix;

void test(){
	string output_folder = "C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\data\\toasters\\004\\face_0\\";
	Mat result = imread(output_folder + to_string(0) + ".hdr", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	
	for (int i = 0; i < 16; i++){
		Mat img = imread(output_folder + to_string(i) + ".hdr", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		if (i == 0){
			result = img.clone();
		}
		else{
			result += img;
		}
	}

	result = result;
	FILE* file = fopen("C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\data\\toasters\\004\\face_0\\tea.txt", "w");
	for (int i = 0; i < result.rows; i++){
		Vec3f v = result.at<Vec3f>(i, 0);
		fprintf(file, "c %f %f %f\n", v[0], v[1], v[2]);
	}
	fclose(file);
}

int main(int argc, char** argv){
	test();
	return 0 ;
	int cubemap_length = 4;
	cv::Size light_probe_size(256, 128);

	/*vector<Mat> cubemap_face;
	CreateCubemapFace(0, cubemap_length, cubemap_face);
	Mat light_probe;
	ConvertCubemapToLightProbe(light_probe, cubemap_face, light_probe_size);
	imwrite("C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\data\\scene\\env.exr", light_probe);*/


	//string obj_path = "C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\data\\toasters\\Toasters004.obj";
	string obj_path = "C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\data\\teapot\\bunny_with_base.obj";
	string output_folder = "C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\data\\toasters\\004";
	string tracer_path = "C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\build\\bin\\Release\\PathTracer.exe";
	int sqrt_num_samples = 30;

	PrepareDirectory(output_folder);

	for (int i = 0; i < 6; i++){
		int count = 0;
		for (int r = 0; r < cubemap_length; r++){
			for (int c = 0; c < cubemap_length; c++){
				printf("Face %d: (%d, %d)\n", i, r, c);
				vector<Mat> cubemap_face;
				Mat light_probe;
				CreateCubemap(i, c, r, cubemap_length, cubemap_face);
				ConvertCubemapToLightProbe(light_probe, cubemap_face, light_probe_size);

				string env_path = output_folder + "\\face_" + to_string(i) + "\\" + "env_" + to_string(count) + ".hdr";
				imwrite(env_path, light_probe);

				string output_path = output_folder + "\\face_" + to_string(i) + "\\" + to_string(count) + ".hdr";
				 
				string command = tracer_path + " " + obj_path + " " + to_string(sqrt_num_samples) + " " + 
					env_path + " " + output_path;
				cout << command << endl;
				system(command.c_str());
				count++;
			}
		}
	}

	return 0;
}