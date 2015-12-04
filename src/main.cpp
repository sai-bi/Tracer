#include "PathTracerScene.h"
#include <iostream>
#include <fstream>
#include <regex>
#include <algorithm>

using namespace std;
using namespace optix;

void printUsageAndExit(const std::string& argv0, bool doExit = true)
{
	std::cerr
		<< "Usage  : " << argv0 << " [options]\n"
		<< "App options:\n"
		<< "  -h  | --help                               Print this usage message\n"
		<< "  -n  | --sqrt_num_samples <ns>              Number of samples to perform for each frame\n"
		<< "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
		<< std::endl;
	GLUTDisplay::printUsage();

	if (doExit) exit(1);
}


unsigned int getUnsignedArg(int& arg_index, int argc, char** argv)
{
	int result = -1;
	if (arg_index + 1 < argc) {
		result = atoi(argv[arg_index + 1]);
	}
	else {
		std::cerr << "Missing argument to " << argv[arg_index] << "\n";
		printUsageAndExit(argv[0]);
	}
	if (result < 0) {
		std::cerr << "Argument to " << argv[arg_index] << " must be positive.\n";
		printUsageAndExit(argv[0]);
	}
	++arg_index;
	return static_cast<unsigned int>(result);
}

int main(int argc, char** argv)
{
	GLUTDisplay::init(argc, argv);
	if (!GLUTDisplay::isBenchmark()) {
		printUsageAndExit(argv[0], false);
	}

	unsigned int sqrt_num_samples = 10u;
	unsigned int width = 512u, height = 512u;
	float timeout = 10.0f;
	PathTracerScene scene;

	string config_path = "C:\\Users\\bisai\\Documents\\GitHub\\Tracer\\src\\Configure.in";
	ifstream fin(config_path.c_str());

	if (!fin.is_open()){
		cerr << "Cannot open the file: " << config_path << endl;
		exit(-1);
	}
	string line = "";
	while (getline(fin, line)){
		int index = line.find_first_not_of("=");
		string var_name = line.substr(0, index);
		string var_value = line.substr(index + 1);

		std::regex e("\\\\");
		var_value = std::regex_replace(var_value, e, "/$2");
		cout << var_name << " " << var_value << endl;
		continue;
		if (var_name == "scene_path"){
			scene.setFileName(var_value);
		}
		else if (var_name == "sqrt_sample_num"){
			scene.setNumSamples(atoi(var_value.c_str()));
		}
		else if (var_name == "env_path"){
			scene.setEnvmapPath(var_value);
		}
		else if (var_name == "output_path"){
			scene.setOutputPath(var_value);
		}
	}

	try {
		GLUTDisplay::setProgressiveDrawingTimeout(timeout);
		GLUTDisplay::setUseSRGB(true);
		GLUTDisplay::run("Cornell Box Scene", &scene, GLUTDisplay::CDProgressive);
	}
	catch (Exception& e){
		sutilReportError(e.getErrorString().c_str());
		exit(1);
	}

	return 0;
}