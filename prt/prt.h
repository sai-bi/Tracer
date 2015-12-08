#pragma once

#include "cv.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Eigen/core"
#include <string>
#include "../src/PathTracerScene.h"


void CreateCubemap(int face_index, int x, int y, int cubemap_length, std::vector<cv::Mat>& cubeface);
int ConvertCubemapToLightProbe(cv::Mat& light_probe, const std::vector<cv::Mat>& cubemap, const cv::Size light_probe_size);
int subPixelF(cv::Vec3f& color, const cv::Mat& image, float x, float y);
void PrepareDirectory(std::string target_dir);

void CreateCubemapFace(int face_index, int cubemap_length, std::vector<cv::Mat>& cubeface);
