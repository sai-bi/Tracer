#pragma once

#include "cv.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Eigen/core"

void CreateCubemap(int face_index, int x, int y, int cubemap_length, std::vector<cv::Mat>& cubeface);
int ConvertCubemapToLightProbe(cv::Mat& light_probe, const std::vector<cv::Mat>& cubemap, const cv::Size light_probe_size);
