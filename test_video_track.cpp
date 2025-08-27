//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.    All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// 基于yolov8-person det 算法和bytetrack跟踪算法，实现视频的行人多目标跟踪，视频中每个人的track_id和box位置
#include <iostream>
#include <dlfcn.h>
#include "yaml-cpp/yaml.h"

#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <regex>

#include "include/bytetrack/draw_utils.hpp"
#include "include/utils/ff_decode.hpp"
#include "include/utils/json.hpp"
#include "include/model_func.hpp"
using json = nlohmann::json;

// yolo det
typedef YoloV8_det* (*InitYOLODetModelFunc)(std::string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names);
typedef object_detect_result_list (*InferenceYOLODetModelFunc)(YoloV8_det* model, cv::Mat input_image, bool enable_logger);

// bytetrack
typedef BYTETracker* (*InitBYTETrackModelFunc)(bytetrack_params track_params);
typedef STracks (*InferenceBYTETrackModelFunc)(BYTETracker* bytetrack, object_detect_result_list result, bool enable_logger);


static int loadSo(const char* soPath, void*& handle) {
	handle = dlopen(soPath, RTLD_LAZY);
	if (!handle) {
		std::cerr << "Cannot open library: " << dlerror() << std::endl;
		return 1;
	}
	dlerror();  // 清除之前的错误
	return 0;
}

// 读取yaml文件
static YAML::Node ReadYamlFile(const char* yaml_path) {
	// Load YAML configuration
	YAML::Node config = YAML::LoadFile(yaml_path);
	YAML::Node model_node = config["models"];

	if (!model_node) {
		std::cerr << "Unknown model_name: models " << std::endl;
		throw std::invalid_argument("Unknown model_name: models");
	}
	return model_node;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("%s <video_path>\n", argv[0]);
		return -1;
	}
	int ret = 0;
	int dev_id = 0;
	std::string input = argv[1];  //
  // get params
  // std::string input = "tests/dsz1206.mp4";
  // 打开动态库
	dlerror();
	void* handle = NULL;
	const char* so_path = "libbm_model_library.so";
	ret = loadSo(so_path, handle);
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Cannot load symbol :" << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

	// 读取yaml文件
	YAML::Node config = ReadYamlFile("models.yaml");
  /*-------------------------------------------
                  Person Det Function
  -------------------------------------------*/
  const char* model_name = "person_det";
	YAML::Node model_node = config[model_name];
	if (!model_node) {
		std::cerr << "Unknown model_name: " << model_name << std::endl;
		throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
	}
  const std::string init_func_name = model_node["init_func_name"].as<std::string>();
	const std::string infer_func_name = model_node["function_name"].as<std::string>();
	const std::string bmodel_detector = model_node["path"].as<std::string>();
	const std::string enable_log_str = model_node["enable_log"].as<std::string>();
	bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
  InitYOLODetModelFunc init_model = (InitYOLODetModelFunc)dlsym(handle, init_func_name.c_str());
  InferenceYOLODetModelFunc inference_model = (InferenceYOLODetModelFunc)dlsym(handle, infer_func_name.c_str());

  model_inference_params params;
  params.input_height = model_node["params"]["input_height"].as<int>();
  params.input_width = model_node["params"]["input_width"].as<int>();
  params.nms_threshold = model_node["params"]["nms_threshold"].as<float>();
  params.box_threshold = model_node["params"]["box_threshold"].as<float>();
  // 读取models.yaml文件的class_names
  std::vector<std::string> class_names;
  for (const auto& class_name : model_node["class_names"]) {
    class_names.push_back(class_name.as<std::string>());
  }
  // initialize net
  YoloV8_det* model = init_model(bmodel_detector, dev_id, params, class_names);

  /*-------------------------------------------
                  Person Track Function
  -------------------------------------------*/
const char* track_model_name = "person_track";
	YAML::Node track_model_node = config[track_model_name];
	if (!track_model_node) {
		std::cerr << "Unknown model_name: " << track_model_name << std::endl;
		throw std::invalid_argument("Unknown model_name: " + std::string(track_model_name));
	}
  const std::string track_init_func_name = track_model_node["init_func_name"].as<std::string>();
	const std::string track_infer_func_name = track_model_node["function_name"].as<std::string>();
	const std::string track_enable_log_str = track_model_node["enable_log"].as<std::string>();
	bool track_enable_log = (track_enable_log_str == "true" || track_enable_log_str == "True");
  InitBYTETrackModelFunc track_init_model = (InitBYTETrackModelFunc)dlsym(handle, track_init_func_name.c_str());
  InferenceBYTETrackModelFunc track_inference_model = (InferenceBYTETrackModelFunc)dlsym(handle, track_infer_func_name.c_str());
  bytetrack_params track_params;
  track_params.track_thresh = track_model_node["params"]["track_thresh"].as<float>();
  track_params.match_thresh = track_model_node["params"]["match_thresh"].as<float>();
  track_params.min_box_area = track_model_node["params"]["min_box_area"].as<int>();
  track_params.track_buffer = track_model_node["params"]["track_buffer"].as<int>();
  track_params.frame_rate = track_model_node["params"]["frame_rate"].as<int>();

  // initialize net
  BYTETracker* bytetrack = track_init_model(track_params);
  std::cout << "bytetrack init done" << std::endl;

  // creat handle
  BMNNHandlePtr bm_handle = std::make_shared<BMNNHandle>(dev_id);
  std::cout << "set device id: " << dev_id << std::endl;
  bm_handle_t h = bm_handle->handle();

  //  test images
  VideoDecFFM decoder;
  decoder.openDec(&h, input.c_str());
  std::string save_image_path = "results/video/";
  if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
  if (access("results/video", 0) != F_OK) mkdir("results/video", S_IRWXU);

  bool end_flag = false;
  int ind = 0;
  int interval = 5;
  cv::Scalar color = cv::Scalar(0, 255, 0);
  while (!end_flag) {
    if (ind % interval != 0) {
      ++ind;
      continue;
    }
    bm_image* img;
    img = decoder.grab();
    if (!img) {
      end_flag = true;
    }
    cv::Mat cv_image;
    cv::bmcv::toMAT(img, cv_image);
    object_detect_result_list result = inference_model(model, cv_image, enable_log);
    STracks output_stracks = track_inference_model(bytetrack, result, track_enable_log);
    for (auto& track_box : output_stracks) {
      int track_id = track_box->track_id;
      int box_x1 = track_box->tlwh[0];
      int box_y1 = track_box->tlwh[1];
      int box_w = track_box->tlwh[2];
      int box_h = track_box->tlwh[3];
      int frame_id = track_box->frame_id;
      int class_id = track_box->class_id;
      // std::string text = "track_id:" + std::to_string(track_id);
      std::string text = std::to_string(track_id);
      std::cout << "frame_id:" << frame_id << " track_id:" << track_id << " class_id:" << class_id << " box:" << box_x1 << "," << box_y1 << "," << box_w << "," << box_h << std::endl;
      cv::rectangle(cv_image, cv::Point(box_x1,box_y1),cv::Point(box_x1+box_w,box_y1+box_h), color, 2);
      cv::putText(cv_image, text, cv::Point(box_x1-10, box_y1-10), cv::FONT_HERSHEY_SIMPLEX, 2, color, 2, cv::LINE_8, false);
    }
    ++ind;
    std::cout << "track_nums: " << output_stracks.size() << std::endl;
    std::string filename = save_image_path + "frame_" + std::to_string(ind) + ".jpg";
    cv::imwrite(filename, cv_image);
    bm_image_destroy(*img);
    cv_image.release();
  }
  
  if (handle != nullptr) {
		dlclose(handle); 
		handle = nullptr;
	}
  return 0;
}