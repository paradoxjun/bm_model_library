//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.    All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// export LD_LIBRARY_PATH=build/src/libbm_model_library.so:$LD_LIBRARY_PATH
#include <iostream>
#include <dlfcn.h>
#include "yaml-cpp/yaml.h"

#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <regex>

#include "include/bytetrack/bytetrack.h"
#include "include/bytetrack/draw_utils.hpp"
#include "include/utils/ff_decode.hpp"
#include "include/utils/json.hpp"
// #include "yolov5.hpp"
#include "include/model_func.hpp"
using json = nlohmann::json;

// yolo det
typedef YoloV8_det* (*InitYOLODetModelFunc)(std::string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names);
typedef object_detect_result_list (*InferenceYOLODetModelFunc)(YoloV8_det* model, cv::Mat input_image, bool enable_logger);
// typedef STracks (*InferencePersonByteTrackModelFunc)(YoloV5 yolov5, BYTETracker bytetrack, bm_image img, bool enable_logger);


std::vector<std::string> stringSplit(const std::string& str, char delim) {
  std::string s;
  s.append(1, delim);
  std::regex reg(s);
  std::vector<std::string> elems(
      std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
      std::sregex_token_iterator());
  return elems;
}
bool check_path(std::string file_path,
                std::vector<std::string> correct_postfixes) {
  auto index = file_path.rfind('.');
  std::string postfix = file_path.substr(index + 1);
  if (find(correct_postfixes.begin(), correct_postfixes.end(), postfix) !=
      correct_postfixes.end()) {
    return true;
  } else {
    std::cout << "skipping path: " << file_path
              << ", please check your dataset!" << std::endl;
    return false;
  }
};
void getAllFiles(std::string path, std::vector<std::string>& files,
                 std::vector<std::string> correct_postfixes) {
  DIR* dir;
  struct dirent* ptr;
  if ((dir = opendir(path.c_str())) == NULL) {
    perror("Open dri error...");
    exit(1);
  }
  while ((ptr = readdir(dir)) != NULL) {
    if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
      continue;
    else if (ptr->d_type == 8 &&
             check_path(path + "/" + ptr->d_name, correct_postfixes))  // file
      files.push_back(path + "/" + ptr->d_name);
    else if (ptr->d_type == 10)  // link file
      continue;
    else if (ptr->d_type == 4) {
      // files.push_back(ptr->d_name);//dir
      getAllFiles(path + "/" + ptr->d_name, files, correct_postfixes);
    }
  }
  closedir(dir);
}

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


int main() {
  std::cout.setf(std::ios::fixed);
  // get params
  int ret = 0;
  int dev_id = 0;
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
  // object_detect_result_list result = inference_model(model, input_image, enable_log);

  /*-------------------------------------------
                  Person Track Function
  -------------------------------------------*/
	// const std::string init_func_name = model_node["init_func_name"].as<std::string>();
	// const std::string infer_func_name = model_node["function_name"].as<std::string>();
	// const std::string bmodel_detector = model_node["path"].as<std::string>();
	// const std::string enable_log_str = model_node["enable_log"].as<std::string>();
	// bool enable_log = (enable_log_str == "true" || enable_log_str == "True");

  // get params
  std::string input = "tests/dsz1206.mp4";
  // std::string input = "tests/jnt1206";
  // std::string classnames = "src/bytetrack/coco.names";
  // std::string config = "src/bytetrack/bytetrack.yaml";

  bytetrack_params track_params;
  // track_params.conf_thresh = 0.7;
  // track_params.nms_thresh = 0.4;
  track_params.track_thresh = 0.7;
  track_params.match_thresh = 0.8;
  track_params.min_box_area = 10;
  track_params.track_buffer = 30;
  track_params.frame_rate = 30;

  // check params
  struct stat info;
  if (stat(bmodel_detector.c_str(), &info) != 0) {
    std::cout << "Cannot find valid detector model file." << std::endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0) {
    std::cout << "Cannot find input path." << std::endl;
    exit(1);
  }

  // InferencePersonByteTrackModelFunc inference_model = (InferencePersonByteTrackModelFunc)dlsym(handle, infer_func_name.c_str());

  // creat handle
  BMNNHandlePtr bm_handle = std::make_shared<BMNNHandle>(dev_id);
  std::cout << "set device id: " << dev_id << std::endl;
  bm_handle_t h = bm_handle->handle();

  // load bmodel
  // std::shared_ptr<BMNNContext> bm_ctx_detector =
  //     std::make_shared<BMNNContext>(bm_handle, bmodel_detector.c_str());

  // initialize net
  BYTETracker bytetrack(track_params);
  std::cout << "bytetrack init done" << std::endl;

  // get batch_size
  // creat save path
  if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
  if (access("results/images", 0) != F_OK) mkdir("results/images", S_IRWXU);
  if (access("results/video", 0) != F_OK) mkdir("results/video", S_IRWXU);

  // initialize data buffer.
  std::vector<bm_image> batch_imgs;
  int id = 0;
  //  test images
  std::vector<std::string> image_paths;
  VideoDecFFM decoder;
  std::string save_image_path;
  auto stringBuffer = stringSplit(input, '/');
  auto bmodel_detector_name_buffer = stringSplit(bmodel_detector, '/');
  auto bmodel_detector_name =
      bmodel_detector_name_buffer[bmodel_detector_name_buffer.size() - 1];
  if (info.st_mode & S_IFDIR) {
    std::vector<std::string> correct_postfixes = {"jpg", "png"};
    getAllFiles(input, image_paths, correct_postfixes);
    sort(image_paths.begin(), image_paths.end());
    save_image_path = "results/images/";
  } else {
    decoder.openDec(&h, input.c_str());
    save_image_path = "results/video/";
  }

  bool end_flag = false;
  int ind = 0;
  cv::Scalar color = cv::Scalar(0, 255, 0);
  while (!end_flag) {
    bm_image* img;
    img = decoder.grab();
    if (!img) {
      end_flag = true;
    }
    cv::Mat cv_image;
    cv::bmcv::toMAT(img, cv_image);
    object_detect_result_list result = inference_model(model, cv_image, false);
    STracks output_stracks;
    bytetrack.update(output_stracks, result);
    for (auto& track_box : output_stracks) {
      int track_id = track_box->track_id;
      int box_x1 = track_box->tlwh[0];
      int box_y1 = track_box->tlwh[1];
      int box_w = track_box->tlwh[2];
      int box_h = track_box->tlwh[3];
      int frame_id = track_box->frame_id;
      int class_id = track_box->class_id;
      std::string text = "track_id:" + std::to_string(track_id);
      std::cout << "frame_id:" << frame_id << " track_id:" << track_id << " class_id:" << class_id << " box:" << box_x1 << "," << box_y1 << "," << box_w << "," << box_h << std::endl;
      cv::rectangle(cv_image, cv::Point(box_x1,box_y1),cv::Point(box_x1+box_w,box_y1+box_h), color, 2);
      cv::putText(cv_image, text, cv::Point(box_x1, box_y1-10), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0));
    }
    ++ind;
    std::cout << "track_nums: " << output_stracks.size() << std::endl;
    std::string filename = save_image_path + "frame_" + std::to_string(ind) + ".jpg";
    cv::imwrite(filename, cv_image);
  }
  
  if (handle != nullptr) {
		dlclose(handle); 
		handle = nullptr;
	}
  return 0;
}