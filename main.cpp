#include <iostream>
#include <dlfcn.h>

#include "include/model_func.hpp"

#include "yaml-cpp/yaml.h"
/*
在链接动态库时可能会遇到找不到库的问题，解决方法如下：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${shared_library_abs_directory}

*/

typedef YoloV8_det* (*InitYOLODetModelFunc)(std::string bmodel_file, int dev_id, model_inference_params params);
typedef object_detect_result_list (*InferenceYOLODetModelFunc)(YoloV8_det* model, cv::Mat input_image, bool enable_logger);


// 加载动态so库
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
	if (argc != 3) {
		printf("%s <model_name> <image_path>\n", argv[0]);
		return -1;
	}
	int ret = 0;
	int dev_id = 0;
	const char* model_name = argv[1];  // det_person
	const char* image_path = argv[2];  // image dirname

	// 打开动态库
	dlerror();
	void* handle = NULL;
	const char* so_path = "libbm_model_library.so";
	ret = loadSo(so_path, handle);

	// 读取yaml文件
	YAML::Node config = ReadYamlFile("models.yaml");
	YAML::Node model_node = config[model_name];
	if (!model_node) {
		std::cerr << "Unknown model_name: " << model_name << std::endl;
		throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
	}

	const std::string init_func_name = model_node["init_func_name"].as<std::string>();
	// const std::string init_func_name = "init_yolov8_det_model";
	InitYOLODetModelFunc init_model = (InitYOLODetModelFunc)dlsym(handle, init_func_name.c_str());

	const std::string infer_func_name = model_node["function_name"].as<std::string>();
    // const std::string infer_func_name = "inference_yolo_person_det_model";
    InferenceYOLODetModelFunc inference_model = (InferenceYOLODetModelFunc)dlsym(handle, infer_func_name.c_str());
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Cannot load symbol '" << infer_func_name << "': " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }
	const std::string bmodel_file = model_node["path"].as<std::string>();
	model_inference_params params;
	params.input_height = model_node["params"]["input_height"].as<int>();
	params.input_width = model_node["params"]["input_width"].as<int>();
	params.nms_threshold = model_node["params"]["nms_threshold"].as<float>();
	params.box_threshold = model_node["params"]["box_threshold"].as<float>();
	// initialize net
	YoloV8_det* model = init_model(bmodel_file, dev_id, params);
	// 加载图像
	cv::Mat input_image = cv::imread(image_path, cv::IMREAD_COLOR);
	if(input_image.empty()) {
		std::cerr << "Failed to read image: " << image_path << std::endl;
		return 1;
	}
	const std::string enable_log_str = model_node["enable_log"].as<std::string>();
	bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
	object_detect_result_list result = inference_model(model, input_image, enable_log);
	std::cout << "result size: " << result.count << std::endl;

	std::cout << "Success to inference model" << std::endl;
	delete model;
	std::cout << "Success to destroy model" << std::endl;

	if (handle != nullptr) {
		dlclose(handle); 
		handle = nullptr;
	}
	std::cout << "Success to dlclose library" << std::endl;
	return ret;
}