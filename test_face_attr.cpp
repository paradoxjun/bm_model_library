#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>  // Corrected include

#include "include/model_func.hpp"

#include "yaml-cpp/yaml.h"
/*
在链接动态库时可能会遇到找不到库的问题，解决方法如下：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${shared_library_abs_directory}

*/

using namespace cv;
using namespace std;

// yolo det
typedef YoloV8_det* (*InitYOLODetModelFunc)(string bmodel_file, int dev_id, model_inference_params params, vector<string> model_class_names);
typedef object_detect_result_list (*InferenceYOLODetModelFunc)(YoloV8_det* model, Mat input_image, bool enable_logger);

// face_attr, callup_smoke
typedef RESNET_NC* (*InitMultiClassModelFunc)(string bmodel_file, int dev_id);
typedef cls_model_result (*InferenceMultiClassModelFunc)(RESNET_NC* model, Mat input_image, bool enable_logger);

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
	if (argc != 2) {
		printf("%s <image_path>\n", argv[0]);
		return -1;
	}
	int ret = 0;
	int dev_id = 0;
	const char* image_path = argv[1];  // image dirname

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

	// 加载图像
    Mat input_image = imread(image_path, IMREAD_COLOR);
    if(input_image.empty()) {
        cerr << "Failed to read image: " << image_path << endl;
        return 1;
    }

	// 读取yaml文件
	YAML::Node config = ReadYamlFile("models.yaml");
	
	/*-------------------------------------------
                  Header Det Function
  	-------------------------------------------*/
	const char* model_name = "header_det";
	YAML::Node model_node = config[model_name];
	if (!model_node) {
		std::cerr << "Unknown model_name: " << model_name << std::endl;
		throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
	}
	const std::string init_func_name = model_node["init_func_name"].as<std::string>();
	const std::string infer_func_name = model_node["function_name"].as<std::string>();
	const std::string bmodel_file = model_node["path"].as<std::string>();
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
	YoloV8_det* model = init_model(bmodel_file, dev_id, params, class_names);
	object_detect_result_list det_result = inference_model(model, input_image, enable_log);
	std::cout << "result size: " << det_result.count << std::endl;
	std::cout << "Success to inference header det model" << std::endl;
	delete model;
	
	/*-------------------------------------------
                  Face Attr Function
  	-------------------------------------------*/
	const char* cls_model_name = "face_attr";
	YAML::Node cls_model_node = config[cls_model_name];
	if (!cls_model_node) {
		std::cerr << "Unknown model_name: " << cls_model_name << std::endl;
		throw std::invalid_argument("Unknown model_name: " + std::string(cls_model_name));
	}
	const std::string cls_init_func_name = cls_model_node["init_func_name"].as<std::string>();
	const std::string cls_infer_func_name = cls_model_node["function_name"].as<std::string>();
	const std::string cls_bmodel_file = cls_model_node["path"].as<std::string>();
	const std::string cls_enable_log_str = cls_model_node["enable_log"].as<std::string>();
	bool cls_enable_log = (cls_enable_log_str == "true" || cls_enable_log_str == "True");

	std::vector<std::string> cls_class_names;
	for (auto cls_class_node : cls_model_node["class_names"]) {
		std::string cls_class_name = cls_class_node.as<std::string>();
		cls_class_names.push_back(cls_class_name);
	}
	std::vector<std::vector<std::string>> class_values;
    YAML::Node face_attr_node = cls_model_node["class_values"];  // Fixed fs -> config
    for (YAML::const_iterator it = face_attr_node.begin(); it != face_attr_node.end(); ++it) {
        std::vector<std::string> values;
        for (YAML::const_iterator jt = it->begin(); jt != it->end(); ++jt) {
            values.push_back(jt->as<std::string>());
        }
        class_values.push_back(values);
    }

	InitMultiClassModelFunc cls_init_model = (InitMultiClassModelFunc)dlsym(handle, cls_init_func_name.c_str());
	InferenceMultiClassModelFunc cls_inference_model = (InferenceMultiClassModelFunc)dlsym(handle, cls_infer_func_name.c_str());
	RESNET_NC* cls_model = cls_init_model(cls_bmodel_file, dev_id);
	std::cout << " face attr model init done" << std::endl;
	for (int i=0;i<det_result.count;i++){
		int cls_id = det_result.results[i].cls_id;
		float score = det_result.results[i].prop;
		int left = det_result.results[i].box.left;
		int top = det_result.results[i].box.top;
		int right = det_result.results[i].box.right;
		int bottom = det_result.results[i].box.bottom;
		cv::Mat img_crop = input_image(cv::Rect(left, top, right - left, bottom - top));
		cls_model_result cls_result = cls_inference_model(cls_model, img_crop, enable_log);
		std::cout << "模型输出类别数量: " << cls_result.num_class << std::endl;
		for (int i=0; i < cls_result.num_class; i++){
			std::cout << "类别: " << cls_class_names[i] << " 输出: " << class_values[i][cls_result.cls_output[i]] << std::endl;
		}
	}
	delete cls_model;

	if (handle != nullptr) {
		dlclose(handle); 
		handle = nullptr;
	}
	std::cout << "Success to dlclose library" << std::endl;
	return ret;
}