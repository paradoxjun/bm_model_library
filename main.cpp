#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include "include/model_func.hpp"
#include "yaml-cpp/yaml.h"
/*
在链接动态库时可能会遇到找不到库的问题，解决方法如下：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${shared_library_abs_directory}

*/

// yolo det
typedef YoloV8_det* (*InitYOLODetModelFunc)(std::string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names);
typedef object_detect_result_list (*InferenceYOLODetModelFunc)(YoloV8_det* model, cv::Mat input_image, bool enable_logger);

// yolo obb
typedef YoloV8_obb* (*InitYOLOObbModelFunc)(std::string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names);
typedef object_obb_result_list (*InferenceYOLOObbModelFunc)(YoloV8_obb* model, cv::Mat input_image, bool enable_logger);

// yolo pose
typedef YoloV8_pose* (*InitYOLOPoseModelFunc)(std::string bmodel_file, int dev_id, model_pose_inference_params params, std::vector<std::string> model_class_names);
typedef object_pose_result_list (*InferenceYOLOPoseModelFunc)(YoloV8_pose* model, cv::Mat input_image, bool enable_logger);

// resnet cls
typedef RESNET* (*InitResNetClsModelFunc)(std::string bmodel_file, int dev_id);
typedef int (*InferenceResNetClsModelFunc)(RESNET* model, cv::Mat input_image, bool enable_logger);
typedef cls_result (*InferenceResNetClsModelRetFunc)(RESNET* model, cv::Mat input_image, bool enable_logger);

// face_attr
typedef RESNET_NC* (*InitMultiClassModelFunc)(std::string bmodel_file, int dev_id);
typedef cls_model_result (*InferenceMultiClassModelFunc)(RESNET_NC* model, cv::Mat input_image, bool enable_logger);

// pose_pointnet
typedef PosePointNet* (*InitPosePointNetModelFunc)(std::string bmodel_file, int dev_id, model_posepointnet_inference_params params, std::vector<std::string> model_class_names);
typedef cls_result (*InferencePosePointNetModelFunc)(PosePointNet* model, const std::vector<object_pose_result_list>& pose_seq, const std::vector<object_detect_result_list>& det_seq, bool enable_logger);

// ppocr
// typedef PPOCR_Detector* (*InitPPOCRDetModelFunc)(std::string bmodel_file, int dev_id);
// typedef PPOCR_Rec* (*InitPPOCRRecModelFunc)(std::string bmodel_file, int dev_id);
// typedef int (*InferencePPOCRDetRecModelFunc)(PPOCR_Detector* ppocr_det, PPOCR_Rec* ppocr_rec);
typedef ocr_result_list (*InferencePPOCRDetRecModelFunc)(std::string bmodel_det, std::string bmodel_rec, cv::Mat input_image, bool enable_logger);
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
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Cannot load symbol :" << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }

	// 读取yaml文件
	YAML::Node config = ReadYamlFile("models.yaml");
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
	const std::string model_type = model_node["model_type"].as<std::string>();

	// 加载图像
	cv::Mat input_image = cv::imread(image_path, cv::IMREAD_COLOR);
	if(input_image.empty()) {
		std::cerr << "Failed to read image: " << image_path << std::endl;
		return 1;
	}

	if (model_type == "yolo_det") { // yolo检测模型
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
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_model(model, input_image, enable_log);
		std::cout << "result size: " << result.count << std::endl;
		std::cout << "Success to inference model" << std::endl;
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "yolo模型执行时间: " << duration.count() << " 毫秒" << std::endl;
		delete model;
		std::cout << "Success to destroy model" << std::endl;
	}else if (model_type == "res_rec") { // 单分类模型
		std::cout << "Start to inference classification model" << std::endl;
		// 读取models.yaml文件的class_names
		std::vector<std::string> class_names;
		for (const auto& class_name : model_node["class_names"]) {
			class_names.push_back(class_name.as<std::string>());
		}
		InitResNetClsModelFunc init_model = (InitResNetClsModelFunc)dlsym(handle, init_func_name.c_str());
		InferenceResNetClsModelRetFunc inference_model = (InferenceResNetClsModelRetFunc)dlsym(handle, infer_func_name.c_str());
		RESNET* model = init_model(bmodel_file, dev_id);
		std::cout << "Success to init model" << std::endl;
		cls_result ret = inference_model(model, input_image, enable_log);
		std::cout << "Success to inference model" << std::endl;
		delete model;
		std::cout << "Success to destroy model" << std::endl;
	}else if (model_type == "multi_class") { // 多分类模型
		std::cout << "Start to inference classification model" << std::endl;
		std::vector<std::string> class_names;
		for (const auto& class_name : model_node["class_names"]) {
			class_names.push_back(class_name.as<std::string>());
		}
		std::vector<std::vector<std::string>> class_values;
		YAML::Node class_values_node = model_node["class_values"];  // Fixed fs -> config
		for (YAML::const_iterator it = class_values_node.begin(); it != class_values_node.end(); ++it) {
			std::vector<std::string> values;
			for (YAML::const_iterator jt = it->begin(); jt != it->end(); ++jt) {
				values.push_back(jt->as<std::string>());
			}
			class_values.push_back(values);
		}
		InitMultiClassModelFunc init_model = (InitMultiClassModelFunc)dlsym(handle, init_func_name.c_str());
		InferenceMultiClassModelFunc inference_model = (InferenceMultiClassModelFunc)dlsym(handle, infer_func_name.c_str());
		RESNET_NC* model = init_model(bmodel_file, dev_id);
		cls_model_result cls_result = inference_model(model, input_image, enable_log);
		std::cout << "模型输出类别数量: " << cls_result.num_class << std::endl;
		for (int i=0; i < cls_result.num_class; i++){
			std::cout << "类别: " << class_names[i] << " 输出: " << class_values[i][cls_result.cls_output[i]] << std::endl;
		}
		delete model;
		std::cout << "Success to destroy model" << std::endl;
	}else if (model_type == "ppocr") {
		// std::cout << "init ppocr det model" << std::endl;
		const std::string det_bmodel_file = model_node["det_path"].as<std::string>();
		// const std::string init_det_func = model_node["init_det_func_name"].as<std::string>();
		// InitPPOCRDetModelFunc init_det_model = (InitPPOCRDetModelFunc)dlsym(handle, init_det_func.c_str()); 
		// PPOCR_Detector* det_model = init_det_model(det_bmodel_file, dev_id);
		// std::cout << "init ppocr rec model" << std::endl;
		const std::string rec_bmodel_file = model_node["rec_path"].as<std::string>();
		// const std::string init_rec_func = model_node["init_rec_func_name"].as<std::string>();
		// InitPPOCRRecModelFunc init_rec_model = (InitPPOCRRecModelFunc)dlsym(handle, init_rec_func.c_str()); 
		// PPOCR_Rec* rec_model = init_rec_model(rec_bmodel_file, dev_id);
		// std::cout << "infer ppocr det and rec model" << std::endl;
		InferencePPOCRDetRecModelFunc inference_model = (InferencePPOCRDetRecModelFunc)dlsym(handle, infer_func_name.c_str());
		std::cout << "Success to load model" << std::endl;
		// int ret = inference_model(det_model, rec_model);
		ocr_result_list result = inference_model(det_bmodel_file, rec_bmodel_file, input_image, enable_log);
		std::cout << "Success to inference model" << std::endl;
	}else if (model_type == "yolo_obb") { // yolo旋转框检测模型
		std::cout << "Start to load model" << std::endl;
		InitYOLOObbModelFunc init_model = (InitYOLOObbModelFunc)dlsym(handle, init_func_name.c_str());
		InferenceYOLOObbModelFunc inference_model = (InferenceYOLOObbModelFunc)dlsym(handle, infer_func_name.c_str());

		model_inference_params params;
		params.input_height = model_node["params"]["input_height"].as<int>();
		params.input_width = model_node["params"]["input_width"].as<int>();
		params.nms_threshold = model_node["params"]["nms_threshold"].as<float>();
		params.box_threshold = model_node["params"]["box_threshold"].as<float>();
		// 读取models.yaml文件的class_names
		std::vector<std::string> class_names;
		for (const auto& class_name : model_node["class_names"]) {
			class_names.push_back(class_name.as<std::string>());
			std::cout << class_name.as<std::string>() << std::endl;
		}
		// initialize net
		YoloV8_obb* model = init_model(bmodel_file, dev_id, params, class_names);
		if (!model) {
			std::cerr << "Failed to initialize model" << std::endl;
			return -1;
		}
		std::cout << "init model done" << std::endl;
		if (input_image.empty()) {
			std::cerr << "Empty input image" << std::endl;
			delete model;
			return -1;
		}
		object_obb_result_list result = inference_model(model, input_image, enable_log);
		std::cout << "inference model done" << std::endl;
		std::cout << "result size: " << result.count << std::endl;
		std::cout << "Success to inference model" << std::endl;
		delete model;
		std::cout << "Success to destroy model" << std::endl;
	}
	else if (model_type == "yolo_pose") {
    InitYOLOPoseModelFunc init_model = (InitYOLOPoseModelFunc)dlsym(handle, init_func_name.c_str());
    InferenceYOLOPoseModelFunc inference_model = (InferenceYOLOPoseModelFunc)dlsym(handle, infer_func_name.c_str());

    if (!init_model || !inference_model) {
        std::cerr << "dlsym failed: " << dlerror() << std::endl;
        return -1;
    }

    model_pose_inference_params params;
    params.input_height = model_node["params"]["input_height"].as<int>();
    params.input_width = model_node["params"]["input_width"].as<int>();
    params.nms_threshold = model_node["params"]["nms_threshold"].as<float>();
    params.box_threshold = model_node["params"]["box_threshold"].as<float>();
    params.kpt_nums = model_node["params"]["kpt_nums"].as<int>();

    std::vector<std::string> class_names;
    for (const auto& class_name : model_node["class_names"]) {
        class_names.push_back(class_name.as<std::string>());
    }

    YoloV8_pose* model = init_model(bmodel_file, dev_id, params, class_names);
    if (!model) {
        std::cerr << "init_model returned nullptr" << std::endl;
        return -1;
    }

    if (input_image.empty()) {
        std::cerr << "Empty input image" << std::endl;
        delete model;
        return -1;
    }

    object_pose_result_list result = inference_model(model, input_image, enable_log);
    delete model;
	}

	else if (model_type == "pose_pointnet") { // 基于骨架的动作分类（如摔倒）
		// 1) 动态符号
		InitPosePointNetModelFunc init_model = (InitPosePointNetModelFunc)dlsym(handle, init_func_name.c_str());
		InferencePosePointNetModelFunc inference_model = (InferencePosePointNetModelFunc)dlsym(handle, infer_func_name.c_str());
		if (!init_model || !inference_model) {
			std::cerr << "dlsym failed: " << dlerror() << std::endl;
			return -1;
		}

		// 2) 读取参数（仅图像尺寸来源）
		model_posepointnet_inference_params params{};
		params.img_size_from_frame = model_node["params"]["img_size_from_frame"].as<bool>();
		if (params.img_size_from_frame) {
			params.img_w = (float)input_image.cols;
			params.img_h = (float)input_image.rows;
		} else {
			params.img_w = model_node["params"]["default_img_w"].as<float>();
			params.img_h = model_node["params"]["default_img_h"].as<float>();
		}

		// 3) 读取 YAML 的 class_names（如 ["fall","other"]）
		std::vector<std::string> class_names;
		for (const auto& class_name : model_node["class_names"]) {
			class_names.push_back(class_name.as<std::string>());
		}

		// 4) 初始化模型（统一把 class_names 传入 init）
		PosePointNet* model = init_model(bmodel_file, dev_id, params, class_names);
		if (!model) {
			std::cerr << "init_pose_pointnet_model returned nullptr" << std::endl;
			return -1;
		}

		// 5) 构造【更真实的模拟数据】（32 帧 / 1 人 / 17 点，含检测框与置信度）
		//    动作：站立 -> 下坠旋转 -> 躺地；关键点置信度随阶段变化
		std::vector<object_pose_result_list>   pose_seq;
		std::vector<object_detect_result_list> det_seq;

		auto clampf = [](float v, float lo, float hi){ return std::max(lo, std::min(hi, v)); };

		// 生成一帧骨架（基于简单的人体几何 + 旋转）
		auto synth_pose_frame = [&](float theta_rad,  // 身体相对竖直的旋转角（0=直立，~90°=躺平）
									float cx, float cy, // 身体中心（大致躯干中点）
									const image_rect_t& box,
									float body_conf,        // 全局置信度
									float limb_low_conf,    // 某些肢体点在跌倒中段的较低置信度（模拟遮挡）
									object_pose_result_list& pose_out) {
			const int V = 17;
			pose_out.id = 0;
			pose_out.count = 1;
			pose_out.results[0].box = box;
			pose_out.results[0].score = body_conf;
			pose_out.results[0].points_num = V;
			pose_out.results[0].points.assign(V * 3, 0.f);

			// 简单几何参数（单位：像素，围绕局部坐标系原点）
			float torso = (box.bottom - box.top) * 0.55f; // 躯干大致长度
			float sh_w  = (box.right - box.left) * 0.35f; // 肩宽
			float hip_w = (box.right - box.left) * 0.28f; // 髋宽
			float neck  = torso * 0.18f;
			float head  = torso * 0.20f;
			float u_arm = torso * 0.33f;   // 上臂
			float l_arm = torso * 0.30f;   // 下臂
			float u_leg = torso * 0.45f;   // 大腿
			float l_leg = torso * 0.45f;   // 小腿

			// 局部参考点（未旋转，y 向下）
			// 以“肩中点”为 (0,0)，髋中点 (0, torso*0.6)，头顶在负 y
			cv::Point2f shoulder_c(0.f, 0.f);
			cv::Point2f hip_c(0.f, torso * 0.6f);

			// 左右标识：左(+) / 右(-) 仅用于横向偏移
			auto L = +1.f, R = -1.f;

			// 定义局部坐标（未旋转）
			std::vector<cv::Point2f> P(V);
			// 头部与脸
			P[5]  = { L*sh_w*0.5f, 0.f };              // L-shoulder
			P[6]  = { R*sh_w*0.5f, 0.f };              // R-shoulder
			P[11] = { L*hip_w*0.5f, hip_c.y };         // L-hip
			P[12] = { R*hip_w*0.5f, hip_c.y };         // R-hip
			cv::Point2f neck_c(0.f, -neck);
			cv::Point2f head_c(0.f, -neck - head);
			P[0]  = head_c;                            // nose
			P[1]  = head_c + cv::Point2f(+head*0.2f, -head*0.1f); // L-eye
			P[2]  = head_c + cv::Point2f(-head*0.2f, -head*0.1f); // R-eye
			P[3]  = head_c + cv::Point2f(+head*0.35f, 0.f);       // L-ear
			P[4]  = head_c + cv::Point2f(-head*0.35f, 0.f);       // R-ear

			// 手臂：肘、腕（默认垂落；跌倒时会向前）
			float arm_fwd = l_arm * 0.4f;  // 前伸量（跟随跌倒阶段会加大）
			P[7]  = P[5] + cv::Point2f(0.f, u_arm * 0.9f);                 // L-elbow
			P[8]  = P[6] + cv::Point2f(0.f, u_arm * 0.9f);                 // R-elbow
			P[9]  = P[7] + cv::Point2f(+arm_fwd, l_arm * 0.9f);            // L-wrist
			P[10] = P[8] + cv::Point2f(+arm_fwd, l_arm * 0.9f);            // R-wrist（两腕都向“+x”方向前伸）

			// 双腿：膝、踝（自然直立或轻微屈膝）
			P[13] = P[11] + cv::Point2f(0.f, u_leg);                       // L-knee
			P[14] = P[12] + cv::Point2f(0.f, u_leg);                       // R-knee
			P[15] = P[13] + cv::Point2f(0.f, l_leg);                       // L-ankle
			P[16] = P[14] + cv::Point2f(0.f, l_leg);                       // R-ankle

			// 跌倒阶段加大前伸与屈膝（用 theta 近似驱动）
			float fall_ratio = clampf(theta_rad / (float)M_PI_2, 0.f, 1.f); // 0..1
			P[7].x  += +arm_fwd * 0.5f * fall_ratio;  P[8].x  += +arm_fwd * 0.5f * fall_ratio;
			P[9].x  += +arm_fwd * 0.8f * fall_ratio;  P[10].x += +arm_fwd * 0.8f * fall_ratio;
			P[13].y -=  u_leg * 0.15f * fall_ratio;   P[14].y -=  u_leg * 0.15f * fall_ratio; // 屈膝一点
			P[15].y -=  l_leg * 0.10f * fall_ratio;   P[16].y -=  l_leg * 0.10f * fall_ratio;

			// 旋转 + 平移到图像坐标（绕肩中点旋转，然后平移到 (cx,cy)）
			float c = std::cos(theta_rad), s = std::sin(theta_rad);
			auto rot = [&](const cv::Point2f& q)->cv::Point2f {
				// 先把点以“肩中点”为参考移到局部，再旋转
				cv::Point2f r;
				float x = q.x - shoulder_c.x;
				float y = q.y - shoulder_c.y;
				r.x = c * x - s * y;
				r.y = s * x + c * y;
				r.x += cx; r.y += cy;
				// 限制在检测框内，避免被预处理置零
				r.x = clampf(r.x, box.left + 2.f,  box.right - 2.f);
				r.y = clampf(r.y, box.top  + 2.f,  box.bottom - 2.f);
				return r;
			};

			// 写入点与置信度
			// 默认置信度
			auto set_kpt = [&](int idx, const cv::Point2f& r, float kconf){
				pose_out.results[0].points[idx*3 + 0] = r.x;
				pose_out.results[0].points[idx*3 + 1] = r.y;
				pose_out.results[0].points[idx*3 + 2] = kconf;
			};

			// 旋转并填入
			for (int i = 0; i < V; ++i) {
				cv::Point2f R = rot(P[i]);
				float kconf = body_conf;
				// 跌倒中段手腕/脚踝稍微降低置信度
				if (i==9 || i==10 || i==15 || i==16) kconf = limb_low_conf;
				set_kpt(i, R, kconf);
			}
		};

		// 生成整个 32 帧序列
		{
			const int  T_sim = 32;
			const int  W = (int)params.img_w;
			const int  H = (int)params.img_h;

			// 初始检测框（直立，较窄高；最后逐步变宽矮）
			int cx0 = W/2, cy0 = H/2 - H/12;
			int box_w0 = std::max(90, W/9);
			int box_h0 = std::max(160, H/3);

			// 躺地阶段的框（更宽更矮，中心更低）
			int cx1 = W/2 + W/20;
			int cy1 = H/2 + H/10;
			int box_w1 = std::max(200, W/3);
			int box_h1 = std::max(90,  H/6);

			// 分段：站立 0..9，跌倒 10..19，躺地 20..31
			for (int t = 0; t < T_sim; ++t) {
				// 插值中心与框尺寸
				float u = (t <= 9) ? 0.f : (t <= 19 ? (t - 10) / 9.f : 1.f);
				int cx = (int)std::round(cx0 + u * (cx1 - cx0));
				int cy = (int)std::round(cy0 + u * (cy1 - cy0));
				int bw = (int)std::round(box_w0 + u * (box_w1 - box_w0));
				int bh = (int)std::round(box_h0 + u * (box_h1 - box_h0));

				image_rect_t box;
				box.left   = clampf(cx - bw/2, 0,          (float)(W-1));
				box.right  = clampf(cx + bw/2, box.left+2, (float)(W-1));
				box.top    = clampf(cy - bh/2, 0,          (float)(H-1));
				box.bottom = clampf(cy + bh/2, box.top+2,  (float)(H-1));

				// 旋转角：0° -> 90°
				float theta = 0.f;
				if (t <= 9)        theta = 0.f;
				else if (t <= 19)  theta = (float)M_PI_2 * (t - 10) / 9.f;   // 快速旋转
				else               theta = (float)M_PI_2 * 0.95f;            // 基本躺平

				// 置信度：整体较高；跌倒中段肢体端点更低一点，模拟遮挡
				float body_conf     = 0.95f;
				float limb_low_conf = (t >= 10 && t <= 20) ? 0.55f : 0.90f;

				// --- 检测框 ---
				object_detect_result_list det{};
				det.id = 0;
				det.count = 1;
				det.results[0].box    = box;
				det.results[0].prop   = 0.96f; // 框置信度
				det.results[0].cls_id = 0;
				det_seq.push_back(det);

				// --- 关键点 ---
				object_pose_result_list pose{};
				synth_pose_frame(theta, (float)cx, (float)cy, box, body_conf, limb_low_conf, pose);
				pose_seq.push_back(pose);
			}
		}

		// 6) 推理
		auto t0 = std::chrono::high_resolution_clock::now();
		cls_result r = inference_model(model, pose_seq, det_seq, enable_log);
		auto t1 = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

		// 7) 打印类别名（用 YAML 读到的 class_names）
		std::string cname = (r.class_id >= 0 && r.class_id < (int)class_names.size())
							? class_names[r.class_id] : "unknown";
		std::cout << "[pose_pointnet] pred: " << cname
				<< " (id=" << r.class_id << ")"
				<< " score=" << r.score
				<< "  time=" << ms << " ms" << std::endl;

		// 8) 释放
		delete model;
		std::cout << "Success to destroy pose_pointnet model" << std::endl;
	}

	
	
	else {
		std::cout << "model_type ERROR !" << std::endl;
	}
	
	if (handle != nullptr) {
		dlclose(handle); 
		handle = nullptr;
	}
	std::cout << "Success to dlclose library" << std::endl;
	return ret;
}