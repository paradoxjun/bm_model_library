#ifndef _BM_MODEL_PARAMS_H_
#define _BM_MODEL_PARAMS_H_

#include <string>
#define OBJ_NUMB_MAX_SIZE 80

typedef struct cls_model_result {
	int num_class;
	int cls_output[OBJ_NUMB_MAX_SIZE];
} cls_model_result;  // 分类模型输出结果

typedef struct {
    int class_id;   // 预测类别
    float score;      // 置信度
} cls_result;


// det
typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
	image_rect_t box;
	float prop;
	int cls_id;
} object_detect_result;

typedef struct {
	int id;
	int count;
	object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

// obb
typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
	int x3;
	int y3;
	int x4;
	int y4;
} image_obb_t;

typedef struct {
    image_obb_t box;
    float score;
    int class_id;
} object_obb_result;

typedef struct {
    int id;
    int count;
    object_obb_result results[OBJ_NUMB_MAX_SIZE];
} object_obb_result_list;

//ppocr
typedef struct {
  int x1, y1, x2, y2, x3, y3, x4, y4;
  std::string rec_res;
  float score;
}ocr_box;
typedef struct {
	int count;
	ocr_box results[OBJ_NUMB_MAX_SIZE];
} ocr_result_list;

// pose
/*
#define KEY_POINTS_MAX_SIZE 17
typedef struct {
    image_rect_t box;
    float score;
    int points_num;
    float points[KEY_POINTS_MAX_SIZE * 3];
} object_pose_result;

// 模板静态分配，减少分配时间
template<int K>
struct object_pose_result_t {
    image_rect_t box;
    float score;
    int points_num;
    std::array<float, K*3> points;
};
*/

struct object_pose_result {
    image_rect_t box;
    float score;
    int points_num;
    std::vector<float> points;   // 大小 = points_num*3
};

typedef struct {
    int id;
    int count;
    object_pose_result results[OBJ_NUMB_MAX_SIZE];
} object_pose_result_list;


/* inference params */
typedef struct model_inference_params {
	int input_height;
	int input_width;
	float nms_threshold;
	float box_threshold;
}model_inference_params;

typedef struct model_pose_inference_params {
    int input_height;
    int input_width;
    float nms_threshold;
    float box_threshold;
    int kpt_nums;
}model_pose_inference_params;

typedef struct model_posepointnet_inference_params {
    bool img_size_from_frame;
    float img_w;
    float img_h;
}model_posepointnet_inference_params;
#endif 