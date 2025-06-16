#ifndef _BM_MODEL_PARAMS_H_
#define _BM_MODEL_PARAMS_H_

#include <string>
#define OBJ_NUMB_MAX_SIZE 80

typedef struct cls_model_result {
	int num_class;
	int cls_output[OBJ_NUMB_MAX_SIZE];
} cls_model_result;  // 分类模型输出结果

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

/* inference params */
typedef struct model_inference_params {
	int input_height;
	int input_width;
	float nms_threshold;
	float box_threshold;
}model_inference_params;
#endif 