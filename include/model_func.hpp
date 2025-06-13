#ifndef _BM_MODEL_FUNC_H_
#define _BM_MODEL_FUNC_H_

#include "models/yolov8_det.hpp"
#include "models/resnet.hpp"
#include "models/resnet_nc.hpp"
#include <opencv2/opencv.hpp>

// #include "bytetrack/bytetrack.h"
// #include "bytetrack/yolov5.hpp"
// #include "utils/bm_wrapper.hpp"

#ifdef __cplusplus
extern "C" {
#endif

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

YoloV8_det* init_yolov8_det_model(std::string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names);
object_detect_result_list inference_yolo_person_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_header_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_screen_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_coco_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);

RESNET* init_resnet_cls_model(std::string bmodel_file, int dev_id);
int inference_resnet_cls_model(RESNET model, cv::Mat input_image, bool enable_logger);

RESNET_NC* init_face_attr_model(std::string bmodel_file, int dev_id);
cls_model_result inference_face_attr_model(RESNET_NC model, cv::Mat input_image, bool enable_logger);
cls_model_result inference_call_up_model(RESNET_NC model, cv::Mat input_image, bool enable_logger);

// PPOCR_Detector* init_ppocr_det_model(std::string bmodel_file, int dev_id);
// PPOCR_Rec* init_ppocr_rec_model(std::string bmodel_file, int dev_id);
// int inference_ppocr_det_rec_model(PPOCR_Detector ppocr_det, PPOCR_Rec ppocr_rec);
ocr_result_list inference_ppocr_det_rec_model(std::string bmodel_det, std::string bmodel_rec, cv::Mat input_image, bool enable_logger);

// STracks inference_bytetrack_yolov5_det_model(YoloV5 yolov5, BYTETracker bytetrack, bm_image img, bool enable_logger);

#ifdef __cplusplus
}
#endif

#endif 