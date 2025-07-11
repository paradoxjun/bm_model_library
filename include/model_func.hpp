#ifndef _BM_MODEL_FUNC_H_
#define _BM_MODEL_FUNC_H_

#include <opencv2/opencv.hpp>

#include "model_params.hpp"
#include "models/yolov8_det.hpp"
#include "models/yolov8_pose.hpp"
#include "models/yolov8_obb.hpp"
#include "models/resnet.hpp"
#include "models/resnet_nc.hpp"
#include "bytetrack/bytetrack.h"


#ifdef __cplusplus
extern "C" {
#endif

YoloV8_det* init_yolov8_det_model(std::string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names);
object_detect_result_list inference_yolo_person_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolov8_object_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_screen_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_coco_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_hand_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_money_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_gzwp_atm_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);
object_detect_result_list inference_yolo_kx_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger);

RESNET* init_resnet_cls_model(std::string bmodel_file, int dev_id);
int inference_resnet_cls_model(RESNET model, cv::Mat input_image, bool enable_logger);

RESNET_NC* init_multi_class_model(std::string bmodel_file, int dev_id);
cls_model_result inference_face_attr_model(RESNET_NC model, cv::Mat input_image, bool enable_logger);
cls_model_result inference_callup_smoke_model(RESNET_NC model, cv::Mat input_image, bool enable_logger);

ocr_result_list inference_ppocr_det_rec_model(std::string bmodel_det, std::string bmodel_rec, cv::Mat input_image, bool enable_logger);

BYTETracker* init_bytetrack_model(bytetrack_params track_params);
STracks inference_person_bytetrack_model(BYTETracker bytetrack, object_detect_result_list result, bool enable_logger);

YoloV8_obb* init_yolov8_obb_model(std::string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names);
object_obb_result_list inference_yolo_stick_obb_model(YoloV8_obb model, cv::Mat input_image, bool enable_logger);

#ifdef __cplusplus
}
#endif

#endif 