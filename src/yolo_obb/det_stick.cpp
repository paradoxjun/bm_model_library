#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "yolov8_obb.hpp"
#include "model_func.hpp"

using json = nlohmann::json;
using namespace std;


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

YoloV8_obb* init_yolov8_obb_model(string bmodel_file, int dev_id, model_inference_params params, std::vector<std::string> model_class_names){
    std::cout << "init_yolov8_obb_model" << std::endl;
    return new YoloV8_obb(bmodel_file, model_class_names, dev_id, params.box_threshold, params.nms_threshold);
}

object_obb_result_list inference_yolo_stick_obb_model(YoloV8_obb model, cv::Mat input_image, bool enable_logger=false){
    // single image inference
    int ret = 0;

    // vector<cv::Mat> batch_mats;
    vector<bm_image> batch_imgs;
    vector<obbBoxVec_xyxy> boxes;

	// batch_mats.push_back(input_image);  // only one image

    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);
    ret = model.Detect(batch_imgs, boxes);
    object_obb_result_list result; // 单张图片推理结果
    if (ret != 0){
        std::cout << "inference_yolo_stick_obb_model failed" << std::endl;
        result.count = -1;
        return result;
    }
    // save result to object_detect_result_list,针对单张图片进行推理
    if (boxes.size() > 0){
        result.count = boxes[0].size();
        for (int i = 0; i < boxes[0].size(); ++i){
            if (i >= OBJ_NUMB_MAX_SIZE){
                break;
            }
            result.results[i].class_id = boxes[0][i].class_id;
            result.results[i].score = boxes[0][i].score;
            result.results[i].box.x1 = boxes[0][i].x1;
            result.results[i].box.y1 = boxes[0][i].y1;
            result.results[i].box.x2 = boxes[0][i].x2;
            result.results[i].box.y2 = boxes[0][i].y2;
            result.results[i].box.x3 = boxes[0][i].x3;
            result.results[i].box.y3 = boxes[0][i].y3;
            result.results[i].box.x4 = boxes[0][i].x4;
            result.results[i].box.y4 = boxes[0][i].y4;
        }
        if (enable_logger){
            std::cout << "detect result:" << result.count << std::endl;
            for (int i = 0; i < result.count; ++i){
                std::cout << "result[" << i << "]: " << result.results[i].class_id << " " << result.results[i].score << " " << 
                result.results[i].box.x1 << " " << result.results[i].box.y1 << " " << 
                result.results[i].box.x2 << " " << result.results[i].box.y2 << " " << 
                result.results[i].box.x3 << " " << result.results[i].box.y3 << " " << 
                result.results[i].box.x4 << " " << result.results[i].box.y4 << std::endl;
            }
            model.drawPred(boxes[0], input_image);
            string img_file = "result.jpg";
            cv::imwrite(img_file, input_image);
            std::cout << "save result to " << img_file << std::endl;
        }
    }else{
        result.count = -1;
    }
    return result;
}


#ifdef __cplusplus
}
#endif
