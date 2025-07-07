#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "yolov8_det.hpp"
#include "model_func.hpp"

using json = nlohmann::json;
using namespace std;


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

object_detect_result_list inference_yolo_hand_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger=false){
    // single image inference
    int ret = 0;

    vector<cv::Mat> batch_mats;
    vector<bm_image> batch_imgs;
    vector<YoloV8BoxVec> boxes;

	batch_mats.push_back(input_image);  // only one image

    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);
    ret = model.Detect(batch_imgs, boxes);
    // assert(0 == ret);
    object_detect_result_list result; // 单张图片推理结果
    if (ret != 0){
        std::cout << "inference_yolo_hand_det_model failed" << std::endl;
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
            result.results[i].cls_id = boxes[0][i].class_id;
            result.results[i].prop = boxes[0][i].score;
            result.results[i].box.left = boxes[0][i].x1;
            result.results[i].box.top = boxes[0][i].y1;
            result.results[i].box.right = boxes[0][i].x2;
            result.results[i].box.bottom = boxes[0][i].y2;
        }
        if (enable_logger){
            std::cout << "detect result:" << std::endl;
            for (int i = 0; i < result.count; ++i){
                std::cout << "result[" << i << "]: " << result.results[i].cls_id << " " << result.results[i].prop << " " << result.results[i].box.left << " " << result.results[i].box.top << " " << result.results[i].box.right << " " << result.results[i].box.bottom << std::endl;
            }
            model.draw_result(input_image, boxes[0]);
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
