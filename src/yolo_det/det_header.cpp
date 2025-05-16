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


// std::vector<std::string> header_det_model_class_names = {
//     // "person",
//     "header"
// };

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

object_detect_result_list inference_yolo_header_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger=false){
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
    assert(0 == ret);

    // save result to object_detect_result_list
    object_detect_result_list result;
    result.count = boxes[0].size();
    for (int i = 0; i < boxes[0].size(); i++){
        result.results->cls_id = boxes[0][i].class_id;
        result.results->prop = boxes[0][i].score;
        result.results->box.left = boxes[0][i].x1;
        result.results->box.top = boxes[0][i].y1;
        result.results->box.right = boxes[0][i].x2;
        result.results->box.bottom = boxes[0][i].y2;
    }

    if (enable_logger){
        std::cout << "detect result:" << std::endl;
        for (int i = 0; i < boxes.size(); i++){
            for (int j = 0; j < boxes[i].size(); j++){
                std::cout << "class_id:" << boxes[i][j].class_id << " score:" << boxes[i][j].score << " x1:" << boxes[i][j].x1 << " y1:" << boxes[i][j].y1 << " x2:" << boxes[i][j].x2 << " y2:" << boxes[i][j].y2 << std::endl;
            }
        }
        // creat save path
        if (access("results", 0) != F_OK)
            mkdir("results", S_IRWXU);
        if (access("results/images", 0) != F_OK)
            mkdir("results/images", S_IRWXU);
        model.draw_result(input_image, boxes[0]);
        string img_file = "result.jpg";
        cv::imwrite(img_file, input_image);
        std::cout << "save result to " << img_file << std::endl;
    }
    return result;
}


#ifdef __cplusplus
}
#endif
