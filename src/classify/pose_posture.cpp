#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <fstream>

#include "json.hpp"
#include "resnet_nc.hpp"
#include "ff_decode.hpp"
#include "model_func.hpp"

using json = nlohmann::json;
using namespace std;

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

std::vector<std::string> posture_class_names = {
    "neg",
    "sit",
    "stand"
};

cls_model_result inference_pose_posture_model(RESNET_NC model, cv::Mat input_image, bool enable_logger=false){
    // single image inference
    std::cout << "inference_pose_posture_model" << endl;
    // int batch_size = model.batch_size();
    cls_model_result results;
    vector<pair<int, int>> model_results;
    vector<bm_image> batch_imgs;
    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);

    int output_num = model.Classify(batch_imgs, model_results);
    bm_image_destroy(batch_imgs[0]);

    results.num_class = output_num;
    for (int i=0; i < output_num; i++){
        int attr_id = model_results[i].first;
        int class_id = model_results[i].second;
        results.cls_output[attr_id] = class_id;
    }

    if (enable_logger){
        std::cout << "模型输出类别: " << results.num_class << endl;
        for (int i=0; i < results.num_class; i++){
            std::cout << "类别: " << i << " 输出: " << results.cls_output[i] << endl;
        }
    }
    return results;
}

#ifdef __cplusplus
}
#endif