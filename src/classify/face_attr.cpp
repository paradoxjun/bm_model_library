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

std::vector<std::string> face_attr_class_names = {
    "hat",
    "glass",
    "mask"
};

RESNET_NC* init_multi_class_model(string bmodel_file, int dev_id){
    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
    cout << "set device id: "  << dev_id << endl;
    bm_handle_t h = handle->handle();
    // load bmodel
    shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());
    cout << "load bmodel: " << bmodel_file << endl;
    return new RESNET_NC(bm_ctx);
}

cls_model_result inference_face_attr_model(RESNET_NC model, cv::Mat input_image, bool enable_logger=false){
    // single image inference
    std::cout << "inference_face_attr_model" << endl;
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
        // cout << "attr_id: " << results[i].first << ", class_id: " << results[i].second << endl;
        int attr_id = model_results[i].first;
        int class_id = model_results[i].second;
        results.cls_output[attr_id] = class_id;
    }

    if (enable_logger){
        std::cout << "模型输出类别: " << results.num_class << endl;
        for (int i=0; i < results.num_class; i++){
            std::cout << "类别: " << i << " 类别名:" << face_attr_class_names[i] << " 输出: " << results.cls_output[i] << endl;
        }
    }
    return results;
}

#ifdef __cplusplus
}
#endif