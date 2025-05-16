#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include "json.hpp"
#include "resnet.hpp"
#include "ff_decode.hpp"
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
using json = nlohmann::json;
using namespace std;

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

RESNET* init_resnet_cls_model(string bmodel_file, int dev_id){
    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
    cout << "set device id: "  << dev_id << endl;
    bm_handle_t h = handle->handle();
    // load bmodel
    shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());
    cout << "load bmodel: " << bmodel_file << endl;
    return new RESNET(bm_ctx);
}

int inference_resnet_cls_model(RESNET model, cv::Mat input_image, bool enable_logger=false){
    // single image inference
    std::cout << "inference_resnet_cls_model" << endl;
    int ret = 0;
    vector<bm_image> batch_imgs;
    vector<pair<int, float>> results;
    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);
    std::cout << "batch_size: " << model.batch_size() << endl;

    ret = model.Classify(batch_imgs, results);
    std::cout << "results size: " << results.size() << endl;
    assert(0 == ret);
    bm_image_destroy(batch_imgs[0]);

    for (int i=0; i < results.size(); i++){
        cout << "class_id: " << results[i].first << ", score: " << results[i].second << endl;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif