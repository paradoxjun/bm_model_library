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
extern "C" {
#endif

// 人员五分类标签
static const std::vector<std::string> ren_class_names = {
    "anbao", "baojie", "worker", "yayun", "ren"
};

/**
 * @brief  人员五分类（返回 Top‑1）
 */
cls_result inference_rec_ren_model(RESNET model, cv::Mat input_image, bool enable_logger = false){
    vector<bm_image> batch_imgs;
    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);

    vector<pair<int,float>> vec;
    int ret = model.Classify(batch_imgs, vec);
    bm_image_destroy(batch_imgs[0]);

    if (ret != 0 || vec.empty()) {
        cerr << "[rec_ren] Classify failed, ret=" << ret << endl;
        return { -1, 0.f };
    }

    int   class_id = vec[0].first;
    float score    = vec[0].second;

    if (enable_logger) {
        cout << "[rec_ren] class_id: " << class_id << " score: "  << score
             << " name: " << (class_id < ren_class_names.size() ? ren_class_names[class_id] : "unknown") << endl;
    }

    return { class_id, score };
}

#ifdef __cplusplus
}
#endif