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


std::vector<std::string> model_class_names = {
    "person",
    // "header"
};

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

YoloV8_det* init_yolov8_det_model(string bmodel_file, int dev_id, model_inference_params params){
    // YoloV8_det yolov8(bmodel_file, model_class_names, dev_id, conf_thresh, nms_thresh);
    // return yolov8;

    return new YoloV8_det(bmodel_file, model_class_names, dev_id, params.box_threshold, params.nms_threshold);
}

object_detect_result_list inference_yolo_person_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger=false){
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


/* batch images inference 
int inference_yolo_person_det_model_batch(){
    int ret = 0;
    // string coco_names_file = "/data/workspace/wangjing/sophon-demo-release/sample/YOLOv8_plus_det/datasets/person_det.names";
    string bmodel_file = "/data/workspace/wangjing/sophon-demo-release/sample/YOLOv8_plus_det/models/BM1684X/det_person_yolov8n_1684x_f16.bmodel";
    int dev_id = 0;
    float conf_thresh = 0.5;
    float nms_thresh = 0.7;
    string input = "/data/workspace/wangjing/sophon-demo-release/sample/YOLOv8_plus_det/datasets/test";
    // initialize net
    // YoloV8_det yolov8(bmodel_file, model_class_names, dev_id, conf_thresh, nms_thresh);
    YoloV8_det yolov8 = init_yolov8_det_model(bmodel_file, dev_id, conf_thresh, nms_thresh);

    // profiling
    TimeStamp yolov8_ts;
    yolov8.m_ts = &yolov8_ts;
    // get batch_size
    int batch_size = yolov8.batch_size;
    
    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);
    
    // load all images
    vector<cv::Mat> all_mats;
    vector<bm_image> all_imgs;
    vector<string> all_names;
    ret = load_bm_images(input, all_mats, all_imgs, all_names);

    vector<cv::Mat> batch_mats;
    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    vector<YoloV8BoxVec> boxes;
    vector<json> results_json;

    for (int id = 0; id < all_names.size();) { 
        string img_name = all_names[id];
        batch_mats.push_back(all_mats[id]);
        batch_imgs.push_back(all_imgs[id]);
        batch_names.push_back(img_name);
        id++;
        bool end_flag = (id == all_names.size());
        std::cout << "batch_imgs.size() = " << batch_imgs.size() << std::endl;
        if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
            // predict
            auto ret = yolov8.Detect(batch_imgs, boxes);
            assert(0 == ret);
            std::cout << "detect result:" << std::endl;
            for (int i = 0; i < batch_size; i++) {
                yolov8.draw_result(batch_mats[i], boxes[i]);
                string img_file = "results/images/" + batch_names[i];
                cv::imwrite(img_file, batch_mats[i]);
                std::cout << "save result to " << img_file << std::endl;
                vector<json> bboxes_json;
                if (boxes[i].size() != 0) {
                    for (auto bbox : boxes[i]) {
                        float bboxwidth = bbox.x2 - bbox.x1;
                        float bboxheight = bbox.y2 - bbox.y1;
                        json bbox_json;
                        bbox_json["category_id"] = bbox.class_id;
                        bbox_json["score"] = bbox.score;
                        bbox_json["bbox"] = {bbox.x1, bbox.y1, bboxwidth, bboxheight};
                        bboxes_json.push_back(bbox_json);
                    }
                }

                json res_json;
                res_json["bboxes"] = bboxes_json;
                res_json["image_name"] = batch_names[i];
                results_json.push_back(res_json);

                bm_image_destroy(batch_imgs[i]);
            }
            batch_mats.clear();
            batch_imgs.clear();
            batch_names.clear();
            boxes.clear();
        }


    }
    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    yolov8_ts.calbr_basetime(base_time);
    yolov8_ts.build_timeline("yolov8 test");
    yolov8_ts.show_summary("yolov8 test");
    yolov8_ts.clear();
    return 0;
}
*/
#ifdef __cplusplus
}
#endif

static int load_batch_bm_images(string input_path, vector<cv::Mat>& batch_mats, vector<bm_image>& batch_imgs, vector<string>& batch_names){
    int dev_id = 0;
    vector<string> files_vector;
    DIR* pDir;
    struct dirent* ptr;
    pDir = opendir(input_path.c_str());
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input_path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);

    int cn = files_vector.size();
    int id = 0;
    std::sort(files_vector.begin(), files_vector.end());
    for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
        string img_file = *iter;
        id++;
        cout << id << "/" << cn << ", img_file: " << img_file << endl;
        bm_image bmimg;
        cv::Mat mat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
        if(mat.empty()){
            cout << "Decode error! Skipping current img." << endl;
            continue;
        }
        cv::bmcv::toBMI(mat, &bmimg);
        size_t index = img_file.rfind("/");
        string img_name = img_file.substr(index + 1);
        batch_mats.push_back(mat);
        batch_imgs.push_back(bmimg);
        batch_names.push_back(img_name);
    }
    return 0;
}