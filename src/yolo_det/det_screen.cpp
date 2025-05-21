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

// TODO： 手机检测需要考虑到目标过小的问题，针对低分（0.2-0.5）的phone，增加patch检测---ing
// static void bmcv_crop(bm_image& input, bm_image& output, vector<int> img_size, vector<int> roi){
//     int channel = 3;
//     int in_w = img_size[0];
//     int in_h = img_size[1];
//     int crop_x1 =  roi[0];
//     int crop_y1 =  roi[1];
//     int crop_x2 =  roi[2];
//     int crop_y2 =  roi[3];
//     int out_w = crop_x2 - crop_x1;
//     int out_h = crop_y2 - crop_y1;
//     int    dev_id = 0;
//     bm_handle_t handle;
//     bm_status_t dev_ret = bm_dev_request(&handle, dev_id);
//     std::shared_ptr<unsigned char> src_ptr(
//             new unsigned char[channel * in_w * in_h],
//             std::default_delete<unsigned char[]>());
//     std::shared_ptr<unsigned char> res_ptr(
//             new unsigned char[channel * out_w * out_h],
//             std::default_delete<unsigned char[]>());
//     unsigned char * src_data = src_ptr.get();
//     unsigned char * res_data = res_ptr.get();
//     for (int i = 0; i < channel * in_w * in_h; i++) {
//         src_data[i] = rand() % 255;
//     }
//     // calculate res
//     bmcv_rect_t crop_attr;
//     crop_attr.start_x   = crop_x1;
//     crop_attr.start_y   = crop_y1;
//     crop_attr.crop_w    = out_w;
//     crop_attr.crop_h    = out_h;
//     // bm_image input, output;
//     bm_image_create(handle,
//             in_h,
//             in_w,
//             FORMAT_RGB_PLANAR,
//             DATA_TYPE_EXT_1N_BYTE,
//             &input);
//     bm_image_alloc_dev_mem(input);
//     bm_image_copy_host_to_device(input, (void **)&src_data);
//     bm_image_create(handle,
//             out_h,
//             out_w,
//             FORMAT_RGB_PLANAR,
//             DATA_TYPE_EXT_1N_BYTE,
//             &output);
//     bm_image_alloc_dev_mem(output);
//     if (BM_SUCCESS != bmcv_image_crop(handle, 1, &crop_attr, input, &output)) {
//         std::cout << "bmcv_copy_to error !!!" << std::endl;
//         bm_image_destroy(input);
//         bm_image_destroy(output);
//         bm_dev_free(handle);
//         exit(-1);
//     }

//     bm_image_copy_device_to_host(output, (void **)&res_data);
//     std::cout << "bmcv_copy_to success !!!" << std::endl;
// }



object_detect_result_list inference_yolo_screen_det_model(YoloV8_det model, cv::Mat input_image, bool enable_logger=false){
    // single image inference
    int dev_id = 0;
    int ret = 0;
    int img_w = input_image.cols;
    int img_h = input_image.rows;
    // vector<int> img_size;
    // img_size.push_back(img_w);
    // img_size.push_back(img_h);
    vector<cv::Mat> batch_mats;
    vector<bm_image> batch_imgs;
    vector<YoloV8BoxVec> boxes;
    // vector<bm_image> batch_patch_imgs;
    // vector<int> batch_patch_x1;
    // vector<int> batch_patch_y1;
	batch_mats.push_back(input_image);  // only one image

    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);
    ret = model.Detect(batch_imgs, boxes);
    assert(0 == ret);

    // save result to object_detect_result_list
    object_detect_result_list result;
    result.count = boxes[0].size();
    int count = 0;
    // bool patch_detect = false;
    for (int i = 0; i < boxes[0].size(); i++){
        result.results[i].cls_id = boxes[0][i].class_id;
        result.results[i].prop = boxes[0][i].score;
        result.results[i].box.left = boxes[0][i].x1;
        result.results[i].box.top = boxes[0][i].y1;
        result.results[i].box.right = boxes[0][i].x2;
        result.results[i].box.bottom = boxes[0][i].y2;
        // if (boxes[0][i].score < conf){
        //     continue;
            // crop patch images
            // patch_detect = true;
            // int phone_x1 = boxes[0][i].x1;
            // int phone_y1 = boxes[0][i].y1;
            // int phone_x2 = boxes[0][i].x2;
            // int phone_y2 = boxes[0][i].y2;
            // int patch_w = 640;
            // int patch_h = 640;
            // int patch_x1 = std::max(phone_x1 - patch_w / 2, 0);
            // int patch_y1 = std::max(phone_y1 - patch_h / 2, 0);
            // int patch_x2 = std::min(phone_x1 + patch_w / 2, img_w);
            // int patch_y2 = std::min(phone_y1 + patch_h / 2, img_h);
            // cv::Mat cv_patch_img = input_image(cv::Rect(patch_x1, patch_y1, patch_w, patch_h));
            // cv::imwrite("patch_img.jpg", cv_patch_img);
            // bm_image patch_img;
            // cv::bmcv::toBMI(cv_patch_img, &patch_img);
            // batch_patch_imgs.push_back(patch_img);
            // batch_patch_x1.push_back(patch_x1);
            // batch_patch_y1.push_back(patch_y1);
        // }else {
        //     count ++;
        //     result.results[count-1].cls_id = boxes[0][i].class_id;
        //     result.results[count-1].prop = boxes[0][i].score;
        //     result.results[count-1].box.left = boxes[0][i].x1;
        //     result.results[count-1].box.top = boxes[0][i].y1;
        //     result.results[count-1].box.right = boxes[0][i].x2;
        //     result.results[count-1].box.bottom = boxes[0][i].y2;
        //     std::cout << "count: " << count << std::endl;
        // }
    }
    // TODO：patch detect again，需要考虑两次检测有重复的box的情况；
    // if (patch_detect){
    //     vector<YoloV8BoxVec> patch_boxes;
    //     ret = model.Detect(batch_patch_imgs, patch_boxes);
    //     for (int i = 0; i < patch_boxes[0].size(); i++){
    //         // std::cout << "patch_boxes[" << i << "]: " << patch_boxes[0][i].class_id << " " << patch_boxes[0][i].score << " " << patch_boxes[0][i].x1 << " " << patch_boxes[0][i].y1 << " " << patch_boxes[0][i].x2 << " " << patch_boxes[0][i].y2 << std::endl;
    //         if (patch_boxes[0][i].score >=conf){
    //             count ++;
    //             result.results[count].cls_id = patch_boxes[0][i].class_id;
    //             result.results[count].prop = patch_boxes[0][i].score;
    //             result.results[count].box.left = patch_boxes[0][i].x1 + batch_patch_x1[i];
    //             result.results[count].box.top = patch_boxes[0][i].y1 + batch_patch_y1[i];
    //             result.results[count].box.right = patch_boxes[0][i].x2 + batch_patch_x1[i];
    //             result.results[count].box.bottom = patch_boxes[0][i].y2 + batch_patch_y1[i];
    //         }
    //     }
    // }
    // result.count = count;
    if (enable_logger){
        std::cout << "detect result:" << std::endl;
        for (int i = 0; i < result.count; i++){
			std::cout << "result[" << i << "]: " << result.results[i].cls_id << " " << result.results[i].prop << " " << result.results[i].box.left << " " << result.results[i].box.top << " " << result.results[i].box.right << " " << result.results[i].box.bottom << std::endl;
		}
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
