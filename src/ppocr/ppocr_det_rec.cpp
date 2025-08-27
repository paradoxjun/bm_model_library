#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include "json.hpp"
#include "ppocr_det.hpp"
#include "ppocr_cls.hpp"
#include "ppocr_rec.hpp"
#include "ff_decode.hpp"
#include "model_func.hpp"
using json = nlohmann::json;
using namespace std;
#define USE_ANGLE_CLS 0
#define USE_OPENCV_WARP 0
#define USE_OPENCV_DECODE 0
//from PPaddleOCR github.

static bm_image get_rotate_crop_image(bm_handle_t handle, bm_image input_bmimg_planar, OCRBox box) { 
    int crop_width = max((int)sqrt(pow(box.x1 - box.x2, 2) + pow(box.y1 - box.y2, 2)),
                        (int)sqrt(pow(box.x3 - box.x4, 2) + pow(box.y3 - box.y4, 2)));
    int crop_height = max((int)sqrt(pow(box.x1 - box.x4, 2) + pow(box.y1 - box.y4, 2)),
                        (int)sqrt(pow(box.x3 - box.x2, 2) + pow(box.y3 - box.y2, 2)));
    // legality bounding
    crop_width = min(max(16, crop_width), input_bmimg_planar.width);
    crop_height = min(max(16, crop_height), input_bmimg_planar.height);

    bmcv_perspective_image_coordinate coord;
    coord.coordinate_num = 1;
    shared_ptr<bmcv_perspective_coordinate> coord_data = make_shared<bmcv_perspective_coordinate>();
    coord.coordinate = coord_data.get();
    coord.coordinate->x[0] = box.x1;
    coord.coordinate->y[0] = box.y1;
    coord.coordinate->x[1] = box.x2;
    coord.coordinate->y[1] = box.y2;
    coord.coordinate->x[2] = box.x4;
    coord.coordinate->y[2] = box.y4;
    coord.coordinate->x[3] = box.x3;
    coord.coordinate->y[3] = box.y3;

    bm_image crop_bmimg;
    bm_image_create(handle, crop_height, crop_width, input_bmimg_planar.image_format, input_bmimg_planar.data_type, &crop_bmimg);
    auto ret = bmcv_image_warp_perspective_with_coordinate(handle, 1, &coord, &input_bmimg_planar, &crop_bmimg, 0);
    assert(BM_SUCCESS == ret);//bilinear interpolation.

    if ((float)crop_height / crop_width < 1.5) {
        return crop_bmimg;
    } else {
        bm_image rot_bmimg;
        bm_image_create(handle, crop_width, crop_height, input_bmimg_planar.image_format, input_bmimg_planar.data_type,
                        &rot_bmimg);

        cv::Point2f center = cv::Point2f(crop_width / 2.0, crop_height / 2.0);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, -90.0, 1.0);
        bmcv_affine_image_matrix matrix_image;
        matrix_image.matrix_num = 1;
        std::shared_ptr<bmcv_affine_matrix> matrix_data = std::make_shared<bmcv_affine_matrix>();
        matrix_image.matrix = matrix_data.get();
        matrix_image.matrix->m[0] = rot_mat.at<double>(0, 0);
        matrix_image.matrix->m[1] = rot_mat.at<double>(0, 1);
        matrix_image.matrix->m[2] = rot_mat.at<double>(0, 2) - crop_height / 2.0 + crop_width / 2.0;
        matrix_image.matrix->m[3] = rot_mat.at<double>(1, 0);
        matrix_image.matrix->m[4] = rot_mat.at<double>(1, 1);
        matrix_image.matrix->m[5] = rot_mat.at<double>(1, 2) - crop_height / 2.0 + crop_width / 2.0;
        ret = bmcv_image_warp_affine(handle, 1, &matrix_image, &crop_bmimg, &rot_bmimg, 0);
        assert(BM_SUCCESS == ret);//bilinear interpolation
        bm_image_destroy(crop_bmimg);
        return rot_bmimg;
    }
}

static void visualize_boxes(bm_image input_bmimg, OCRBoxVec& ocr_result, const string& save_path, float rec_thresh) {
    cv::Mat img_src;
    cv::bmcv::toMAT(&input_bmimg, img_src);
    // cv::Mat img_res(input_bmimg.height, input_bmimg.width, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int n = 0; n < ocr_result.size(); n++) {
        if (ocr_result[n].rec_res.empty())
            continue;
        cv::Point rook_points[4];
        rook_points[0] = cv::Point(int(ocr_result[n].x1), int(ocr_result[n].y1));
        rook_points[1] = cv::Point(int(ocr_result[n].x2), int(ocr_result[n].y2));
        rook_points[2] = cv::Point(int(ocr_result[n].x3), int(ocr_result[n].y3));
        rook_points[3] = cv::Point(int(ocr_result[n].x4), int(ocr_result[n].y4));

        const cv::Point* ppt[1] = {rook_points};
        int npt[] = {4};
        string label = ocr_result[n].rec_res;
        if (label != "###" && ocr_result[n].score > rec_thresh) {
            cv::polylines(img_src, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
            std::cout << label << "; ";
            // cv::putText(img_src, label, cv::Point(int(ocr_result[n].x1), int(ocr_result[n].y1)),
            // cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
        }
    }
    cv::imwrite(save_path, img_src);
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif

// PPOCR_Detector* init_ppocr_det_model(string bmodel_file, int dev_id){
//     // creat handle
//     BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
//     cout << "set device id: "  << dev_id << endl;
//     bm_handle_t h = handle->handle();
//     // load bmodel
//     shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());
//     cout << "load bmodel: " << bmodel_file << endl;
//     return new PPOCR_Detector(bm_ctx);
// }

// PPOCR_Rec* init_ppocr_rec_model(string bmodel_file, int dev_id){
//     // creat handle
//     BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
//     cout << "set device id: "  << dev_id << endl;
//     bm_handle_t h = handle->handle();
//     // load bmodel
//     shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());
//     cout << "load bmodel: " << bmodel_file << endl;
//     return new PPOCR_Rec(bm_ctx);
// }

// int inference_ppocr_det_rec_model(PPOCR_Detector ppocr_det, PPOCR_Rec ppocr_rec) {
ocr_result_list inference_ppocr_det_rec_model(string bmodel_det, string bmodel_rec, cv::Mat input_image, bool enable_logger=false) {
    // ppocr_cls model unsupport now.
    std::cout << "start ppocr_det_rec_model" << std::endl;
    // string img_file = "tests/ppocr_test_image.jpg";
    // string bmodel_det = "models/ch_PP-OCRv4_det_fp16.bmodel";
    // string bmodel_rec = "models/ch_PP-OCRv4_rec_fp16.bmodel";
    string label_names = "src/ppocr/ppocr_keys_v1.txt";
    float rec_thresh = 0.2;
    int dev_id = 0;
    bool beam_search = true;
    int beam_size = 3;

    // creat handle
    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
    cout << "set device id: " << dev_id << endl;
    bm_handle_t h = handle->handle();

    // // Load bmodel
    std::shared_ptr<BMNNContext> bm_ctx_det = std::make_shared<BMNNContext>(handle, bmodel_det.c_str());
    PPOCR_Detector ppocr_det(bm_ctx_det);
    CV_Assert(0 == ppocr_det.Init());

    std::shared_ptr<BMNNContext> bm_ctx_rec = std::make_shared<BMNNContext>(handle, bmodel_rec.c_str());
    PPOCR_Rec ppocr_rec(bm_ctx_rec);
    CV_Assert(0 == ppocr_rec.Init(label_names.c_str()));

    std::cout << "load bmodel success" << std::endl;
    std::cout << "det batch size: " << ppocr_det.batch_size() << std::endl;
    // std::cout << "rec batch size: " << ppocr_rec.max_batch << std::endl;
    TimeStamp ts;
    ppocr_det.enableProfile(&ts);
    ppocr_rec.enableProfile(&ts);

    vector<bm_image> batch_imgs;
    vector<OCRBoxVec> batch_boxes;
    vector<pair<int, int>> batch_ids;
    vector<bm_image> batch_crops;
    ocr_result_list ocr_result;
    // Read image
    bm_image bmimg;  
    // cv::Mat cvmat = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);
    std::cout << "ocr detect start" << std::endl;
    CV_Assert(0 == ppocr_det.run(batch_imgs, batch_boxes));
    std::cout << "ocr detect success" << std::endl;
    bm_image input_bmimg_planar;
    bm_image_create(h, batch_imgs[0].height, batch_imgs[0].width, FORMAT_BGR_PLANAR,
                    batch_imgs[0].data_type, &input_bmimg_planar);
    auto ret = bmcv_image_vpp_convert(h, 1, batch_imgs[0], &input_bmimg_planar);

    bm_image_destroy(batch_imgs[0]);
    batch_imgs[0] = input_bmimg_planar;
    std::cout << "original image: " << batch_imgs[0].height << " " << batch_imgs[0].width << std::endl;

    // for (int j = 0; j < batch_boxes[0].size(); j++) {
    //     batch_boxes[0][j].printInfo();
    //     bm_image crop_bmimg = get_rotate_crop_image(h, input_bmimg_planar, batch_boxes[0][j]);
    //     batch_crops.push_back(crop_bmimg);
    //     batch_ids.push_back(std::make_pair(0, j));
    // }
    int idx = 0;
    for (int j = 0; j < batch_boxes[0].size(); j++) {
        bm_image crop_bmimg = get_rotate_crop_image(h, input_bmimg_planar, batch_boxes[0][j]);
        batch_crops.push_back(crop_bmimg);
        batch_ids.push_back(std::make_pair(0, j));
        // idx ++;
        // TODO: 增加这部分旋转代码后，文字识别不到了，需要debug ？？？
        // cv::Mat src_cv_img;
        // cv::Mat dst_cv_img;
        // bm_image rotate_bmimg;
        // cv::bmcv::toMAT(&crop_bmimg, src_cv_img);
        // // cv_img旋转90度
        // cv::rotate(src_cv_img, dst_cv_img, cv::ROTATE_90_CLOCKWISE);
        // cv::bmcv::toBMI(dst_cv_img, &rotate_bmimg);
        // batch_crops.push_back(rotate_bmimg);
        // batch_ids.push_back(std::make_pair(0, j));
    }
    CV_Assert(0 == ppocr_rec.run(batch_crops, batch_boxes, batch_ids, beam_search, beam_size));
    if (enable_logger){
        std::cout << "ocr rec success" << std::endl;
        for (int i = 0; i < batch_boxes.size(); i++) {
            string save_file = "result.jpg";
            std::cout << "detect results: ";
            visualize_boxes(batch_imgs[i], batch_boxes[i], save_file.c_str(), rec_thresh);
        }
    }

    return ocr_result;
}
#ifdef __cplusplus
}
#endif