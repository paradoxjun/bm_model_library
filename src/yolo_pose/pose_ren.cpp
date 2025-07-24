#include <fstream>
#include <string>
#include "yolov8_pose.hpp"
#include "model_func.hpp"

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

YoloV8_pose* init_yolov8_pose_model(std::string bmodel_file, int dev_id, model_pose_inference_params params, std::vector<std::string> model_class_names)
{
    auto handle = std::make_shared<BMNNHandle>(dev_id);
    auto ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    auto* model = new YoloV8_pose(ctx, params.kpt_nums);
    model->Init(params.box_threshold, params.nms_threshold);
    // std::cout << "[init_yolov8_pose_model] load success: " << bmodel_file << "  (kpt=" << params.kpt_nums << ")\n";
    return model;
}

object_pose_result_list inference_yolov8_ren_pose_model(YoloV8_pose model, cv::Mat input_image, bool enable_logger){
    int ret = 0;
    vector<bm_image> batch_imgs;
    vector<poseBoxVec> boxes;

    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);
    ret = model.Detect(batch_imgs, boxes);

    object_pose_result_list result{};
    if (ret != 0){
        std::cout << "inference_yolov8_ren_pose_model failed" << std::endl;
        result.count = -1;
        return result;
    }
     
    // 1. 拷贝数据到结果结构体
    result.count = std::min<int>(boxes[0].size(), OBJ_NUMB_MAX_SIZE);
    for (int i = 0; i < result.count; ++i) {
        const poseBox& src = boxes[0][i];
        auto& dst = result.results[i];

        dst.box.left = src.x1;
        dst.box.top = src.y1;
        dst.box.right = src.x2;
        dst.box.bottom = src.y2;
        dst.score = src.score;

        dst.points_num = src.keyPoints.size() / 3;
        dst.points = src.keyPoints;          // std::vector 一步拷贝
    }

    // 2. 日志 + 可视化（完全用 OpenCV）
    if (enable_logger) {
        std::cout << "detect result:" << result.count << std::endl;

        // ⬇ 用 boxes[0] 直接在 input_image 上绘制
        for (int i = 0; i < result.count; ++i) {
            const auto& r = result.results[i];
            const auto& pbox = boxes[0][i];

            // ---- 终端打印 ----
            std::cout << "result[" << i << "]: " << r.score << "  "
                      << r.box.left << " " << r.box.top << " "
                      << r.box.right << " " << r.box.bottom;

            for (int k = 0; k < r.points_num; ++k) {
                float x = r.points[k * 3 + 0];
                float y = r.points[k * 3 + 1];
                float c = r.points[k * 3 + 2];
                std::cout << "  (" << x << "," << y << "," << c << ")";
            }
            std::cout << std::endl;

            // ---- OpenCV 画检测框 ----
            cv::rectangle(input_image, cv::Point(r.box.left, r.box.top), cv::Point(r.box.right, r.box.bottom), 
                          cv::Scalar(0, 255, 0), 2);

            // 分数
            cv::putText(input_image, cv::format("s:%.2f", r.score), cv::Point(r.box.left, std::max(0, r.box.top - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

            // ---- 画关键点 + 连线 ----
            // 取连线表：YoloV8_pose::pointLinks 是 public static
            const auto& links = YoloV8_pose::pointLinks;

            // 先把关键点收成 cv::Point，方便画线
            std::vector<cv::Point> pts;
            for (int k = 0; k < r.points_num; ++k) {
                float conf = r.points[k * 3 + 2];
                if (conf < 0.4f) {
                    pts.emplace_back(-1, -1);   // 记一个非法点位
                    continue;
                }
                int x = static_cast<int>(r.points[k * 3 + 0]);
                int y = static_cast<int>(r.points[k * 3 + 1]);
                pts.emplace_back(x, y);

                // 实心圆表示关键点
                cv::circle(input_image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
            }

            // 连线
            for (auto& lk : links) {
                int a = lk.first;
                int b = lk.second;
                if (a >= pts.size() || b >= pts.size()) continue;
                if (pts[a].x < 0 || pts[b].x < 0)        continue;   // 跳过无效点
                cv::line(input_image, pts[a], pts[b],
                        cv::Scalar(255, 0, 0), 1);
            }
        }

        // ---- 保存结果 ----
        std::string img_file = "pose_ren_result.jpg";
        cv::imwrite(img_file, input_image);
        std::cout << "save result to " << img_file << std::endl;
    }

    return result;
}

#ifdef __cplusplus
}
#endif
