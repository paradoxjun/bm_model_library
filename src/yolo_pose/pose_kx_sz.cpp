#include <fstream>
#include <string>
#include "yolov8_pose.hpp"
#include "model_func.hpp"

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif


object_pose_result_list inference_yolov8_kx_sz_pose_model(YoloV8_pose model, cv::Mat input_image, bool enable_logger){
    int ret = 0;
    vector<bm_image> batch_imgs;
    vector<poseBoxVec> boxes;

    bm_image bmimg;
    cv::bmcv::toBMI(input_image, &bmimg);
    batch_imgs.push_back(bmimg);
    ret = model.Detect(batch_imgs, boxes);

    object_pose_result_list result{};
    if (ret != 0){
        std::cout << "inference_yolov8_kx_sz_pose_model failed" << std::endl;
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

    // 2. 日志 + 可视化（OpenCV）
    if (enable_logger) {
        std::cout << "detect result:" << result.count << std::endl;

        // 连线表（横平款箱）
        static const std::vector<std::pair<int,int>> links = {
            {0,1},
        };

        for (int i = 0; i < result.count; ++i) {
            const auto& r = result.results[i];

            // ---- 打印 ----
            std::cout << "result[" << i << "]: " << r.score << "  "
                    << r.box.left << " " << r.box.top << " " << r.box.right << " " << r.box.bottom;

            for (int k = 0; k < r.points_num; ++k) {
                float x = r.points[k*3+0];
                float y = r.points[k*3+1];
                float c = r.points[k*3+2];
                std::cout << "  (" << x << "," << y << "," << c << ")";
            }
            std::cout << std::endl;

            // ---- 画检测框 ----
            cv::rectangle(input_image, cv::Point(r.box.left, r.box.top),
                        cv::Point(r.box.right, r.box.bottom), cv::Scalar(0,255,0), 2);

            // 分数
            cv::putText(input_image, cv::format("s:%.2f", r.score), cv::Point(r.box.left, std::max(0, r.box.top-5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);

            // ---- 关键点 & 连线 ----
            std::vector<cv::Point> pts;
            pts.reserve(r.points_num);

            for (int k = 0; k < r.points_num; ++k) {
                float conf = r.points[k*3+2];
                if (conf < 0.4f) {              // 过滤低置信度
                    pts.emplace_back(-1,-1);
                    continue;
                }
                int x = static_cast<int>(r.points[k*3+0]);
                int y = static_cast<int>(r.points[k*3+1]);
                pts.emplace_back(x,y);

                // 点颜色：红
                cv::circle(input_image, cv::Point(x,y), 3, cv::Scalar(0,0,255), -1);
            }

            // 画连线（蓝色）
            for (auto& lk : links) {
                int a = lk.first, b = lk.second;
                if (a >= pts.size() || b >= pts.size()) continue;
                if (pts[a].x < 0 || pts[b].x < 0) continue; // 无效点
                cv::line(input_image, pts[a], pts[b], cv::Scalar(255,0,0), 2); // 蓝
            }
        }

        // ---- 保存 ----
        std::string img_file = "pose_kx_sz_result.jpg";
        cv::imwrite(img_file, input_image);
        std::cout << "save result to " << img_file << std::endl;
    }

    return result;
}

#ifdef __cplusplus
}
#endif
