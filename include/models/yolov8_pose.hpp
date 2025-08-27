//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef YOLOV8_POSE_H
#define YOLOV8_POSE_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "../utils/bmnn_utils.h"
#include "../utils/utils.hpp"
#include "../utils/bm_wrapper.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1
#define DEBUG 0


struct poseBox {
    float x1, y1, x2, y2;
    float score;
    int index;
    std::vector<float> keyPoints;
};

using poseBoxVec = std::vector<poseBox>;

class YoloV8_pose {
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;

    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;

    //configuration
    float m_confThreshold= 0.25;
    float m_nmsThreshold = 0.7;
    int m_points_num = 17; 
    int mask_num = 0;
    int m_net_h, m_net_w;
    int max_batch;
    int output_num;
    int min_dim;
    int max_det=300;
    int max_wh=7680; // (pixels) maximum box width and height
    bmcv_convert_to_attr converto_attr;

    TimeStamp *m_ts;
    unsigned int m_colorIndex;

    private:
    int pre_process(const std::vector<bm_image>& images);
    int post_process(const std::vector<bm_image>& images, std::vector<poseBoxVec>& boxes);
    void get_max_value_neon(float* cls_conf,float &max_value ,int & max_index,int i,int nout);

    static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
    void NMS(poseBoxVec &dets, float nmsConfidence);
    void ReTransPoseBox(poseBoxVec& v, float tx, float ty, float r, int fw, int fh, float* d, int n);
    void ProcessPoseBox(poseBoxVec& v, float* d, int n);
    bmcv_color_t GetBmColor();
    public:
    YoloV8_pose(std::shared_ptr<BMNNContext> context);
    // NEW 重载：支持外部指定关键点个数
    YoloV8_pose(std::shared_ptr<BMNNContext> context, int kpt_nums);
    bm_handle_t getHandle() const;      // 取句柄

    virtual ~YoloV8_pose();
    int Init(float confThresh=0.5, float nmsThresh=0.5);
    void enableProfile(TimeStamp *ts);
    int batch_size();
    int Detect(const std::vector<bm_image>& images, std::vector<poseBoxVec>& boxes);
    void draw_bmcv(bm_handle_t& handle, poseBox& b, bm_image& frame, bool putScore);
    static const std::vector<std::pair<int, int>> pointLinks;

};

#endif //!YOLOV8_POSE_H