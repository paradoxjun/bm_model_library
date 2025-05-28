//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef RESNET_HPP
#define RESNET_HPP

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "../utils/bmnn_utils.h"
#include "../utils/bm_wrapper.hpp"
#include "../utils/utils.hpp"
// Define USE_OPENCV for enabling OPENCV related funtions in bm_wrapper.hpp
#define USE_OPENCV 1

class RESNET {
  std::shared_ptr<BMNNContext> m_bmContext;
  std::shared_ptr<BMNNNetwork> m_bmNetwork;
  std::vector<bm_image> m_resized_imgs;
  std::vector<bm_image> m_converto_imgs;

  // model info 
  int m_net_h;
  int m_net_w;
  int max_batch;
  int output_num;
  int class_num;
  bmcv_convert_to_attr converto_attr;

  // for profiling
  TimeStamp *ts_ = NULL;

  private:
  int pre_process(std::vector<bm_image> &images);
  int post_process(std::vector<bm_image> &images, std::vector<std::pair<int, float>> &results);
  static float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *alignWidth);
  
  public:
  // RESNET(std::shared_ptr<BMNNContext> context);
  RESNET(std::shared_ptr<BMNNContext> context): m_bmContext(context){
    std::cout << "RESNET()" << std::endl;
    //1. get network
    m_bmNetwork = m_bmContext->network(0);
    std::cout << "m_bmNetwork->maxBatch() = " << m_bmNetwork->maxBatch() << std::endl;
    //2. get input
    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];
    std::cout << "m_net_h = " << m_net_h << std::endl;
    //3. get output
    output_num = m_bmNetwork->outputTensorNum();
    assert(output_num > 0);
    class_num = m_bmNetwork->outputTensor(0)->get_shape()->dims[1];

    //4. initialize bmimages
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);
    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
    for(int i=0; i<max_batch; i++){
      auto ret= bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
      assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8){
      img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    // 5.converto
    float input_scale = tensor->get_scale();
    const std::vector<float> std = {0.229, 0.224, 0.225};
    const std::vector<float> mean = {0.485, 0.456, 0.406};
    converto_attr.alpha_0 = 1 / (255. * std[0]) * input_scale;
    converto_attr.alpha_1 = 1 / (255. * std[1]) * input_scale;
    converto_attr.alpha_2 = 1 / (255. * std[2]) * input_scale;
    converto_attr.beta_0 = (-mean[0] / std[0]) * input_scale;
    converto_attr.beta_1 = (-mean[1] / std[1]) * input_scale;
    converto_attr.beta_2 = (-mean[2] / std[2]) * input_scale;
  }
  virtual ~RESNET();
  // int Init();
  void enableProfile(TimeStamp *ts);
  int batch_size();
  int Classify(std::vector<bm_image>& input_images, std::vector<std::pair<int, float>>& results);
};

#endif /* RESNET_HPP */
