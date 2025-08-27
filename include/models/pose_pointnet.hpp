//===----------------------------------------------------------------------===//
//
// PosePointNet — 双输入时序骨架二分类（BMRuntime 封装）
// 仅头文件；实现见 pose_pointnet.cpp
//
// 依赖：bmnn_utils.h / bm_wrapper.hpp / utils.hpp / model_params.hpp
//
// 约定：
//   xg: [B, CG=8, T, V, P]   // 全局 (x/img_w, y/img_h, conf, 其余通道默认0)
//   xl: [B, CL=3, T, V, P]   // 局部 ((x-x1)/w, (y-y1)/h, conf)
//   输出 logits: [B, num_class]  // 本项目二分类: ["fall","other"]
//
// 预处理：
//   • 不做选人；人数<=P，不足的 p 维置 0
//   • L>T → 取前 T；L<T → 直接零填充
//   • 置信度<阈值 或 框外 → 三通道置 0
//
//===----------------------------------------------------------------------===//
#ifndef POSE_POINTNET_HPP
#define POSE_POINTNET_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "../utils/bmnn_utils.h"   // BMNNContext / BMNNNetwork / BMNNTensor
#include "../utils/bm_wrapper.hpp" // bm_handle_t, bm_* API
#include "../utils/utils.hpp"      // TimeStamp
#include "../model_params.hpp"        // object_* / cls_result

class PosePointNet {
public:
  // img_w/img_h 为像素宽高（用于全局归一化）
  explicit PosePointNet(std::shared_ptr<BMNNContext> context,
                        int img_w, int img_h,
                        bool img_size_from_frame=false)
  : m_bmContext(std::move(context)),
    m_img_w(img_w), m_img_h(img_h),
    m_use_img_from_frame(img_size_from_frame) {
    m_bmNetwork = m_bmContext->network(0);
    m_net_name_ = m_bmContext->network_name(0);
    init_from_bmodel_();
  }

  virtual ~PosePointNet() = default;

  // Profiling
  void enableProfile(TimeStamp* ts) { ts_ = ts; }

  // 查询网络信息
  int  batch_size()  const { return m_bmNetwork ? m_bmNetwork->maxBatch() : 1; }
  int  T()           const { return m_T; }
  int  P()           const { return m_P; }
  int  V()           const { return m_V; }
  int  CG()          const { return m_CG; }
  int  CL()          const { return m_CL; }
  int  num_class()   const { return m_num_class; }
  const std::string& net_name() const { return m_net_name_; }

  void set_image_size(int w, int h) { m_img_w = w; m_img_h = h; }

  // 类别名（来自 YAML；init 时设置）
  void set_class_names(const std::vector<std::string>& names) { m_class_names = names; }
  const std::vector<std::string>& class_names() const { return m_class_names; }

  // 一次性片段推理（部署主路径）
  // 要求：pose_seq.size() == det_seq.size()
  // 返回 0 成功；<0 失败；logits.size() == num_class
  int RunClip(const std::vector<object_pose_result_list>& pose_seq,
              const std::vector<object_detect_result_list>& det_seq,
              std::vector<float>& logits);

  // 逐帧积累（可不使用）：满 T 帧触发一次推理
  // 触发时返回 1 并填充 logits；未满 T 返回 0；失败 <0
  int AccumulateAndRun(const object_pose_result_list& pose_frame,
                       const object_detect_result_list& det_frame,
                       std::vector<float>& logits);

private:
  // 从 bmodel 输入/输出形状解析 T/P/V/CG/CL/num_class
  void init_from_bmodel_() {
    assert(m_bmNetwork && "BMNNNetwork is null");
    m_input_num  = 2;
    m_output_num = m_bmNetwork->outputTensorNum();

    // xg: [B, CG, T, V, P]
    auto in0 = m_bmNetwork->inputTensor(0);
    auto s0  = in0->get_shape();
    assert(s0->num_dims == 5 && "xg shape must be 5D [B,C,T,V,P]");
    m_CG = s0->dims[1];
    m_T  = s0->dims[2];
    m_V  = s0->dims[3];
    m_P  = s0->dims[4];

    // xl: [B, CL, T, V, P]
    auto in1 = m_bmNetwork->inputTensor(1);
    auto s1  = in1->get_shape();
    assert(s1->num_dims == 5 && "xl shape must be 5D [B,C,T,V,P]");
    m_CL = s1->dims[1];

    // logits: [B, num_class] 或 [1, num_class]
    auto out = m_bmNetwork->outputTensor(0);
    auto so  = out->get_shape();
    int classes = 1;
    for (int i = 0; i < so->num_dims; ++i) {
      if (i == 0) continue;
      classes *= so->dims[i];
    }
    m_num_class = classes;
  }

  // 预处理：pose/det → xg_host/xl_host（展平）
  int pre_process_xgxl_(const std::vector<object_pose_result_list>& pose_seq,
                        const std::vector<object_detect_result_list>& det_seq,
                        std::vector<float>& xg_host,
                        std::vector<float>& xl_host) const;

  // 前向：host→device→forward→device→host
  int forward_(const std::vector<float>& xg_host,
               const std::vector<float>& xl_host,
               std::vector<float>& logits);

  // 工具
  static inline bool inside_(float x, float y, const image_rect_t& r) {
    return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
  }
  static inline float clamp01_(float v) {
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
  }
  static inline int lin_idx_(int c, int t, int v, int p,
                             int C, int T, int V, int P) {
    return (((c*T + t)*V + v)*P + p);
  }

private:
  // BMRuntime
  std::shared_ptr<BMNNContext>  m_bmContext;
  std::shared_ptr<BMNNNetwork>  m_bmNetwork;

  // 模型信息
  std::string m_net_name_;
  int m_T  {32};
  int m_P  {1};
  int m_V  {17};
  int m_CG {8};
  int m_CL {3};
  int m_num_class {2};
  int m_input_num  {2};
  int m_output_num {1};

  // 图像尺寸
  int  m_img_w {1920};
  int  m_img_h {1080};
  bool m_use_img_from_frame {false}; // 仅保存标志（当前无帧对象，无法自动读取）

  // 类别名（从 YAML 注入）
  std::vector<std::string> m_class_names;

  // 流式窗口
  std::deque<object_pose_result_list>   m_pose_win;
  std::deque<object_detect_result_list> m_det_win;

  // profiling
  TimeStamp* ts_ {nullptr};
};

#endif // POSE_POINTNET_HPP
