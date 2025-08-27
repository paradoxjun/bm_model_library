//===----------------------------------------------------------------------===//
//
// PosePointNet — 双输入时序骨架二分类（BMRuntime 封装）
// 公共实现：类 + 通用 C 接口(init + inference)
//
// 依赖：models/pose_pointnet.hpp, model_params.hpp, model_func.hpp
//
//===----------------------------------------------------------------------===//
#include "models/pose_pointnet.hpp"
#include "model_params.hpp"
#include "model_func.hpp"   // 为了通用 C 接口声明
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace {
constexpr float KPT_CONF_THR = 0.25f;   // 与训练一致，固定在 CPP
inline float safe_div(float a, float b, float eps = 1e-6f) { return (std::fabs(b) > eps) ? (a / b) : 0.f; }
} // namespace

// ========== 一次性片段推理 ==========
int PosePointNet::RunClip(const std::vector<object_pose_result_list>& pose_seq,
                          const std::vector<object_detect_result_list>& det_seq,
                          std::vector<float>& logits) {
  if (pose_seq.size() != det_seq.size()) {
    std::cerr << "[PosePointNet] RunClip: pose_seq.size()!=det_seq.size()" << std::endl;
    return -1;
  }
  if (!m_bmNetwork) {
    std::cerr << "[PosePointNet] RunClip: m_bmNetwork is null" << std::endl;
    return -2;
  }

  std::vector<float> xg_host, xl_host;
  int ret = pre_process_xgxl_(pose_seq, det_seq, xg_host, xl_host);
  if (ret != 0) return ret;

  logits.assign(m_num_class, 0.f);
  ret = forward_(xg_host, xl_host, logits);
  return ret;
}

// ========== 流式（可选） ==========
int PosePointNet::AccumulateAndRun(const object_pose_result_list& pose_frame,
                                   const object_detect_result_list& det_frame,
                                   std::vector<float>& logits) {
  m_pose_win.emplace_back(pose_frame);
  m_det_win.emplace_back(det_frame);
  while ((int)m_pose_win.size() > m_T) m_pose_win.pop_front();
  while ((int)m_det_win.size()  > m_T) m_det_win.pop_front();

  if ((int)m_pose_win.size() < m_T) return 0;

  std::vector<object_pose_result_list> pose_seq(m_pose_win.begin(), m_pose_win.end());
  std::vector<object_detect_result_list> det_seq(m_det_win.begin(), m_det_win.end());
  int ret = RunClip(pose_seq, det_seq, logits);

  if (!m_pose_win.empty()) m_pose_win.pop_front();
  if (!m_det_win.empty())  m_det_win.pop_front();
  return (ret == 0) ? 1 : ret;
}

// ========== 预处理 ==========
int PosePointNet::pre_process_xgxl_(const std::vector<object_pose_result_list>& pose_seq,
                                    const std::vector<object_detect_result_list>& det_seq,
                                    std::vector<float>& xg_host,
                                    std::vector<float>& xl_host) const {
  const int L = static_cast<int>(pose_seq.size());
  const int use_frames = std::min(L, m_T);

  xg_host.assign(m_CG * m_T * m_V * m_P, 0.f);
  xl_host.assign(m_CL * m_T * m_V * m_P, 0.f);

  for (int t = 0; t < use_frames; ++t) {
    const auto& poses = pose_seq[t];
    const auto& dets  = det_seq[t];

    const int n_pose = std::max(0, poses.count);
    const int n_det  = std::max(0, dets.count);
    const int use_p  = std::max(0, std::min({m_P, n_pose, n_det}));

    for (int p = 0; p < use_p; ++p) {
      const auto& pose_obj = poses.results[p];
      const auto& det_obj  = dets.results[p];

      if (pose_obj.points_num != m_V) {
        std::cerr << "[PosePointNet] pre_process_xgxl_: points_num(" << pose_obj.points_num
                  << ") != V(" << m_V << ") at t=" << t << " p=" << p << std::endl;
        return -3;
      }

      const image_rect_t& box = det_obj.box;
      const float bw = std::max(1.f, float(box.right  - box.left));
      const float bh = std::max(1.f, float(box.bottom - box.top));

      for (int v = 0; v < m_V; ++v) {
        const float x = pose_obj.points[v * 3 + 0];
        const float y = pose_obj.points[v * 3 + 1];
        const float c = pose_obj.points[v * 3 + 2];

        if (c < KPT_CONF_THR) continue;

        // 全局归一化
        float gx = clamp01_(safe_div(x, (float)m_img_w));
        float gy = clamp01_(safe_div(y, (float)m_img_h));

        // 局部归一化（框内有效）
        float lx = 0.f, ly = 0.f, lc = 0.f;
        if (inside_(x, y, box)) {
          lx = clamp01_(safe_div(x - box.left, bw));
          ly = clamp01_(safe_div(y - box.top,  bh));
          lc = c;
        }

        // xg: ch0=gx, ch1=gy, ch2=c
        if (m_CG >= 1) xg_host[lin_idx_(0, t, v, p, m_CG, m_T, m_V, m_P)] = gx;
        if (m_CG >= 2) xg_host[lin_idx_(1, t, v, p, m_CG, m_T, m_V, m_P)] = gy;
        if (m_CG >= 3) xg_host[lin_idx_(2, t, v, p, m_CG, m_T, m_V, m_P)] = c;

        // xl: ch0=lx, ch1=ly, ch2=lc
        if (m_CL >= 1) xl_host[lin_idx_(0, t, v, p, m_CL, m_T, m_V, m_P)] = lx;
        if (m_CL >= 2) xl_host[lin_idx_(1, t, v, p, m_CL, m_T, m_V, m_P)] = ly;
        if (m_CL >= 3) xl_host[lin_idx_(2, t, v, p, m_CL, m_T, m_V, m_P)] = lc;
      }
    }
  }
  return 0;
}

// ========== 前向 ==========
int PosePointNet::forward_(const std::vector<float>& xg_host,
                           const std::vector<float>& xl_host,
                           std::vector<float>& logits) {
  if (!m_bmNetwork) return -10;

  bm_handle_t h = m_bmContext->handle();

  // 输入 0：xg
  auto in0 = m_bmNetwork->inputTensor(0);
  in0->set_shape_by_dim(0, 1);
  const size_t bytes_xg = xg_host.size() * sizeof(float);
  bm_device_mem_t mem_xg;
  if (bm_malloc_device_byte(h, &mem_xg, bytes_xg) != BM_SUCCESS) {
    std::cerr << "[PosePointNet] forward_: alloc xg failed" << std::endl;
    return -11;
  }
  if (bm_memcpy_s2d_partial(h, mem_xg, (void*)xg_host.data(), bytes_xg) != BM_SUCCESS) {
    bm_free_device(h, mem_xg);
    std::cerr << "[PosePointNet] forward_: copy xg failed" << std::endl;
    return -12;
  }
  in0->set_device_mem(&mem_xg);

  // 输入 1：xl
  auto in1 = m_bmNetwork->inputTensor(1);
  in1->set_shape_by_dim(0, 1);
  const size_t bytes_xl = xl_host.size() * sizeof(float);
  bm_device_mem_t mem_xl;
  if (bm_malloc_device_byte(h, &mem_xl, bytes_xl) != BM_SUCCESS) {
    bm_free_device(h, mem_xg);
    std::cerr << "[PosePointNet] forward_: alloc xl failed" << std::endl;
    return -13;
  }
  if (bm_memcpy_s2d_partial(h, mem_xl, (void*)xl_host.data(), bytes_xl) != BM_SUCCESS) {
    bm_free_device(h, mem_xg);
    bm_free_device(h, mem_xl);
    std::cerr << "[PosePointNet] forward_: copy xl failed" << std::endl;
    return -14;
  }
  in1->set_device_mem(&mem_xl);

  // 前向
  LOG_TS(ts_, "pose_pointnet inference");
  int ret = m_bmNetwork->forward();
  LOG_TS(ts_, "pose_pointnet inference");

  // 释放输入显存
  bm_free_device(h, mem_xg);
  bm_free_device(h, mem_xl);

  if (ret != 0) {
    std::cerr << "[PosePointNet] forward_: forward failed" << std::endl;
    return -15;
  }

  // 读取输出
  auto out = m_bmNetwork->outputTensor(0);
  float* out_fp32 = out->get_cpu_data();
  if (!out_fp32) {
    std::cerr << "[PosePointNet] forward_: output is null" << std::endl;
    return -16;
  }
  logits.assign(out_fp32, out_fp32 + m_num_class);
  return 0;
}

//===================== 通用 C 接口（需要加入 model_func.hpp 的声明） =====================
#ifdef __cplusplus
extern "C" {
#endif

// model_func.hpp 应新增：
// PosePointNet* init_pose_pointnet_model(std::string bmodel_file, int dev_id,
//     model_posepointnet_inference_params params, std::vector<std::string> model_class_names);
// cls_result inference_pose_pointnet_model(PosePointNet model,
//     const std::vector<object_pose_result_list>& pose_seq,
//     const std::vector<object_detect_result_list>& det_seq,
//     bool enable_logger);

static inline void softmax_inplace(std::vector<float>& v) {
  if (v.empty()) return;
  float m = *std::max_element(v.begin(), v.end());
  float s = 0.f;
  for (float& x : v) { x = std::exp(x - m); s += x; }
  float inv = (s > 0.f) ? 1.f / s : 0.f;
  for (float& x : v) x *= inv;
}

PosePointNet* init_pose_pointnet_model(std::string bmodel_file,
                                       int dev_id,
                                       model_posepointnet_inference_params params,
                                       std::vector<std::string> model_class_names) {
  auto handle = std::make_shared<BMNNHandle>(dev_id);
  auto ctx    = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());

  // 若启用“从帧读尺寸”，但当前接口没有图像帧对象，这里用 params 的兜底尺寸
  int img_w = (params.img_w > 0) ? (int)params.img_w : 1920;
  int img_h = (params.img_h > 0) ? (int)params.img_h : 1080;
  if (params.img_size_from_frame) {
    std::cout << "[init_pose_pointnet_model] img_size_from_frame=true，但本接口无帧对象；"
                 "使用 params.img_w/img_h = " << img_w << "x" << img_h << " 归一化。" << std::endl;
  }

  auto* model = new PosePointNet(ctx, img_w, img_h, /*img_size_from_frame=*/false);
  // <<< 把 YAML 传入的类别名写进模型 >>>
  model->set_class_names(model_class_names);
  return model;
}

cls_result inference_pose_pointnet_model(PosePointNet model,
                                         const std::vector<object_pose_result_list>& pose_seq,
                                         const std::vector<object_detect_result_list>& det_seq,
                                         bool enable_logger) {
  cls_result out{};
  std::vector<float> logits;
  int ret = model.RunClip(pose_seq, det_seq, logits);
  if (ret != 0) {
    std::cerr << "[inference_pose_pointnet_model] RunClip failed, code=" << ret << std::endl;
    out.class_id = -1; out.score = 0.f;
    return out;
  }

  softmax_inplace(logits);
  int argmax = 0; float best = logits.empty() ? 0.f : logits[0];
  for (int i = 1; i < (int)logits.size(); ++i) if (logits[i] > best) { best = logits[i]; argmax = i; }
  out.class_id = argmax;
  out.score    = best;

  if (enable_logger) {
    const auto& names = model.class_names();
    std::cout << "[pose_pointnet] net=" << model.net_name()
              << "  T/V/P=" << model.T() << "/" << model.V() << "/" << model.P()
              << "  num_class=" << model.num_class() << "\n  probs: ";
    for (size_t i = 0; i < logits.size(); ++i) {
      if (i) std::cout << ", ";
      if (i < names.size()) std::cout << names[i] << "=";
      std::cout << logits[i];
    }
    std::cout << "\n=> pred=" << out.class_id;
    if (out.class_id >= 0 && out.class_id < (int)names.size())
      std::cout << " (" << names[out.class_id] << ")";
    std::cout << "  score=" << out.score << std::endl;
  }
  return out;
}

#ifdef __cplusplus
}
#endif
