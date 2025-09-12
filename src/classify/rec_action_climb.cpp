//===----------------------------------------------------------------------===//
//
// PosePointNet 任务封装：摔倒 / 翻越 等
// 作用：为具体任务声明 “期望 P/V/T/CG/CL”，做一次形状校验，再调用通用 inference
//
// 依赖：models/pose_pointnet.hpp, model_params.hpp, model_func.hpp
//
//===----------------------------------------------------------------------===//
#include "models/pose_pointnet.hpp"
#include "model_params.hpp"
#include "model_func.hpp"
#include <iostream>

//======================= 任务常量（可按任务分别调整） =======================
// 「摔倒检测」— 与 bmodel: best_acc_32T17V1P_fp16_1b.bmodel 对齐
static constexpr int FALL_T_EXPECT  = 32;
static constexpr int FALL_V_EXPECT  = 17;
static constexpr int FALL_P_EXPECT  = 1;
static constexpr int FALL_CG_EXPECT = 8;
static constexpr int FALL_CL_EXPECT = 3;


//========================== 工具：形状校验 ==========================
static bool validate_shapes(const PosePointNet& m,
                            int T, int V, int P, int CG, int CL,
                            const char* task_name) {
  bool ok = (m.T()==T && m.V()==V && m.P()==P && m.CG()==CG && m.CL()==CL);
  if (!ok) {
    std::cerr << "[pose_pointnet][" << task_name << "] shape mismatch!"
              << " (model T/V/P/CG/CL = "
              << m.T() << "/" << m.V() << "/" << m.P() << "/" << m.CG() << "/" << m.CL()
              << ", expect = "
              << T << "/" << V << "/" << P << "/" << CG << "/" << CL << ")\n";
  }
  return ok;
}

//========================== 任务入口：摔倒 ==========================
// 需要加入 model_func.hpp 的声明：
// cls_result inference_pose_pointnet_fall_model(PosePointNet model,
//     const std::vector<object_pose_result_list>& pose_seq,
//     const std::vector<object_detect_result_list>& det_seq,
//     bool enable_logger);
extern "C" cls_result
inference_pose_pointnet_climb_model(PosePointNet model,
                                    const std::vector<object_pose_result_list>& pose_seq,
                                    const std::vector<object_detect_result_list>& det_seq,
                                    bool enable_logger) {
  if (!validate_shapes(model,
                       FALL_T_EXPECT, FALL_V_EXPECT, FALL_P_EXPECT,
                       FALL_CG_EXPECT, FALL_CL_EXPECT,
                       "fall"))
  {
    cls_result bad{}; bad.class_id = -1; bad.score = 0.f; return bad;
  }
  // 直接复用通用 inference
  return inference_pose_pointnet_model(model, pose_seq, det_seq, enable_logger);
}
