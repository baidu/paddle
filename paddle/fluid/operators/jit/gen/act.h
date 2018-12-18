/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <string>
#include "glog/logging.h"
#include "paddle/fluid/operators/jit/gen/jitcode.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

extern const float exp_float_consts[];
extern const int exp_int_0x7f[];
extern int g_tmp_mem[];

#define EXP_HIG 88.3762626647949f
#define EXP_LOW -88.3762626647949f
#define CEPHES_LOG2EF 1.44269504088896341
#define CEPHES_EXP_C1 0.693359375
#define CEPHES_EXP_C2 -2.12194440e-4
#define CEPHES_EXP_P0 1.9875691500E-4
#define CEPHES_EXP_P1 1.3981999507E-3
#define CEPHES_EXP_P2 8.3334519073E-3
#define CEPHES_EXP_P3 4.1665795894E-2
#define CEPHES_EXP_P4 1.6666665459E-1
#define CEPHES_EXP_P5 5.0000001201E-1

#define REPEAT_8TIMES(val) val, val, val, val, val, val, val, val

#define OFFSET_EXP_ONE 0 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_TWO 1 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_0P5 2 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_HIG 3 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_LOW 4 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_LOG2EF 5 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_C1 6 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_C2 7 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P0 8 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P1 9 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P2 10 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P3 11 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P4 12 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P5 13 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_MAX_INPUT 14 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_SIGMOID_MAX 15 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_SIGMOID_MIN 16 * YMM_FLOAT_BLOCK * sizeof(float)

class VActJitCode : public JitCode {
 public:
  explicit VActJitCode(int d, operand_type type, size_t code_size,
                       void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), num_(d), type_(type) {
    if (!(type_ == operand_type::relu || type_ == operand_type::exp ||
          type_ == operand_type::sigmoid || type_ == operand_type::tanh ||
          type_ == operand_type::identity)) {
      LOG(FATAL) << "Do not support this operand type: " << type_;
    }
    this->genCode();
  }

  const char* name() const override {
    std::string base = "VActJitCode";
    switch (type_) {
      case operand_type::relu:
        base += "_Relu";
        break;
      case operand_type::exp:
        base += "_Exp";
        break;
      case operand_type::sigmoid:
        base += "_Sigmoid";
        break;
      case operand_type::tanh:
        base += "_Tanh";
        break;
      case operand_type::identity:
        base += "_Identity";
        break;
      default:
        break;
    }
    return base.c_str();
  }
  void genCode() override;

 protected:
  // compute relu with ymm, xmm
  template <typename JMM>
  void relu_jmm(JMM& dst, JMM& src, int zero_idx = 15) {  // NOLINT
    JMM zero = JMM(zero_idx);
    vxorps(zero, zero, zero);
    vmaxps(dst, src, zero);
  }

  // compute exp with ymm, xmm
  template <typename JMM>
  void exp_jmm(JMM& dst, JMM& src, int src_idx = 11, int fx_idx = 12,  // NOLINT
               int fy_idx = 13, int mask_idx = 14, int tmp_idx = 15) {
    using namespace platform;  // NOLINT
    // check all idx can not equal
    JMM jmm_src = JMM(src_idx);
    JMM jmm_fx = JMM(fx_idx);
    JMM jmm_fy = JMM(fy_idx);
    JMM jmm_mask = JMM(mask_idx);
    JMM jmm_tmp = JMM(tmp_idx);
    reg64_t reg_ptr_global = rax;
    push(reg_ptr_global);
    vmovaps(jmm_src, src);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_HIG]);
    vminps(jmm_src, jmm_src, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_LOW]);
    vmaxps(jmm_src, jmm_src, jmm_tmp);
    // express exp(x) as exp(g + n*log(2))
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_LOG2EF]);
    vmulps(jmm_fx, jmm_src, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_0P5]);
    vaddps(jmm_fx, jmm_fx, jmm_tmp);
    vroundps(jmm_fy, jmm_fx, 0x01);
    // if greater, substract 1
    vcmpgtps(jmm_mask, jmm_fy, jmm_fx);
    vmovaps(jmm_tmp, ptr[reg_ptr_global]);
    vandps(jmm_mask, jmm_mask, jmm_tmp);
    vsubps(jmm_fx, jmm_fy, jmm_mask);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_C1]);
    vmulps(jmm_fy, jmm_fx, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_C2]);
    JMM ymm_z = JMM(jmm_mask.getIdx());
    vmulps(ymm_z, jmm_fx, jmm_tmp);
    vsubps(jmm_src, jmm_src, jmm_fy);
    vsubps(jmm_src, jmm_src, ymm_z);
    vmulps(ymm_z, jmm_src, jmm_src);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_P0]);
    vmulps(dst, jmm_src, jmm_tmp);
    for (size_t i = OFFSET_EXP_P1; i < OFFSET_EXP_P5;
         i += (YMM_FLOAT_BLOCK * sizeof(float))) {
      vmovaps(jmm_tmp, ptr[reg_ptr_global + i]);  // P1~P4
      vaddps(dst, dst, jmm_tmp);
      vmulps(dst, dst, jmm_src);
    }
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_P5]);
    vaddps(dst, dst, jmm_tmp);
    vmulps(dst, dst, ymm_z);
    vaddps(dst, dst, jmm_src);
    vmovaps(jmm_tmp, ptr[reg_ptr_global]);
    vaddps(dst, dst, jmm_tmp);
    // build 2^n
    JMM ymm_int = jmm_fx;
    vcvttps2dq(ymm_int, jmm_fx);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_int_0x7f));
    vmovdqa(jmm_tmp, ptr[reg_ptr_global]);
    if (MayIUse(avx2) || std::is_same<JMM, xmm_t>::value) {
      vpaddd(ymm_int, ymm_int, jmm_tmp);
      vpslld(ymm_int, ymm_int, 23);
    } else if (MayIUse(avx)) {
      xmm_t xtmp1 = xmm_t(ymm_int.getIdx());
      xmm_t xtmp2 = xmm_t(jmm_tmp.getIdx());
      reg64_t reg_ptr_tmp = reg_ptr_global;
      mov(reg_ptr_tmp, reinterpret_cast<size_t>(g_tmp_mem));
      vmovdqa(ptr[reg_ptr_tmp], ymm_int);
      vmovdqa(ptr[reg_ptr_tmp + YMM_FLOAT_BLOCK * sizeof(float)], jmm_tmp);
      vpaddd(xtmp1, xtmp1, xtmp2);
      vpslld(xtmp1, xtmp1, 23);
      vmovdqa(ptr[reg_ptr_tmp], xtmp1);
      // next 128bits
      vmovdqa(xtmp1, ptr[reg_ptr_tmp + XMM_FLOAT_BLOCK * sizeof(float)]);
      vmovdqa(xtmp2, ptr[reg_ptr_tmp +
                         (YMM_FLOAT_BLOCK + XMM_FLOAT_BLOCK) * sizeof(float)]);
      vpaddd(xtmp1, xtmp1, xtmp2);
      vpslld(xtmp1, xtmp1, 23);
      vmovdqa(ptr[reg_ptr_tmp + XMM_FLOAT_BLOCK * sizeof(float)], xtmp1);
      // load out
      vmovdqa(ymm_int, ptr[reg_ptr_tmp]);
    }
    vmulps(dst, dst, ymm_int);
    pop(reg_ptr_global);
  }

  // compute sigmoid with ymm, xmm
  template <typename JMM>
  void sigmoid_jmm(JMM& dst, JMM& src, int src_idx = 11,  // NOLINT
                   int fx_idx = 12, int fy_idx = 13, int mask_idx = 14,
                   int tmp_idx = 15) {
    // y = 1 / (1 + e^-x)
    JMM jmm_tmp = JMM(tmp_idx);
    JMM jmm_src = JMM(src_idx);
    reg64_t reg_ptr_global = rax;
    push(reg_ptr_global);
    vmovaps(jmm_src, src);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_SIGMOID_MAX]);
    vminps(jmm_src, jmm_src, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_SIGMOID_MIN]);
    vmaxps(jmm_src, jmm_src, jmm_tmp);
    vxorps(jmm_tmp, jmm_tmp, jmm_tmp);
    vsubps(jmm_src, jmm_tmp, jmm_src);
    exp_jmm<JMM>(dst, jmm_src, src_idx, fx_idx, fy_idx, mask_idx, tmp_idx);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
    vaddps(dst, dst, jmm_tmp);
    vdivps(dst, jmm_tmp, dst);
    pop(reg_ptr_global);
  }

  // compute tanh with ymm, xmm
  template <typename JMM>
  void tanh_jmm(JMM& dst, JMM& src, int src_idx = 11,  // NOLINT
                int fx_idx = 12, int fy_idx = 13, int mask_idx = 14,
                int tmp_idx = 15) {
    // y = 2 / (1 + e^(-2x)) - 1
    JMM jmm_src = JMM(src_idx);
    JMM jmm_tmp = JMM(tmp_idx);
    JMM jmm_zero = JMM(mask_idx);
    reg64_t reg_ptr_global = rax;
    push(reg_ptr_global);
    vmovaps(jmm_src, src);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_TWO]);
    vxorps(jmm_zero, jmm_zero, jmm_zero);
    vsubps(jmm_tmp, jmm_zero, jmm_tmp);
    vmulps(jmm_src, jmm_src, jmm_tmp);
    exp_jmm<JMM>(dst, jmm_src, src_idx, fx_idx, fy_idx, mask_idx, tmp_idx);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
    vaddps(dst, dst, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_TWO]);
    vdivps(dst, jmm_tmp, dst);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
    vsubps(dst, dst, jmm_tmp);
    pop(reg_ptr_global);
  }

  // compute identity with ymm, xmm
  template <typename JMM>
  void identity_jmm(JMM& dst, JMM& src, int zero_idx) {  // NOLINT
    JMM zero = JMM(zero_idx);
    vxorps(zero, zero, zero);
    vaddps(dst, src, zero);
    // TODO(TJ): use below
    // dst.setIdx(src.getIdx());
  }

  template <typename JMM>
  void act(JMM& dst, JMM& src, operand_type type) {  // NOLINT
    // use 11~15
    switch (type) {
      case operand_type::relu:
        relu_jmm<JMM>(dst, src, 15);
        break;
      case operand_type::exp:
        exp_jmm<JMM>(dst, src, 11, 12, 13, 14, 15);
        break;
      case operand_type::sigmoid:
        sigmoid_jmm<JMM>(dst, src, 11, 12, 13, 14, 15);
        break;
      case operand_type::tanh:
        tanh_jmm<JMM>(dst, src, 11, 12, 13, 14, 15);
        break;
      case operand_type::identity:
        identity_jmm<JMM>(dst, src, 15);
        break;
      default:
        LOG(FATAL) << "Do not support this operand type: " << type_;
        break;
    }
  }

 protected:
  int num_;
  operand_type type_;
  reg64_t param1{abi_param1};
  reg64_t param2{abi_param2};

  xmm_t xmm_src = xmm_t(0);
  ymm_t ymm_src = ymm_t(0);

  xmm_t xmm_dst = xmm_t(1);
  ymm_t ymm_dst = ymm_t(1);
};

#define DECLARE_ACT_JITCODE(name, op_type)                                    \
  class name##JitCode : public VActJitCode {                                  \
   public:                                                                    \
    explicit name##JitCode(int d, size_t code_size, void* code_ptr = nullptr) \
        : VActJitCode(d, op_type, code_size, code_ptr) {}                     \
  };

DECLARE_ACT_JITCODE(VRelu, operand_type::relu);
DECLARE_ACT_JITCODE(VIdentity, operand_type::identity);
DECLARE_ACT_JITCODE(VExp, operand_type::exp);
DECLARE_ACT_JITCODE(VSigmoid, operand_type::sigmoid);
DECLARE_ACT_JITCODE(VTanh, operand_type::tanh);

#undef DECLARE_ACT_JITCODE

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
