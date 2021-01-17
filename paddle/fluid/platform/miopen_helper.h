/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/dynload/miopen.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/macros.h"

using std::string;
using std::vector;
#define MIOPEN_BN_MIN_EPSILON 1e-05

namespace paddle {
namespace platform {
struct float16;
}  // namespace platform
}  // namespace paddle

DECLARE_bool(cudnn_deterministic);

namespace paddle {
namespace platform {

inline const char* miopenGetErrorString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "MIOPEN_STATUS_SUCCESS";
    case miopenStatusNotInitialized:
      return "MIOPEN_STATUS_NOT_INITIALIZED";
    case miopenStatusInvalidValue:
      return "MIOPEN_STATUS_INVALID_VALUE";
    case miopenStatusBadParm:
      return "MIOPEN_STATUS_BAD_PARAM";
    case miopenStatusAllocFailed:
      return "MIOPEN_STATUS_ALLOC_FAILED";
    case miopenStatusInternalError:
      return "MIOPEN_STATUS_INTERNAL_ERROR";
    case miopenStatusNotImplemented:
      return "MIOPEN_STATUS_NOT_IMPLEMENTED";
    case miopenStatusUnknownError:
    default:
      return "MIOPEN_STATUS_UNKNOWN_ERROR";
  }
}

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= ((major)*1000 + (minor)*100 + (patch)))

enum class DataLayout {  // Not use
  kNHWC,
  kNCHW,
  kNCDHW,
  kNDHWC,  // add, liyamei
  kNCHW_VECT_C,
};

enum class PoolingMode {
  kMaximum,
  kMaximumDeterministic,
  kAverageExclusive,
  kAverageInclusive,
};

enum class ActivationMode {
  kNone,  // activation identity
  kSigmoid,
  kRelu,
  kRelu6,
  kReluX,
  kTanh,
  kBandPass,
};

inline miopenPoolingMode_t GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    case PoolingMode::kMaximumDeterministic:
      return miopenPoolingMax;
    case PoolingMode::kAverageExclusive:
      return miopenPoolingAverage;
    case PoolingMode::kAverageInclusive:
      return miopenPoolingAverageInclusive;
    case PoolingMode::kMaximum:
      return miopenPoolingMax;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unexpected MIOPEN pooling mode."));
  }
}

inline ActivationMode StringToActivationMode(const std::string& str) {
  if (str == "identity") {
    return ActivationMode::kNone;
  } else if (str == "sigmoid") {
    return ActivationMode::kSigmoid;
  } else if (str == "relu") {
    return ActivationMode::kRelu;
  } else if (str == "relu6") {
    return ActivationMode::kRelu6;
  } else if (str == "relux") {
    return ActivationMode::kReluX;
  } else if (str == "tanh") {
    return ActivationMode::kTanh;
  } else if (str == "bandpass") {
    return ActivationMode::kBandPass;
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unexpected CUDNN pooling mode."));
  }
}

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<float16> {
 public:
  static const miopenDataType_t type = miopenHalf;
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<float> {
 public:
  static const miopenDataType_t type = miopenFloat;
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

inline std::string GetCudnnTensorFormat(const DataLayout& order) {  // Not use
  switch (order) {
    //    case DataLayout::kNHWC:
    //      return false;
    case DataLayout::kNCHW:
      return "NCHW";
    case DataLayout::kNCDHW:
      return "NCHW";  // NOTE: cudnn treat NdTensor as the same
                      //    case DataLayout::kNDHWC:
                      //      return false;  // add, liyamei
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "MIOPEN has no equivalent dataLayout for input order."));
  }
  return "NCHW";
}

class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~ScopedTensorDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyTensorDescriptor(desc_));
  }

  inline miopenTensorDescriptor_t descriptor(const std::string format,
                                             const miopenDataType_t type,
                                             const std::vector<int>& dims,
                                             const int groups = 1) {
    // the format is not used now, will add later
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    // Update tensor descriptor dims setting if groups > 1
    // NOTE: Here, Assume using NCHW or NCDHW order
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }

    if (dims.size() == 4) {
      if (format == "NCHW") {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetTensorDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      } else {  // false
        // miopen not support nhwc
        return 0;
      }
    } else if (dims.size() == 5) {
      if (format == "NCHW") {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetTensorDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      } else {  // false
        // miopen not support nhwc
        return 0;
      }
    }
    return desc_;
  }

  template <typename T>
  inline miopenTensorDescriptor_t descriptor(const DataLayout& order,
                                             const std::vector<int>& dims,
                                             const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type, dims,
                      groups);
  }

 private:
  miopenTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};
// xuan  [TODO]
class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~ScopedFilterDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyTensorDescriptor(desc_));
  }

  inline miopenTensorDescriptor_t descriptor(const std::string format,
                                             const miopenDataType_t type,
                                             const std::vector<int>& kernel,
                                             const int groups = 1) {
    std::vector<int> strides(kernel.size());
    strides[kernel.size() - 1] = 1;
    for (int i = kernel.size() - 2; i >= 0; i--) {
      strides[i] = kernel[i + 1] * strides[i + 1];
    }

    // filter layout: MCHW(MCDHW), where M is the number of
    // output image channels, C is the number of input image channels,
    // D is the depth of the filter, H is the height of the filter, and W is the
    // width of the filter.
    std::vector<int> kernel_with_group(kernel.begin(), kernel.end());
    if (groups > 1) {
      kernel_with_group[0] /= groups;
      // NOTE: input filter(C) of the filter is already asserted to be C/groups.
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetTensorDescriptor(
        desc_, type, kernel_with_group.size(), kernel_with_group.data(),
        strides.data()));
    return desc_;
  }

  template <typename T>
  inline miopenTensorDescriptor_t descriptor(const DataLayout& order,
                                             const std::vector<int>& kernel,
                                             const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type,
                      kernel, groups);
  }

 private:
  miopenTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateConvolutionDescriptor(&desc_));
  }
  ~ScopedConvolutionDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenDestroyConvolutionDescriptor(desc_));
  }

  inline miopenConvolutionDescriptor_t descriptor(
      miopenDataType_t type, const std::vector<int>& pads,
      const std::vector<int>& strides, const std::vector<int>& dilations) {
    PADDLE_ENFORCE_EQ(pads.size(), strides.size(),
                      platform::errors::InvalidArgument(
                          "The size of pads and strides should be equal. But "
                          "received size of pads is %d, size of strides is %d.",
                          pads.size(), strides.size()));
    PADDLE_ENFORCE_EQ(
        pads.size(), dilations.size(),
        platform::errors::InvalidArgument(
            "The size of pads and dilations should be equal. But received size "
            "of pads is %d, size of dilations is %d.",
            pads.size(), dilations.size()));
    miopenConvolutionMode_t compute_mode = miopenConvolution;
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenInitConvolutionNdDescriptor(
        static_cast<miopenConvolutionDescriptor_t>(desc_),
        static_cast<int>(pads.size()), const_cast<int*>(pads.data()),
        const_cast<int*>(strides.data()), const_cast<int*>(dilations.data()),
        compute_mode));
    return desc_;
  }

  template <typename T>
  inline miopenConvolutionDescriptor_t descriptor(
      const std::vector<int>& pads, const std::vector<int>& strides,
      const std::vector<int>& dilations) {
    return descriptor(CudnnDataType<T>::type, pads, strides, dilations);
  }

 private:
  miopenConvolutionDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenCreatePoolingDescriptor(&desc_));
  }
  ~ScopedPoolingDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyPoolingDescriptor(desc_));
  }

  inline miopenPoolingDescriptor_t descriptor(const PoolingMode& mode,
                                              const std::vector<int>& kernel,
                                              const std::vector<int>& pads,
                                              const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(), pads.size(),
                      platform::errors::InvalidArgument(
                          "The size of kernel and pads should be equal. But "
                          "received size of kernel is %d, size of pads is %d.",
                          kernel.size(), pads.size()));
    PADDLE_ENFORCE_EQ(
        kernel.size(), strides.size(),
        platform::errors::InvalidArgument(
            "The size of kernel and strides should be equal. But "
            "received size of kernel is %d, size of strides is %d.",
            kernel.size(), strides.size()));
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetNdPoolingDescriptor(
        (miopenPoolingDescriptor_t)desc_, (GetPoolingMode(mode)),
        static_cast<int>(kernel.size()), const_cast<int*>(kernel.data()),
        const_cast<int*>(pads.data()), const_cast<int*>(strides.data())));
    return desc_;
  }

 private:
  miopenPoolingDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

// class ScopedSpatialTransformerDescriptor {
// public:
//  ScopedSpatialTransformerDescriptor() {
//    PADDLE_ENFORCE_CUDA_SUCCESS(
//        dynload::cudnnCreateSpatialTransformerDescriptor(&desc_));
//  }
//  ~ScopedSpatialTransformerDescriptor() PADDLE_MAY_THROW {
//    PADDLE_ENFORCE_CUDA_SUCCESS(
//        dynload::cudnnDestroySpatialTransformerDescriptor(desc_));
//  }
//
//  template <typename T>
//  inline cudnnSpatialTransformerDescriptor_t descriptor(const int nbDims,
//                                                        const int dimA[]) {
//    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetSpatialTransformerNdDescriptor(
//        desc_, CUDNN_SAMPLER_BILINEAR, CudnnDataType<T>::type, nbDims, dimA));
//    return desc_;
//  }
//
// private:
//  cudnnSpatialTransformerDescriptor_t desc_;
//  DISABLE_COPY_AND_ASSIGN(ScopedSpatialTransformerDescriptor);
//};

class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateActivationDescriptor(&desc_));
  }
  ~ScopedActivationDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenDestroyActivationDescriptor(desc_));
  }

  template <typename T>
  inline miopenActivationDescriptor_t descriptor(
      const std::string& act, double value_max = static_cast<double>(0.)) {
    double relu_ceiling = 0.0;
    ActivationMode activation_mode = StringToActivationMode(act);
    miopenActivationMode_t mode;
    switch (activation_mode) {
      case ActivationMode::kNone:
        mode = miopenActivationPASTHRU;
        break;
      case ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = miopenActivationCLIPPEDRELU;
        break;
      case ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = miopenActivationCLIPPEDRELU;
        break;
      case ActivationMode::kRelu:
        mode = miopenActivationRELU;
        break;
      case ActivationMode::kSigmoid:
        mode = miopenActivationLOGISTIC;
        break;
      case ActivationMode::kTanh:
        mode = miopenActivationTANH;
        break;
      default:
        PADDLE_THROW("unrecognized activation mode: %d .",
                     static_cast<int>(activation_mode));
    }
    return desc_;
  }

 private:
  miopenActivationDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};

inline bool CanCUDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_cudnn = ctx.Attr<bool>("use_cudnn");
  use_cudnn &= paddle::platform::is_gpu_place(ctx.GetPlace());
  if (use_cudnn) {
    auto& dev_ctx = ctx.device_context<platform::CUDADeviceContext>();
    use_cudnn &= dev_ctx.cudnn_handle() != nullptr;
  }
  return use_cudnn;
}

class ScopedCTCLossDescriptor {
 public:
  ScopedCTCLossDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenCreateCTCLossDescriptor(&desc_));
  }
  ~ScopedCTCLossDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyCTCLossDescriptor(desc_));
  }

  template <typename T>
  inline miopenCTCLossDescriptor_t descriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenSetCTCLossDescriptor(desc_, 0, CudnnDataType<T>::type));
    return desc_;
  }

 private:
  miopenCTCLossDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedCTCLossDescriptor);
};

}  // namespace platform
}  // namespace paddle
