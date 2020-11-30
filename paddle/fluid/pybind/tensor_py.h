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

#include <Python.h>
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

// Note: use same enum number of float16 in numpy.
// import numpy as np
// print np.dtype(np.float16).num  # 23
constexpr int NPY_FLOAT16_ = 23;
constexpr int NPY_UINT16_ = 4;
constexpr int NPY_COMPLEX64 = 14;
constexpr int NPY_COMPLEX128 = 15;

// Note: Since float16 is not a builtin type in C++, we register
// paddle::platform::float16 as numpy.float16.
// Ref: https://github.com/pybind/pybind11/issues/1776
template <>
struct npy_format_descriptor<paddle::platform::float16> {
  static py::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16_);
    return reinterpret_borrow<py::dtype>(ptr);
  }
  static std::string format() {
    // Note: "e" represents float16.
    // Details at:
    // https://docs.python.org/3/library/struct.html#format-characters.
    return "e";
  }
  static constexpr auto name = _("float16");
};

// Note: Since bfloat16 is not a builtin type in C++ and in numpy,
// we register paddle::platform::bfloat16 as numpy.uint16.
template <>
struct npy_format_descriptor<paddle::platform::bfloat16> {
  static py::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_UINT16_);
    return reinterpret_borrow<py::dtype>(ptr);
  }
  static std::string format() {
    // Note: "H" represents UINT16.
    // Details at:
    // https://docs.python.org/3/library/struct.html#format-characters.
    return "H";
  }
  static constexpr auto name = _("bfloat16");
};

// we register paddle::platform::complex64 as numpy.complex64.
template <>
struct npy_format_descriptor<paddle::platform::complex64> {
  static py::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_COMPLEX64);
    return reinterpret_borrow<py::dtype>(ptr);
  }

  static std::string format() {
    // Note: "F" represents complex64.
    // Details at:
    // https://stackoverflow.com/questions/13997087/what-are-the-available-datatypes-for-dtype-with-numpys-loadtxt-an-genfromtx
    // for k, v in np.sctypeDict.iteritems():
    //     print '{0:14s} : {1:40s}'.format(str(k), v)
    return "F";
  }
  static constexpr auto name = _("complext64");
};

// we register paddle::platform::complex128 as numpy.complex128.
template <>
struct npy_format_descriptor<paddle::platform::complex128> {
  static py::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_COMPLEX128);
    return reinterpret_borrow<py::dtype>(ptr);
  }

  static std::string format() {
    // Note: "D" represents complex128.
    // Details at:
    // https://stackoverflow.com/questions/13997087/what-are-the-available-datatypes-for-dtype-with-numpys-loadtxt-an-genfromtx
    // for k, v in np.sctypeDict.iteritems():
    //     print '{0:14s} : {1:40s}'.format(str(k), v)
    return "D";
  }
  static constexpr auto name = _("complext128");
};

}  // namespace detail
}  // namespace pybind11

namespace paddle {
namespace pybind {

namespace details {

template <typename T>
class PYBIND11_HIDDEN NumpyAllocation : public memory::Allocation {
 public:
  explicit NumpyAllocation(const py::array &arr)
      : Allocation(const_cast<void *>(arr.data()), sizeof(T) * (arr.size()),
                   paddle::platform::CPUPlace()),
        arr_(arr.ptr()) {
    PADDLE_ENFORCE_NOT_NULL(arr_, platform::errors::InvalidArgument(
                                      "The underlying PyObject pointer of "
                                      "numpy array cannot be nullptr"));
    PADDLE_ENFORCE_NE(
        arr_, Py_None,
        platform::errors::PreconditionNotMet(
            "The underlying PyObject pointer of numpy array cannot be None"));
    Py_INCREF(arr_);
  }
  ~NumpyAllocation() override {
    py::gil_scoped_acquire gil;
    Py_DECREF(arr_);
  }

 private:
  PyObject *arr_;
};

template <typename T>
struct ValidDTypeToPyArrayChecker {
  static constexpr bool kValue = false;
};

#define DECLARE_VALID_DTYPE_TO_PY_ARRAY(type) \
  template <>                                 \
  struct ValidDTypeToPyArrayChecker<type> {   \
    static constexpr bool kValue = true;      \
  }

DECLARE_VALID_DTYPE_TO_PY_ARRAY(platform::float16);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(platform::bfloat16);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(platform::complex64);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(platform::complex128);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(float);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(double);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(bool);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int8_t);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int16_t);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(int64_t);
DECLARE_VALID_DTYPE_TO_PY_ARRAY(uint8_t);

inline std::string TensorDTypeToPyDTypeStr(
    framework::proto::VarType::Type type) {
#define TENSOR_DTYPE_TO_PY_DTYPE(T, proto_type)                             \
  if (type == proto_type) {                                                 \
    if (std::is_same<T, platform::float16>::value) {                        \
      return "e";                                                           \
    } else if (std::is_same<T, platform::bfloat16>::value) {                \
      /* NumPy character code of uint16 due to no support for bfloat16 */   \
      return "H";                                                           \
    } else if (std::is_same<T, platform::complex64>::value) {               \
      return "F";                                                           \
    } else if (std::is_same<T, platform::complex128>::value) {              \
      return "D";                                                           \
    } else {                                                                \
      constexpr auto kIsValidDType = ValidDTypeToPyArrayChecker<T>::kValue; \
      PADDLE_ENFORCE_EQ(                                                    \
          kIsValidDType, true,                                              \
          platform::errors::Unimplemented(                                  \
              "This type [%s] of tensor cannot be expose to Python",        \
              typeid(T).name()));                                           \
      return py::format_descriptor<T>::format();                            \
    }                                                                       \
  }

  _ForEachDataType_(TENSOR_DTYPE_TO_PY_DTYPE);
#undef TENSOR_DTYPE_TO_PY_DTYPE
  PADDLE_THROW(platform::errors::Unimplemented(
      "Unsupported tensor data type: %s", framework::DataTypeToString(type)));
}

}  // namespace details

template <typename T>
T TensorGetElement(const framework::Tensor &self, size_t offset) {
  PADDLE_ENFORCE_LT(offset, self.numel(),
                    platform::errors::InvalidArgument(
                        "The offset exceeds the size of tensor."));
  T b = static_cast<T>(0);
  if (platform::is_cpu_place(self.place())) {
    b = self.data<T>()[offset];
  } else if (platform::is_xpu_place(self.place())) {
#ifdef PADDLE_WITH_XPU
    const T *a = self.data<T>();
    auto p = BOOST_GET_CONST(platform::XPUPlace, self.place());
    paddle::memory::Copy(platform::CPUPlace(), &b, p, a + offset, sizeof(T));
#endif
  } else if (platform::is_gpu_place(self.place())) {
#ifdef PADDLE_WITH_CUDA
    const T *a = self.data<T>();
    auto p = BOOST_GET_CONST(platform::CUDAPlace, self.place());
    paddle::memory::Copy(platform::CPUPlace(), &b, p, a + offset, sizeof(T),
                         nullptr);
#endif
  }
  return b;
}

template <typename T>
void TensorSetElement(framework::Tensor *self, size_t offset, T elem) {
  PADDLE_ENFORCE_LT(offset, self->numel(),
                    platform::errors::InvalidArgument(
                        "The offset exceeds the size of tensor."));
  if (platform::is_cpu_place(self->place())) {
    self->mutable_data<T>(self->place())[offset] = elem;
  } else if (platform::is_xpu_place(self->place())) {
#ifdef PADDLE_WITH_XPU
    auto p = BOOST_GET_CONST(platform::XPUPlace, self->place());
    T *a = self->mutable_data<T>(p);
    paddle::memory::Copy(p, a + offset, platform::CPUPlace(), &elem, sizeof(T));
#endif
  } else if (platform::is_gpu_place(self->place())) {
#ifdef PADDLE_WITH_CUDA
    auto p = BOOST_GET_CONST(platform::CUDAPlace, self->place());
    T *a = self->mutable_data<T>(p);
    paddle::memory::Copy(p, a + offset, platform::CPUPlace(), &elem, sizeof(T),
                         nullptr);
#endif
  }
}

template <typename T, typename P>
void SetTensorFromPyArrayT(
    framework::Tensor *self,
    const py::array_t<T, py::array::c_style | py::array::forcecast> &array,
    const P &place, bool zero_copy) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }
  self->Resize(framework::make_ddim(dims));

  if (paddle::platform::is_cpu_place(place)) {
    if (zero_copy) {
      auto holder = std::make_shared<details::NumpyAllocation<T>>(array);
      auto type = framework::ToDataType(std::type_index(typeid(T)));
      self->ResetHolderWithType(holder, type);
    } else {
      auto dst = self->mutable_data<T>(place);
      std::memcpy(dst, array.data(), array.nbytes());
    }
  } else if (paddle::platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    auto dst = self->mutable_data<T>(place);
    xpu_memcpy(dst, array.data(), array.nbytes(),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Cannot use XPUPlace in CPU/GPU version, "
        "Please recompile or reinstall Paddle with XPU support."));
#endif
  } else {
#ifdef PADDLE_WITH_CUDA
    auto dst = self->mutable_data<T>(place);
    if (paddle::platform::is_cuda_pinned_place(place)) {
      std::memcpy(dst, array.data(), array.nbytes());
    } else if (paddle::platform::is_gpu_place(place)) {
      paddle::platform::GpuMemcpySync(dst, array.data(), array.nbytes(),
                                      cudaMemcpyHostToDevice);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Incompatible place type: Tensor.set() supports "
          "CPUPlace, CUDAPlace "
          "and CUDAPinnedPlace, but got %s!",
          place));
    }
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Cannot use CUDAPlace or CUDAPinnedPlace in CPU only version, "
        "Please recompile or reinstall Paddle with CUDA support."));
#endif
  }
}

template <typename P>
void SetTensorFromPyArray(framework::Tensor *self, const py::object &obj,
                          const P &place, bool zero_copy) {
  auto array = obj.cast<py::array>();
  if (py::isinstance<py::array_t<float>>(array)) {
    SetTensorFromPyArrayT<float, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int>>(array)) {
    SetTensorFromPyArrayT<int, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int64_t>>(array)) {
    SetTensorFromPyArrayT<int64_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<double>>(array)) {
    SetTensorFromPyArrayT<double, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int8_t>>(array)) {
    SetTensorFromPyArrayT<int8_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<int16_t>>(array)) {
    SetTensorFromPyArrayT<int16_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<uint8_t>>(array)) {
    SetTensorFromPyArrayT<uint8_t, P>(self, array, place, zero_copy);
  } else if (py::isinstance<py::array_t<paddle::platform::float16>>(array)) {
    SetTensorFromPyArrayT<paddle::platform::float16, P>(self, array, place,
                                                        zero_copy);
  } else if (py::isinstance<py::array_t<paddle::platform::complex64>>(array)) {
    SetTensorFromPyArrayT<paddle::platform::complex64, P>(self, array, place,
                                                          zero_copy);
  } else if (py::isinstance<py::array_t<paddle::platform::complex128>>(array)) {
    SetTensorFromPyArrayT<paddle::platform::complex128, P>(self, array, place,
                                                           zero_copy);
  } else if (py::isinstance<py::array_t<uint16_t>>(array)) {
    // since there is still no support for bfloat16 in NumPy,
    // uint16 is used for casting bfloat16
    SetTensorFromPyArrayT<paddle::platform::bfloat16, P>(self, array, place,
                                                         zero_copy);
  } else if (py::isinstance<py::array_t<bool>>(array)) {
    SetTensorFromPyArrayT<bool, P>(self, array, place, zero_copy);
  } else {
    // obj may be any type, obj.cast<py::array>() may be failed,
    // then the array.dtype will be string of unknown meaning,
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Input object type error or incompatible array data type. "
        "tensor.set() supports array with bool, float16, float32, "
        "float64, int8, int16, int32, int64, uint8 or uint16, "
        "please check your input or input array data type."));
  }
}

template <typename T, size_t D>
void _sliceCompute(const framework::Tensor *in, framework::Tensor *out,
                   const platform::CPUDeviceContext &ctx,
                   const std::vector<int> &axes,
                   const std::vector<int> &starts) {
  auto &eigen_place = *ctx.eigen_device();
  auto place = in->place();
  auto out_dims = out->dims();
  auto in_dims = in->dims();

  auto offsets = Eigen::array<int, D>();
  auto extents = Eigen::array<int, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = out_dims[i];
  }
  int start;
  for (size_t i = 0; i < axes.size(); ++i) {
    start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *in);
  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out);
  out_t.device(eigen_place) = in_t.slice(offsets, extents);
}

template <typename T>
void _concatCompute(const std::vector<paddle::framework::Tensor> &ins,
                    paddle::framework::Tensor *out,
                    const platform::CPUDeviceContext &ctx, int64_t axis) {
  if (axis == 0 && ins.size() < 10) {
    size_t output_offset = 0;
    for (auto &in : ins) {
      auto in_stride = framework::stride_numel(in.dims());
      auto out_stride = framework::stride_numel(out->dims());
      paddle::operators::StridedNumelCopyWithAxis<T>(
          ctx, axis, out->data<T>() + output_offset, out_stride, in.data<T>(),
          in_stride, in_stride[axis]);
      output_offset += in_stride[axis];
    }
  } else {
    paddle::operators::math::ConcatFunctor<platform::CPUDeviceContext, T>
        concat_functor;
    concat_functor(ctx, ins, static_cast<int>(axis), out);
  }
}

inline void _getSliceinfo(const framework::Tensor &self, py::object obj,
                          const int64_t dim, int64_t *pstart, int64_t *pstop,
                          int64_t *pstep, int64_t *pslicelength) {
  auto &start = *pstart;
  auto &stop = *pstop;
  auto &step = *pstep;
  auto &slicelength = *pslicelength;
  const framework::DDim &srcDDim = self.dims();
  if (dim < 0 || dim >= srcDDim.size()) {
    throw py::index_error();
  }
  if (py::isinstance<py::slice>(obj)) {
    size_t lstart, lstop, lstep, lslicelength;
    py::slice s = static_cast<py::slice>(obj);
    if (!s.compute(srcDDim[dim], &lstart, &lstop, &lstep, &lslicelength)) {
      throw py::index_error();
    }
    start = static_cast<int64_t>(lstart);
    stop = static_cast<int64_t>(lstop);
    step = static_cast<int64_t>(lstep);
    slicelength = static_cast<int64_t>(lslicelength);
  } else if (py::isinstance<py::int_>(obj)) {
    start = static_cast<int64_t>(static_cast<py::int_>(obj));
    if (std::abs(start) >= srcDDim[dim]) {
      throw py::index_error();
    }
    start = (start >= 0) ? start : srcDDim[dim] - start;
    stop = start + 1;
    step = 1;
    slicelength = 1;
  } else {
    throw py::index_error();
  }
}

inline framework::Tensor *_getTensor(const framework::Tensor &self,
                                     const framework::DDim &ddim) {
  framework::Tensor *output = new framework::Tensor();
  output->Resize(ddim);
  auto place = self.place();
  if (platform::is_cpu_place(place)) {
    output->mutable_data(BOOST_GET_CONST(platform::CPUPlace, place),
                         self.type());
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    output->mutable_data(BOOST_GET_CONST(platform::XPUPlace, place),
                         self.type());
#endif
  } else {
#ifdef PADDLE_WITH_CUDA
    if (platform::is_cuda_pinned_place(place)) {
      output->mutable_data(BOOST_GET_CONST(platform::CUDAPinnedPlace, place),
                           self.type());
    } else if ((platform::is_gpu_place(place))) {
      output->mutable_data(BOOST_GET_CONST(platform::CUDAPlace, place),
                           self.type());
    }
#endif
  }
  return output;
}

template <typename T>
void _sliceDapper(const framework::Tensor *in, framework::Tensor *out,
                  const platform::CPUDeviceContext &ctx,
                  const std::vector<int> &axes, const std::vector<int> &starts,
                  int size) {
  switch (size) {
    case 1:
      _sliceCompute<T, 1>(in, out, ctx, axes, starts);
      break;
    case 2:
      _sliceCompute<T, 2>(in, out, ctx, axes, starts);
      break;
    case 3:
      _sliceCompute<T, 3>(in, out, ctx, axes, starts);
      break;
    case 4:
      _sliceCompute<T, 4>(in, out, ctx, axes, starts);
      break;
    case 5:
      _sliceCompute<T, 5>(in, out, ctx, axes, starts);
      break;
    case 6:
      _sliceCompute<T, 6>(in, out, ctx, axes, starts);
      break;
    case 7:
      _sliceCompute<T, 7>(in, out, ctx, axes, starts);
      break;
    case 8:
      _sliceCompute<T, 8>(in, out, ctx, axes, starts);
      break;
    case 9:
      _sliceCompute<T, 9>(in, out, ctx, axes, starts);
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The dim size should be 1 to 9, current is %d", size));
      break;
  }
}

template <typename T>
inline framework::Tensor *_sliceWrapper(const framework::Tensor &self,
                                        const platform::CPUDeviceContext &ctx,
                                        py::object obj, int dim, int64_t start,
                                        int64_t slicelength) {
  framework::DDim dstDDim = self.dims();
  dstDDim[dim] = static_cast<int64_t>(slicelength);
  std::vector<int> axes({dim});
  std::vector<int> starts({static_cast<int>(start)});
  framework::Tensor *output = _getTensor(self, dstDDim);
  _sliceDapper<T>(&self, output, ctx, axes, starts, dstDDim.size());
  return output;
}

template <typename T>
inline framework::Tensor *_sliceAndConcat(const framework::Tensor &self,
                                          py::object obj, int dim) {
  platform::CPUDeviceContext ctx;
  int64_t start, stop, step, slicelength;
  _getSliceinfo(self, obj, dim, &start, &stop, &step, &slicelength);
  if (step == 1 || slicelength == 1) {
    return _sliceWrapper<T>(self, ctx, obj, dim, start, slicelength);
  } else {
    std::vector<framework::Tensor> ins;
    for (auto i = 0; i < slicelength; ++i, start += step) {
      ins.emplace_back(*_sliceWrapper<T>(self, ctx, obj, dim, start, 1));
    }

    // do the concat operation
    framework::DDim dstDDim = self.dims();
    dstDDim[dim] = static_cast<int64_t>(slicelength);
    framework::Tensor *output1 = _getTensor(self, dstDDim);
    _concatCompute<T>(ins, output1, ctx, dim);
    return output1;
  }
}

inline framework::Tensor *_sliceTensor(const framework::Tensor &self,
                                       py::object obj, int dim) {
  auto src_type = self.type();
  switch (src_type) {
    case framework::proto::VarType::FP16:
      return _sliceAndConcat<paddle::platform::float16>(self, obj, dim);
    case framework::proto::VarType::BF16:
      return _sliceAndConcat<paddle::platform::bfloat16>(self, obj, dim);
    case framework::proto::VarType::COMPLEX64:
      return _sliceAndConcat<paddle::platform::complex64>(self, obj, dim);
    case framework::proto::VarType::COMPLEX128:
      return _sliceAndConcat<paddle::platform::complex128>(self, obj, dim);
    case framework::proto::VarType::FP32:
      return _sliceAndConcat<float>(self, obj, dim);
    case framework::proto::VarType::FP64:
      return _sliceAndConcat<double>(self, obj, dim);
    case framework::proto::VarType::INT8:
      return _sliceAndConcat<int8_t>(self, obj, dim);
    case framework::proto::VarType::INT16:
      return _sliceAndConcat<int16_t>(self, obj, dim);
    case framework::proto::VarType::INT32:
      return _sliceAndConcat<int>(self, obj, dim);
    case framework::proto::VarType::INT64:
      return _sliceAndConcat<int64_t>(self, obj, dim);
    case framework::proto::VarType::BOOL:
      return _sliceAndConcat<bool>(self, obj, dim);
    case framework::proto::VarType::UINT8:
      return _sliceAndConcat<uint8_t>(self, obj, dim);
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Not support tensor type: %s",
          framework::DataTypeToString(src_type)));
  }
}

inline framework::Tensor *_pySliceTensor(const framework::Tensor &self,
                                         py::object obj) {
  if (py::isinstance<py::tuple>(obj)) {
    py::list l = static_cast<py::list>(obj);
    std::unique_ptr<framework::Tensor> target;
    framework::Tensor *src = const_cast<framework::Tensor *>(&self);
    for (auto i = 0; i < static_cast<int>(l.size()); ++i) {
      src = _sliceTensor(*src, l[i], i);
      if (i + 1 == static_cast<int>(l.size())) {
        return src;
      } else {
        target.reset(src);
      }
    }
    return nullptr;
  } else {
    return _sliceTensor(self, obj, 0);
  }
}

inline framework::Tensor *PySliceTensor(const framework::Tensor &self,
                                        py::object obj) {
  if (platform::is_gpu_place(self.place())) {
    std::unique_ptr<framework::Tensor> holder;
    framework::Tensor src;
    framework::TensorCopySync(self, platform::CPUPlace(), &src);
    framework::Tensor *output = _pySliceTensor(src, obj);
    holder.reset(output);
    framework::Tensor *dst = _getTensor(*output, output->dims());
    framework::TensorCopySync(*output, self.place(), dst);
    return dst;
  } else {
    return _pySliceTensor(self, obj);
  }
}

inline py::array TensorToPyArray(const framework::Tensor &tensor,
                                 bool need_deep_copy = false) {
  if (!tensor.IsInitialized()) {
    return py::array();
  }
  bool is_gpu_tensor = platform::is_gpu_place(tensor.place());
  bool is_xpu_tensor = platform::is_xpu_place(tensor.place());
  const auto &tensor_dims = tensor.dims();
  auto tensor_dtype = tensor.type();
  size_t sizeof_dtype = framework::SizeOfType(tensor_dtype);

  std::vector<size_t> py_dims(tensor_dims.size());
  std::vector<size_t> py_strides(tensor_dims.size());

  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    py_dims[i] = (size_t)tensor_dims[i];
    py_strides[i] = sizeof_dtype * numel;
    numel *= py_dims[i];
  }

  const void *tensor_buf_ptr = tensor.data<void>();

  std::string py_dtype_str = details::TensorDTypeToPyDTypeStr(tensor.type());

  if (!is_gpu_tensor && !is_xpu_tensor) {
    if (!need_deep_copy) {
      auto base = py::cast(std::move(tensor));
      return py::array(py::dtype(py_dtype_str.c_str()), py_dims, py_strides,
                       const_cast<void *>(tensor_buf_ptr), base);
    } else {
      py::array py_arr(py::dtype(py_dtype_str.c_str()), py_dims, py_strides);
      PADDLE_ENFORCE_EQ(
          py_arr.writeable(), true,
          platform::errors::InvalidArgument(
              "PyArray is not writable, in which case memory leak "
              "or double free would occur"));
      PADDLE_ENFORCE_EQ(
          py_arr.owndata(), true,
          platform::errors::InvalidArgument(
              "PyArray does not own data, in which case  memory leak "
              "or double free would occur"));
      platform::CPUPlace place;
      size_t copy_bytes = sizeof_dtype * numel;
      paddle::memory::Copy(place, py_arr.mutable_data(), place, tensor_buf_ptr,
                           copy_bytes);
      return py_arr;
    }
  } else if (is_xpu_tensor) {
#ifdef PADDLE_WITH_XPU
    py::array py_arr(py::dtype(py_dtype_str.c_str()), py_dims, py_strides);
    PADDLE_ENFORCE_EQ(py_arr.writeable(), true,
                      platform::errors::InvalidArgument(
                          "PyArray is not writable, in which case memory leak "
                          "or double free would occur"));
    PADDLE_ENFORCE_EQ(
        py_arr.owndata(), true,
        platform::errors::InvalidArgument(
            "PyArray does not own data, in which case  memory leak "
            "or double free would occur"));

    size_t copy_bytes = sizeof_dtype * numel;
    auto p = BOOST_GET_CONST(platform::XPUPlace, tensor.place());
    paddle::memory::Copy(platform::CPUPlace(), py_arr.mutable_data(), p,
                         tensor_buf_ptr, copy_bytes);
    return py_arr;
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Cannot use XPUPlace in CPU/GPU version, "
        "Please recompile or reinstall Paddle with XPU support."));
#endif
  } else if (is_gpu_tensor) {
#ifdef PADDLE_WITH_CUDA
    py::array py_arr(py::dtype(py_dtype_str.c_str()), py_dims, py_strides);
    PADDLE_ENFORCE_EQ(py_arr.writeable(), true,
                      platform::errors::InvalidArgument(
                          "PyArray is not writable, in which case memory leak "
                          "or double free would occur"));
    PADDLE_ENFORCE_EQ(
        py_arr.owndata(), true,
        platform::errors::InvalidArgument(
            "PyArray does not own data, in which case  memory leak "
            "or double free would occur"));

    size_t copy_bytes = sizeof_dtype * numel;
    paddle::platform::GpuMemcpySync(py_arr.mutable_data(), tensor_buf_ptr,
                                    copy_bytes, cudaMemcpyDeviceToHost);
    return py_arr;
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Cannot use CUDAPlace in CPU only version, "
        "Please recompile or reinstall Paddle with CUDA support."));
#endif
  }
  PADDLE_THROW(platform::errors::Unimplemented("Place is not supported"));
  return py::array();
}

}  // namespace pybind
}  // namespace paddle
