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

#include <mkl.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag mklml_dso_flag;
extern void* mklml_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load mklml routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_MKLML_WRAP(__name)                                    \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {       \
      using mklmlFunc = decltype(&::__name);                               \
      std::call_once(mklml_dso_flag, []() {                                \
        mklml_dso_handle = paddle::platform::dynload::GetMKLMLDsoHandle(); \
      });                                                                  \
      static void* p_##_name = dlsym(mklml_dso_handle, #__name);           \
      return reinterpret_cast<mklmlFunc>(p_##_name)(args...);              \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_MKLML_WRAP(__name) DYNAMIC_LOAD_MKLML_WRAP(__name)

#define MKLML_ROUTINE_EACH(__macro) \
  __macro(cblas_sgemm);             \
  __macro(cblas_dgemm);             \
  __macro(cblas_cgemm);             \
  __macro(cblas_zgemm);             \
  __macro(cblas_saxpy);             \
  __macro(cblas_daxpy);             \
  __macro(cblas_caxpy);             \
  __macro(cblas_zaxpy);             \
  __macro(cblas_scopy);             \
  __macro(cblas_dcopy);             \
  __macro(cblas_ccopy);             \
  __macro(cblas_zcopy);             \
  __macro(cblas_sgemv);             \
  __macro(cblas_dgemv);             \
  __macro(cblas_cgemv);             \
  __macro(cblas_zgemv);             \
  __macro(cblas_strsm);             \
  __macro(cblas_dtrsm);             \
  __macro(cblas_sgemm_alloc);       \
  __macro(cblas_dgemm_alloc);       \
  __macro(cblas_sgemm_pack);        \
  __macro(cblas_dgemm_pack);        \
  __macro(cblas_sgemm_compute);     \
  __macro(cblas_dgemm_compute);     \
  __macro(cblas_sgemm_free);        \
  __macro(cblas_dgemm_free);        \
  __macro(cblas_sgemm_batch);       \
  __macro(cblas_dgemm_batch);       \
  __macro(cblas_cgemm_batch);       \
  __macro(cblas_zgemm_batch);       \
  __macro(cblas_sdot);              \
  __macro(cblas_ddot);              \
  __macro(cblas_sasum);             \
  __macro(cblas_dasum);             \
  __macro(cblas_isamax);            \
  __macro(cblas_idamax);            \
  __macro(cblas_sscal);             \
  __macro(cblas_dscal);             \
  __macro(vsAdd);                   \
  __macro(vdAdd);                   \
  __macro(vsSub);                   \
  __macro(vdSub);                   \
  __macro(vsMul);                   \
  __macro(vdMul);                   \
  __macro(vsDiv);                   \
  __macro(vdDiv);                   \
  __macro(vsExp);                   \
  __macro(vdExp);                   \
  __macro(vsSqr);                   \
  __macro(vdSqr);                   \
  __macro(vsPowx);                  \
  __macro(vdPowx);                  \
  __macro(vsInv);                   \
  __macro(vdInv);                   \
  __macro(vmsErf);                  \
  __macro(vmdErf);                  \
  __macro(MKL_Free_Buffers);        \
  __macro(MKL_Set_Num_Threads);     \
  __macro(MKL_Get_Max_Threads);

MKLML_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MKLML_WRAP);

#if !defined(_WIN32)
DYNAMIC_LOAD_MKLML_WRAP(mkl_scsrmm);
DYNAMIC_LOAD_MKLML_WRAP(mkl_dcsrmm);
#endif

#undef DYNAMIC_LOAD_MKLML_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
