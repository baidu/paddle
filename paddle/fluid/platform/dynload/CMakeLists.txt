cc_library(dynamic_loader SRCS dynamic_loader.cc DEPS glog gflags enforce)

list(APPEND CUDA_SRCS cublas.cc cudnn.cc curand.cc cusolver.cc nvtx.cc)

if (WITH_ROCM)
  list(APPEND HIP_SRCS rocblas.cc miopen.cc hiprand.cc)
endif()

# There is no macOS version of NCCL.
# Disable nvrtc and cuda_driver api on MacOS and Windows, and only do a early test on Linux.
if (NOT APPLE AND NOT WIN32)
    list(APPEND CUDA_SRCS nvrtc.cc cuda_driver.cc)
  if (WITH_NCCL)
    list(APPEND CUDA_SRCS nccl.cc)
  endif()
  if (WITH_ROCM)
    list(APPEND HIP_SRCS hiprtc.cc rocm_driver.cc)
    if (WITH_RCCL)
      list(APPEND HIP_SRCS rccl.cc)
    endif()
  endif()
endif()

if (TENSORRT_FOUND)
  list(APPEND CUDA_SRCS tensorrt.cc)
endif()

configure_file(cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
if (CUPTI_FOUND)
    list(APPEND CUDA_SRCS cupti.cc)
endif(CUPTI_FOUND)
if(WITH_ROCM)
  hip_library(dynload_cuda SRCS ${HIP_SRCS} DEPS dynamic_loader)
  cc_library(dynload_warpctc SRCS warpctc.cc DEPS dynamic_loader warpctc)
elseif (WITH_ASCEND_CL)
  cc_library(dynload_warpctc SRCS warpctc.cc hccl.cc DEPS dynamic_loader warpctc)
else()
  nv_library(dynload_cuda SRCS ${CUDA_SRCS} DEPS dynamic_loader)
  cc_library(dynload_warpctc SRCS warpctc.cc DEPS dynamic_loader warpctc)
endif()
if (WITH_MKLML)
    cc_library(dynload_mklml SRCS mklml.cc DEPS dynamic_loader mklml)
endif()
# TODO(TJ): add iomp, mkldnn?
