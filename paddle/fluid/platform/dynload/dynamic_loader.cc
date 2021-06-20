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
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/dynload/cupti_lib_path.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_string(cudnn_dir, "",
              "Specify path for loading libcudnn.so. For instance, "
              "/usr/local/cudnn/lib. If empty [default], dlopen "
              "will search cudnn from LD_LIBRARY_PATH");

DEFINE_string(cuda_dir, "",
              "Specify path for loading cuda library, such as libcublas, "
              "libcurand, libcusolver. For instance, /usr/local/cuda/lib64. "
              "If default, dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(nccl_dir, "",
              "Specify path for loading nccl library, such as libnccl.so. "
              "For instance, /usr/local/cuda/lib64. If default, "
              "dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(hccl_dir, "",
              "Specify path for loading hccl library, such as libhccl.so. "
              "For instance, "
              "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/. If "
              "default, "
              "dlopen will search hccl from LD_LIBRARY_PATH");

DEFINE_string(cupti_dir, "", "Specify path for loading cupti.so.");

DEFINE_string(
    tensorrt_dir, "",
    "Specify path for loading tensorrt library, such as libnvinfer.so.");

DEFINE_string(mklml_dir, "", "Specify path for loading libmklml_intel.so.");

DEFINE_string(op_dir, "", "Specify path for loading user-defined op library.");

#ifdef PADDLE_WITH_HIP

DEFINE_string(miopen_dir, "",
              "Specify path for loading libMIOpen.so. For instance, "
              "/opt/rocm/miopen/lib. If empty [default], dlopen "
              "will search miopen from LD_LIBRARY_PATH");

DEFINE_string(rocm_dir, "",
              "Specify path for loading rocm library, such as librocblas, "
              "libmiopen, libhipsparse. For instance, /opt/rocm/lib. "
              "If default, dlopen will search rocm from LD_LIBRARY_PATH");

DEFINE_string(rccl_dir, "",
              "Specify path for loading rccl library, such as librccl.so. "
              "For instance, /opt/rocm/rccl/lib. If default, "
              "dlopen will search rccl from LD_LIBRARY_PATH");
#endif

namespace paddle {
namespace platform {
namespace dynload {

struct PathNode {
  PathNode() {}
  std::string path = "";
};

static constexpr char cupti_lib_path[] = CUPTI_LIB_PATH;

// NOTE: In order to adapt to the default installation path of cuda
#if defined(_WIN32) && defined(PADDLE_WITH_CUDA)
static constexpr char cuda_lib_path[] = CUDA_TOOLKIT_ROOT_DIR "/bin";
#else
static constexpr char cuda_lib_path[] = "/usr/local/cuda/lib64";
#endif

static PathNode s_py_site_pkg_path;

#if defined(_WIN32) && defined(PADDLE_WITH_CUDA)
static constexpr char* win_cudnn_lib = "cudnn64_" CUDNN_MAJOR_VERSION ".dll";
static constexpr char* win_cublas_lib =
    "cublas64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cublas64_" CUDA_VERSION_MAJOR ".dll";
#if CUDA_VERSION >= 11000
static constexpr char* win_curand_lib =
    "curand64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;curand64_" CUDA_VERSION_MAJOR ".dll;curand64_10.dll";
static constexpr char* win_nvjpeg_lib =
    "nvjpeg64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;nvjpeg64_" CUDA_VERSION_MAJOR ".dll;nvjpeg64_10.dll";
static constexpr char* win_cusolver_lib =
    "cusolver64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cusolver64_" CUDA_VERSION_MAJOR ".dll;cusolver64_10.dll";
#else
static constexpr char* win_curand_lib =
    "curand64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;curand64_" CUDA_VERSION_MAJOR ".dll";
static constexpr char* win_nvjpeg_lib =
    "nvjpeg64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;nvjpeg64_" CUDA_VERSION_MAJOR ".dll";
static constexpr char* win_cusolver_lib =
    "cusolver64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cusolver64_" CUDA_VERSION_MAJOR ".dll";
#endif  // CUDA_VERSION
#endif

static inline std::string join(const std::string& part1,
                               const std::string& part2) {
  // directory separator
  const char sep = '/';
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

static inline std::vector<std::string> split(
    const std::string& str, const std::string separator = " ") {
  std::vector<std::string> str_list;
  std::string::size_type firstPos;
  firstPos = str.find_first_not_of(separator, 0);
  std::string::size_type lastPos;
  lastPos = str.find_first_of(separator, firstPos);
  while (std::string::npos != firstPos && std::string::npos != lastPos) {
    str_list.push_back(str.substr(firstPos, lastPos - firstPos));
    firstPos = str.find_first_not_of(separator, lastPos);
    lastPos = str.find_first_of(separator, firstPos);
  }
  if (std::string::npos == lastPos) {
    str_list.push_back(str.substr(firstPos, lastPos - firstPos));
  }
  return str_list;
}

void SetPaddleLibPath(const std::string& py_site_pkg_path) {
  s_py_site_pkg_path.path = py_site_pkg_path;
  VLOG(3) << "Set paddle lib path : " << py_site_pkg_path;
}

static inline void* GetDsoHandleFromSpecificPath(const std::string& spec_path,
                                                 const std::string& dso_name,
                                                 int dynload_flags) {
  void* dso_handle = nullptr;
  if (!spec_path.empty()) {
    // search xxx.so from custom path
    VLOG(3) << "Try to find library: " << dso_name
            << " from specific path: " << spec_path;
    std::string dso_path = join(spec_path, dso_name);
    dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  }
  return dso_handle;
}

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path,
                                                int dynload_flags) {
  // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
  // and /usr/local/lib path
  void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  VLOG(3) << "Try to find library: " << dso_path
          << " from default system path.";

// TODO(chenweihang): This path is used to search which libs?
// DYLD_LIBRARY_PATH is disabled after Mac OS 10.11 to
// bring System Integrity Projection (SIP), if dso_handle
// is null, search from default package path in Mac OS.
#if defined(__APPLE__) || defined(__OSX__)
  if (nullptr == dso_handle) {
    dso_handle =
        dlopen(join("/usr/local/cuda/lib/", dso_path).c_str(), dynload_flags);
  }
#endif

  return dso_handle;
}

/*
 * We define three priorities for dynamic library search:
 *
 * First: Search for the path specified by the user
 * Second: Search the system default path
 * Third: Search for a special path corresponding to
 *        a specific library to adapt to changes and easy to expand.
 */

static inline void* GetDsoHandleFromSearchPath(
    const std::string& config_path, const std::string& dso_name,
    bool throw_on_error = true,
    const std::vector<std::string>& extra_paths = std::vector<std::string>(),
    const std::string& warning_msg = std::string()) {
#if !defined(_WIN32)
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
#else
  int dynload_flags = 0;
#endif  // !_WIN32
  std::vector<std::string> dso_names = split(dso_name, ";");
  void* dso_handle = nullptr;
  for (auto dso : dso_names) {
    // 1. search in user config path by FLAGS
    dso_handle = GetDsoHandleFromSpecificPath(config_path, dso, dynload_flags);
    // 2. search in extra paths
    if (nullptr == dso_handle) {
      for (auto path : extra_paths) {
        VLOG(3) << "extra_paths: " << path;
        dso_handle = GetDsoHandleFromSpecificPath(path, dso, dynload_flags);
      }
    }
    // 3. search in system default path
    if (nullptr == dso_handle) {
      dso_handle = GetDsoHandleFromDefaultPath(dso, dynload_flags);
    }
    if (nullptr != dso_handle) break;
  }

  // 4. [If Failed for All dso_names] logging warning if exists
  if (nullptr == dso_handle && !warning_msg.empty()) {
    LOG(WARNING) << warning_msg;
  }

  // 5. [If Failed for All dso_names] logging or throw error info
  if (nullptr == dso_handle) {
    auto error_msg =
        "The third-party dynamic library (%s) that Paddle depends on is not "
        "configured correctly. (error code is %s)\n"
        "  Suggestions:\n"
        "  1. Check if the third-party dynamic library (e.g. CUDA, CUDNN) "
        "is installed correctly and its version is matched with paddlepaddle "
        "you installed.\n"
        "  2. Configure third-party dynamic library environment variables as "
        "follows:\n"
        "  - Linux: set LD_LIBRARY_PATH by `export LD_LIBRARY_PATH=...`\n"
        "  - Windows: set PATH by `set PATH=XXX;%PATH%`\n"
        "  - Mac: set  DYLD_LIBRARY_PATH by `export DYLD_LIBRARY_PATH=...` "
        "[Note: After Mac OS 10.11, using the DYLD_LIBRARY_PATH is "
        "impossible unless System Integrity Protection (SIP) is disabled.]";
#if !defined(_WIN32)
    auto errorno = dlerror();
#else
    auto errorno = GetLastError();
#endif  // !_WIN32
    if (throw_on_error) {
      // NOTE: Special error report case, no need to change its format
      PADDLE_THROW(
          platform::errors::PreconditionNotMet(error_msg, dso_name, errorno));
    } else {
      LOG(WARNING) << string::Sprintf(error_msg, dso_name, errorno);
    }
  }

  return dso_handle;
}

void* GetCublasDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, win_cublas_lib, true,
                                    {cuda_lib_path});
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "librocblas.so");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.so");
#endif
}

void* GetCUDNNDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  std::string mac_warn_meg(
      "Note: [Recommend] copy cudnn into /usr/local/cuda/ \n "
      "For instance, sudo tar -xzf "
      "cudnn-7.5-osx-x64-v5.0-ga.tgz -C /usr/local \n sudo "
      "chmod a+r /usr/local/cuda/include/cudnn.h "
      "/usr/local/cuda/lib/libcudnn*");
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.dylib", false,
                                    {}, mac_warn_meg);
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  std::string win_warn_meg(
      "Note: [Recommend] copy cudnn into CUDA installation directory. \n "
      "For instance, download cudnn-10.0-windows10-x64-v7.6.5.32.zip from "
      "NVIDIA's official website, \n"
      "then, unzip it and copy it into C:\\Program Files\\NVIDIA GPU Computing "
      "Toolkit\\CUDA\\v10.0\n"
      "You should do this according to your CUDA installation directory and "
      "CUDNN version.");
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, win_cudnn_lib, true,
                                    {cuda_lib_path}, win_warn_meg);
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_miopen_dir, "libMIOpen.so", false);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cudnn_dir, "libcudnn.so", false,
                                    {cuda_lib_path});
#endif
}

void* GetCUPTIDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cupti_dir, "libcupti.dylib", false,
                                    {cupti_lib_path});
#else
  return GetDsoHandleFromSearchPath(FLAGS_cupti_dir, "libcupti.so", false,
                                    {cupti_lib_path});
#endif
}

void* GetCurandDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, win_curand_lib, true,
                                    {cuda_lib_path});
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libhiprand.so");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.so");
#endif
}

void* GetNvjpegDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvjpeg.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, win_nvjpeg_lib, true,
                                    {cuda_lib_path});
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvjpeg.so");
#endif
}

void* GetCusolverDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusolver.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, win_cusolver_lib, true,
                                    {cuda_lib_path});
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusolver.so");
#endif
}

void* GetNVRTCDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvrtc.dylib", false);
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libamdhip64.so", false);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvrtc.so", false);
#endif
}

void* GetCUDADsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcuda.dylib", false);
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libamdhip64.so", false);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcuda.so", false);
#endif
}

void* GetWarpCTCDsoHandle() {
  std::string warpctc_dir = "";
  if (!s_py_site_pkg_path.path.empty()) {
    warpctc_dir = s_py_site_pkg_path.path;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(warpctc_dir, "warpctc.dll");
#else
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.so");
#endif
}

void* GetNCCLDsoHandle() {
#ifdef PADDLE_WITH_HIP
  std::string warning_msg(
      "You may need to install 'rccl' from ROCM official website: "
      "https://rocmdocs.amd.com/en/latest/Installation_Guide/"
      "Installation-Guide.html before install PaddlePaddle.");
#else
  std::string warning_msg(
      "You may need to install 'nccl2' from NVIDIA official website: "
      "https://developer.nvidia.com/nccl/nccl-download"
      "before install PaddlePaddle.");
#endif

#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.dylib", true, {},
                                    warning_msg);
#elif defined(PADDLE_WITH_HIP) && defined(PADDLE_WITH_RCCL)
  return GetDsoHandleFromSearchPath(FLAGS_rccl_dir, "librccl.so", true, {},
                                    warning_msg);
#else
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.so", true, {},
                                    warning_msg);
#endif
}
void* GetHCCLDsoHandle() {
  std::string warning_msg(
      "You may need to install 'hccl2' from Huawei official website: "
      "before install PaddlePaddle.");
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.dylib", true, {},
                                    warning_msg);
#elif defined(PADDLE_WITH_HIP) && defined(PADDLE_WITH_RCCL)
  return GetDsoHandleFromSearchPath(FLAGS_rccl_dir, "librccl.so", true);

#elif defined(PADDLE_WITH_ASCEND_CL)
  return GetDsoHandleFromSearchPath(FLAGS_hccl_dir, "libeccl.so", true, {},
                                    warning_msg);
#else
  return GetDsoHandleFromSearchPath(FLAGS_nccl_dir, "libnccl.so", true, {},
                                    warning_msg);
#endif
}

void* GetTensorRtDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "nvinfer.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.so");
#endif
}

void* GetMKLMLDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "libmklml_intel.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "mklml.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "libmklml_intel.so");
#endif
}

void* GetOpDsoHandle(const std::string& dso_name) {
  return GetDsoHandleFromSearchPath(FLAGS_op_dir, dso_name);
}

void* GetNvtxDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  PADDLE_THROW(platform::errors::Unimplemented("Nvtx do not support Apple."));
#elif defined(_WIN32)
  PADDLE_THROW(platform::errors::Unimplemented("Nvtx do not support Windows."));
#elif !defined(PADDLE_WITH_CUDA)
  PADDLE_THROW(
      platform::errors::Unimplemented("Nvtx do not support without CUDA."));
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvToolsExt.so");
#endif
}

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
