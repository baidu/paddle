# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(system)

project(paddle CXX C)

include(init)
include(generic)            # simplify cmake module

# enable language CUDA
# TODO(Shibo Tao): remove find_package(CUDA) completely.
find_package(CUDA QUIET)
option(WITH_GPU         "Compile PaddlePaddle with NVIDIA GPU"          ${CUDA_FOUND})
option(WITH_XPU         "Compile PaddlePaddle with BAIDU KUNLUN"        OFF)
if (WITH_GPU  AND WITH_XPU)
    message(FATAL_ERROR "Error when compile GPU and XPU at the same time")
endif()
# cmake 3.12, 3.13, 3.14 will append gcc link options to nvcc, and nvcc doesn't recognize them.
if(WITH_GPU AND (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.12) AND (${CMAKE_VERSION} VERSION_LESS 3.15))
    message(FATAL_ERROR "cmake ${CMAKE_VERSION} is not supported when WITH_GPU=ON because of bug https://cmake.org/pipermail/cmake/2018-September/068195.html. "
       "You can use cmake 3.16 (recommended), 3.10, 3.11, 3.15 or 3.17. Please refer to the install document: https://cmake.org/install/")
endif()

if(WITH_GPU AND NOT APPLE)
    enable_language(CUDA)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}, version: "
        "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

if(WIN32)
    option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)

    set(CMAKE_SUPPRESS_REGENERATION ON)
    set(CMAKE_STATIC_LIBRARY_PREFIX lib)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")

    if (MSVC_STATIC_CRT)
        message(STATUS "Use static C runtime time, refer to https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=vs-2019")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
        set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
        set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
        foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
            if(${flag_var} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            endif()
        endforeach(flag_var)
    endif()

    # windows build turn off warnings.
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
        string(REGEX REPLACE "/W[1-4]" " /W0 " ${flag_var} "${${flag_var}}")
    endforeach(flag_var)
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
        set(${flag_var} "${${flag_var}} /w")
    endforeach(flag_var)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838 /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838 /MP")
    message(STATUS "Using parallel compiling (/MP)")
    set(PADDLE_LINK_FLAGS "/IGNORE:4006 /IGNORE:4098 /IGNORE:4217 /IGNORE:4221")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
else(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
endif(WIN32)

find_package(Git REQUIRED)
find_package(Threads REQUIRED)

include(simd)

################################ Exposed Configurations #######################################
option(WITH_AVX         "Compile PaddlePaddle with AVX intrinsics"      ${AVX_FOUND})
option(WITH_PYTHON      "Compile PaddlePaddle with python interpreter"  ON)
option(WITH_TESTING     "Compile PaddlePaddle with unit testing"        OFF)
option(WITH_MKL         "Compile PaddlePaddle with MKL support."        ${AVX_FOUND})
option(WITH_SYSTEM_BLAS   "Use system blas library"           OFF)
option(WITH_DISTRIBUTE  "Compile with distributed support"              OFF)
option(WITH_BRPC_RDMA     "Use brpc rdma as the rpc protocal"           OFF)
option(ON_INFER         "Turn on inference optimization and inference-lib generation" OFF)
################################ Internal Configurations #######################################
option(WITH_AMD_GPU     "Compile PaddlePaddle with AMD GPU"             OFF)
option(WITH_NV_JETSON     "Compile PaddlePaddle with NV JETSON"             OFF)
option(WITH_PROFILER    "Compile PaddlePaddle with GPU profiler and gperftools"        OFF)
option(WITH_COVERAGE    "Compile PaddlePaddle with code coverage"       OFF)
OPTION(WITH_LIBXSMM     "Compile with libxsmm"                          OFF)
option(COVERALLS_UPLOAD "Package code coverage data to coveralls"       OFF)
option(WITH_PSLIB       "Compile with pslib support"                    OFF)
option(WITH_BOX_PS      "Compile with box_ps support"                   OFF)
option(WITH_XBYAK       "Compile with xbyak support"                    ON)
option(WITH_CONTRIB     "Compile the third-party contributation"        OFF)
option(WITH_GRPC     "Use grpc as the default rpc framework"            ${WITH_DISTRIBUTE})
option(WITH_INFERENCE_API_TEST   "Test fluid inference C++ high-level api interface"  OFF)
option(PY_VERSION       "Compile PaddlePaddle with python3 support"     ${PY_VERSION})
option(WITH_DGC   "Use DGC(Deep Gradient Compression) or not" ${WITH_DISTRIBUTE})
option(SANITIZER_TYPE "Choose the type of sanitizer, options are: Address, Leak, Memory, Thread, Undefined" OFF)
option(WITH_LITE   "Compile Paddle Fluid with Lite Engine" OFF)
option(WITH_NCCL   "Compile PaddlePaddle with NCCL support"             ON)
option(WITH_CRYPTO   "Compile PaddlePaddle with crypto support"         ON)
option(WITH_ARM   "Compile PaddlePaddle with arm support"         OFF)
option(WITH_MUSL        "Compile with musl libc instead of gblic"  OFF)

# PY_VERSION
if(NOT PY_VERSION)
  set(PY_VERSION 2.7)
endif()
set(PYBIND11_PYTHON_VERSION ${PY_VERSION})

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
endif()

# the type of sanitizer, options are: Address, Leak, Memory, Thread, Undefined. Default: OFF
if(SANITIZER_TYPE AND NOT "${SANITIZER_TYPE}" MATCHES "^(Address|Leak|Memory|Thread|Undefined)$")
  message("Choose the correct type of sanitizer")
  return()
endif()

if(WIN32)
    if(WITH_DISTRIBUTE)
        MESSAGE(WARNING
            "Disable DISTRIBUTE when compiling for Windows. Force WITH_DISTRIBUTE=OFF.")
        set(WITH_DISTRIBUTE OFF CACHE STRING
            "Disable DISTRIBUTE when compiling for Windows" FORCE)
    endif()
    if(WITH_NCCL)
        MESSAGE(WARNING
            "Disable NCCL when compiling for Windows. Force WITH_NCCL=OFF.")
        set(WITH_NCCL OFF CACHE STRING
            "Disable NCCL when compiling for Windows" FORCE)
    endif()
endif()

if (NOT WITH_GPU AND WITH_NCCL)
    MESSAGE(WARNING
        "Disable NCCL when compiling without GPU. Force WITH_NCCL=OFF.")
    set(WITH_NCCL OFF CACHE STRING
        "Disable NCCL when compiling without GPU" FORCE)
endif()

if(WITH_NCCL)
     add_definitions("-DPADDLE_WITH_NCCL")
     include(nccl)
else()
     if(WITH_GPU)
         MESSAGE(WARNING "If the environment is multi-card, the WITH_NCCL option needs to be turned on, otherwise only a single card can be used.")
     endif()
endif()

if(WITH_BRPC_RDMA)
    message(STATUS "Use brpc with rdma.")
    if(WITH_GRPC)
        message(FATAL_ERROR "Can't use grpc with brpc rdma.")
    endif()
    if(NOT WITH_DISTRIBUTE)
        message(FATAL_ERROR "Can't use brpc rdma in no distribute env.")
    endif()
endif()

# lite subgraph compilation depends on CUDNN_ROOT,
# so include(cudnn) needs to be in front of include(third_party/lite)
include(cudnn)              # set cudnn libraries, must before configure
include(third_party)        # download, build, install third_party

if(WITH_DISTRIBUTE)
    if(WITH_GRPC)
        message(STATUS "Use grpc framework.")
        include(external/grpc)
    else()
        message(STATUS "Use brpc framework.")
        include(external/leveldb)
        include(external/brpc)
    endif()
endif()

# there is no official support of nccl, cupti in windows
if(NOT WIN32)
    include(cupti)
endif()

include(flags)              # set paddle compile flags

if(WITH_GPU)
    include(cuda)
    include(tensorrt)
endif()

if(WITH_PROFILER)
    find_package(Gperftools REQUIRED)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
    add_definitions(-DWITH_GPERFTOOLS)
endif()

include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(version)            # set PADDLE_VERSION
include(coveralls)          # set code coverage
include(configure)          # add paddle env configuration

include_directories("${PADDLE_SOURCE_DIR}")

if(WITH_AMD_GPU)
    find_package(HIP)
    include(hip)
endif(WITH_AMD_GPU)

if(WITH_DISTRIBUTE)
    if(LINUX)
        set(WITH_GLOO ON CACHE STRING "Enable GLOO when compiling WITH_DISTRIBUTE=ON." FORCE)
    endif()
endif()

if(WITH_ARM)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(WITH_XBYAK OFF CACHE STRING "Disable XBYAK when compiling WITH_ARM=ON" FORCE)
    set(WITH_MKL OFF CACHE STRING "Disable MKL when compiling WITH_ARM=ON." FORCE)
    set(WITH_GPU OFF CACHE STRING "Disable GPU when compiling WITH_ARM=ON." FORCE)
    add_definitions(-DPADDLE_WITH_ARM)
endif()

set(PADDLE_PYTHON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/build")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

if(ON_INFER)
    # you can trun off the paddle fluid and inference lib by set ON_INFER=OFF
    message(STATUS "On inference mode, will take place some specific optimization.")
    include(inference_lib)
    add_definitions(-DPADDLE_ON_INFERENCE)
else()
    #TODO(luotao), combine this warning with `make inference_lib_dist` command.
    message(WARNING "On inference mode, will take place some specific optimization. Turn on the ON_INFER flag when building inference_lib only.")
endif()

add_subdirectory(paddle)
if(WITH_PYTHON)
    add_subdirectory(python)
endif()
