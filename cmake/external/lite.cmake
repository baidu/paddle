# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

if(NOT LINUX)
  message("Paddle-lite will not build because the required Linux do not exist.")
  set(WITH_LITE OFF)
  return()
endif()

if (LITE_WITH_XPU)
  add_definitions(-DLITE_SUBGRAPH_WITH_XPU)
  IF(WITH_AARCH64)
    SET(XPU_SDK_ENV "kylin_aarch64")
  ELSEIF(WITH_SUNWAY)
    SET(XPU_SDK_ENV "deepin_sw6_64")
  ELSEIF(WITH_BDCENTOS)
    SET(XPU_SDK_ENV "bdcentos_x86_64")
  ELSEIF(WITH_UBUNTU)
    SET(XPU_SDK_ENV "ubuntu_x86_64")
  ELSEIF(WITH_CENTOS)
    SET(XPU_SDK_ENV "centos7_x86_64")
  ELSE ()
    SET(XPU_SDK_ENV "ubuntu_x86_64")
  ENDIF()
endif()

if (NOT LITE_SOURCE_DIR OR NOT LITE_BINARY_DIR)
  include(ExternalProject)
  set(LITE_PROJECT extern_lite)
  set(LITE_SOURCES_DIR ${THIRD_PARTY_PATH}/lite)
  set(LITE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/lite)

  if(NOT LITE_GIT_TAG)
    set(LITE_GIT_TAG 7d54f557529e2f1a6b3daa4e35c1bf32bac002b8) # lite branch is release/v2.9 and the xpu_tool_chain is 2.0.0.128
  endif()

  if(NOT CUDA_ARCH_NAME)
    set(CUDA_ARCH_NAME "Auto")
  endif()

  # No quotes, so cmake can resolve it as a command with arguments.
  if(WITH_ARM)
    set(LITE_BUILD_COMMAND $(MAKE) publish_inference -j)
    message(WARNING "BUILD_COMMAND: ${LITE_BUILD_COMMAND}")
    set(LITE_OPTIONAL_ARGS -DWITH_MKL=OFF
                           -DLITE_WITH_CUDA=OFF
                           -DWITH_MKLDNN=OFF
                           -DLITE_WITH_X86=OFF
                           -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON
                           -DLITE_WITH_PROFILE=OFF
                           -DARM_TARGET_OS=armlinux
                           -DWITH_LITE=ON
                           -DWITH_PYTHON=OFF
                           -DWITH_TESTING=OFF
                           -DLITE_BUILD_EXTRA=ON
                           -DLITE_WITH_XPU=${LITE_WITH_XPU}
                           -DXPU_SDK_URL=${XPU_BASE_URL}
                           -DXPU_SDK_ENV=${XPU_SDK_ENV}
                           -DLITE_WITH_CODE_META_INFO=OFF
                           -DLITE_WITH_ARM=ON)
    ExternalProject_Add(
      ${LITE_PROJECT}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      GIT_REPOSITORY      "${GIT_URL}/PaddlePaddle/Paddle-Lite.git"
      GIT_TAG             ${LITE_GIT_TAG}
      PREFIX              ${LITE_SOURCES_DIR}
      PATCH_COMMAND       mkdir -p ${LITE_SOURCES_DIR}/src/extern_lite-build/lite/gen_code && touch ${LITE_SOURCES_DIR}/src/extern_lite-build/lite/gen_code/__generated_code__.cc && sed -i "/aarch64-linux-gnu-gcc/d" ${LITE_SOURCES_DIR}/src/extern_lite/cmake/cross_compiling/armlinux.cmake && sed -i "/aarch64-linux-gnu-g++/d" ${LITE_SOURCES_DIR}/src/extern_lite/cmake/cross_compiling/armlinux.cmake
      UPDATE_COMMAND      ""
      BUILD_COMMAND       ${LITE_BUILD_COMMAND}
      INSTALL_COMMAND     ""
      CMAKE_ARGS          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                          -DCMAKE_CXX_FLAGS=${LITE_CMAKE_CXX_FLAGS}
                          -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                          -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                          -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                          -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                          -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                          -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                          ${EXTERNAL_OPTIONAL_ARGS}
                          ${LITE_OPTIONAL_ARGS}
    )
  else()
    set(LITE_BUILD_COMMAND $(MAKE) publish_inference -j)
    set(LITE_OPTIONAL_ARGS -DWITH_MKL=ON
                           -DLITE_WITH_CUDA=${WITH_GPU}
                           -DWITH_MKLDNN=OFF
                           -DLITE_WITH_X86=ON
                           -DLITE_WITH_PROFILE=OFF
                           -DWITH_LITE=OFF
                           -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
                           -DWITH_PYTHON=OFF
                           -DWITH_TESTING=OFF
                           -DLITE_BUILD_EXTRA=ON
                           -DCUDNN_ROOT=${CUDNN_ROOT}
                           -DLITE_WITH_STATIC_CUDA=OFF
                           -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME}
                           -DLITE_WITH_XPU=${LITE_WITH_XPU}
                           -DXPU_SDK_URL=${XPU_BASE_URL}
                           -DXPU_SDK_ENV=${XPU_SDK_ENV}
                           -DLITE_WITH_CODE_META_INFO=OFF
                           -DLITE_WITH_ARM=OFF)

    ExternalProject_Add(
        ${LITE_PROJECT}
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY      "${GIT_URL}/PaddlePaddle/Paddle-Lite.git"
        GIT_TAG             ${LITE_GIT_TAG}
        PREFIX              ${LITE_SOURCES_DIR}
        UPDATE_COMMAND      ""
        BUILD_COMMAND       ${LITE_BUILD_COMMAND}
        INSTALL_COMMAND     ""
        CMAKE_ARGS          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            -DCMAKE_CXX_FLAGS=${LITE_CMAKE_CXX_FLAGS}
                            -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                            -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                            -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                            -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                            -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                            ${EXTERNAL_OPTIONAL_ARGS}
                            ${LITE_OPTIONAL_ARGS}
    )
  endif()
  ExternalProject_Get_property(${LITE_PROJECT} BINARY_DIR)
  ExternalProject_Get_property(${LITE_PROJECT} SOURCE_DIR)
  set(LITE_BINARY_DIR ${BINARY_DIR})
  set(LITE_SOURCE_DIR ${SOURCE_DIR})

endif()

if (WITH_ARM)
  if(LITE_WITH_XPU)
    set(LITE_OUTPUT_BIN_DIR inference_lite_lib.armlinux.armv8.xpu)
  else()
    set(LITE_OUTPUT_BIN_DIR inference_lite_lib.armlinux.armv8)
  endif()
else()
  set(LITE_OUTPUT_BIN_DIR inference_lite_lib)
endif()

message(STATUS "Paddle-lite BINARY_DIR: ${LITE_BINARY_DIR}")
message(STATUS "Paddle-lite SOURCE_DIR: ${LITE_SOURCE_DIR}")
include_directories(${LITE_SOURCE_DIR})
include_directories(${LITE_BINARY_DIR})
if(LITE_WITH_XPU)
  include_directories(${LITE_BINARY_DIR}/third_party/install/xpu/xdnn/include/)
  include_directories(${LITE_BINARY_DIR}/third_party/install/xpu/xre/include/)
endif()

function(external_lite_libs alias path)
  add_library(${alias} SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET ${alias} PROPERTY IMPORTED_LOCATION
               ${path})
  if (LITE_PROJECT)
    add_dependencies(${alias} ${LITE_PROJECT})
  endif()
endfunction()

external_lite_libs(lite_full_static ${LITE_BINARY_DIR}/${LITE_OUTPUT_BIN_DIR}/cxx/lib/libpaddle_full_api_shared.so)
set(LITE_SHARED_LIB ${LITE_BINARY_DIR}/${LITE_OUTPUT_BIN_DIR}/cxx/lib/libpaddle_full_api_shared.so)

add_definitions(-DPADDLE_WITH_LITE)
add_definitions(-DLITE_WITH_LOG)
