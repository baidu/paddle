# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# make package for paddle fluid shared and static library
set(PADDLE_INSTALL_DIR "${CMAKE_BINARY_DIR}/paddle_install_dir" CACHE STRING
  "A path setting paddle shared and static libraries")

set(PADDLE_INFERENCE_INSTALL_DIR "${CMAKE_BINARY_DIR}/paddle_inference_install_dir" CACHE STRING
  "A path setting paddle inference shared and static libraries")
  
# At present, the size of static lib in Windows is very large,
# so we need to crop the library size.
if(WIN32)
    #todo: remove the option 
    option(WITH_STATIC_LIB "Compile demo with static/shared library, default use dynamic."   OFF)
    if(NOT PYTHON_EXECUTABLE)
        FIND_PACKAGE(PythonInterp REQUIRED)
    endif()
endif()

set(COPY_SCRIPT_DIR ${PADDLE_SOURCE_DIR}/cmake)
function(copy TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DSTS)
    cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
    list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
    if (NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
        message(FATAL_ERROR "${TARGET} source numbers are not equal to destination numbers")
    endif ()
    math(EXPR len "${copy_lib_SRCS_len} - 1")
    foreach (index RANGE ${len})
        list(GET copy_lib_SRCS ${index} src)
        list(GET copy_lib_DSTS ${index} dst)
        if (WIN32)   #windows
            file(TO_NATIVE_PATH ${src} native_src)
            file(TO_NATIVE_PATH ${dst} native_dst)
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND ${PYTHON_EXECUTABLE} ${COPY_SCRIPT_DIR}/copyfile.py ${native_src} ${native_dst})
        else (WIN32) #not windows
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND mkdir -p "${dst}"
                    COMMAND cp -r "${src}" "${dst}"
                    COMMENT "copying ${src} -> ${dst}")
        endif (WIN32) # not windows
    endforeach ()
endfunction()

function(copy_part_of_thrid_party TARGET DST) 
    if(${CBLAS_PROVIDER} STREQUAL MKLML)
        set(dst_dir "${DST}/third_party/install/mklml")
        if(WIN32)
            copy(${TARGET}
                    SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_SHARED_LIB}
                    ${MKLML_SHARED_IOMP_LIB} ${MKLML_INC_DIR}
                    DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}/lib
                    ${dst_dir}/lib ${dst_dir})
        else()
            copy(${TARGET}
                    SRCS ${MKLML_LIB} ${MKLML_IOMP_LIB} ${MKLML_INC_DIR}
                    DSTS ${dst_dir}/lib ${dst_dir}/lib ${dst_dir})
        endif()
    elseif(${CBLAS_PROVIDER} STREQUAL EXTERN_OPENBLAS)
        set(dst_dir "${DST}/third_party/install/openblas")
	if(WIN32)
            copy(${TARGET}
                    SRCS ${CBLAS_INSTALL_DIR}/lib ${OPENBLAS_SHARED_LIB} ${CBLAS_INSTALL_DIR}/include
                    DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir})
	else()
            copy(${TARGET}
                    SRCS ${CBLAS_INSTALL_DIR}/lib ${CBLAS_INSTALL_DIR}/include
                    DSTS ${dst_dir} ${dst_dir})
	endif()
    endif()

    if(WITH_MKLDNN)
        set(dst_dir "${DST}/third_party/install/mkldnn")
        if(WIN32)
            copy(${TARGET}
                    SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB}  ${MKLDNN_LIB}
                    DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir}/lib)
        else()
            copy(${TARGET}
                    SRCS ${MKLDNN_INC_DIR} ${MKLDNN_SHARED_LIB} ${MKLDNN_SHARED_LIB_1} ${MKLDNN_SHARED_LIB_2}
                    DSTS ${dst_dir} ${dst_dir}/lib ${dst_dir}/lib ${dst_dir}/lib)
        endif()
    endif()

    set(dst_dir "${DST}/third_party/install/gflags")
    copy(${TARGET}
            SRCS ${GFLAGS_INCLUDE_DIR} ${GFLAGS_LIBRARIES}
            DSTS ${dst_dir} ${dst_dir}/lib)

    set(dst_dir "${DST}/third_party/install/glog")
    copy(${TARGET}
            SRCS ${GLOG_INCLUDE_DIR} ${GLOG_LIBRARIES}
            DSTS ${dst_dir} ${dst_dir}/lib)

    if (WITH_CRYPTO)
        set(dst_dir "${DST}/third_party/install/cryptopp")
        copy(${TARGET}
            SRCS ${CRYPTOPP_INCLUDE_DIR} ${CRYPTOPP_LIBRARIES}
            DSTS ${dst_dir} ${dst_dir}/lib)
    endif()

    set(dst_dir "${DST}/third_party/install/xxhash")
    copy(${TARGET}
        SRCS ${XXHASH_INCLUDE_DIR} ${XXHASH_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib)    

    if (NOT PROTOBUF_FOUND OR WIN32)
        set(dst_dir "${DST}/third_party/install/protobuf")
        copy(${TARGET}
                SRCS ${PROTOBUF_INCLUDE_DIR} ${PROTOBUF_LIBRARY}
                DSTS ${dst_dir} ${dst_dir}/lib)
    endif ()

    if (LITE_BINARY_DIR)
        set(dst_dir "${DST}/third_party/install/lite")
        copy(${TARGET}
                SRCS ${LITE_BINARY_DIR}/${LITE_OUTPUT_BIN_DIR}/*
                DSTS ${dst_dir})
    endif()
endfunction()

# inference library for only inference
set(inference_lib_deps third_party paddle_inference paddle_inference_c paddle_inference_shared paddle_inference_c_shared)
add_custom_target(inference_lib_dist DEPENDS ${inference_lib_deps})


set(dst_dir "${PADDLE_INFERENCE_INSTALL_DIR}/third_party/threadpool")
copy(inference_lib_dist
        SRCS ${THREADPOOL_INCLUDE_DIR}/ThreadPool.h
        DSTS ${dst_dir})

# GPU must copy externalErrorMsg.pb
IF(WITH_GPU)
    set(dst_dir "${PADDLE_INFERENCE_INSTALL_DIR}/third_party/externalError/data")
    copy(inference_lib_dist
            SRCS ${externalError_INCLUDE_DIR}
            DSTS ${dst_dir})
ENDIF()

# CMakeCache Info
copy(inference_lib_dist
        SRCS ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
        DSTS ${PADDLE_INFERENCE_INSTALL_DIR})

copy_part_of_thrid_party(inference_lib_dist ${PADDLE_INFERENCE_INSTALL_DIR})

set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
if(WIN32)
    if(WITH_STATIC_LIB)
        set(paddle_inference_lib $<TARGET_FILE_DIR:paddle_inference>/libpaddle_inference.lib
                             $<TARGET_FILE_DIR:paddle_inference>/paddle_inference.*)
    else()
        set(paddle_inference_lib $<TARGET_FILE_DIR:paddle_inference_shared>/paddle_inference.dll
                             $<TARGET_FILE_DIR:paddle_inference_shared>/paddle_inference.lib)
    endif()
    copy(inference_lib_dist
            SRCS  ${src_dir}/inference/api/paddle_*.h ${paddle_inference_lib}
            DSTS  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib
            ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib)
else(WIN32)
    set(paddle_inference_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/libpaddle_inference.*)
    copy(inference_lib_dist
                SRCS  ${src_dir}/inference/api/paddle_*.h ${paddle_inference_lib}
                DSTS  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/lib)
endif(WIN32)

copy(inference_lib_dist
        SRCS  ${CMAKE_BINARY_DIR}/paddle/fluid/framework/framework.pb.h
        DSTS  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/internal)
copy(inference_lib_dist
        SRCS  ${PADDLE_SOURCE_DIR}/paddle/fluid/framework/io/crypto/cipher.h
        DSTS  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/crypto/)
include_directories(${CMAKE_BINARY_DIR}/../paddle/fluid/framework/io)

copy(inference_lib_dist
        SRCS  ${PADDLE_SOURCE_DIR}/paddle/fluid/extension/include/*
        DSTS  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/)
copy(inference_lib_dist
        SRCS  ${PADDLE_SOURCE_DIR}/paddle/fluid/platform/complex.h
        DSTS  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/)
copy(inference_lib_dist
        SRCS  ${PADDLE_SOURCE_DIR}/paddle/fluid/platform/float16.h
        DSTS  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/)

# CAPI inference library for only inference
set(PADDLE_INFERENCE_C_INSTALL_DIR "${CMAKE_BINARY_DIR}/paddle_inference_c_install_dir" CACHE STRING
"A path setting CAPI paddle inference shared")
copy_part_of_thrid_party(inference_lib_dist ${PADDLE_INFERENCE_C_INSTALL_DIR})

set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
if(WIN32)
  set(paddle_inference_c_lib $<TARGET_FILE_DIR:paddle_inference_c>/paddle_inference_c.*)
else(WIN32)
  set(paddle_inference_c_lib ${PADDLE_BINARY_DIR}/paddle/fluid/inference/capi_exp/libpaddle_inference_c.*)
endif(WIN32)

copy(inference_lib_dist
      SRCS  ${src_dir}/inference/capi_exp/pd_*.h  ${paddle_inference_c_lib}
      DSTS  ${PADDLE_INFERENCE_C_INSTALL_DIR}/paddle/include ${PADDLE_INFERENCE_C_INSTALL_DIR}/paddle/lib)

set(dst_dir "${PADDLE_INSTALL_DIR}/paddle/fluid")
set(module "inference")
if(WIN32)
        copy(inference_lib_dist
                SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/api/paddle_*.h ${paddle_inference_lib}
                DSTS ${dst_dir}/${module} ${dst_dir}/${module} ${dst_dir}/${module} ${dst_dir}/${module}
                )
        else()
        copy(inference_lib_dist
                SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/api/paddle_*.h ${paddle_inference_lib}
                DSTS ${dst_dir}/${module} ${dst_dir}/${module} ${dst_dir}/${module} 
                )
endif()

set(dst_dir "${PADDLE_INSTALL_DIR}/paddle/fluid")

set(module "platform")
set(platform_lib_deps profiler_proto error_codes_proto)
if(WITH_GPU)
  set(platform_lib_deps ${platform_lib_deps} external_error_proto)
endif(WITH_GPU)

set(dst_dir "${PADDLE_INSTALL_DIR}/third_party/eigen3")
copy(inference_lib_dist
        SRCS ${EIGEN_INCLUDE_DIR}/Eigen/Core ${EIGEN_INCLUDE_DIR}/Eigen/src ${EIGEN_INCLUDE_DIR}/unsupported/Eigen
        DSTS ${dst_dir}/Eigen ${dst_dir}/Eigen ${dst_dir}/unsupported)

set(dst_dir "${PADDLE_INSTALL_DIR}/third_party/boost")
copy(inference_lib_dist
        SRCS ${BOOST_INCLUDE_DIR}/boost
        DSTS ${dst_dir})

set(dst_dir "${PADDLE_INSTALL_DIR}/third_party/dlpack")
copy(inference_lib_dist
        SRCS ${DLPACK_INCLUDE_DIR}/dlpack
        DSTS ${dst_dir})

set(dst_dir "${PADDLE_INSTALL_DIR}/third_party/install/zlib")
copy(inference_lib_dist
        SRCS ${ZLIB_INCLUDE_DIR} ${ZLIB_LIBRARIES}
        DSTS ${dst_dir} ${dst_dir}/lib)

# paddle fluid version
function(version version_file)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -1
            WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}
            OUTPUT_VARIABLE PADDLE_GIT_COMMIT)
    file(WRITE ${version_file}
            "GIT COMMIT ID: ${PADDLE_GIT_COMMIT}\n"
            "WITH_MKL: ${WITH_MKL}\n"
            "WITH_MKLDNN: ${WITH_MKLDNN}\n"
            "WITH_GPU: ${WITH_GPU}\n"
            "WITH_ROCM: ${WITH_ROCM}\n")
    if(WITH_GPU)
        file(APPEND ${version_file}
                "CUDA version: ${CUDA_VERSION}\n"
                "CUDNN version: v${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}\n")
    endif()
    if(WITH_ROCM)
        file(APPEND ${version_file}
                "HIP version: ${HIP_VERSION}\n"
                "MIOpen version: v${MIOPEN_MAJOR_VERSION}.${MIOPEN_MINOR_VERSION}\n")
    endif()
    file(APPEND ${version_file} "CXX compiler version: ${CMAKE_CXX_COMPILER_VERSION}\n")
    if(TENSORRT_FOUND)
        file(APPEND ${version_file}
                "WITH_TENSORRT: ${TENSORRT_FOUND}\n" "TensorRT version: v${TENSORRT_MAJOR_VERSION}\n")
    endif()
    if(WITH_LITE)
        file(APPEND ${version_file} "WITH_LITE: ${WITH_LITE}\n" "LITE_GIT_TAG: ${LITE_GIT_TAG}\n")
    endif()
    
endfunction()
version(${PADDLE_INSTALL_DIR}/version.txt)
version(${PADDLE_INFERENCE_INSTALL_DIR}/version.txt)
version(${PADDLE_INFERENCE_C_INSTALL_DIR}/version.txt)
