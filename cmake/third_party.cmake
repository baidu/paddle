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

include(ExternalProject)
# Creat a target named "third_party", which can compile external dependencies on all platform(windows/linux/mac)

set(THIRD_PARTY_PATH  "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
    "A path setting third party libraries download & build directories.")
set(THIRD_PARTY_CACHE_PATH     "${CMAKE_SOURCE_DIR}"    CACHE STRING
    "A path cache third party source code to avoid repeated download.")

set(THIRD_PARTY_BUILD_TYPE Release)
set(third_party_deps)

# cache funciton to avoid repeat download code of third_party.
# This function has 4 parameters, URL / REPOSITOR / TAG / DIR:
# 1. URL:           specify download url of 3rd party
# 2. REPOSITORY:    specify git REPOSITORY of 3rd party
# 3. TAG:           specify git tag/branch/commitID of 3rd party
# 4. DIR:           overwrite the original SOURCE_DIR when cache directory
#
# The function Return 1 PARENT_SCOPE variables:
#  - ${TARGET}_DOWNLOAD_CMD: Simply place "${TARGET}_DOWNLOAD_CMD" in ExternalProject_Add,
#                            and you no longer need to set any donwnload steps in ExternalProject_Add.
# For example:
#    Cache_third_party(${TARGET}
#            REPOSITORY ${TARGET_REPOSITORY}
#            TAG        ${TARGET_TAG}
#            DIR        ${TARGET_SOURCE_DIR})

FUNCTION(cache_third_party TARGET)
    SET(options "")
    SET(oneValueArgs URL REPOSITORY TAG DIR)
    SET(multiValueArgs "")
    cmake_parse_arguments(cache_third_party "${optionps}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    STRING(REPLACE "extern_" "" TARGET_NAME ${TARGET})
    STRING(REGEX REPLACE "[0-9]+" "" TARGET_NAME ${TARGET_NAME})
    STRING(TOUPPER ${TARGET_NAME} TARGET_NAME)
    IF(cache_third_party_REPOSITORY)
        SET(${TARGET_NAME}_DOWNLOAD_CMD
                GIT_REPOSITORY  ${cache_third_party_REPOSITORY})
        IF(cache_third_party_TAG)
            LIST(APPEND   ${TARGET_NAME}_DOWNLOAD_CMD
                    GIT_TAG     ${cache_third_party_TAG})
        ENDIF()
    ELSEIF(cache_third_party_URL)
        SET(${TARGET_NAME}_DOWNLOAD_CMD
                URL             ${cache_third_party_URL})
    ELSE()
        MESSAGE(FATAL_ERROR    "Download link (Git repo or URL) must be specified for cache!")
    ENDIF()
    IF(WITH_TP_CACHE)
        IF(NOT cache_third_party_DIR)
            MESSAGE(FATAL_ERROR   "Please input the ${TARGET_NAME}_SOURCE_DIR for overwriting when -DWITH_TP_CACHE=ON")
        ENDIF()
        # Generate and verify cache dir for third_party source code
        SET(cache_third_party_REPOSITORY ${cache_third_party_REPOSITORY} ${cache_third_party_URL})
        IF(cache_third_party_REPOSITORY AND cache_third_party_TAG)
            STRING(MD5 HASH_REPO ${cache_third_party_REPOSITORY})
            STRING(MD5 HASH_GIT ${cache_third_party_TAG})
            STRING(SUBSTRING ${HASH_REPO} 0 8 HASH_REPO)
            STRING(SUBSTRING ${HASH_GIT} 0 8 HASH_GIT)
            STRING(CONCAT HASH ${HASH_REPO} ${HASH_GIT})
            # overwrite the original SOURCE_DIR when cache directory
            SET(${cache_third_party_DIR} ${THIRD_PARTY_CACHE_PATH}/third_party/${TARGET}_${HASH})
        ELSEIF(cache_third_party_REPOSITORY)
            STRING(MD5 HASH_REPO ${cache_third_party_REPOSITORY})
            STRING(SUBSTRING ${HASH_REPO} 0 16 HASH)
            # overwrite the original SOURCE_DIR when cache directory
            SET(${cache_third_party_DIR} ${THIRD_PARTY_CACHE_PATH}/third_party/${TARGET}_${HASH})
        ENDIF()

        IF(EXISTS ${${cache_third_party_DIR}})
            # judge whether the cache dir is empty
            FILE(GLOB files ${${cache_third_party_DIR}}/*)
            LIST(LENGTH files files_len)
            IF(files_len GREATER 0)
                list(APPEND ${TARGET_NAME}_DOWNLOAD_CMD DOWNLOAD_COMMAND "")
            ENDIF()
        ENDIF()
        SET(${cache_third_party_DIR} ${${cache_third_party_DIR}} PARENT_SCOPE)
    ENDIF()

    # Pass ${TARGET_NAME}_DOWNLOAD_CMD to parent scope, the double quotation marks can't be removed
    SET(${TARGET_NAME}_DOWNLOAD_CMD "${${TARGET_NAME}_DOWNLOAD_CMD}" PARENT_SCOPE)
ENDFUNCTION()

MACRO(UNSET_VAR VAR_NAME)
    UNSET(${VAR_NAME} CACHE)
    UNSET(${VAR_NAME})
ENDMACRO()

# Funciton to Download the dependencies during compilation
# This function has 2 parameters, URL / DIRNAME:
# 1. URL:           The download url of 3rd dependencies
# 2. NAME:          The name of file, that determin the dirname
#
FUNCTION(file_download_and_uncompress URL NAME)
  set(options "")
  set(oneValueArgs MD5)
  set(multiValueArgs "")
  cmake_parse_arguments(URL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  MESSAGE(STATUS "Download dependence[${NAME}] from ${URL}, MD5: ${URL_MD5}")
  SET(${NAME}_INCLUDE_DIR ${THIRD_PARTY_PATH}/${NAME}/data PARENT_SCOPE)
  ExternalProject_Add(
      download_${NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${THIRD_PARTY_PATH}/${NAME}
      URL                   ${URL}
      URL_MD5               ${URL_MD5}
      TIMEOUT               120
      DOWNLOAD_DIR          ${THIRD_PARTY_PATH}/${NAME}/data/
      SOURCE_DIR            ${THIRD_PARTY_PATH}/${NAME}/data/
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ""
      UPDATE_COMMAND        ""
      INSTALL_COMMAND       ""
    )
  set(third_party_deps ${third_party_deps} download_${NAME} PARENT_SCOPE)
ENDFUNCTION()


# Correction of flags on different Platform(WIN/MAC) and Print Warning Message
if (APPLE)
    if(WITH_MKL)
        MESSAGE(WARNING
            "Mac is not supported with MKL in Paddle yet. Force WITH_MKL=OFF.")
        set(WITH_MKL OFF CACHE STRING "Disable MKL for building on mac" FORCE)
    endif()
endif()

if(WIN32 OR APPLE)
    MESSAGE(STATUS "Disable XBYAK in Windows and MacOS")
    SET(WITH_XBYAK OFF CACHE STRING "Disable XBYAK in Windows and MacOS" FORCE)

    if(WITH_LIBXSMM)
        MESSAGE(WARNING
            "Windows, Mac are not supported with libxsmm in Paddle yet."
            "Force WITH_LIBXSMM=OFF")
        SET(WITH_LIBXSMM OFF CACHE STRING "Disable LIBXSMM in Windows and MacOS" FORCE)
    endif()

    if(WITH_BOX_PS)
        MESSAGE(WARNING
            "Windows or Mac is not supported with BOX_PS in Paddle yet."
            "Force WITH_BOX_PS=OFF")
        SET(WITH_BOX_PS OFF CACHE STRING "Disable BOX_PS package in Windows and MacOS" FORCE)
    endif()

    if(WITH_PSLIB)
        MESSAGE(WARNING
            "Windows or Mac is not supported with PSLIB in Paddle yet."
            "Force WITH_PSLIB=OFF")
        SET(WITH_PSLIB OFF CACHE STRING "Disable PSLIB package in Windows and MacOS" FORCE)
    endif()

    if(WITH_LIBMCT)
        MESSAGE(WARNING
            "Windows or Mac is not supported with LIBMCT in Paddle yet."
            "Force WITH_LIBMCT=OFF")
        SET(WITH_LIBMCT OFF CACHE STRING "Disable LIBMCT package in Windows and MacOS" FORCE)
    endif()

    if(WITH_PSLIB_BRPC)
        MESSAGE(WARNING
            "Windows or Mac is not supported with PSLIB_BRPC in Paddle yet."
            "Force WITH_PSLIB_BRPC=OFF")
        SET(WITH_PSLIB_BRPC OFF CACHE STRING "Disable PSLIB_BRPC package in Windows and MacOS" FORCE)
    endif()
endif()

set(WITH_MKLML ${WITH_MKL})
if(NOT DEFINED WITH_MKLDNN)
    if(WITH_MKL AND AVX2_FOUND)
        set(WITH_MKLDNN ON)
    else()
        message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN")
        set(WITH_MKLDNN OFF)
    endif()
endif()

if(WIN32 OR APPLE OR NOT WITH_GPU OR ON_INFER)
    set(WITH_DGC OFF)
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.5.2")
    set(SHALLOW_CLONE "GIT_SHALLOW TRUE") # adds --depth=1 arg to git clone of External_Projects
endif()

########################### include third_party according to flags ###############################
include(external/zlib)      # download, build, install zlib
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/boost)     # download boost
include(external/eigen)     # download eigen3
include(external/threadpool)# download threadpool
include(external/dlpack)    # download dlpack
include(external/xxhash)    # download, build, install xxhash
include(external/warpctc)   # download, build, install warpctc

list(APPEND third_party_deps extern_eigen3 extern_gflags extern_glog extern_boost extern_xxhash)
list(APPEND third_party_deps extern_zlib extern_dlpack extern_warpctc extern_threadpool)

include(cblas)              	# find first, then download, build, install openblas

message(STATUS "CBLAS_PROVIDER: ${CBLAS_PROVIDER}")
if(${CBLAS_PROVIDER} STREQUAL MKLML)
    list(APPEND third_party_deps extern_mklml)
elseif(${CBLAS_PROVIDER} STREQUAL EXTERN_OPENBLAS)
    list(APPEND third_party_deps extern_openblas)
endif()


if(WITH_MKLDNN)
    include(external/mkldnn)    # download, build, install mkldnn
    list(APPEND third_party_deps extern_mkldnn)
endif()

include(external/protobuf)  	# find first, then download, build, install protobuf
if(TARGET extern_protobuf)
    list(APPEND third_party_deps extern_protobuf)
endif()

if(WITH_PYTHON)
    include(external/python)    # find python and python_module
    include(external/pybind11)  # download pybind11
    list(APPEND third_party_deps extern_pybind)
endif()

IF(WITH_TESTING OR WITH_DISTRIBUTE)
    include(external/gtest)     # download, build, install gtest
    list(APPEND third_party_deps extern_gtest)
ENDIF()

if(WITH_GPU)
    if (${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0)
        include(external/cub)       # download cub
        list(APPEND third_party_deps extern_cub)
    endif()
    set(URL  "https://paddlepaddledeps.bj.bcebos.com/externalErrorMsg.tar.gz" CACHE STRING "" FORCE)
    file_download_and_uncompress(${URL} "externalError" MD5 c0749523ebb536eb7382487d645d9cd4)   # download file externalErrorMsg.tar.gz
    if(WITH_TESTING)
        # copy externalErrorMsg.pb for unittest 'enforce_test'
        set(SRC_DIR ${THIRD_PARTY_PATH}/externalError/data)
        if(WIN32 AND (NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja"))
            set(DST_DIR ${CMAKE_BINARY_DIR}/paddle/fluid/third_party/externalError/data)
        else()
            set(DST_DIR ${CMAKE_BINARY_DIR}/paddle/third_party/externalError/data)
        endif()
        add_custom_command(TARGET download_externalError POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${SRC_DIR} ${DST_DIR}
            COMMENT "copy_directory from ${SRC_DIR} to ${DST_DIR}")
    endif()
endif(WITH_GPU)

if(WITH_XPU)
    include(external/xpu)          # download, build, install xpu
    list(APPEND third_party_deps extern_xpu)
endif(WITH_XPU)

if(WITH_PSLIB)
    include(external/pslib)          # download, build, install pslib
    list(APPEND third_party_deps extern_pslib)
    if(WITH_LIBMCT)
        include(external/libmct)     # download, build, install libmct
        list(APPEND third_party_deps extern_libxsmm)
    endif()
    if(WITH_PSLIB_BRPC)
        include(external/pslib_brpc) # download, build, install pslib_brpc
        list(APPEND third_party_deps extern_pslib_brpc)
    else()    
        include(external/snappy)
        list(APPEND third_party_deps extern_snappy)

        include(external/leveldb)
        list(APPEND third_party_deps extern_leveldb)
        include(external/brpc)
        list(APPEND third_party_deps extern_brpc)
    endif()
endif(WITH_PSLIB)

if(NOT WIN32 AND NOT APPLE)
    include(external/gloo)
    list(APPEND third_party_deps extern_gloo)
endif()

if(WITH_BOX_PS)
    include(external/box_ps)
    list(APPEND third_party_deps extern_box_ps)
endif(WITH_BOX_PS)

if(WITH_ASCEND OR WITH_ASCEND_CL)
    include(external/ascend)
    if(WITH_ASCEND OR WITH_ASCEND_CL)
        list(APPEND third_party_deps extern_ascend)
    endif()
    if(WITH_ASCEND_CL)
        list(APPEND third_party_deps extern_ascend_cl)
    endif()
endif ()

if (WITH_PSCORE)
    include(external/snappy)
    list(APPEND third_party_deps extern_snappy)

    include(external/leveldb)
    list(APPEND third_party_deps extern_leveldb)

    include(external/brpc)
    list(APPEND third_party_deps extern_brpc)

    include(external/libmct)     # download, build, install libmct
    list(APPEND third_party_deps extern_libmct)
    
    if (WITH_HETERPS)
        include(external/rocksdb)     # download, build, install libmct
        list(APPEND third_party_deps extern_rocksdb)
    endif()
endif()

if(WITH_XBYAK)
    include(external/xbyak)         # download, build, install xbyak
    list(APPEND third_party_deps extern_xbyak)
endif()

if(WITH_LIBXSMM)
    include(external/libxsmm)       # download, build, install libxsmm
    list(APPEND third_party_deps extern_libxsmm)
endif()

if(WITH_DGC)
    message(STATUS "add dgc lib.")
    include(external/dgc)           # download, build, install dgc
    add_definitions(-DPADDLE_WITH_DGC)
    list(APPEND third_party_deps extern_dgc)
endif()

if (WITH_LITE)
    message(STATUS "Compile Paddle with Lite Engine.")
    include(external/lite)
endif (WITH_LITE)

if (WITH_CRYPTO)
    include(external/cryptopp)   # download, build, install cryptopp
    list(APPEND third_party_deps extern_cryptopp)
    add_definitions(-DPADDLE_WITH_CRYPTO)
endif (WITH_CRYPTO)

add_custom_target(third_party ALL DEPENDS ${third_party_deps})
