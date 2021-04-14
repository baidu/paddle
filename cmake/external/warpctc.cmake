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
# limitations under the License.

INCLUDE(ExternalProject)

IF(WITH_ROCM)
    add_definitions(-DWARPCTC_WITH_HIP)
ENDIF()

SET(WARPCTC_PREFIX_DIR  ${THIRD_PARTY_PATH}/warpctc)
SET(WARPCTC_SOURCE_DIR  ${THIRD_PARTY_PATH}/warpctc/src/extern_warpctc)
SET(WARPCTC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/warpctc)
# in case of low internet speed  
#set(WARPCTC_REPOSITORY  https://gitee.com/tianjianhe/warp-ctc.git)
set(WARPCTC_REPOSITORY  ${GIT_URL}/baidu-research/warp-ctc.git)
set(WARPCTC_TAG         c690fc5755abbdbdc98ef78d51ec10a6748a8cd1)

SET(WARPCTC_INCLUDE_DIR "${WARPCTC_INSTALL_DIR}/include"
    CACHE PATH "Warp-ctc Directory" FORCE)
# Used in unit test test_WarpCTCLayer
SET(WARPCTC_LIB_DIR "${WARPCTC_INSTALL_DIR}/lib"
    CACHE PATH "Warp-ctc Library Directory" FORCE)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR WIN32)
    SET(USE_OMP OFF)
ELSE()
    SET(USE_OMP ON)
ENDIF()

cache_third_party(extern_warpctc
    REPOSITORY   ${WARPCTC_REPOSITORY}
    TAG          ${WARPCTC_TAG}
    DIR          WARPCTC_SOURCE_DIR)

if(WITH_ASCEND OR WITH_ASCEND_CL)
    ExternalProject_Add(
        extern_warpctc
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        "${WARPCTC_DOWNLOAD_CMD}"
        PREFIX          ${WARPCTC_PREFIX_DIR}
        SOURCE_DIR      ${WARPCTC_SOURCE_DIR}
        #UPDATE_COMMAND  ""
        PATCH_COMMAND   ""
        BUILD_ALWAYS    1
        CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
                        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                        -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
                        -DWITH_GPU=${WITH_GPU}
                        -DWITH_ROCM=${WITH_ROCM}
                        -DWITH_OMP=${USE_OMP}
                        -DWITH_TORCH=OFF
                        -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
                        -DBUILD_SHARED=ON
                        -DBUILD_TESTS=OFF
                        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                        ${EXTERNAL_OPTIONAL_ARGS}
        CMAKE_CACHE_ARGS -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
                         -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                         -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
    )
else()
    ExternalProject_Add(
        extern_warpctc
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        "${WARPCTC_DOWNLOAD_CMD}"
        PREFIX          ${WARPCTC_PREFIX_DIR}
        SOURCE_DIR      ${WARPCTC_SOURCE_DIR}
        #UPDATE_COMMAND  ""
        PATCH_COMMAND   ""
        BUILD_ALWAYS    1
        CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_C_FLAGS=$<FILTER:${CMAKE_C_FLAGS},EXCLUDE,/Zc:inline>
                        -DCMAKE_C_FLAGS_DEBUG=$<FILTER:${CMAKE_C_FLAGS_DEBUG},EXCLUDE,/Zc:inline>
                        -DCMAKE_C_FLAGS_RELEASE=$<FILTER:${CMAKE_C_FLAGS_RELEASE},EXCLUDE,/Zc:inline>
                        -DCMAKE_CXX_FLAGS=$<FILTER:${CMAKE_CXX_FLAGS},EXCLUDE,/Zc:inline>
                        -DCMAKE_CXX_FLAGS_RELEASE=$<FILTER:${CMAKE_CXX_FLAGS_RELEASE},EXCLUDE,/Zc:inline>
                        -DCMAKE_CXX_FLAGS_DEBUG=$<FILTER:${CMAKE_CXX_FLAGS_DEBUG},EXCLUDE,/Zc:inline>
                        -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
                        -DWITH_GPU=${WITH_GPU}
                        -DWITH_ROCM=${WITH_ROCM}
                        -DWITH_OMP=${USE_OMP}
                        -DWITH_TORCH=OFF
                        -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
                        -DBUILD_SHARED=ON
                        -DBUILD_TESTS=OFF
                        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                        ${EXTERNAL_OPTIONAL_ARGS}
        CMAKE_CACHE_ARGS -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
                         -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                         -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
    )
endif()


IF(WIN32)
    SET(WARPCTC_LIBRARIES "${WARPCTC_INSTALL_DIR}/bin/warpctc${CMAKE_SHARED_LIBRARY_SUFFIX}"
            CACHE FILEPATH "Warp-ctc Library" FORCE)
else(WIN32)
    SET(WARPCTC_LIBRARIES "${WARPCTC_INSTALL_DIR}/lib/libwarpctc${CMAKE_SHARED_LIBRARY_SUFFIX}"
            CACHE FILEPATH "Warp-ctc Library" FORCE)
ENDIF(WIN32)

MESSAGE(STATUS "warp-ctc library: ${WARPCTC_LIBRARIES}")
get_filename_component(WARPCTC_LIBRARY_PATH ${WARPCTC_LIBRARIES} DIRECTORY)
INCLUDE_DIRECTORIES(${WARPCTC_INCLUDE_DIR}) # For warpctc code to include its headers.

ADD_LIBRARY(warpctc SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET warpctc PROPERTY IMPORTED_LOCATION ${WARPCTC_LIBRARIES})
ADD_DEPENDENCIES(warpctc extern_warpctc)
