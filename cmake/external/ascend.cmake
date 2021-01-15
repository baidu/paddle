# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

SET(ASCEND_PROJECT       "extern_ascend")
IF((NOT DEFINED ASCEND_VER) OR (NOT DEFINED ASCEND_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(ASCEND_VER "0.1.1" CACHE STRING "" FORCE)
  SET(ASCEND_NAME "ascend" CACHE STRING "" FORCE)
  SET(ASCEND_URL "http://paddle-ascend.bj.bcebos.com/ascend.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "ASCEND_NAME: ${ASCEND_NAME}, ASCEND_URL: ${ASCEND_URL}")
SET(ASCEND_SOURCE_DIR    "${THIRD_PARTY_PATH}/ascend")
SET(ASCEND_DOWNLOAD_DIR  "${ASCEND_SOURCE_DIR}/src/${ASCEND_PROJECT}")
SET(ASCEND_DST_DIR       "ascend")
SET(ASCEND_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(ASCEND_INSTALL_DIR   ${ASCEND_INSTALL_ROOT}/${ASCEND_DST_DIR})
SET(ASCEND_ROOT          ${ASCEND_INSTALL_DIR})
SET(ASCEND_INC_DIR       ${ASCEND_ROOT}/include)
SET(ASCEND_LIB_DIR       ${ASCEND_ROOT}/lib)
SET(ASCEND_LIB           ${ASCEND_LIB_DIR}/libge_runner.so)
SET(ASCEND_GRAPH_LIB           ${ASCEND_LIB_DIR}/libgraph.so)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${ASCEND_ROOT}/lib")

INCLUDE_DIRECTORIES(${ASCEND_INC_DIR})
FILE(WRITE ${ASCEND_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(ASCEND)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${ASCEND_NAME}/include ${ASCEND_NAME}/lib \n"
  "        DESTINATION ${ASCEND_DST_DIR})\n")
ExternalProject_Add(
    ${ASCEND_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${ASCEND_SOURCE_DIR}
    DOWNLOAD_DIR          ${ASCEND_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${ASCEND_URL} -c -q -O ${ASCEND_NAME}.tar.gz
                          && tar zxvf ${ASCEND_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${ASCEND_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${ASCEND_INSTALL_ROOT}
)
ADD_LIBRARY(ascend SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascend PROPERTY IMPORTED_LOCATION ${ASCEND_LIB})

ADD_LIBRARY(ascend_graph SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET ascend_graph PROPERTY IMPORTED_LOCATION ${ASCEND_GRAPH_LIB})
ADD_DEPENDENCIES(ascend ascend_graph ${ASCEND_PROJECT})

