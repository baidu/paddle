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

INCLUDE(ExternalProject)

SET(PSLIB_BRPC_PROJECT       "extern_pslib_brpc")
IF((NOT DEFINED PSLIB_BRPC_NAME) OR (NOT DEFINED PSLIB_BRPC_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(PSLIB_BRPC_VER "0.1.0" CACHE STRING "" FORCE)
  SET(PSLIB_BRPC_NAME "pslib_brpc" CACHE STRING "" FORCE)
  SET(PSLIB_BRPC_URL "https://pslib.bj.bcebos.com/pslib_brpc.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "PSLIB_BRPC_NAME: ${PSLIB_BRPC_NAME}, PSLIB_BRPC_URL: ${PSLIB_BRPC_URL}")
SET(PSLIB_BRPC_SOURCE_DIR    "${THIRD_PARTY_PATH}/pslib_brpc")
SET(PSLIB_BRPC_DOWNLOAD_DIR  "${PSLIB_BRPC_SOURCE_DIR}/src/${PSLIB_BRPC_PROJECT}")
SET(PSLIB_BRPC_DST_DIR       "pslib_brpc")
SET(PSLIB_BRPC_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(PSLIB_BRPC_INSTALL_DIR   ${PSLIB_BRPC_INSTALL_ROOT}/${PSLIB_BRPC_DST_DIR})
SET(PSLIB_BRPC_ROOT          ${PSLIB_BRPC_INSTALL_DIR})
SET(PSLIB_BRPC_INC_DIR       ${PSLIB_BRPC_ROOT}/include)
SET(PSLIB_BRPC_LIB_DIR       ${PSLIB_BRPC_ROOT}/lib)
SET(PSLIB_BRPC_LIB           ${PSLIB_BRPC_LIB_DIR}/libbrpc.a)
SET(PSLIB_BRPC_IOMP_LIB      ${PSLIB_BRPC_LIB_DIR}/libiomp5.so) #todo what is this
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${PSLIB_BRPC_ROOT}/lib")

INCLUDE_DIRECTORIES(${PSLIB_BRPC_INC_DIR})

FILE(WRITE ${PSLIB_BRPC_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(PSLIB_BRPC)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${PSLIB_BRPC_NAME}/include ${PSLIB_BRPC_NAME}/lib \n"
  "        DESTINATION ${PSLIB_BRPC_DST_DIR})\n")

ExternalProject_Add(
    ${PSLIB_BRPC_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${PSLIB_BRPC_SOURCE_DIR}
    DOWNLOAD_DIR          ${PSLIB_BRPC_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      cp /home/malin10/Paddle/pslib_brpc.tar.gz ${PSLIB_BRPC_NAME}.tar.gz
                          && tar zxvf ${PSLIB_BRPC_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${PSLIB_BRPC_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${PSLIB_BRPC_INSTALL_ROOT}
)

ADD_LIBRARY(pslib_brpc SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET pslib_brpc PROPERTY IMPORTED_LOCATION ${PSLIB_BRPC_LIB})
ADD_DEPENDENCIES(pslib_brpc ${PSLIB_BRPC_PROJECT})
