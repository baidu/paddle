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

SET(PSLIB_PROJECT       "extern_pslib")
IF((NOT DEFINED PSLIB_VER) OR (NOT DEFINED PSLIB_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(PSLIB_VER "0.1.1" CACHE STRING "" FORCE)
  SET(PSLIB_NAME "pslib" CACHE STRING "" FORCE)
  SET(PSLIB_URL "https://pslib.bj.bcebos.com/pslib.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "PSLIB_NAME: ${PSLIB_NAME}, PSLIB_URL: ${PSLIB_URL}")
SET(PSLIB_SOURCE_DIR    "${THIRD_PARTY_PATH}/pslib")
SET(PSLIB_DOWNLOAD_DIR  "${PSLIB_SOURCE_DIR}/src/${PSLIB_PROJECT}")
SET(PSLIB_DST_DIR       "pslib")
SET(PSLIB_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(PSLIB_INSTALL_DIR   ${PSLIB_INSTALL_ROOT}/${PSLIB_DST_DIR})
SET(PSLIB_ROOT          ${PSLIB_INSTALL_DIR})
SET(PSLIB_INC_DIR       ${PSLIB_ROOT}/include)
SET(PSLIB_LIB_DIR       ${PSLIB_ROOT}/lib)
SET(PSLIB_LIB           ${PSLIB_LIB_DIR}/libps.so)
SET(PSLIB_VERSION_PY    ${PSLIB_DOWNLOAD_DIR}/pslib/version.py)
SET(PSLIB_IOMP_LIB      ${PSLIB_LIB_DIR}/libiomp5.so) #todo what is this
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${PSLIB_ROOT}/lib")

INCLUDE_DIRECTORIES(${PSLIB_INC_DIR})

FILE(WRITE ${PSLIB_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(PSLIB)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${PSLIB_NAME}/include ${PSLIB_NAME}/lib \n"
  "        DESTINATION ${PSLIB_DST_DIR})\n")

ExternalProject_Add(
    ${PSLIB_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${PSLIB_SOURCE_DIR}
    DOWNLOAD_DIR          ${PSLIB_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      cp /home/xujiaqi/zhangwenhui03/pslib_tar/no_abacus_0622/pslib.tar.gz ${PSLIB_NAME}.tar.gz
                          && tar zxvf ${PSLIB_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${PSLIB_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${PSLIB_INSTALL_ROOT}
)

ADD_LIBRARY(pslib SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET pslib PROPERTY IMPORTED_LOCATION ${PSLIB_LIB})
ADD_DEPENDENCIES(pslib ${PSLIB_PROJECT})
