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

SET(ZLIB_PREFIX_DIR ${THIRD_PARTY_PATH}/zlib)
SET(ZLIB_SOURCE_DIR ${THIRD_PARTY_PATH}/zlib/src/extern_zlib)
SET(ZLIB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/zlib)
SET(ZLIB_ROOT ${ZLIB_INSTALL_DIR} CACHE FILEPATH "zlib root directory." FORCE)
SET(ZLIB_INCLUDE_DIR "${ZLIB_INSTALL_DIR}/include" CACHE PATH "zlib include directory." FORCE)
set(ZLIB_REPOSITORY https://gitee.com/tianjianhe/zlib.git)
set(ZLIB_TAG        v1.2.8)

INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR}) # For zlib code to include its own headers.
INCLUDE_DIRECTORIES(${THIRD_PARTY_PATH}/install) # For Paddle code to include zlib.h.

cache_third_party(extern_zlib
    REPOSITORY    ${ZLIB_REPOSITORY}
    TAG           ${ZLIB_TAG}
    DIR           ZLIB_SOURCE_DIR)

ExternalProject_Add(
    extern_zlib
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${ZLIB_DOWNLOAD_CMD}"
    PREFIX          ${ZLIB_PREFIX_DIR}
    SOURCE_DIR      ${ZLIB_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
		    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}  -D_GLIBCXX_USE_CXX11_ABI=0"
                    -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
                    -DBUILD_SHARED_LIBS=OFF
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_MACOSX_RPATH=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ZLIB_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)
IF(WIN32)
  SET(ZLIB_LIBRARIES "${ZLIB_INSTALL_DIR}/lib/zlibstatic.lib" CACHE FILEPATH "zlib library." FORCE)
ELSE(WIN32)
  SET(ZLIB_LIBRARIES "${ZLIB_INSTALL_DIR}/lib/libz.a" CACHE FILEPATH "zlib library." FORCE)
ENDIF(WIN32)

ADD_LIBRARY(zlib STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET zlib PROPERTY IMPORTED_LOCATION ${ZLIB_LIBRARIES})
ADD_DEPENDENCIES(zlib extern_zlib)
