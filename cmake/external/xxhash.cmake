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

set(XXHASH_PREFIX_DIR ${THIRD_PARTY_PATH}/xxhash)
set(XXHASH_SOURCE_DIR ${THIRD_PARTY_PATH}/xxhash/src/extern_xxhash)
set(XXHASH_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xxhash)
set(XXHASH_INCLUDE_DIR "${XXHASH_INSTALL_DIR}/include")
set(XXHASH_REPOSITORY  https://gitee.com/tianjianhe/xxHash.git)
set(XXHASH_TAG         v0.6.5)

cache_third_party(extern_xxhash
    REPOSITORY    ${XXHASH_REPOSITORY}
    TAG           ${XXHASH_TAG}
    DIR           XXHASH_SOURCE_DIR)

IF(APPLE)
  SET(BUILD_CMD sed -i \"\" "s/-Wstrict-prototypes -Wundef/-Wstrict-prototypes -Wundef -fPIC/g" ${XXHASH_SOURCE_DIR}/Makefile && make lib)
ELSEIF(UNIX)
  SET(BUILD_CMD sed -i "s/-Wstrict-prototypes -Wundef/-Wstrict-prototypes -Wundef -fPIC/g" ${XXHASH_SOURCE_DIR}/Makefile && make lib)
ENDIF()

if(WIN32)
  ExternalProject_Add(
      extern_xxhash
      ${EXTERNAL_PROJECT_LOG_ARGS}
      ${SHALLOW_CLONE}
      "${XXHASH_DOWNLOAD_CMD}"
      PREFIX           ${XXHASH_PREFIX_DIR}
      SOURCE_DIR       ${XXHASH_SOURCE_DIR}
      UPDATE_COMMAND   ""
      PATCH_COMMAND    ""
      CONFIGURE_COMMAND
                      ${CMAKE_COMMAND} ${XXHASH_SOURCE_DIR}/cmake_unofficial
                      -DCMAKE_INSTALL_PREFIX:PATH=${XXHASH_INSTALL_DIR}
                      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                      -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
                      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                      -DBUILD_XXHSUM=OFF
                      -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                      -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
                      -DBUILD_SHARED_LIBS=OFF
                      ${OPTIONAL_CACHE_ARGS}
      TEST_COMMAND      ""
  )
else()
  ExternalProject_Add(
      extern_xxhash
      ${EXTERNAL_PROJECT_LOG_ARGS}
      "${XXHASH_DOWNLOAD_CMD}"
      PREFIX           ${XXHASH_PREFIX_DIR}
      SOURCE_DIR       ${XXHASH_SOURCE_DIR}
      UPDATE_COMMAND    ""
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE   1
      BUILD_COMMAND     ${BUILD_CMD}
      INSTALL_COMMAND   make PREFIX=${XXHASH_INSTALL_DIR} install
      TEST_COMMAND      ""
  )
endif()

if (WIN32)
  set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/xxhash.lib")
else()
  set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/libxxhash.a")
endif ()
INCLUDE_DIRECTORIES(${XXHASH_INCLUDE_DIR})

add_library(xxhash STATIC IMPORTED GLOBAL)
set_property(TARGET xxhash PROPERTY IMPORTED_LOCATION ${XXHASH_LIBRARIES})
include_directories(${XXHASH_INCLUDE_DIR})
add_dependencies(xxhash extern_xxhash)
