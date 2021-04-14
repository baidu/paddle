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

INCLUDE(ExternalProject)

SET(GLOO_PROJECT       "extern_gloo")
SET(GLOO_PREFIX_DIR    ${THIRD_PARTY_PATH}/gloo)
SET(GLOO_SOURCE_DIR    ${THIRD_PARTY_PATH}/gloo/src/extern_gloo/gloo)
SET(GLOO_INSTALL_DIR   ${THIRD_PARTY_PATH}/install/gloo)
SET(GLOO_INCLUDE_DIR   "${GLOO_INSTALL_DIR}/include" CACHE PATH "gloo include directory." FORCE)
SET(GLOO_LIBRARY_DIR   "${GLOO_INSTALL_DIR}/lib" CACHE PATH "gloo library directory." FORCE)
# As we add extra features for gloo, we use the non-official repo
SET(GLOO_REPOSITORY    ${GIT_URL}/sandyhouse/gloo.git)
SET(GLOO_TAG           v0.0.2)
SET(GLOO_LIBRARIES     "${GLOO_INSTALL_DIR}/lib/libgloo.a" CACHE FILEPATH "gloo library." FORCE)

INCLUDE_DIRECTORIES(${GLOO_INCLUDE_DIR})

cache_third_party(extern_gloo
    REPOSITORY    ${GLOO_REPOSITORY}
    TAG           ${GLOO_TAG}
    DIR           GLOO_SOURCE_DIR)

  if(WITH_ASCEND OR WITH_ASCEND_CL)
  ExternalProject_Add(
      extern_gloo
      ${EXTERNAL_PROJECT_LOG_ARGS}
      ${SHALLOW_CLONE}
      "${GLOO_DOWNLOAD_CMD}"
      PREFIX                "${GLOO_PREFIX_DIR}"
      SOURCE_DIR            "${GLOO_SOURCE_DIR}"
      UPDATE_COMMAND        ""
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         mkdir -p ${GLOO_SOURCE_DIR}/build
          && cd ${GLOO_SOURCE_DIR}/build && cmake .. -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} && make
          && mkdir -p ${GLOO_LIBRARY_DIR} ${GLOO_INCLUDE_DIR}/gloo
      INSTALL_COMMAND      ${CMAKE_COMMAND} -E copy ${GLOO_SOURCE_DIR}/build/gloo/libgloo.a ${GLOO_LIBRARY_DIR}
      COMMAND              ${CMAKE_COMMAND} -E copy_directory "${GLOO_SOURCE_DIR}/gloo/" "${GLOO_INCLUDE_DIR}/gloo"
  )
else()
  ExternalProject_Add(
      extern_gloo
      ${EXTERNAL_PROJECT_LOG_ARGS}
      ${SHALLOW_CLONE}
      "${GLOO_DOWNLOAD_CMD}"
      PREFIX                "${GLOO_PREFIX_DIR}"
      SOURCE_DIR            "${GLOO_SOURCE_DIR}"
      UPDATE_COMMAND        ""
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         mkdir -p ${GLOO_SOURCE_DIR}/build
          && cd ${GLOO_SOURCE_DIR}/build && cmake .. && make
          && mkdir -p ${GLOO_LIBRARY_DIR} ${GLOO_INCLUDE_DIR}/gloo
      INSTALL_COMMAND      ${CMAKE_COMMAND} -E copy ${GLOO_SOURCE_DIR}/build/gloo/libgloo.a ${GLOO_LIBRARY_DIR}
      COMMAND              ${CMAKE_COMMAND} -E copy_directory "${GLOO_SOURCE_DIR}/gloo/" "${GLOO_INCLUDE_DIR}/gloo"
  )
endif()


ADD_LIBRARY(gloo STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gloo PROPERTY IMPORTED_LOCATION ${GLOO_LIBRARIES})
ADD_DEPENDENCIES(gloo ${GLOO_PROJECT})
