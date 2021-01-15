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

include(ExternalProject)

set(DLPACK_PREFIX_DIR ${THIRD_PARTY_PATH}/dlpack)
set(DLPACK_SOURCE_DIR ${THIRD_PARTY_PATH}/dlpack/src/extern_dlpack)

set(DLPACK_REPOSITORY https://gitee.com/tianjianhe/dlpack.git)
set(DLPACK_TAG        v0.2)

cache_third_party(extern_dlpack
    REPOSITORY    ${DLPACK_REPOSITORY}
    TAG           ${DLPACK_TAG}
    DIR           DLPACK_SOURCE_DIR)

set(DLPACK_INCLUDE_DIR  ${DLPACK_SOURCE_DIR}/include)
include_directories(${DLPACK_INCLUDE_DIR})

ExternalProject_Add(
  extern_dlpack
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${DLPACK_DOWNLOAD_CMD}"
  PREFIX          ${DLPACK_PREFIX_DIR}
  SOURCE_DIR      ${DLPACK_SOURCE_DIR}
  UPDATE_COMMAND    ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

add_library(dlpack INTERFACE)

add_dependencies(dlpack extern_dlpack)
