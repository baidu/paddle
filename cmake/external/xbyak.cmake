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

set(XBYAK_PROJECT       extern_xbyak)
set(XBYAK_PREFIX_DIR    ${THIRD_PARTY_PATH}/xbyak)
SET(XBYAK_SOURCE_DIR     ${THIRD_PARTY_PATH}/xbyak/src/extern_xbyak)
set(XBYAK_INSTALL_ROOT  ${THIRD_PARTY_PATH}/install/xbyak)
set(XBYAK_INC_DIR       ${XBYAK_INSTALL_ROOT}/include)
set(XBYAK_REPOSITORY    https://gitee.com/tianjianhe/xbyak.git)
set(XBYAK_TAG           v5.661) # Jul 26th

include_directories(${XBYAK_INC_DIR})
include_directories(${XBYAK_INC_DIR}/xbyak)

add_definitions(-DPADDLE_WITH_XBYAK)

# xbyak options
add_definitions(-DXBYAK64)
add_definitions(-DXBYAK_NO_OP_NAMES)

cache_third_party(${XBYAK_PROJECT}
    REPOSITORY    ${XBYAK_REPOSITORY}
    TAG           ${XBYAK_TAG}
    DIR           XBYAK_SOURCE_DIR)

ExternalProject_Add(
    ${XBYAK_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${XBYAK_DOWNLOAD_CMD}"
    DEPENDS             ""
    PREFIX              ${XBYAK_PREFIX_DIR}
    SOURCE_DIR          ${XBYAK_SOURCE_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${XBYAK_INSTALL_ROOT}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${XBYAK_INSTALL_ROOT}
)

add_library(xbyak INTERFACE)

add_dependencies(xbyak ${XBYAK_PROJECT})
