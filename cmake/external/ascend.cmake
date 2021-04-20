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


#NOTE: Logic is from
# https://github.com/mindspore-ai/graphengine/blob/master/CMakeLists.txt
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_DIR /usr/local/Ascend)
endif()

if(EXISTS ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include/graph/ascend_string.h)
  # It means CANN 20.2 +
  add_definitions(-DPADDLE_WITH_ASCEND_STRING)
endif()

if(WITH_ASCEND)
  set(ASCEND_DRIVER_DIR ${ASCEND_DIR}/driver/lib64)
  set(ASCEND_DRIVER_COMMON_DIR ${ASCEND_DIR}/driver/lib64/common)
  set(ASCEND_DRIVER_SHARE_DIR ${ASCEND_DIR}/driver/lib64/share)
  set(ASCEND_RUNTIME_DIR ${ASCEND_DIR}/fwkacllib/lib64)
  set(ASCEND_ATC_DIR ${ASCEND_DIR}/atc/lib64)
  set(ASCEND_ACL_DIR ${ASCEND_DIR}/acllib/lib64)
  set(STATIC_ACL_LIB ${ASCEND_ACL_DIR})

  set(ASCEND_MS_RUNTIME_PATH ${ASCEND_RUNTIME_DIR} ${ASCEND_ACL_DIR} ${ASCEND_ATC_DIR})
  set(ASCEND_MS_DRIVER_PATH ${ASCEND_DRIVER_DIR} ${ASCEND_DRIVER_COMMON_DIR})
  set(ATLAS_RUNTIME_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/lib64)
  set(ATLAS_RUNTIME_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include)
  set(ATLAS_ACL_DIR ${ASCEND_DIR}/ascend-toolkit/latest/acllib/lib64)
  set(ATLAS_ATC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/atc/lib64)
  set(ATLAS_MS_RUNTIME_PATH ${ATLAS_RUNTIME_DIR} ${ATLAS_ACL_DIR} ${ATLAS_ATC_DIR})

  set(atlas_graph_lib ${ATLAS_RUNTIME_DIR}/libgraph.so)
  set(atlas_ge_runner_lib ${ATLAS_RUNTIME_DIR}/libge_runner.so)
  set(atlas_acl_lib ${ATLAS_RUNTIME_DIR}/libascendcl.so)
  INCLUDE_DIRECTORIES(${ATLAS_RUNTIME_INC_DIR})



  ADD_LIBRARY(ascend_ge SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET ascend_ge PROPERTY IMPORTED_LOCATION ${atlas_ge_runner_lib})

  ADD_LIBRARY(ascend_graph SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET ascend_graph PROPERTY IMPORTED_LOCATION ${atlas_graph_lib})

  ADD_LIBRARY(atlas_acl SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET atlas_acl PROPERTY IMPORTED_LOCATION ${atlas_acl_lib})

  add_custom_target(extern_ascend DEPENDS ascend_ge ascend_graph atlas_acl)
endif()

if(WITH_ASCEND_CL)
  set(ASCEND_CL_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/lib64)

  set(ascend_hccl_lib ${ASCEND_CL_DIR}/libhccl.so)
  set(ascendcl_lib ${ASCEND_CL_DIR}/libascendcl.so)
  set(acl_op_compiler_lib ${ASCEND_CL_DIR}/libacl_op_compiler.so)
  set(FWKACLLIB_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include)
  set(ACLLIB_INC_DIR ${ASCEND_DIR}/ascend-toolkit/latest/acllib/include)

  message(STATUS "FWKACLLIB_INC_DIR ${FWKACLLIB_INC_DIR}")
  message(STATUS "ASCEND_CL_DIR ${ASCEND_CL_DIR}")
  INCLUDE_DIRECTORIES(${FWKACLLIB_INC_DIR})
  INCLUDE_DIRECTORIES(${ACLLIB_INC_DIR})

  ADD_LIBRARY(ascendcl SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET ascendcl PROPERTY IMPORTED_LOCATION ${ascendcl_lib})

  ADD_LIBRARY(ascend_hccl SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET ascend_hccl PROPERTY IMPORTED_LOCATION ${ascend_hccl_lib})

  ADD_LIBRARY(acl_op_compiler SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET acl_op_compiler PROPERTY IMPORTED_LOCATION ${acl_op_compiler_lib})
  add_custom_target(extern_ascend_cl DEPENDS ascendcl acl_op_compiler)

endif()
