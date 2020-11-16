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
#

set(C_API_SRCS pd_config.cc pd_predictor.cc pd_tensor.cc c_api.cc)

cc_library(paddle_fluid_c SRCS ${C_API_SRCS} DEPS paddle_fluid)

if(NOT ON_INFER)
    return()
endif()

# Create inference capi shared library
cc_library(paddle_fluid_c_shared SHARED SRCS ${C_API_SRCS} DEPS paddle_fluid)
set_target_properties(paddle_fluid_c_shared PROPERTIES OUTPUT_NAME paddle_fluid_c)
set_target_properties(paddle_fluid_c_shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(WIN32)
    target_link_libraries(paddle_fluid_c_shared shlwapi.lib)
endif()
