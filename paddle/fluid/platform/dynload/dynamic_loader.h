/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <string>

namespace paddle {
namespace platform {
namespace dynload {

#ifndef _WIN32
#define DECLARE_TYPE(__name, ...) decltype(__name(__VA_ARGS__))
#else
#define DECLARE_TYPE(__name, ...) decltype(auto)
#endif

void* GetCublasDsoHandle();
void* GetCUDNNDsoHandle();
void* GetCUPTIDsoHandle();
void* GetCurandDsoHandle();
void* GetCusolverDsoHandle();
void* GetNVRTCDsoHandle();
void* GetCUDADsoHandle();
void* GetWarpCTCDsoHandle();
void* GetNCCLDsoHandle();
void* GetHCCLDsoHandle();
void* GetTensorRtDsoHandle();
void* GetMKLMLDsoHandle();
void* GetOpDsoHandle(const std::string& dso_name);
void* GetNvtxDsoHandle();

void SetPaddleLibPath(const std::string&);
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
