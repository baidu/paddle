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

#include <map>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <exception>

namespace paddle {
    namespace platform {

        enum DebugInfoType {
            TOperaor = 0
        };

        class DebugSupport {
        public:
            // Returns the singleton of ThreadPool.
            static DebugSupport *GetInstance();

            std::string getActiveOperator();

            void setActiveOperator(std::string info);

            std::string getBacktraceStacks();

        private:
            DebugSupport() {
              infos.insert(std::make_pair<DebugInfoType, std::string>(TOperaor, ""));
            }

            static std::once_flag init_flag_;
            std::map <DebugInfoType, std::string> infos;
        };
    }  // namespace platform
}  // namespace paddle
