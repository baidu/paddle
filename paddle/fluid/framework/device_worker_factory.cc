/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker_factory.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace framework {

class DeviceWorker;

typedef std::shared_ptr<DeviceWorker> (*Createdevice_workerFunction)();
typedef std::unordered_map<std::string, Createdevice_workerFunction>
    device_workerMap;
device_workerMap g_device_worker_map;
#define REGISTER_DEVICE_WORKER_CLASS(device_worker_class)                \
  namespace {                                                            \
  std::shared_ptr<DeviceWorker> Creator_##device_worker_class() {        \
    return std::shared_ptr<DeviceWorker>(new device_worker_class);       \
  }                                                                      \
  class __Registerer_##device_worker_class {                             \
   public:                                                               \
    __Registerer_##device_worker_class() {                               \
      g_device_worker_map[#device_worker_class] =                        \
          &Creator_##device_worker_class;                                \
    }                                                                    \
  };                                                                     \
  __Registerer_##device_worker_class g_registerer_##device_worker_class; \
  }  // namespace

std::string DeviceWorkerFactory::DeviceWorkerTypeList() {
  std::string device_worker_types;
  for (auto iter = g_device_worker_map.begin();
       iter != g_device_worker_map.end(); ++iter) {
    if (iter != g_device_worker_map.begin()) {
      device_worker_types += ", ";
    }
    device_worker_types += iter->first;
  }
  return device_worker_types;
}

std::shared_ptr<DeviceWorker> DeviceWorkerFactory::CreateDeviceWorker(
    std::string device_worker_class) {
  if (g_device_worker_map.count(device_worker_class) < 1) {
    exit(-1);
  }
  return g_device_worker_map[device_worker_class]();
}

REGISTER_DEVICE_WORKER_CLASS(HogwildWorker);
REGISTER_DEVICE_WORKER_CLASS(DownpourWorker);
REGISTER_DEVICE_WORKER_CLASS(DownpourWorkerOpt);
#ifdef PADDLE_WITH_PSLIB
REGISTER_DEVICE_WORKER_CLASS(HeterCpuWorker);
REGISTER_DEVICE_WORKER_CLASS(HeterBoxWorker);
#endif
#ifdef PADDLE_WITH_PSLIB
REGISTER_DEVICE_WORKER_CLASS(PSGPUWorker);
#endif
#if defined(PADDLE_WITH_NCCL)
REGISTER_DEVICE_WORKER_CLASS(SectionWorker);
#endif
}  // namespace framework
}  // namespace paddle
