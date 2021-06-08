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

#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace platform {

void Timer::Reset() {
  _start.tv_sec = 0;
  _start.tv_usec = 0;

  _count = 0;
  _elapsed = 0;
  _paused = true;
}

void Timer::Start() {
  Reset();
  Resume();
}

void Timer::Pause() {
  if (_paused) {
    return;
  }
  _elapsed += Tickus();
  ++_count;
  _paused = true;
}

void Timer::Resume() {
  gettimeofday(&_start, nullptr);
  _paused = false;
}

int Timer::Count() { return _count; }

double Timer::ElapsedUS() { return static_cast<double>(_elapsed); }

double Timer::ElapsedMS() { return _elapsed / 1000.0; }

double Timer::ElapsedSec() { return _elapsed / 1000000.0; }

int64_t Timer::Tickus() {
  gettimeofday(&_now, nullptr);
  return (_now.tv_sec - _start.tv_sec) * 1000 * 1000L +
         (_now.tv_usec - _start.tv_usec);
}

}  // namespace platform
}  // namespace paddle
