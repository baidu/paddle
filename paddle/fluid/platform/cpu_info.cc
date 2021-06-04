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

#include "paddle/fluid/platform/cpu_info.h"

#ifdef PADDLE_WITH_XBYAK
#include "xbyak/xbyak_util.h"
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif  // _WIN32

#include <algorithm>

#include "gflags/gflags.h"

DECLARE_double(fraction_of_cpu_memory_to_use);
DECLARE_uint64(initial_cpu_memory_in_mb);
DECLARE_double(fraction_of_cuda_pinned_memory_to_use);

// If use_pinned_memory is true, CPUAllocator calls mlock, which
// returns pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the amount
// of memory available to the system for paging.  So, by default, we
// should set false to use_pinned_memory.
DEFINE_bool(use_pinned_memory, true, "If set, allocate cpu pinned memory.");

namespace paddle {
namespace platform {

size_t CpuTotalPhysicalMemory() {
#ifdef __APPLE__
  int mib[2];
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  int64_t size = 0;
  size_t len = sizeof(size);
  if (sysctl(mib, 2, &size, &len, nullptr, 0) == 0)
    return static_cast<size_t>(size);
  return 0L;
#elif defined(_WIN32)
  MEMORYSTATUSEX sMeminfo;
  sMeminfo.dwLength = sizeof(sMeminfo);
  GlobalMemoryStatusEx(&sMeminfo);
  return sMeminfo.ullTotalPhys;
#else
  int64_t pages = sysconf(_SC_PHYS_PAGES);
  int64_t page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
#endif
}

size_t CpuMaxAllocSize() {
  // For distributed systems, it requires configuring and limiting
  // the fraction of memory to use.
  return FLAGS_fraction_of_cpu_memory_to_use * CpuTotalPhysicalMemory();
}

size_t CpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 4 KB.
  return 1 << 12;
}

size_t CpuMaxChunkSize() {
  // Allow to allocate the maximum chunk size is roughly 3% of CPU memory,
  // or the initial_cpu_memory_in_mb.
  return std::min(
      static_cast<size_t>(CpuMaxAllocSize() / 32),
      static_cast<size_t>(FLAGS_initial_cpu_memory_in_mb * 1 << 20));
}

size_t CUDAPinnedMaxAllocSize() {
  // For distributed systems, it requires configuring and limiting
  // the fraction of memory to use.
  return FLAGS_fraction_of_cuda_pinned_memory_to_use * CpuTotalPhysicalMemory();
}

size_t CUDAPinnedMinChunkSize() {
  // Allow to allocate the minimum chunk size is 64 KB.
  return 1 << 16;
}

size_t CUDAPinnedMaxChunkSize() {
  // Allow to allocate the maximum chunk size is roughly 1/256 of CUDA_PINNED
  // memory.
  return CUDAPinnedMaxAllocSize() / 256;
}

size_t NPUPinnedMaxAllocSize() {
  // For distributed systems, it requires configuring and limiting
  // the fraction of memory to use.
  return FLAGS_fraction_of_cuda_pinned_memory_to_use * CpuTotalPhysicalMemory();
}

size_t NPUPinnedMinChunkSize() {
  // Allow to allocate the minimum chunk size is 64 KB.
  return 1 << 16;
}

size_t NPUPinnedMaxChunkSize() {
  // Allow to allocate the maximum chunk size is roughly 1/256 of NPU_PINNED
  // memory.
  return NPUPinnedMaxAllocSize() / 256;
}

#ifdef PADDLE_WITH_XBYAK
static Xbyak::util::Cpu cpu;
bool MayIUse(const cpu_isa_t cpu_isa) {
  using namespace Xbyak::util;  // NOLINT
  switch (cpu_isa) {
    case sse42:
      return cpu.has(Cpu::tSSE42);
    case avx:
      return cpu.has(Cpu::tAVX);
    case avx2:
      return cpu.has(Cpu::tAVX2);
    case avx512f:
      return cpu.has(Cpu::tAVX512F);
    case avx512_core:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    case avx512_core_vnni:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ) &&
             cpu.has(Cpu::tAVX512_VNNI);
    case avx512_mic:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) &&
             cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
    case avx512_mic_4ops:
      return true && MayIUse(avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) &&
             cpu.has(Cpu::tAVX512_4VNNIW);
    case avx512_bf16:
      return true && cpu.has(Cpu::tAVX512_BF16);
    case isa_any:
      return true;
  }
  return false;
}
#else
bool MayIUse(const cpu_isa_t cpu_isa) {
  if (cpu_isa == isa_any) {
    return true;
  } else {
#if !defined(WITH_NV_JETSON) && !defined(PADDLE_WITH_ARM) && \
    !defined(PADDLE_WITH_SW) && !defined(PADDLE_WITH_MIPS)
    int reg[4];
    cpuid(reg, 0);
    int nIds = reg[0];
    if (nIds >= 0x00000001) {
      // EAX = 1
      cpuid(reg, 0x00000001);
      // AVX: ECX Bit 28
      if (cpu_isa == avx) {
        int avx_mask = (1 << 28);
        return (reg[2] & avx_mask) != 0;
      }
    }
    if (nIds >= 0x00000007) {
      // EAX = 7
      cpuid(reg, 0x00000007);
      if (cpu_isa == avx2) {
        // AVX2: EBX Bit 5
        int avx2_mask = (1 << 5);
        return (reg[1] & avx2_mask) != 0;
      } else if (cpu_isa == avx512f) {
        // AVX512F: EBX Bit 16
        int avx512f_mask = (1 << 16);
        return (reg[1] & avx512f_mask) != 0;
      } else if (cpu_isa == avx512_core) {
        unsigned int avx512f_mask = (1 << 16);
        unsigned int avx512dq_mask = (1 << 17);
        unsigned int avx512bw_mask = (1 << 30);
        unsigned int avx512vl_mask = (1 << 31);
        return ((reg[1] & avx512f_mask) && (reg[1] & avx512dq_mask) &&
                (reg[1] & avx512bw_mask) && (reg[1] & avx512vl_mask));
      }
      // EAX = 7, ECX = 1
      cpuid(reg, 0x00010007);
      if (cpu_isa == avx512_bf16) {
        // AVX512BF16: EAX Bit 5
        int avx512bf16_mask = (1 << 5);
        return (reg[0] & avx512bf16_mask) != 0;
      }
    }
#endif
    return false;
  }
}
#endif

}  // namespace platform
}  // namespace paddle
