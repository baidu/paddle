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

#include <functional>
#include <iostream>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace platform {

struct CPUPlace {
  // WORKAROUND: for some reason, omitting this constructor
  // causes errors with boost 1.59 and OSX
  CPUPlace() {}

  // needed for variant equality comparison
  inline bool operator==(const CPUPlace &) const { return true; }
  inline bool operator!=(const CPUPlace &) const { return false; }
  inline bool operator<(const CPUPlace &) const { return false; }
};

struct CUDAPlace {
  CUDAPlace() : CUDAPlace(0) {}
  explicit CUDAPlace(int d) : device(d) {}

  inline int GetDeviceId() const { return device; }
  // needed for variant equality comparison
  inline bool operator==(const CUDAPlace &o) const {
    return device == o.device;
  }
  inline bool operator!=(const CUDAPlace &o) const { return !(*this == o); }
  inline bool operator<(const CUDAPlace &o) const { return device < o.device; }

  int device;
};

struct CUDAPinnedPlace {
  CUDAPinnedPlace() {}

  // needed for variant equality comparison
  inline bool operator==(const CUDAPinnedPlace &) const { return true; }
  inline bool operator!=(const CUDAPinnedPlace &) const { return false; }
  inline bool operator<(const CUDAPinnedPlace &) const { return false; }
};

// Place for Baidu Kunlun Accelerator
struct XPUPlace {
  XPUPlace() : XPUPlace(0) {}
  explicit XPUPlace(int d) : device(d) {}

  inline int GetDeviceId() const { return device; }
  // needed for variant equality comparison
  inline bool operator==(const XPUPlace &o) const { return device == o.device; }
  inline bool operator!=(const XPUPlace &o) const { return !(*this == o); }
  inline bool operator<(const XPUPlace &o) const { return device < o.device; }

  int device;
};

struct IsCUDAPlace : public boost::static_visitor<bool> {
  bool operator()(const CPUPlace &) const { return false; }
  bool operator()(const XPUPlace &) const { return false; }
  bool operator()(const CUDAPlace &gpu) const { return true; }
  bool operator()(const CUDAPinnedPlace &) const { return false; }
};

struct IsCPUPlace : public boost::static_visitor<bool> {
  bool operator()(const CPUPlace &cpu) const { return true; }
  bool operator()(const XPUPlace &) const { return false; }
  bool operator()(const CUDAPlace &) const { return false; }
  bool operator()(const CUDAPinnedPlace &) const { return false; }
};

struct IsCUDAPinnedPlace : public boost::static_visitor<bool> {
  bool operator()(const CPUPlace &) const { return false; }
  bool operator()(const XPUPlace &) const { return false; }
  bool operator()(const CUDAPlace &) const { return false; }
  bool operator()(const CUDAPinnedPlace &cuda_pinned) const { return true; }
};

struct IsXPUPlace : public boost::static_visitor<bool> {
  bool operator()(const CPUPlace &) const { return false; }
  bool operator()(const XPUPlace &xpu) const { return true; }
  bool operator()(const CUDAPlace &) const { return false; }
  bool operator()(const CUDAPinnedPlace &) const { return false; }
};

class Place
    : public boost::variant<CUDAPlace, XPUPlace, CPUPlace, CUDAPinnedPlace> {
 private:
  using PlaceBase =
      boost::variant<CUDAPlace, XPUPlace, CPUPlace, CUDAPinnedPlace>;

 public:
  Place() = default;
  Place(const CPUPlace &cpu_place) : PlaceBase(cpu_place) {}     // NOLINT
  Place(const XPUPlace &xpu_place) : PlaceBase(xpu_place) {}     // NOLINT
  Place(const CUDAPlace &cuda_place) : PlaceBase(cuda_place) {}  // NOLINT
  Place(const CUDAPinnedPlace &cuda_pinned_place)                // NOLINT
      : PlaceBase(cuda_pinned_place) {}

  bool operator<(const Place &place) const {
    return PlaceBase::operator<(static_cast<const PlaceBase &>(place));
  }
  bool operator==(const Place &place) const {
    return PlaceBase::operator==(static_cast<const PlaceBase &>(place));
  }
};

using PlaceList = std::vector<Place>;

bool is_gpu_place(const Place &);
bool is_xpu_place(const Place &);
bool is_cpu_place(const Place &);
bool is_cuda_pinned_place(const Place &);
bool places_are_same_class(const Place &, const Place &);
bool is_same_place(const Place &, const Place &);

std::ostream &operator<<(std::ostream &, const Place &);

template <typename Visitor>
struct PlaceVisitorWrapper
    : public boost::static_visitor<typename Visitor::result_type> {
  const Visitor &visitor_;
  explicit PlaceVisitorWrapper(const Visitor &visitor) : visitor_(visitor) {}

  typename Visitor::result_type operator()(const CPUPlace &cpu) const {
    return visitor_(cpu);
  }

  typename Visitor::result_type operator()(const XPUPlace &xpu) const {
#ifdef PADDLE_WITH_XPU
    return visitor_(xpu);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Paddle is not compiled with XPU. Cannot visit xpu device"));
    return typename Visitor::result_type();
#endif
  }

  typename Visitor::result_type operator()(const CUDAPlace &cuda) const {
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP))
    return visitor_(cuda);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Paddle is not compiled with CUDA. Cannot visit cuda device"));
    return typename Visitor::result_type();
#endif
  }

  typename Visitor::result_type operator()(
      const CUDAPinnedPlace &cuda_pinned) const {
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP))
    return visitor_(cuda_pinned);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Paddle is not compiled with CUDA. Cannot visit cuda_pinned"));
    return typename Visitor::result_type();
#endif
  }
};

template <typename Visitor>
typename Visitor::result_type VisitPlace(const Place &place,
                                         const Visitor &visitor) {
  return boost::apply_visitor(PlaceVisitorWrapper<Visitor>(visitor), place);
}

}  // namespace platform
}  // namespace paddle
