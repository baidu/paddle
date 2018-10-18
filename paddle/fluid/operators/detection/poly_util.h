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

#ifndef POLY_UTIL_H_
#define POLY_UTIL_H_

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/gpc.h"

namespace paddle {
namespace operators {

template <class T>
class Point_ {
 public:
  // default constructor
  Point_() {}
  Point_(T _x, T _y) {}
  Point_(const Point_& pt) {}

  Point_& operator=(const Point_& pt);
  // conversion to another data type
  // template<typename _T> operator Point_<_T>() const;
  // conversion to the old-style C structures
  // operator Vec<T, 2>() const;

  // checks whether the point is inside the specified rectangle
  // bool inside(const Rect_<T>& r) const;
  T x;  //!< x coordinate of the point
  T y;  //!< y coordinate of the point
};

template <class T>
void Array2PointVec(const T*& box, const size_t box_size,
                    std::vector<Point_<T>>& vec);

template <class T>
void Array2Poly(const T*& box, const size_t box_size, gpc::gpc_polygon& poly);

template <class T>
void PointVec2Poly(const std::vector<Point_<T>>& vec, gpc::gpc_polygon& poly);

template <class T>
void Poly2PointVec(const gpc::gpc_vertex_list& contour,
                   std::vector<Point_<T>>& vec);

template <class T>
T GetContourArea(std::vector<Point_<T>>& vec);

template <class T>
T PolyArea(const T* box, const size_t box_size, const bool normalized);

template <class T>
T PolyOverlapArea(const T* box1, const T* box2, const size_t box_size,
                  const bool normalized);
}  // namespace operators
}  // namespace paddle

#include "paddle/fluid/operators/detection/poly_util.cc"

#endif  // POLY_UTIL_H_
