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

#include "paddle/fluid/operators/detection/mask_util.h"

#include <cmath>
#include <cstdlib>

#include "paddle/fluid/memory/memory.h"

namespace paddle {
namespace operators {

uint32_t UMax(uint32_t a, uint32_t b) { return (a > b) ? a : b; }

static inline int Compare(const void* a, const void* b) {
  uint32_t c = *(reinterpret_cast<const uint32_t*>(a));
  uint32_t d = *(reinterpret_cast<const uint32_t*>(b));
  return c > d ? 1 : c < d ? -1 : 0;
}

void Decode(const uint32_t* cnts, int m, uint8_t* mask) {
  uint8_t v = 0;
  for (int j = 0; j < m; j++) {
    for (uint32_t k = 0; k < cnts[j]; k++) {
      *(mask++) = v;
    }
    v = !v;
  }
}

using uint = uint32_t;
void Poly2Mask(const float* xy, int k, int h, int w, uint8_t* mask) {
  int j, m = 0;
  double scale = 5;
  int *x, *y, *u, *v;
  uint *a, *b;
  platform::CPUPlace cpu;
  auto xptr = memory::Alloc(cpu, sizeof(int) * (k + 1) * 2);
  x = reinterpret_cast<int*>(xptr->ptr());
  y = x + (k + 1);

  for (j = 0; j < k; j++) x[j] = static_cast<int>(scale * xy[j * 2 + 0] + .5);
  x[k] = x[0];
  for (j = 0; j < k; j++) y[j] = static_cast<int>(scale * xy[j * 2 + 1] + .5);
  y[k] = y[0];
  for (j = 0; j < k; j++) {
    m += UMax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
  }
  auto vptr = memory::Alloc(cpu, sizeof(int) * m * 2);
  u = reinterpret_cast<int*>(vptr->ptr());
  v = u + m;
  m = 0;
  for (j = 0; j < k; j++) {
    int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
    int flip;
    double s;
    dx = abs(xe - xs);
    dy = abs(ys - ye);
    flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
    if (flip) {
      t = xs;
      xs = xe;
      xe = t;
      t = ys;
      ys = ye;
      ye = t;
    }
    if (dx >= dy) {
      s = dx == 0 ? 0 : static_cast<double>(ye - ys) / dx;
      for (d = 0; d <= dx; d++) {
        t = flip ? dx - d : d;
        u[m] = t + xs;
        v[m] = static_cast<int>(ys + s * t + .5);
        m++;
      }
    } else {
      s = dy == 0 ? 0 : static_cast<double>(xe - xs) / dy;
      for (d = 0; d <= dy; d++) {
        t = flip ? dy - d : d;
        v[m] = t + ys;
        u[m] = static_cast<int>(xs + s * t + .5);
        m++;
      }
    }
  }
  /* get points along y-boundary and downsample */
  k = m;
  m = 0;
  double xd, yd;
  auto xyptr = memory::Alloc(cpu, sizeof(int) * k * 2);
  x = reinterpret_cast<int*>(xyptr->ptr());
  y = x + k;
  for (j = 1; j < k; j++) {
    if (u[j] != u[j - 1]) {
      xd = static_cast<double>(u[j] < u[j - 1] ? u[j] : u[j] - 1);
      xd = (xd + .5) / scale - .5;
      if (floor(xd) != xd || xd < 0 || xd > w - 1) continue;
      yd = static_cast<double>(v[j] < v[j - 1] ? v[j] : v[j - 1]);
      yd = (yd + .5) / scale - .5;
      if (yd < 0)
        yd = 0;
      else if (yd > h)
        yd = h;
      yd = ceil(yd);
      x[m] = static_cast<int>(xd);
      y[m] = static_cast<int>(yd);
      m++;
    }
  }
  /* compute rle encoding given y-boundary points */
  k = m;
  auto aptr = memory::Alloc(cpu, sizeof(uint) * (k + 1));
  a = reinterpret_cast<uint*>(aptr->ptr());
  for (j = 0; j < k; j++) a[j] = static_cast<uint>(x[j] * h + y[j]);
  a[k++] = static_cast<uint>(h * w);

  qsort(a, k, sizeof(uint), Compare);
  uint p = 0;
  for (j = 0; j < k; j++) {
    uint t = a[j];
    a[j] -= p;
    p = t;
  }
  auto bptr = memory::Alloc(cpu, sizeof(uint32_t) * k);
  b = reinterpret_cast<uint32_t*>(bptr->ptr());
  j = m = 0;
  b[m++] = a[j++];
  while (j < k) {
    if (a[j] > 0) {
      b[m++] = a[j++];
    } else {
      j++;
      if (j < k) b[m - 1] += a[j++];
    }
  }

  // convert to mask
  auto mskptr = memory::Alloc(cpu, sizeof(uint8_t) * h * w);
  auto* msk = reinterpret_cast<uint8_t*>(mskptr->ptr());
  Decode(b, m, msk);

  for (int ii = 0; ii < h; ++ii) {
    for (int jj = 0; jj < w; ++jj) {
      mask[ii * w + jj] = msk[jj * h + ii];
    }
  }
}

void Poly2Boxes(const std::vector<std::vector<std::vector<float>>>& polys,
                float* boxes) {
  // lists
  for (size_t i = 0; i < polys.size(); ++i) {
    float x0 = std::numeric_limits<float>::max();
    float x1 = std::numeric_limits<float>::min();
    float y0 = std::numeric_limits<float>::max();
    float y1 = std::numeric_limits<float>::min();
    // each list may have more than one polys
    // NOLINTNEXTLINE
    for (size_t j = 0; j < polys[i].size(); ++j) {
      for (size_t k = 0; k < polys[i][j].size() / 2; ++k) {
        x0 = std::min(x0, polys[i][j][2 * k]);
        x1 = std::max(x1, polys[i][j][2 * k]);
        y0 = std::min(y0, polys[i][j][2 * k + 1]);
        y1 = std::max(y1, polys[i][j][2 * k + 1]);
      }
    }
    boxes[i * 4] = x0;
    boxes[i * 4 + 1] = y0;
    boxes[i * 4 + 2] = x1;
    boxes[i * 4 + 3] = y1;
  }
}

void Polys2MaskWrtBox(const std::vector<std::vector<float>>& polygons,
                      const float* box, int M, uint8_t* mask) {
  float w = box[2] - box[0];
  float h = box[3] - box[1];
  w = std::max(w, static_cast<float>(1.));
  h = std::max(h, static_cast<float>(1.));

  uint8_t* msk = nullptr;
  if (polygons.size() == 1UL) {
    msk = mask;
  } else {
    msk = reinterpret_cast<uint8_t*>(
        malloc(M * M * polygons.size() * sizeof(uint8_t)));
  }
  for (size_t i = 0; i < polygons.size(); ++i) {
    int k = polygons[i].size() / 2;
    std::vector<float> p;
    for (int j = 0; j < k; ++j) {
      float pw = (polygons[i][2 * j] - box[0]) * M / w;
      float ph = (polygons[i][2 * j + 1] - box[1]) * M / h;
      p.push_back(pw);
      p.push_back(ph);
    }
    uint8_t* msk_i = msk + i * M * M;
    Poly2Mask(p.data(), k, M, M, msk_i);
  }

  if (polygons.size() > 1UL) {
    for (size_t i = 0; i < polygons.size(); ++i) {
      uint8_t* msk_i = msk + i * M * M;
      for (int j = 0; j < M * M; ++j) {
        if (i == 0) {
          mask[j] = msk_i[j];
        } else {
          mask[j] = (mask[j] + msk_i[j]) > 0 ? 1 : 0;
        }
      }
    }
    free(msk);
  }
}

}  // namespace operators
}  // namespace paddle
