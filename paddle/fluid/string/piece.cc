//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/string/piece.h"

#include <algorithm>
#include <cstring>
#define CHAR_POINTER_CMP(a, b) \
  do {                         \
    if (!a && !b) return 0;    \
    if (!a) return -1;         \
    if (!b) return 1;          \
  } while (0)

namespace paddle {
namespace string {

Piece::Piece() : data_(nullptr), size_(0) {}

Piece::Piece(const char* d, size_t n) : data_(d), size_(n) {
  if (d == nullptr && n != 0)
    throw std::invalid_argument("Piece requires len to be 0 for NULL data");
}

Piece::Piece(const char* s) : data_(s) {
  size_ = (s == nullptr) ? 0 : strlen(s);
}

Piece::Piece(const std::string& s) : data_(s.data()), size_(s.size()) {}

char Piece::operator[](size_t n) const {
  if (n >= len()) throw std::invalid_argument("index out of Piece length");
  return data_[n];
}

int Compare(Piece a, Piece b) {
  CHAR_POINTER_CMP(a.data(), b.data());
  const size_t min_len = (a.len() < b.len()) ? a.len() : b.len();
  int r = memcmp(a.data(), b.data(), min_len);
  if (r == 0) {
    if (a.len() < b.len())
      return -1;
    else if (a.len() > b.len())
      return 1;
  }
  return r;
}

bool operator==(Piece x, Piece y) {
  return (!x.len() && !y.len()) ? true
                                : ((x.len() == y.len()) &&
                                   (x.data() == y.data() ||
                                    memcmp(x.data(), y.data(), x.len()) == 0));
}

bool operator!=(Piece x, Piece y) { return !(x == y); }

bool operator<(Piece x, Piece y) { return Compare(x, y) < 0; }
bool operator>(Piece x, Piece y) { return Compare(x, y) > 0; }

bool operator<=(Piece x, Piece y) { return Compare(x, y) <= 0; }
bool operator>=(Piece x, Piece y) { return Compare(x, y) >= 0; }

bool HasPrefix(Piece s, Piece x) {
  return !x.len() ? true
                  : ((s.len() >= x.len()) &&
                     (memcmp(s.data(), x.data(), x.len()) == 0));
}

bool HasSuffix(Piece s, Piece x) {
  return !x.len()
             ? true
             : ((s.len() >= x.len()) && (memcmp(s.data() + (s.len() - x.len()),
                                                x.data(), x.len()) == 0));
}

Piece SkipPrefix(Piece s, size_t n) {
  if (n > s.len())
    throw std::invalid_argument("Skip distance larger than Piece length");
  return {s.data() + n, s.len() - n};
}

Piece SkipSuffix(Piece s, size_t n) {
  if (n > s.len())
    throw std::invalid_argument("Skip distance larger than Piece length");
  return {s.data(), s.len() - n};
}

Piece TrimPrefix(Piece s, Piece x) {
  return HasPrefix(s, x) ? SkipPrefix(s, x.len()) : s;
}

Piece TrimSuffix(Piece s, Piece x) {
  return HasSuffix(s, x) ? SkipSuffix(s, x.len()) : s;
}

bool Contains(Piece s, Piece sub) {
  return std::search(s.begin(), s.end(), sub.begin(), sub.end()) != s.end();
}

size_t Index(Piece s, Piece sub) {
  auto e = std::search(s.begin(), s.end(), sub.begin(), sub.end());
  return e != s.end() ? e - s.data() : Piece::npos;
}

size_t Find(Piece s, char c, size_t pos) {
  if (pos >= s.len()) {
    return Piece::npos;
  }
  const char* result =
      reinterpret_cast<const char*>(memchr(s.data() + pos, c, s.len() - pos));
  return result != nullptr ? result - s.data() : Piece::npos;
}

size_t RFind(Piece s, char c, size_t pos) {
  if (s.len() == 0) return Piece::npos;
  for (const char* p = s.data() + std::min(pos, s.len() - 1); p >= s.data();
       p--) {
    if (*p == c) {
      return p - s.data();
    }
  }
  return Piece::npos;
}

Piece SubStr(Piece s, size_t pos, size_t n) {
  if (pos > s.len()) pos = s.len();
  if (n > s.len() - pos) n = s.len() - pos;
  return {s.data() + pos, n};
}

std::ostream& operator<<(std::ostream& o, Piece piece) {
  return o << piece.ToString();
}

}  // namespace string
}  // namespace paddle
