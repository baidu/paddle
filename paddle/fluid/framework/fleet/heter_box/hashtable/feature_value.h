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

#pragma once

#ifdef PADDLE_WITH_PSLIB

#include <iostream>

namespace paddle {
namespace framework {
#define MF_DIM 8

typedef uint64_t FeatureKey;

struct FeatureValue {
  float delta_score;
  float show;
  float clk;
  int slot;
  float lr;
  float lr_g2sum;
  int mf_size;
  float mf[MF_DIM + 1]; 
  
  friend std::ostream& operator<<(std::ostream& out, FeatureValue& val) {
    out << "show: " << val.show << " clk: " << val.clk
        << " slot: " << val.slot << " lr: " << val.lr
        << " mf_size: " << val.mf_size << " mf:";
    for (int i = 0; i < val.mf_size; ++i) {
      out << " " << val.mf[i];
    }
    return out;
  }

};

struct FeaturePushValue {
  float show;
  float clk;
  int slot;
  float lr_g;
  float mf_g[MF_DIM];
};
//class DownpourFixedFeatureValue {
//    public:
//        DownpourFixedFeatureValue() {}
//        ~DownpourFixedFeatureValue() {}
//        float* data() {
//            return _data.data();
//        }
//        size_t size() {
//            return _data.size();
//        }
//        void resize(size_t size) {
//            _data.resize(size);
//        }
//        void shrink_to_fit() {
//            _data.shrink_to_fit();
//        }
//    private:
//        std::vector<float> _data;
//    };

}  // end namespace framework
}  // end namespace paddle
#endif
