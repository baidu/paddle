/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <NvInfer.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class ReduceSumOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a paddle reduce_sum op to tensorrt reduce layer";
    framework::OpDesc op_desc(op, nullptr);

    auto* x = engine_->GetITensor(op_desc.Input("X").front());
    nvinfer1::Dims input_shape = x->getDimensions();
    int input_dims = input_shape.nbDims;

    bool keep_dim = BOOST_GET_CONST(bool, op_desc.GetAttr("keep_dim"));
    std::vector<int32_t> dim =
        BOOST_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("dim"));
    bool reduce_all = BOOST_GET_CONST(bool, op_desc.GetAttr("reduce_all"));

    // Now we only support dynamic_shape mode.
    nvinfer1::IReduceLayer* layer = nullptr;
    if (reduce_all) {
      uint32_t reduce_dim = 0;
      for (int i = 0; i < input_dims; ++i) {
        reduce_dim |= 1 << i;
      }
      layer = TRT_ENGINE_ADD_LAYER(engine_, Reduce, *x,
                                   nvinfer1::ReduceOperation::kSUM, reduce_dim,
                                   keep_dim);
    } else {
      auto CvtToBitMask = [&](const std::vector<int32_t>& dims) -> uint32_t {
        uint32_t res = 0;
        for (auto x : dims) {
          if (x < 0) {
            res |= 1 << (x + input_dims);
          } else {
            res |= 1 << x;
          }
        }
        return res;
      };
      layer = TRT_ENGINE_ADD_LAYER(engine_, Reduce, *x,
                                   nvinfer1::ReduceOperation::kSUM,
                                   CvtToBitMask(dim), keep_dim);
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "reduce_sum", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(reduce_sum, ReduceSumOpConverter);
