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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TensorRTEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx_ = new platform::CUDADeviceContext(platform::CUDAPlace(0));

    engine_ = new TensorRTEngine(10, 1 << 10);
    engine_->InitNetwork();
  }

  void TearDown() override {
    if (engine_) {
      delete engine_;
      engine_ = nullptr;
    }
  }

  void PrepareInputOutput(const std::vector<float> &input,
                          std::vector<int> output_shape) {
    TensorFromVector(input, *ctx_, &input_);
    output_.Resize(framework::make_ddim(output_shape));
  }

  void GetOutput(std::vector<float> *output) {
    TensorToVector(output_, *ctx_, output);
  }

 protected:
  framework::Tensor input_;
  framework::Tensor output_;
  TensorRTEngine *engine_;
  platform::CUDADeviceContext *ctx_;
};

TEST_F(TensorRTEngineTest, add_layer) {
  const int size = 1;

  float raw_weight[size] = {2.};  // Weight in CPU memory.
  float raw_bias[size] = {3.};

  std::vector<void *> buffers(2);  // TRT binded inputs

  LOG(INFO) << "create weights";
  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, size);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, size);
  auto *x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 1, 1});
  auto *fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *x, size,
                                        weight.get(), bias.get());
  PADDLE_ENFORCE_NOT_NULL(fc_layer,
                          platform::errors::InvalidArgument(
                              "TRT fully connected layer building failed."));

  engine_->DeclareOutput(fc_layer, 0, "y");
  LOG(INFO) << "freeze network";
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  // fill in real data
  std::vector<float> x_v = {1234};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {1});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  LOG(INFO) << "Set attr";
  engine_->Set("test_attr", new std::string("test_attr"));
  if (engine_->Has("test_attr")) {
    auto attr_val = engine_->Get<std::string>("test_attr");
    engine_->Erase("test_attr");
  }
  std::string *attr_key = new std::string("attr_key");
  engine_->SetNotOwned("attr1", attr_key);

  LOG(INFO) << "to execute";
  engine_->Execute(1, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  LOG(INFO) << "to checkout output";
  ASSERT_EQ(y_cpu[0], x_v[0] * 2 + 3);

  delete attr_key;
}

TEST_F(TensorRTEngineTest, add_layer_multi_dim) {
  // Weight in CPU memory.
  // It seems tensorrt FC use col-major: [[1.0, 3.3], [1.1, 4.4]]
  // instead of row-major, which is [[1.0, 1.1], [3.3, 4.4]]
  float raw_weight[4] = {1.0, 1.1, 3.3, 4.4};
  float raw_bias[2] = {1.3, 2.4};
  std::vector<void *> buffers(2);  // TRT binded inputs

  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, 4);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, 2);
  auto *x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 2, 1});
  auto *fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *x, 2,
                                        weight.get(), bias.get());
  PADDLE_ENFORCE_NOT_NULL(fc_layer,
                          platform::errors::InvalidArgument(
                              "TRT fully connected layer building failed."));

  engine_->DeclareOutput(fc_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  // fill in real data
  std::vector<float> x_v = {1.0, 2.0};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {2});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(1, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  auto dims = engine_->GetITensor("y")->getDimensions();
  ASSERT_EQ(dims.nbDims, 3);
  ASSERT_EQ(dims.d[0], 2);
  ASSERT_EQ(dims.d[1], 1);

  ASSERT_EQ(y_cpu[0], 4.5);
  ASSERT_EQ(y_cpu[1], 14.5);
}

TEST_F(TensorRTEngineTest, test_conv2d) {
  // Weight in CPU memory.
  float raw_weight[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float raw_bias[1] = {0};
  std::vector<void *> buffers(2);  // TRT binded inputs

  TensorRTEngine::Weight weight(nvinfer1::DataType::kFLOAT, raw_weight, 9);
  TensorRTEngine::Weight bias(nvinfer1::DataType::kFLOAT, raw_bias, 1);
  auto *x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 3, 3});
  auto *conv_layer =
      TRT_ENGINE_ADD_LAYER(engine_, Convolution, *x, 1, nvinfer1::DimsHW{3, 3},
                           weight.get(), bias.get());
  PADDLE_ENFORCE_NOT_NULL(conv_layer,
                          platform::errors::InvalidArgument(
                              "TRT convolution layer building failed."));
  conv_layer->setStride(nvinfer1::DimsHW{1, 1});
  conv_layer->setPadding(nvinfer1::DimsHW{1, 1});

  engine_->DeclareOutput(conv_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  // fill in real data
  std::vector<float> x_v = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {18});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(2, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  ASSERT_EQ(y_cpu[0], 4.0);
  ASSERT_EQ(y_cpu[1], 6.0);
}

TEST_F(TensorRTEngineTest, test_pool2d) {
  // Weight in CPU memory.
  auto *x = engine_->DeclareInput("x", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims3{1, 2, 2});

  std::vector<void *> buffers(2);  // TRT binded inputs
  nvinfer1::PoolingType pool_t = nvinfer1::PoolingType::kAVERAGE;
  auto *pool_layer = TRT_ENGINE_ADD_LAYER(engine_, Pooling, *x, pool_t,
                                          nvinfer1::DimsHW{2, 2});

  PADDLE_ENFORCE_NOT_NULL(
      pool_layer,
      platform::errors::InvalidArgument("TRT pooling layer building failed."));
  pool_layer->setStride(nvinfer1::DimsHW{1, 1});
  pool_layer->setPadding(nvinfer1::DimsHW{0, 0});

  engine_->DeclareOutput(pool_layer, 0, "y");
  engine_->FreezeNetwork();
  ASSERT_EQ(engine_->engine()->getNbBindings(), 2);

  // fill in real data
  std::vector<float> x_v = {1.0, 2.0, 5.0, 0.0, 2.0, 3.0, 5.0, 10.0};
  std::vector<float> y_cpu;
  PrepareInputOutput(x_v, {2});

  auto *x_v_gpu_data = input_.mutable_data<float>(ctx_->GetPlace());
  auto *y_gpu_data = output_.mutable_data<float>(ctx_->GetPlace());

  buffers[0] = reinterpret_cast<void *>(x_v_gpu_data);
  buffers[1] = reinterpret_cast<void *>(y_gpu_data);

  engine_->Execute(2, &buffers, ctx_->stream());

  LOG(INFO) << "to get output";
  GetOutput(&y_cpu);

  ASSERT_EQ(y_cpu[0], 2.0);
  ASSERT_EQ(y_cpu[1], 5.0);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
