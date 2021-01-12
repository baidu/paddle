# Paddle 预测 API

为了更简单方便的预测部署，Fluid 提供了一套高层 API 用来隐藏底层不同的优化实现。

预测库包含:

- 头文件 `paddle_inference_api.h` 定义了所有的接口
- 库文件 `libpaddle_fluid.so/.a(Linux/Mac)` `libpaddle_fluid.lib/paddle_fluid.dll(Windows)` 

下面是详细的一些 API 概念介绍

## PaddleTensor

PaddleTensor 定义了预测最基本的输入输出的数据格式，其定义是

```c++
struct PaddleTensor {
  std::string name;  // variable name.
  std::vector<int> shape;
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
};
```

- `name` 用于指定输入数据对应的 模型中variable 的名字 （暂时没有用，但会在后续支持任意 target 时启用）
- `shape` 表示一个 Tensor 的 shape
- `data`  数据以连续内存的方式存储在`PaddleBuf` 中，`PaddleBuf` 可以接收外面的数据或者独立`malloc`内存，详细可以参考头文件中相关定义。
- `dtype` 表示 Tensor 的数据类型

## engine

高层 API 底层有多种优化实现，我们称之为 engine，目前有两种 engine

- 原生 engine，由 paddle 原生的 forward operator 组成，可以天然支持所有paddle 训练出的模型，
- TensorRT mixed engine，用子图的方式支持了 [TensorRT](https://developer.nvidia.com/tensorrt) ，支持所有paddle 模型，并自动切割部分计算子图到 TensorRT 上加速（WIP）

其实现为

```c++
enum class PaddleEngineKind {
  kNative = 0,       // Use the native Fluid facility.
  kAutoMixedTensorRT // Automatically mixing TensorRT with the Fluid ops.
};
```

## 预测部署过程

总体上分为以下步骤

1. 用合适的配置创建 `PaddlePredictor`
2. 创建输入用的 `PaddleTensor`，传入到 `PaddlePredictor` 中
3. 获取输出的 `PaddleTensor` ，将结果取出

下面完整演示一个简单的模型，部分细节代码隐去

```c++
#include "paddle_inference_api.h"

// 创建一个 config，并修改相关设置
paddle::NativeConfig config;
config.model_dir = "xxx";
config.use_gpu = false;
// 创建一个原生的 PaddlePredictor
auto predictor =
      paddle::CreatePaddlePredictor<paddle::NativeConfig, paddle::PaddleEngineKind::kNative>(config);
// 创建输入 tensor
int64_t data[4] = {1, 2, 3, 4};
paddle::PaddleTensor tensor{.name = "",
                            .shape = std::vector<int>({4, 1}),
                            .data = paddle::PaddleBuf(data, sizeof(data)),
                            .dtype = paddle::PaddleDType::INT64};
// 创建输出 tensor，输出 tensor 的内存可以复用
std::vector<paddle::PaddleTensor> outputs;
// 执行预测
CHECK(predictor->Run(slots, &outputs));
// 获取 outputs ...
```

编译时，联编 `libpaddle_fluid.a/.so(Linux/Mac)` 或 `libpaddle_fluid.lib/paddle_fluid.dll(Windows)` 便可。

## 详细代码参考

- [paddle_inference_api.h](./paddle_inference_api.h)
- [some demos](./demo_ci)
