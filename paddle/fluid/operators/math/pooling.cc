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
#include "paddle/fluid/operators/math/pooling.h"
#include <algorithm>

namespace paddle {
namespace operators {
namespace math {

/*
* Tensors are in NCHW or NHWC format.
* Ksize, strides are two elements. These two elements represent height
* and width, respectively.
* Paddings are four elements. These four elements represent height_up,
* height_down, width_left and width_right, respectively.
*/
template <typename PoolProcess, typename T>
class Pool2dFunctor<platform::CPUDeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, PoolProcess pool_process,
                  bool exclusive, bool adaptive, framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int hstart, hend;
    int wstart, wend;
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          if (adaptive) {
            hstart = AdaptStartIndex(ph, input_height, output_height);
            hend = AdaptEndIndex(ph, input_height, output_height);
          }
          for (int pw = 0; pw < output_width; ++pw) {
            int pool_size = 1;
            if (adaptive) {
              wstart = AdaptStartIndex(pw, input_width, output_width);
              wend = AdaptEndIndex(pw, input_width, output_width);
            } else {
              hstart = ph * stride_height - padding_height;
              wstart = pw * stride_width - padding_width;
              hend = std::min(hstart + ksize_height,
                              input_height + padding_height);
              wend =
                  std::min(wstart + ksize_width, input_width + padding_width);
              pool_size = (hend - hstart) * (wend - wstart);

              wstart = std::max(wstart, 0);
              hstart = std::max(hstart, 0);
              hend = std::min(hend, input_height);
              wend = std::min(wend, input_width);
            }

            T ele = pool_process.initial();
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                pool_process.compute(input_data[h * input_width + w], &ele);
              }
            }
            if (exclusive || adaptive) {
              pool_size = (hend - hstart) * (wend - wstart);
            }

            pool_process.finalize(static_cast<T>(pool_size), &ele);
            output_data[ph * output_width + pw] = ele;
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }

  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, PoolProcess pool_process,
                  bool exclusive, bool adaptive, framework::Tensor* output) {
    bool channel_last = (data_format == "NHWC");

    const int batch_size = input.dims()[0];
    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output->dims()[3] : output->dims()[1];
    const int output_height =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int output_width =
        channel_last ? output->dims()[2] : output->dims()[3];

    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];

    const int stride_height = strides[0];
    const int stride_width = strides[1];

    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int hstart, hend;
    int wstart, wend;
    if (!channel_last) {
      const int input_stride = input_height * input_width;
      const int output_stride = output_height * output_width;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int pw = 0; pw < output_width; ++pw) {
              int pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, 0);
                hstart = std::max(hstart, 0);
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }

              T ele = pool_process.initial();
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  pool_process.compute(input_data[h * input_width + w], &ele);
                }
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              pool_process.finalize(static_cast<T>(pool_size), &ele);
              output_data[ph * output_width + pw] = ele;
            }
          }
          input_data += input_stride;
          output_data += output_stride;
        }
      }
    } else {
      const int input_stride = input_height * input_width * input_channels;
      const int output_stride = output_height * output_width * output_channels;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int pw = 0; pw < output_width; ++pw) {
              int pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, 0);
                hstart = std::max(hstart, 0);
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }
              T ele = pool_process.initial();
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  pool_process.compute(
                      input_data[h * input_width * input_channels +
                                 w * input_channels + c],
                      &ele);
                }
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              pool_process.finalize(static_cast<T>(pool_size), &ele);
              output_data[ph * output_width * output_channels +
                          pw * output_channels + c] = ele;
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
};

/*
* tensors are in NCHW or NHWC format.
* Ksize, strides are two elements. These two elements represent height
* and width, respectively.
* Paddings are four elements. These four elements represent height_up,
* height_down, width_left and width_right, respectively.
*/
template <typename PoolProcess, class T>
class Pool2dGradFunctor<platform::CPUDeviceContext, PoolProcess, T> {
 public:
  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, PoolProcess pool_grad_process,
      bool exclusive, bool adaptive, framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int hstart, hend;
    int wstart, wend;
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          if (adaptive) {
            hstart = AdaptStartIndex(ph, input_height, output_height);
            hend = AdaptEndIndex(ph, input_height, output_height);
          }
          for (int pw = 0; pw < output_width; ++pw) {
            int pool_size = 1;
            if (adaptive) {
              wstart = AdaptStartIndex(pw, input_width, output_width);
              wend = AdaptEndIndex(pw, input_width, output_width);
            } else {
              hstart = ph * stride_height - padding_height;
              wstart = pw * stride_width - padding_width;
              hend = std::min(hstart + ksize_height,
                              input_height + padding_height);
              wend =
                  std::min(wstart + ksize_width, input_width + padding_width);
              pool_size = (hend - hstart) * (wend - wstart);

              wstart = std::max(wstart, 0);
              hstart = std::max(hstart, 0);
              hend = std::min(hend, input_height);
              wend = std::min(wend, input_width);
            }
            if (exclusive || adaptive) {
              pool_size = (hend - hstart) * (wend - wstart);
            }
            float scale = 1.0 / pool_size;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                pool_grad_process.compute(
                    input_data[h * input_width + w],
                    output_data[ph * output_width + pw],
                    output_grad_data[ph * output_width + pw],
                    static_cast<T>(scale),
                    input_grad_data + h * input_width + w);
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }

  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, const std::string data_format,
      PoolProcess pool_grad_process, bool exclusive, bool adaptive,
      framework::Tensor* input_grad) {
    bool channel_last = (data_format == "NHWC");

    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int output_width = channel_last ? output.dims()[2] : output.dims()[3];

    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];

    const int stride_height = strides[0];
    const int stride_width = strides[1];

    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int hstart, hend;
    int wstart, wend;
    if (!channel_last) {
      const int input_stride = input_height * input_width;
      const int output_stride = output_height * output_width;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int pw = 0; pw < output_width; ++pw) {
              int pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, 0);
                hstart = std::max(hstart, 0);
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              float scale = 1.0 / pool_size;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  pool_grad_process.compute(
                      input_data[h * input_width + w],
                      output_data[ph * output_width + pw],
                      output_grad_data[ph * output_width + pw],
                      static_cast<T>(scale),
                      input_grad_data + h * input_width + w);
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int input_stride = input_height * input_width * input_channels;
      const int output_stride = output_height * output_width * output_channels;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }
            for (int pw = 0; pw < output_width; ++pw) {
              int pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                hstart = ph * stride_height - padding_height;
                wstart = pw * stride_width - padding_width;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (hend - hstart) * (wend - wstart);

                wstart = std::max(wstart, 0);
                hstart = std::max(hstart, 0);
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }
              if (exclusive || adaptive) {
                pool_size = (hend - hstart) * (wend - wstart);
              }
              float scale = 1.0 / pool_size;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  auto input_idx =
                      h * input_width * input_channels + w * input_channels + c;
                  auto output_idx = ph * output_width * output_channels +
                                    pw * output_channels + c;
                  pool_grad_process.compute(
                      input_data[input_idx], output_data[output_idx],
                      output_grad_data[output_idx], static_cast<T>(scale),
                      input_grad_data + input_idx);
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

/*
* Tensors are in NCHW or NHWC format.
* Ksize, strides are two elements. These two elements represent height
* and width, respectively.
* Paddings are four elements. These four elements represent height_up,
* height_down, width_left and width_right, respectively.
*/
template <class T>
class MaxPool2dGradFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          int hstart = ph * stride_height - padding_height;
          int hend = std::min(hstart + ksize_height, input_height);
          hstart = std::max(hstart, 0);
          for (int pw = 0; pw < output_width; ++pw) {
            int wstart = pw * stride_width - padding_width;
            int wend = std::min(wstart + ksize_width, input_width);
            wstart = std::max(wstart, 0);

            bool stop = false;
            for (int h = hstart; h < hend && !stop; ++h) {
              for (int w = wstart; w < wend && !stop; ++w) {
                int input_idx = h * input_width + w;
                int output_idx = ph * output_width + pw;
                if (input_data[input_idx] == output_data[output_idx]) {
                  input_grad_data[input_idx] += output_grad_data[output_idx];
                  stop = true;
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }

  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, const std::string data_format,
      framework::Tensor* input_grad) {
    bool channel_last = (data_format == "NHWC");

    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int output_width = channel_last ? output.dims()[2] : output.dims()[3];

    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];

    const int stride_height = strides[0];
    const int stride_width = strides[1];

    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    if (!channel_last) {
      const int input_stride = input_height * input_width;
      const int output_stride = output_height * output_width;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int ph = 0; ph < output_height; ++ph) {
            int hstart = ph * stride_height - padding_height;
            int hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, 0);
            for (int pw = 0; pw < output_width; ++pw) {
              int wstart = pw * stride_width - padding_width;
              int wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, 0);

              bool stop = false;
              for (int h = hstart; h < hend && !stop; ++h) {
                for (int w = wstart; w < wend && !stop; ++w) {
                  int input_idx = h * input_width + w;
                  int output_idx = ph * output_width + pw;
                  if (input_data[input_idx] == output_data[output_idx]) {
                    input_grad_data[input_idx] += output_grad_data[output_idx];
                    stop = true;
                  }
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int input_stride = input_height * input_width * input_channels;
      const int output_stride = output_height * output_width * output_channels;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int ph = 0; ph < output_height; ++ph) {
            int hstart = ph * stride_height - padding_height;
            int hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, 0);
            for (int pw = 0; pw < output_width; ++pw) {
              int wstart = pw * stride_width - padding_width;
              int wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, 0);

              bool stop = false;
              for (int h = hstart; h < hend && !stop; ++h) {
                for (int w = wstart; w < wend && !stop; ++w) {
                  int input_idx =
                      h * input_width * input_channels + w * input_channels + c;
                  int output_idx = ph * output_width * output_channels +
                                   pw * output_channels + c;
                  if (input_data[input_idx] == output_data[output_idx]) {
                    input_grad_data[input_idx] += output_grad_data[output_idx];
                    stop = true;
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};
template class MaxPool2dGradFunctor<platform::CPUDeviceContext, float>;
template class MaxPool2dGradFunctor<platform::CPUDeviceContext, double>;

template class Pool2dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool2dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool2dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::MaxPoolGrad<float>,
                                 float>;
template class Pool2dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::AvgPoolGrad<float>,
                                 float>;
template class Pool2dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool2dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool2dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::MaxPoolGrad<double>,
                                 double>;
template class Pool2dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::AvgPoolGrad<double>,
                                 double>;

/*
* Tensors are in NCDHW or NDHWC format.
* Ksize, strides, paddings are three elements. These three elements represent
* depth, height and width, respectively.
* Paddings are six elements. These six elements represent depth_forth,
* depth_back,
* height_up, height_down, width_left and width_right, respectively.
*/
template <typename PoolProcess, class T>
class Pool3dFunctor<platform::CPUDeviceContext, PoolProcess, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, PoolProcess pool_process,
                  bool exclusive, bool adaptive, framework::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          if (adaptive) {
            dstart = AdaptStartIndex(pd, input_depth, output_depth);
            dend = AdaptEndIndex(pd, input_depth, output_depth);
          }

          for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }

            for (int pw = 0; pw < output_width; ++pw) {
              int pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                dstart = pd * stride_depth - padding_depth;
                dend =
                    std::min(dstart + ksize_depth, input_depth + padding_depth);
                hstart = ph * stride_height - padding_height;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wstart = pw * stride_width - padding_width;
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);
                pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                dstart = std::max(dstart, 0);
                hstart = std::max(hstart, 0);
                wstart = std::max(wstart, 0);
                dend = std::min(dend, input_depth);
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }
              int output_idx = (pd * output_height + ph) * output_width + pw;
              T ele = pool_process.initial();
              for (int d = dstart; d < dend; ++d) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    pool_process.compute(
                        input_data[(d * input_height + h) * input_width + w],
                        &ele);
                  }
                }
              }
              if (exclusive || adaptive) {
                pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
              }
              pool_process.finalize(static_cast<T>(pool_size), &ele);
              output_data[output_idx] = ele;
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, PoolProcess pool_process,
                  bool exclusive, bool adaptive, framework::Tensor* output) {
    bool channel_last = (data_format == "NDHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output->dims()[4] : output->dims()[1];
    const int output_depth =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int output_height =
        channel_last ? output->dims()[2] : output->dims()[3];
    const int output_width =
        channel_last ? output->dims()[3] : output->dims()[4];

    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];

    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];

    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    if (!channel_last) {
      const int input_stride = input_depth * input_height * input_width;
      const int output_stride = output_depth * output_height * output_width;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int pw = 0; pw < output_width; ++pw) {
                int pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, 0);
                  hstart = std::max(hstart, 0);
                  wstart = std::max(wstart, 0);
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                int output_idx = (pd * output_height + ph) * output_width + pw;
                T ele = pool_process.initial();
                for (int d = dstart; d < dend; ++d) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      pool_process.compute(
                          input_data[(d * input_height + h) * input_width + w],
                          &ele);
                    }
                  }
                }
                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                pool_process.finalize(static_cast<T>(pool_size), &ele);
                output_data[output_idx] = ele;
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
        }
      }
    } else {
      const int input_stride =
          input_depth * input_height * input_width * input_channels;
      const int output_stride =
          output_depth * output_height * output_width * output_channels;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int pw = 0; pw < output_width; ++pw) {
                int pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, 0);
                  hstart = std::max(hstart, 0);
                  wstart = std::max(wstart, 0);
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                T ele = pool_process.initial();
                for (int d = dstart; d < dend; ++d) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      int input_idx =
                          ((d * input_height + h) * input_width + w) *
                              input_channels +
                          c;
                      pool_process.compute(input_data[input_idx], &ele);
                    }
                  }
                }
                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                pool_process.finalize(static_cast<T>(pool_size), &ele);
                int output_idx =
                    ((pd * output_height + ph) * output_width + pw) *
                        output_channels +
                    c;
                output_data[output_idx] = ele;
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
};

/*
* Tensors are in NCDHW or NDHWC format.
* Ksize, strides, paddings are three elements. These three elements represent
* depth, height and width, respectively.
* Paddings are six elements. These six elements represent depth_forth,
* depth_back,
* height_up, height_down, width_left and width_right, respectively.
*/
template <typename PoolProcess, class T>
class Pool3dGradFunctor<platform::CPUDeviceContext, PoolProcess, T> {
 public:
  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, PoolProcess pool_grad_process,
      bool exclusive, bool adaptive, framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];
    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          if (adaptive) {
            dstart = AdaptStartIndex(pd, input_depth, output_depth);
            dend = AdaptEndIndex(pd, input_depth, output_depth);
          }

          for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            }

            for (int pw = 0; pw < output_width; ++pw) {
              int pool_size = 1;
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                dstart = pd * stride_depth - padding_depth;
                dend =
                    std::min(dstart + ksize_depth, input_depth + padding_depth);
                hstart = ph * stride_height - padding_height;
                hend = std::min(hstart + ksize_height,
                                input_height + padding_height);
                wstart = pw * stride_width - padding_width;
                wend =
                    std::min(wstart + ksize_width, input_width + padding_width);

                pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                dstart = std::max(dstart, 0);
                hstart = std::max(hstart, 0);
                wstart = std::max(wstart, 0);
                dend = std::min(dend, input_depth);
                hend = std::min(hend, input_height);
                wend = std::min(wend, input_width);
              }

              if (exclusive || adaptive) {
                pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
              }
              float scale = 1.0 / pool_size;
              for (int d = dstart; d < dend; ++d) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    int input_idx = (d * input_height + h) * input_width + w;
                    int output_idx =
                        (pd * output_height + ph) * output_width + pw;
                    pool_grad_process.compute(
                        input_data[input_idx], output_data[output_idx],
                        output_grad_data[output_idx], static_cast<T>(scale),
                        input_grad_data + input_idx);
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, const std::string data_format,
      PoolProcess pool_grad_process, bool exclusive, bool adaptive,
      framework::Tensor* input_grad) {
    bool channel_last = (data_format == "NDHWC");

    const int batch_size = input.dims()[0];
    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int output_depth = channel_last ? output.dims()[1] : output.dims()[2];
    const int output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int output_width = channel_last ? output.dims()[3] : output.dims()[4];

    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];

    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];

    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    if (!channel_last) {
      const int input_stride = input_depth * input_height * input_width;
      const int output_stride = output_depth * output_height * output_width;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int pw = 0; pw < output_width; ++pw) {
                int pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, 0);
                  hstart = std::max(hstart, 0);
                  wstart = std::max(wstart, 0);
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                float scale = 1.0 / pool_size;
                for (int d = dstart; d < dend; ++d) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      int input_idx = (d * input_height + h) * input_width + w;
                      int output_idx =
                          (pd * output_height + ph) * output_width + pw;
                      pool_grad_process.compute(
                          input_data[input_idx], output_data[output_idx],
                          output_grad_data[output_idx], static_cast<T>(scale),
                          input_grad_data + input_idx);
                    }
                  }
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int input_stride =
          input_depth * input_height * input_width * input_channels;
      const int output_stride =
          output_depth * output_height * output_width * output_channels;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
              dstart = AdaptStartIndex(pd, input_depth, output_depth);
              dend = AdaptEndIndex(pd, input_depth, output_depth);
            }

            for (int ph = 0; ph < output_height; ++ph) {
              if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
              }

              for (int pw = 0; pw < output_width; ++pw) {
                int pool_size = 1;
                if (adaptive) {
                  wstart = AdaptStartIndex(pw, input_width, output_width);
                  wend = AdaptEndIndex(pw, input_width, output_width);
                } else {
                  dstart = pd * stride_depth - padding_depth;
                  dend = std::min(dstart + ksize_depth,
                                  input_depth + padding_depth);
                  hstart = ph * stride_height - padding_height;
                  hend = std::min(hstart + ksize_height,
                                  input_height + padding_height);
                  wstart = pw * stride_width - padding_width;
                  wend = std::min(wstart + ksize_width,
                                  input_width + padding_width);

                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                  dstart = std::max(dstart, 0);
                  hstart = std::max(hstart, 0);
                  wstart = std::max(wstart, 0);
                  dend = std::min(dend, input_depth);
                  hend = std::min(hend, input_height);
                  wend = std::min(wend, input_width);
                }

                if (exclusive || adaptive) {
                  pool_size =
                      (dend - dstart) * (hend - hstart) * (wend - wstart);
                }
                float scale = 1.0 / pool_size;
                for (int d = dstart; d < dend; ++d) {
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      int input_idx =
                          ((d * input_height + h) * input_width + w) *
                              input_channels +
                          c;
                      int output_idx =
                          ((pd * output_height + ph) * output_width + pw) *
                              output_channels +
                          c;
                      pool_grad_process.compute(
                          input_data[input_idx], output_data[output_idx],
                          output_grad_data[output_idx], static_cast<T>(scale),
                          input_grad_data + input_idx);
                    }
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};

/*
* Tensors are in NCDHW or NDHWC format.
* Ksize, strides, paddings are three elements. These three elements represent
* depth, height and width, respectively.
* Paddings are six elements. These six elements represent depth_forth,
* depth_back,
* height_up, height_down, width_left and width_right, respectively.
*/
template <class T>
class MaxPool3dGradFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, framework::Tensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];
    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          int dstart = pd * stride_depth - padding_depth;
          int dend = std::min(dstart + ksize_depth, input_depth);
          dstart = std::max(dstart, 0);
          for (int ph = 0; ph < output_height; ++ph) {
            int hstart = ph * stride_height - padding_height;
            int hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, 0);
            for (int pw = 0; pw < output_width; ++pw) {
              int wstart = pw * stride_width - padding_width;
              int wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, 0);
              bool stop = false;
              for (int d = dstart; d < dend && !stop; ++d) {
                for (int h = hstart; h < hend && !stop; ++h) {
                  for (int w = wstart; w < wend && !stop; ++w) {
                    int input_idx = (d * input_height + h) * input_width + w;
                    int output_idx =
                        (pd * output_height + ph) * output_width + pw;

                    if (input_data[input_idx] == output_data[output_idx]) {
                      input_grad_data[input_idx] +=
                          output_grad_data[output_idx];
                      stop = true;
                    }
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
  void operator()(
      const platform::CPUDeviceContext& context, const framework::Tensor& input,
      const framework::Tensor& output, const framework::Tensor& output_grad,
      const std::vector<int>& ksize, const std::vector<int>& strides,
      const std::vector<int>& paddings, const std::string data_format,
      framework::Tensor* input_grad) {
    bool channel_last = (data_format == "NDHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int output_depth = channel_last ? output.dims()[1] : output.dims()[2];
    const int output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int output_width = channel_last ? output.dims()[3] : output.dims()[4];

    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];

    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];

    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(context.GetPlace());

    if (!channel_last) {
      const int input_stride = input_depth * input_height * input_width;
      const int output_stride = output_depth * output_height * output_width;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int pd = 0; pd < output_depth; ++pd) {
            int dstart = pd * stride_depth - padding_depth;
            int dend = std::min(dstart + ksize_depth, input_depth);
            dstart = std::max(dstart, 0);
            for (int ph = 0; ph < output_height; ++ph) {
              int hstart = ph * stride_height - padding_height;
              int hend = std::min(hstart + ksize_height, input_height);
              hstart = std::max(hstart, 0);
              for (int pw = 0; pw < output_width; ++pw) {
                int wstart = pw * stride_width - padding_width;
                int wend = std::min(wstart + ksize_width, input_width);
                wstart = std::max(wstart, 0);
                bool stop = false;
                for (int d = dstart; d < dend && !stop; ++d) {
                  for (int h = hstart; h < hend && !stop; ++h) {
                    for (int w = wstart; w < wend && !stop; ++w) {
                      int input_idx = (d * input_height + h) * input_width + w;
                      int output_idx =
                          (pd * output_height + ph) * output_width + pw;

                      if (input_data[input_idx] == output_data[output_idx]) {
                        input_grad_data[input_idx] +=
                            output_grad_data[output_idx];
                        stop = true;
                      }
                    }
                  }
                }
              }
            }
          }
          input_data += input_stride;
          output_data += output_stride;
          input_grad_data += input_stride;
          output_grad_data += output_stride;
        }
      }
    } else {
      const int input_stride =
          input_depth * input_height * input_width * input_channels;
      const int output_stride =
          output_depth * output_height * output_width * output_channels;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
          for (int pd = 0; pd < output_depth; ++pd) {
            int dstart = pd * stride_depth - padding_depth;
            int dend = std::min(dstart + ksize_depth, input_depth);
            dstart = std::max(dstart, 0);
            for (int ph = 0; ph < output_height; ++ph) {
              int hstart = ph * stride_height - padding_height;
              int hend = std::min(hstart + ksize_height, input_height);
              hstart = std::max(hstart, 0);
              for (int pw = 0; pw < output_width; ++pw) {
                int wstart = pw * stride_width - padding_width;
                int wend = std::min(wstart + ksize_width, input_width);
                wstart = std::max(wstart, 0);
                bool stop = false;

                for (int d = dstart; d < dend && !stop; ++d) {
                  for (int h = hstart; h < hend && !stop; ++h) {
                    for (int w = wstart; w < wend && !stop; ++w) {
                      int input_idx =
                          ((d * input_height + h) * input_width + w) *
                              input_channels +
                          c;
                      int output_idx =
                          ((pd * output_height + ph) * output_width + pw) *
                              output_channels +
                          c;

                      if (input_data[input_idx] == output_data[output_idx]) {
                        input_grad_data[input_idx] +=
                            output_grad_data[output_idx];
                        stop = true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        input_data += input_stride;
        output_data += output_stride;
        input_grad_data += input_stride;
        output_grad_data += output_stride;
      }
    }
  }
};
template class MaxPool3dGradFunctor<platform::CPUDeviceContext, float>;
template class MaxPool3dGradFunctor<platform::CPUDeviceContext, double>;

template class Pool3dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::MaxPool<float>, float>;
template class Pool3dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::AvgPool<float>, float>;
template class Pool3dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::MaxPoolGrad<float>,
                                 float>;
template class Pool3dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::AvgPoolGrad<float>,
                                 float>;
template class Pool3dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::MaxPool<double>, double>;
template class Pool3dFunctor<platform::CPUDeviceContext,
                             paddle::operators::math::AvgPool<double>, double>;
template class Pool3dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::MaxPoolGrad<double>,
                                 double>;
template class Pool3dGradFunctor<platform::CPUDeviceContext,
                                 paddle::operators::math::AvgPoolGrad<double>,
                                 double>;

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool2dWithIndexFunctor<platform::CPUDeviceContext, T1, T2> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* output, framework::Tensor* mask) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];
    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T1* input_data = input.data<T1>();
    T1* output_data = output->mutable_data<T1>(context.GetPlace());
    T2* mask_data = mask->mutable_data<T2>(context.GetPlace());

    int hstart, hend;
    int wstart, wend;
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          if (adaptive) {
            hstart = AdaptStartIndex(ph, input_height, output_height);
            hend = AdaptEndIndex(ph, input_height, output_height);
          } else {
            hstart = ph * stride_height - padding_height;
            hend = std::min(hstart + ksize_height, input_height);
            hstart = std::max(hstart, 0);
          }
          for (int pw = 0; pw < output_width; ++pw) {
            if (adaptive) {
              wstart = AdaptStartIndex(pw, input_width, output_width);
              wend = AdaptEndIndex(pw, input_width, output_width);
            } else {
              wstart = pw * stride_width - padding_width;
              wend = std::min(wstart + ksize_width, input_width);
              wstart = std::max(wstart, 0);
            }

            T1 ele = static_cast<T1>(-FLT_MAX);
            int index = -1;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                if (ele < input_data[h * input_width + w]) {
                  ele = input_data[h * input_width + w];
                  index = h * input_width + w;
                }
              }
            }
            output_data[ph * output_width + pw] = ele;
            mask_data[ph * output_width + pw] = index;
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool2dWithIndexGradFunctor<platform::CPUDeviceContext, T1, T2> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* input_grad) {
    const int batch_size = input_grad->dims()[0];
    const int input_height = input_grad->dims()[2];
    const int input_width = input_grad->dims()[3];
    const int output_channels = output_grad.dims()[1];
    const int output_height = output_grad.dims()[2];
    const int output_width = output_grad.dims()[3];
    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = input_grad->mutable_data<T1>(context.GetPlace());

    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          for (int pw = 0; pw < output_width; ++pw) {
            const int output_idx = ph * output_width + pw;
            const int input_idx = static_cast<int>(mask_data[output_idx]);
            input_grad_data[input_idx] += output_grad_data[output_idx];
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class MaxPool2dWithIndexFunctor<platform::CPUDeviceContext, float,
                                         int>;
template class MaxPool2dWithIndexGradFunctor<platform::CPUDeviceContext, float,
                                             int>;
template class MaxPool2dWithIndexFunctor<platform::CPUDeviceContext, double,
                                         int>;
template class MaxPool2dWithIndexGradFunctor<platform::CPUDeviceContext, double,
                                             int>;

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool3dWithIndexFunctor<platform::CPUDeviceContext, T1, T2> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* output, framework::Tensor* mask) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];
    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T1* input_data = input.data<T1>();
    T1* output_data = output->mutable_data<T1>(context.GetPlace());
    T2* mask_data = mask->mutable_data<T2>(context.GetPlace());

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          if (adaptive) {
            dstart = AdaptStartIndex(pd, input_depth, output_depth);
            dend = AdaptEndIndex(pd, input_depth, output_depth);
          } else {
            dstart = pd * stride_depth - padding_depth;
            dend = std::min(dstart + ksize_depth, input_depth);
            dstart = std::max(dstart, 0);
          }
          for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
              hstart = AdaptStartIndex(ph, input_height, output_height);
              hend = AdaptEndIndex(ph, input_height, output_height);
            } else {
              hstart = ph * stride_height - padding_height;
              hend = std::min(hstart + ksize_height, input_height);
              hstart = std::max(hstart, 0);
            }
            for (int pw = 0; pw < output_width; ++pw) {
              if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
              } else {
                wstart = pw * stride_width - padding_width;
                wend = std::min(wstart + ksize_width, input_width);
                wstart = std::max(wstart, 0);
              }

              int output_idx = (pd * output_height + ph) * output_width + pw;
              T1 ele = static_cast<T1>(-FLT_MAX);
              int index = -1;
              for (int d = dstart; d < dend; ++d) {
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    int input_idx = (d * input_height + h) * input_width + w;
                    if (ele < input_data[input_idx]) {
                      index = input_idx;
                      ele = input_data[input_idx];
                    }
                  }
                }
              }
              output_data[output_idx] = ele;
              mask_data[output_idx] = index;
            }
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
template <typename T1, typename T2>
class MaxPool3dWithIndexGradFunctor<platform::CPUDeviceContext, T1, T2> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* input_grad) {
    const int batch_size = input_grad->dims()[0];
    const int input_depth = input_grad->dims()[2];
    const int input_height = input_grad->dims()[3];
    const int input_width = input_grad->dims()[4];
    const int output_channels = output_grad.dims()[1];
    const int output_depth = output_grad.dims()[2];
    const int output_height = output_grad.dims()[3];
    const int output_width = output_grad.dims()[4];
    const int input_stride = input_depth * input_height * input_width;
    const int output_stride = output_depth * output_height * output_width;

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = input_grad->mutable_data<T1>(context.GetPlace());

    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < output_channels; ++c) {
        for (int pd = 0; pd < output_depth; ++pd) {
          for (int ph = 0; ph < output_height; ++ph) {
            for (int pw = 0; pw < output_width; ++pw) {
              const int output_idx =
                  (pd * output_height + ph) * output_width + pw;
              const int input_idx = static_cast<int>(mask_data[output_idx]);
              input_grad_data[input_idx] += output_grad_data[output_idx];
            }
          }
        }
        // offset
        input_grad_data += input_stride;
        output_grad_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
};

template class MaxPool3dWithIndexFunctor<platform::CPUDeviceContext, float,
                                         int>;
template class MaxPool3dWithIndexGradFunctor<platform::CPUDeviceContext, float,
                                             int>;
template class MaxPool3dWithIndexFunctor<platform::CPUDeviceContext, double,
                                         int>;
template class MaxPool3dWithIndexGradFunctor<platform::CPUDeviceContext, double,
                                             int>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
