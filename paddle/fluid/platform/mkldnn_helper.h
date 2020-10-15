/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "mkldnn.hpp"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
#ifdef PADDLE_WITH_MKLDNN
using MKLDNNMemoryFormat = mkldnn::memory::format_tag;
#endif
namespace platform {

using MKLDNNStream = mkldnn::stream;
using MKLDNNEngine = mkldnn::engine;
using MKLDNNMemory = mkldnn::memory;
using MKLDNNMemoryDescriptor = mkldnn::memory::desc;
using MKLDNNPrimitive = mkldnn::primitive;
using MKLDNNPrimitiveDesc = mkldnn::handle<mkldnn_primitive_desc_t>;

typedef std::unique_ptr<MKLDNNStream> MKLDNNStreamPtr;
typedef std::unique_ptr<MKLDNNEngine> MKLDNNEnginePtr;
typedef std::unique_ptr<MKLDNNMemory> MKLDNNMemoryPtr;
typedef std::unique_ptr<MKLDNNPrimitive> MKLDNNPrimitivePtr;
typedef std::unique_ptr<MKLDNNPrimitiveDesc> MKLDNNPrimitiveDescPtr;

template <typename Type>
void* to_void_cast(const Type* t) {
  return static_cast<void*>(const_cast<Type*>(t));
}

template <typename Type>
void* to_void_reinterpret_cast(const Type* t) {
  return reinterpret_cast<void*>(const_cast<Type*>(t));
}

template <class Type>
using tf_desc = typename Type::desc;

template <class Type>
using tf_pd = typename Type::primitive_desc;

template <typename Type, typename Engine, typename... Args>
std::shared_ptr<tf_pd<Type>> MKLDNNFwdPrimitiveDesc(const Engine& e,
                                                    Args&&... args) {
  auto desc = tf_desc<Type>(mkldnn::prop_kind::forward, (args)...);
  auto pd = new tf_pd<Type>(desc, e);
  return std::shared_ptr<tf_pd<Type>>(pd);
}

template <typename Type, typename Engine, typename Primitive, typename... Args>
tf_pd<Type> MKLDNNBwdPrimitiveDesc(const Engine& e, const Primitive& p,
                                   Args&&... args) {
  auto desc = tf_desc<Type>(args...);
  return tf_pd<Type>(desc, e, p);
}

inline void MatchShapeToLayout(framework::Tensor* tensor_in,
                               framework::DataLayout from,
                               framework::DataLayout to) {
  // In these data layouts, channel dimension is either on 2nd position: nChw or
  // at last nhwC, so for dim==2 these layouts are the same and nothing should
  // be done. Similarly for dim==1 when you have just one possible combination.
  if (tensor_in->dims().size() < 3) {
    return;
  }

  auto print_dims = [](const std::vector<int>& dims) {
    std::ostringstream oss;

    if (!dims.empty()) {
      oss << "[";
      // Convert all but the last element to avoid a trailing ","
      std::copy(dims.begin(), dims.end() - 1,
                std::ostream_iterator<int>(oss, ","));

      // Now add the last element with no delimiter
      oss << dims.back() << "]";
    }

    return oss.str();
  };

  switch (from) {
    case framework::DataLayout::kMKLDNN:
      if (to == framework::DataLayout::kNHWC) {
        auto dims = framework::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
        tensor_in->Resize(framework::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: kMKLDNN to: kNHWC output_shape"
                << print_dims(dims);
      }
      break;
    case framework::DataLayout::kNHWC:
      if (to == framework::DataLayout::kMKLDNN) {
        auto dims = framework::vectorize<int>(tensor_in->dims());
        std::rotate(dims.begin() + 1, dims.end() - 1, dims.end());
        tensor_in->Resize(framework::make_ddim(dims));
        VLOG(3) << "Rotating Shape from: kNHWC to: kMKLDNN output_shape"
                << print_dims(dims);
      }
      break;
    default:
      break;
  }
}

struct mkldnn_dummy_primitive {
  struct primitive_desc {};
  struct desc {};
};

inline mkldnn::memory::desc MKLDNNMemDesc(const std::vector<int64_t>& dims,
                                          mkldnn::memory::data_type data_type,
                                          MKLDNNMemoryFormat format) {
  return mkldnn::memory::desc({dims}, data_type, format);
}

inline bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_mkldnn = ctx.Attr<bool>("use_mkldnn");
  return use_mkldnn && platform::is_cpu_place(ctx.GetPlace());
}

inline void ClearMKLDNNCache(const platform::Place& place) {
  // Clear mkl-dnn cache,
  if (platform::is_cpu_place(place)) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    platform::MKLDNNDeviceContext* dev_ctx =
        (platform::MKLDNNDeviceContext*)pool.Get(place);
    dev_ctx->ResetBlobMap();
    platform::MKLDNNDeviceContext::tls().set_cur_paddle_data_layout(
        paddle::framework::DataLayout::kNCHW);
  }
}

inline void DontClearMKLDNNCache(const platform::Place& place) {
  // Clear mkl-dnn cache,
  if (platform::is_cpu_place(place)) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    platform::MKLDNNDeviceContext* dev_ctx =
        (platform::MKLDNNDeviceContext*)pool.Get(place);
    dev_ctx->BlockNextCacheClearing();
  }
}

template <typename Type>
mkldnn::memory::data_type MKLDNNGetDataType() {
  return mkldnn::memory::data_type::undef;
}

template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<float>() {
  return mkldnn::memory::data_type::f32;
}
template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<int32_t>() {
  return mkldnn::memory::data_type::s32;
}
template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<int8_t>() {
  return mkldnn::memory::data_type::s8;
}
template <>
inline mkldnn::memory::data_type MKLDNNGetDataType<uint8_t>() {
  return mkldnn::memory::data_type::u8;
}

template <>
inline mkldnn::memory::data_type
MKLDNNGetDataType<paddle::platform::bfloat16>() {
  return mkldnn::memory::data_type::bf16;
}

inline void Reorder(mkldnn::memory src, mkldnn::memory dst,
                    const mkldnn::engine& engine) {
  auto reorder_prim = mkldnn::reorder(src, dst);
  mkldnn::stream astream(engine);
  reorder_prim.execute(astream, src, dst);
  astream.wait();
}

inline mkldnn::memory::format_tag GetMKLDNNFormat(
    mkldnn::memory::desc mem_desc) {
  auto ndims = mem_desc.data.ndims;
  auto strides = mem_desc.data.format_desc.blocking.strides;
  auto inner_nblks = mem_desc.data.format_desc.blocking.inner_nblks;
  auto inner_blks = mem_desc.data.format_desc.blocking.inner_blks;
  auto inner_idxs = mem_desc.data.format_desc.blocking.inner_idxs;

  if (ndims == 1) {
    return mkldnn::memory::format_tag::x;
  } else if (ndims == 2) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1]) {
        return mkldnn::memory::format_tag::nc;
      } else {
        return mkldnn::memory::format_tag::cn;
      }
    }
  } else if (ndims == 3) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2]) {
        return mkldnn::memory::format_tag::ncw;
      } else if (strides[1] >= strides[0] && strides[0] >= strides[2]) {
        return mkldnn::memory::format_tag::ntc;
      } else {
        return mkldnn::memory::format_tag::nwc;
      }
    }
  } else if (ndims == 4) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3]) {
        return mkldnn::memory::format_tag::nchw;
      } else if (strides[2] >= strides[3] && strides[3] >= strides[1] &&
                 strides[1] >= strides[0]) {
        return mkldnn::memory::format_tag::cdba;
      } else {
        return mkldnn::memory::format_tag::nhwc;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw16c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw8c;
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdb8a;
        }
      } else if (inner_blks[0] == 4 && inner_idxs[0] == 1) {
        return mkldnn::memory::format_tag::nChw4c;
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdb16a;
        }
      }
    } else if (inner_nblks == 2) {
      if (inner_blks[0] == 16 && inner_blks[1] == 16) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return mkldnn::memory::format_tag::OIhw16i16o;
        }
      } else if (inner_blks[0] == 8 && inner_blks[1] == 8) {
        if (inner_idxs[0] == 1 && inner_idxs[1] == 0) {
          return mkldnn::memory::format_tag::OIhw8i8o;
        }
      }
    }
  } else if (ndims == 5) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4]) {
        return mkldnn::memory::format_tag::ncdhw;
      } else {
        return mkldnn::memory::format_tag::ndhwc;
      }
    } else if (inner_nblks == 1) {
      if (inner_blks[0] == 8 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdeb8a;
        }
      } else if (inner_blks[0] == 8 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return mkldnn::memory::format_tag::aBcde8b;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 0) {
        if (strides[0] >= strides[2] && strides[2] >= strides[3] &&
            strides[3] >= strides[4] && strides[4] >= strides[1]) {
          return mkldnn::memory::format_tag::Acdeb16a;
        }
      } else if (inner_blks[0] == 16 && inner_idxs[0] == 1) {
        if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
            strides[2] >= strides[3] && strides[3] >= strides[4]) {
          return mkldnn::memory::format_tag::aBcde16b;
        }
      }
    }
  } else if (ndims == 6) {
    if (inner_nblks == 0) {
      if (strides[0] >= strides[1] && strides[1] >= strides[2] &&
          strides[2] >= strides[3] && strides[3] >= strides[4] &&
          strides[4] >= strides[5]) {
        return mkldnn::memory::format_tag::abcdef;
      }
    }
  }
  // DEBUG CODE - KEEP UNTILL TENSOR.MEMORY_DESC IMPLEMENTED
  // std::cout<<"@@@@@@@@@@ UNDEFINED FORMAT @@@@@@@@@@@@@@@@@@@"<<std::endl;
  // std::cout<<"NDIMS: "<<ndims<<std::endl;
  // std::cout<<"INNER_NBLKS: "<<inner_nblks<<std::endl;
  // for (int i=0;i<ndims;++i) {
  //   std::cout<<"STRIDE["<<i<<"]: "<<strides[i]<<std::endl;
  // }
  // for (int i=0;i<inner_nblks;++i) {
  //   std::cout<<"INNER_BLKS["<<i<<"]: "<<inner_blks[i]<<std::endl;
  // }
  // for (int i=0;i<inner_nblks;++i) {
  //   std::cout<<"INNER_IDXS["<<i<<"]: "<<inner_idxs[i]<<std::endl;
  // }
  return mkldnn::memory::format_tag::undef;
}

inline mkldnn::memory::format_tag GetMKLDNNFormat(const mkldnn::memory memory) {
  auto mem_desc = memory.get_desc();
  return GetMKLDNNFormat(mem_desc);
}

inline MKLDNNMemoryFormat MKLDNNFormatForSize(size_t dims_size,
                                              MKLDNNMemoryFormat data_format) {
  if (dims_size == 1) {
    return MKLDNNMemoryFormat::x;
  } else if (dims_size == 2) {
    return MKLDNNMemoryFormat::nc;
  } else if (dims_size == 3) {
    if (data_format == MKLDNNMemoryFormat::nchw) {
      return MKLDNNMemoryFormat::ncw;
    } else if (data_format == MKLDNNMemoryFormat::nhwc) {
      return MKLDNNMemoryFormat::nwc;
    }
  } else if (dims_size == 4) {
    if (data_format == MKLDNNMemoryFormat::goihw) {
      return MKLDNNMemoryFormat::oihw;
    }
  } else if (dims_size == 5) {
    if (data_format == MKLDNNMemoryFormat::goidhw) {
      return MKLDNNMemoryFormat::oidhw;
    }
    if (data_format == MKLDNNMemoryFormat::nchw) {
      return MKLDNNMemoryFormat::ncdhw;
    } else if (data_format == MKLDNNMemoryFormat::nhwc) {
      return MKLDNNMemoryFormat::ndhwc;
    }
  }
  return data_format;
}

inline MKLDNNMemoryFormat data_format_to_memory_format(
    const std::string& data_format) {
  switch (framework::StringToDataLayout(data_format)) {
    case framework::DataLayout::kNHWC:
      return MKLDNNMemoryFormat::nhwc;
    case framework::DataLayout::kNCHW:
      return MKLDNNMemoryFormat::nchw;
    default:
      return MKLDNNMemoryFormat::any;
  }
}

inline MKLDNNMemoryFormat StringToMKLDNNFormat(std::string* format) {
  std::transform(format->begin(), format->end(), format->begin(), ::tolower);

  if (!format->compare("nchw")) {
    return MKLDNNMemoryFormat::nchw;
  } else if (!format->compare("nchw16c")) {
    return MKLDNNMemoryFormat::nChw16c;
  } else if (!format->compare("nchw8c")) {
    return MKLDNNMemoryFormat::nChw8c;
  } else if (!format->compare("nhwc")) {
    return MKLDNNMemoryFormat::nhwc;
  } else {
    return MKLDNNMemoryFormat::any;
  }
}

inline std::string ThreadIDasStr(void) {
  return std::to_string(
      std::hash<std::thread::id>()(std::this_thread::get_id()));
}

template <typename T>
inline void AppendKey(std::string* key, const T& num) {
  key->append(std::to_string(num));
}

template <>
inline void AppendKey(std::string* key,
                      const mkldnn::memory::format_tag& format) {
  key->append(std::to_string(static_cast<int>(format)));
}

template <>
inline void AppendKey(std::string* key,
                      const mkldnn::memory::data_type& data_type) {
  key->append(std::to_string(static_cast<int>(data_type)));
}

template <>
inline void AppendKey(std::string* key, const mkldnn::algorithm& algorithm) {
  key->append(std::to_string(static_cast<int>(algorithm)));
}

template <>
inline void AppendKey(std::string* key,
                      const mkldnn::normalization_flags& flags) {
  key->append(std::to_string(static_cast<int>(flags)));
}

inline void AppendKey(std::string* key, const std::string& str) {
  key->append(str);
}

inline void AppendKey(std::string* key, const char* str) { key->append(str); }

template <typename T>
inline void AppendKey(std::string* key, const std::vector<T>& dims) {
  for (size_t i = 0; i < dims.size(); i++) {
    AppendKey(key, std::to_string(dims[i]));
  }
}

template <typename... ArgTypes>
inline std::string CreateKey(ArgTypes&&... args) {
  std::string key;
  key.reserve(64);
  using expand_type = int[];
  expand_type{0, (AppendKey(&key, std::forward<ArgTypes>(args)), 0)...};
  return key;
}

inline std::vector<std::vector<int64_t>> ToMkldnnPadding(
    const std::vector<int64_t>& paddings) {
  if (paddings.size() == 6) {
    int padding_front = paddings[0];
    int padding_back = paddings[1];
    int padding_top = paddings[2];
    int padding_bottom = paddings[3];
    int padding_left = paddings[4];
    int padding_right = paddings[5];

    return {{padding_front, padding_top, padding_left},
            {padding_back, padding_bottom, padding_right}};
  } else {
    int padding_top = paddings[0];
    int padding_bottom = paddings[1];
    int padding_left = paddings[2];
    int padding_right = paddings[3];

    return {{padding_top, padding_left}, {padding_bottom, padding_right}};
  }
}

inline bool HasOpINT8DataType(const paddle::framework::OpDesc* op) {
  return (op->GetAttrIfExists<std::string>("mkldnn_data_type") == "int8" ||
          op->GetAttrIfExists<bool>("use_quantizer"));
}

inline bool HasOpBFLOAT16DataType(const paddle::framework::OpDesc* op) {
  return op->GetAttrIfExists<std::string>("mkldnn_data_type") == "bfloat16";
}

inline bool HasOpFLOAT32DataType(const paddle::framework::OpDesc* op) {
  return op->GetAttrIfExists<std::string>("mkldnn_data_type") == "float32";
}
enum class RNNReorderType { PP_NTC, PP_TNC, NTC_PP, TNC_PP };

}  // namespace platform
}  // namespace paddle
