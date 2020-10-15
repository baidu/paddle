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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__)
template <typename T>
void prepare_csr_data(const std::vector<uint64_t> &offset,
                      const int64_t *ids_data, const size_t idx_width,
                      T *csr_vals, int *csr_colmuns, int *csr_row_idx,
                      int64_t padding_idx = kNoPadding) {
  int val_idx = 0;
  int row_idx = 0;
  csr_row_idx[0] = 0;

  std::map<int, int> ids_map;

  // for each sequence in batch
  for (size_t i = 0; i < offset.size() - 1; ++i) {
    for (size_t idx = 0; idx < idx_width; ++idx) {
      ids_map.clear();

      // construct a map for creating csr
      for (size_t j = offset[i]; j < offset[i + 1]; ++j) {
        auto ids_value = ids_data[idx + j * idx_width];
        if (ids_value != padding_idx) {
          unsigned int word_idx = static_cast<unsigned int>(ids_value);
          ++ids_map[word_idx];
        }
      }

      VLOG(4) << "====sequence %d====" << i;
      for (std::map<int, int>::const_iterator it = ids_map.begin();
           it != ids_map.end(); ++it) {
        VLOG(4) << it->first << " => " << it->second;
        csr_vals[val_idx] = it->second;
        csr_colmuns[val_idx] = it->first;
        ++val_idx;
      }
      csr_row_idx[row_idx + 1] = csr_row_idx[row_idx] + ids_map.size();
      ++row_idx;
    }
  }
}
#else
template <typename T>
struct EmbeddingVSumFunctor {
  void operator()(const framework::ExecutionContext &context,
                  const LoDTensor *table_t, const LoDTensor *ids_t,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t table_height = table_t->dims()[0];
    int64_t table_width = table_t->dims()[1];
    int64_t out_width = output_t->dims()[1];
    const int64_t *ids = ids_t->data<int64_t>();
    auto ids_lod = ids_t->lod()[0];
    int64_t idx_width = ids_t->numel() / ids_lod.back();
    auto *output = output_t->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_LE(table_width * idx_width, out_width,
                      platform::errors::InvalidArgument(
                          "table_width * idx_width should be less than or "
                          "equal to out_width. But received "
                          "table_width * idx_width = %s, out_width = %d.",
                          table_width * idx_width, out_width));
    PADDLE_ENFORCE_GT(ids_lod.size(), 1UL,
                      platform::errors::InvalidArgument(
                          "The tensor ids's LoD[0] should be greater than 1. "
                          "But received the ids's LoD[0] = %d.",
                          ids_lod.size()));

    jit::emb_seq_pool_attr_t attr(table_height, table_width, 0, idx_width,
                                  out_width, jit::SeqPoolType::kSum);
    for (size_t i = 0; i != ids_lod.size() - 1; ++i) {
      attr.index_height = ids_lod[i + 1] - ids_lod[i];
      auto emb_seqpool =
          jit::KernelFuncs<jit::EmbSeqPoolTuple<T>, platform::CPUPlace>::Cache()
              .At(attr);
      emb_seqpool(table, ids + ids_lod[i] * idx_width, output + i * out_width,
                  &attr);
    }
  }
};
#endif

inline int FusedEmbeddingSeqPoolLastDim(const framework::DDim &table_dims,
                                        const framework::DDim &ids_dims) {
  int64_t last_dim = table_dims[1];
  for (int i = 1; i != ids_dims.size(); ++i) {
    last_dim *= ids_dims[i];
  }
  return last_dim;
}

template <typename T>
class FusedEmbeddingSeqPoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const LoDTensor *ids_t = context.Input<LoDTensor>("Ids");  // int tensor
    LoDTensor *output_t = context.Output<LoDTensor>("Out");    // float tensor
    const LoDTensor *table_var = context.Input<LoDTensor>("W");
    const std::string &combiner_type = context.Attr<std::string>("combiner");

    int64_t last_dim =
        FusedEmbeddingSeqPoolLastDim(table_var->dims(), ids_t->dims());
    const auto &ids_lod = ids_t->lod();
    // in run time, the LoD of ids must be 1
    PADDLE_ENFORCE_EQ(ids_lod.size(), 1UL,
                      platform::errors::InvalidArgument(
                          "The LoD level of Input(Ids) should be 1. But "
                          "received Ids's LoD level = %d.",
                          ids_lod.size()));
    int64_t batch_size = ids_lod[0].size() - 1;
    // in run time, the shape from Ids -> output
    // should be [seq_length, 1] -> [batch_size, last_dim]
    output_t->Resize({batch_size, last_dim});

    if (combiner_type == "sum") {
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__)
      int64_t padding_idx = context.Attr<int64_t>("padding_idx");
      auto output = output_t->mutable_data<T>(context.GetPlace());
      int64_t table_height = table_var->dims()[0];
      int64_t table_width = table_var->dims()[1];
      auto weights = table_var->data<T>();

      const std::vector<uint64_t> offset = ids_lod[0];
      auto len = ids_t->numel();
      int idx_width = len / offset.back();

      Tensor csr_vals_t, csr_colmuns_t, csr_row_idx_t;
      csr_vals_t.Resize({len});
      csr_colmuns_t.Resize({len});
      csr_row_idx_t.Resize({(batch_size + 1) * idx_width});
      auto csr_vals = csr_vals_t.mutable_data<T>(context.GetPlace());
      auto csr_colmuns = csr_colmuns_t.mutable_data<int>(context.GetPlace());
      auto csr_row_idx = csr_row_idx_t.mutable_data<int>(context.GetPlace());
      prepare_csr_data<T>(offset, ids_t->data<int64_t>(), idx_width, csr_vals,
                          csr_colmuns, csr_row_idx, padding_idx);

      const char transa = 'N';
      const T alpha = 1.0;
      const T beta = 0.0;
      const char matdescra[] = {'G', 'L', 'N', 'C'};

      const int m = batch_size * idx_width;
      const int n = table_width;
      const int k = table_height;
      auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
      blas.CSRMM(&transa, &m, &n, &k, &alpha, matdescra, (const T *)csr_vals,
                 (const int *)csr_colmuns, (const int *)csr_row_idx,
                 (const int *)csr_row_idx + 1, weights, &n, &beta, output, &n);

#else
      EmbeddingVSumFunctor<T> functor;
      functor(context, table_var, ids_t, output_t);
#endif
    }
  }
};

template <typename T>
class FusedEmbeddingSeqPoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    DDim table_dim;
    if (table_var->IsType<LoDTensor>()) {
      table_dim = context.Input<LoDTensor>("W")->dims();
    } else if (table_var->IsType<SelectedRows>()) {
      auto *table_t = context.Input<SelectedRows>("W");
      table_dim = table_t->value().dims();
    } else {
      PADDLE_THROW(platform::errors::PermissionDenied(
          "The parameter W of a LookupTable "
          "must be either LoDTensor or SelectedRows."));
    }

    bool is_sparse = context.Attr<bool>("is_sparse");
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<SelectedRows>(framework::GradVarName("W"));
      // runtime shape
      d_table->set_height(table_dim[0]);

      auto *ids_data = ids->data<int64_t>();
      int64_t ids_num = ids->numel();
      auto lod = ids->lod()[0];
      int64_t out_width = d_output->dims()[1];

      framework::Vector<int64_t> *new_rows = d_table->mutable_rows();
      new_rows->resize(ids_num);
      std::memcpy(&(*new_rows)[0], ids_data, ids_num * sizeof(int64_t));

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({ids_num, table_dim[1]});
      T *d_table_data = d_table_value->mutable_data<T>(context.GetPlace());
      const T *d_output_data = d_output->data<T>();

      auto vbroadcast =
          jit::KernelFuncs<jit::VBroadcastTuple<T>, platform::CPUPlace>::Cache()
              .At(out_width);
      for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
        int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
        const T *src = d_output_data + i * out_width;
        T *dst = d_table_data + lod[i] * out_width;
        vbroadcast(src, dst, h, out_width);
      }
    } else {
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__)
      auto *ids = context.Input<LoDTensor>("Ids");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<LoDTensor>(framework::GradVarName("W"));
      int64_t padding_idx = context.Attr<int64_t>("padding_idx");

      d_table->Resize(table_dim);
      auto *d_table_data = d_table->mutable_data<T>(context.GetPlace());
      memset(d_table_data, 0, d_table->numel() * sizeof(T));

      const auto &ids_lod = ids->lod();
      PADDLE_ENFORCE_EQ(ids_lod.size(), 1UL,
                        platform::errors::InvalidArgument(
                            "The LoD level of Input(Ids) should be 1. But "
                            "received Ids's LoD level = %d.",
                            ids_lod.size()));
      const std::vector<uint64_t> offset = ids_lod[0];
      auto len = ids->numel();
      int idx_width = len / offset.back();

      Tensor csr_vals_t, csr_colmuns_t, csr_row_idx_t;
      csr_vals_t.Resize({len});
      csr_colmuns_t.Resize({len});
      int64_t batch_size = ids_lod[0].size() - 1;
      csr_row_idx_t.Resize({(batch_size + 1) * idx_width});
      auto csr_vals = csr_vals_t.mutable_data<T>(context.GetPlace());
      auto csr_colmuns = csr_colmuns_t.mutable_data<int>(context.GetPlace());
      auto csr_row_idx = csr_row_idx_t.mutable_data<int>(context.GetPlace());
      prepare_csr_data<T>(offset, ids->data<int64_t>(), idx_width, csr_vals,
                          csr_colmuns, csr_row_idx, padding_idx);

      auto *d_output_data = d_output->data<T>();
      auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
      int width = static_cast<int>(table_dim[1]);
      int num_seq = batch_size * idx_width;
      LOG(INFO) << "num seq = " << num_seq << " width = " << width;
      for (int i = 0; i < num_seq; ++i) {
        for (int j = csr_row_idx[i]; j < csr_row_idx[i + 1]; ++j) {
          unsigned int word_idx = csr_colmuns[j];
          T val = csr_vals[j];
          blas.AXPY(width, val, d_output_data + i * width,
                    d_table_data + word_idx * width);
        }
      }
#else
      LOG(ERROR) << "Dense is not supported in fused_embedding_seq_pool_op now";
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle
