// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/tdm_child_op.h"

namespace paddle {
namespace operators {

/**
 * @param nway: N-way tree
 * @param length: the vector length for a node, now equals to nway+3
 */
template <typename T = int64_t, typename InfoT = int, typename OutT = int>
__global__ void TDMChildCUDAInner(const T* input, int inputN,
            const InfoT* tree, int treeN, int nway, int length,
                   OutT* children, OutT* mask) {

    // idx of input
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= inputN) {
        // ignore the idle thread
        return ;
    }

    // TODO: check it outside
    T node_id = input[idx];

    int meta_length = length - nway;

    // leaf node:
    //  1. node_id is pading as zero
    //  2. the first child of this node_id equals to zero
    bool has_child = (node_id == 0 ||
            tree[static_cast<int>(node_id * length + meta_length)] == 0) ?
            false : true;

    int child_idx = 0;
    InfoT child = 0;
    if (has_child) {
        for ( ; child_idx < nway; ++child_idx) {
            child =
                tree[static_cast<int>(node_id * length + meta_length + child_idx)];
            children[static_cast<int>(idx * nway + child_idx)] = child;

            mask[static_cast<int>(idx * nway + child_idx)] =
                static_cast<OutT>(tree[static_cast<int>(child * length)] == 0? 0: 1);
        }
    } else {
        for ( ; child_idx < nway; ++child_idx) {
            children[static_cast<int>(idx * nway + child_idx)] = 0;

            mask[static_cast<int>(idx * nway + child_idx)] = 0;
        }
    }
}

template <typename T>
class TDMChildKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("X");
    auto *tree_info_var = ctx.InputVar("TreeInfo");

    auto &input_tensor = input_var->Get<LoDTensor>();
    const auto &input_type = input_tensor.type();
    bool input_type_match = input_type == framework::proto::VarType::INT32 ||
                            input_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(input_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(X) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(input_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto &tree_info_tensor = tree_info_var->Get<LoDTensor>();
    const auto &info_type = tree_info_tensor.type();
    bool info_type_match = info_type == framework::proto::VarType::INT32 ||
                           info_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(
        info_type_match, true,
        platform::errors::InvalidArgument(
            "Input(TreeInfo) holds the wrong type, it holds %s, but "
            "desires to be %s or %s",
            paddle::framework::DataTypeToString(info_type),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT64)));

    auto *child_var = ctx.OutputVar("Child");
    auto *leaf_mask_var = ctx.OutputVar("LeafMask");
    auto *child_tensor = child_var->GetMutable<framework::LoDTensor>();
    auto *leaf_mask_tensor = leaf_mask_var->GetMutable<framework::LoDTensor>();

    auto output_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    bool out_type_match = output_type == framework::proto::VarType::INT32 ||
                          output_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(out_type_match, true,
                      platform::errors::InvalidArgument(
                          "Ouput(Child) & Output(LeafMask) holds the wrong "
                          "type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(output_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto child_nums = ctx.Attr<int>("child_nums");
    auto info_dims = tree_info_tensor.dims();
    int node_nums = info_dims[0];
    int length = info_dims[1];
    int input_ids_num = input_tensor.numel();

    if (info_type == framework::proto::VarType::INT32 &&
        output_type == framework::proto::VarType::INT32) {

      int* child_data = static_cast<int*>(child_tensor->mutable_data<int>(ctx.GetPlace()));
      int* leaf_mask_data = static_cast<int*>(leaf_mask_tensor->mutable_data<int>(ctx.GetPlace()));
      TDMChildCUDAInner<T, int, int><<<
          1, input_ids_num>>>(input_tensor.data<T>(), input_ids_num,
              tree_info_tensor.data<int>(), node_nums,
              child_nums,
              length,
              child_data,
              leaf_mask_data
              );
    } else if (info_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT32) {
      int* child_data = static_cast<int*>(child_tensor->mutable_data<int>(ctx.GetPlace()));
      int* leaf_mask_data = static_cast<int*>(leaf_mask_tensor->mutable_data<int>(ctx.GetPlace()));
      TDMChildCUDAInner<T, int64_t, int><<<
          1, input_ids_num>>>(input_tensor.data<T>(), input_ids_num,
              tree_info_tensor.data<int64_t>(), node_nums,
              child_nums,
              length,
              child_data,
              leaf_mask_data
              );
    } else if (info_type == framework::proto::VarType::INT32 &&
               output_type == framework::proto::VarType::INT64) {
      int64_t* child_data = static_cast<int64_t*>(child_tensor->mutable_data<int64_t>(ctx.GetPlace()));
      int64_t* leaf_mask_data = static_cast<int64_t*>(leaf_mask_tensor->mutable_data<int64_t>(ctx.GetPlace()));
      TDMChildCUDAInner<T, int, int64_t><<<
          1, input_ids_num>>>(input_tensor.data<T>(), input_ids_num,
              tree_info_tensor.data<int>(), node_nums,
              child_nums,
              length,
              child_data,
              leaf_mask_data
              );
    } else if (info_type == framework::proto::VarType::INT64 &&
               output_type == framework::proto::VarType::INT64) {
      int64_t* child_data = static_cast<int64_t*>(child_tensor->mutable_data<int64_t>(ctx.GetPlace()));
      int64_t* leaf_mask_data = static_cast<int64_t*>(leaf_mask_tensor->mutable_data<int64_t>(ctx.GetPlace()));
      TDMChildCUDAInner<T, int64_t, int64_t><<<
          1, input_ids_num>>>(input_tensor.data<T>(), input_ids_num,
              tree_info_tensor.data<int64_t>(), node_nums,
              child_nums,
              length,
              child_data,
              leaf_mask_data
              );
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    tdm_child, ops::TDMChildKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TDMChildKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TDMChildKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TDMChildKernel<paddle::platform::CUDADeviceContext, int64_t>);
