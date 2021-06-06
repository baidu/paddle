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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class BipartiteMatchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("DistMat"), true,
        platform::errors::InvalidArgument(
            "Input(DistMat) of BipartiteMatch should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("ColToRowMatchIndices"), true,
                      platform::errors::InvalidArgument(
                          "Output(ColToRowMatchIndices) of BipartiteMatch "
                          "should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("ColToRowMatchDist"), true,
        platform::errors::InvalidArgument(
            "Output(ColToRowMatchDist) of BipartiteMatch should not be null."));

    auto dims = ctx->GetInputDim("DistMat");
    PADDLE_ENFORCE_EQ(dims.size() == 2UL || dims.size() == 3UL, true,
                      platform::errors::InvalidArgument(
                          "Expected the rank of Input(DistMat) must be 2 or 3. "
                          "But received %d.",
                          dims.size()));

    if (dims.size() == 3UL) {
      ctx->SetOutputDim("ColToRowMatchIndices",
                        framework::make_ddim({dims[0], dims[2]}));
      ctx->SetOutputDim("ColToRowMatchDist",
                        framework::make_ddim({dims[0], dims[2]}));
    } else {
      ctx->SetOutputDim("ColToRowMatchIndices", dims);
      ctx->SetOutputDim("ColToRowMatchDist", dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "DistMat"),
        platform::CPUPlace());
  }
};

template <class T>
bool DistPairDescend(std::tuple<int, int, T> pair1,
                     std::tuple<int, int, T> pair2) {
  return std::get<2>(pair1) > std::get<2>(pair2);
}

template <typename T>
class BipartiteMatchKernel : public framework::OpKernel<T> {
 public:
  // The match_indices must be initialized to -1 at first.
  // The match_dist must be initialized to 0 at first.
  void BipartiteMatch(const Tensor& dist, int* match_indices,
                      T* match_dist) const {
    PADDLE_ENFORCE_EQ(
        dist.dims().size(), 2,
        platform::errors::InvalidArgument("The rank of dist must be 2."));
    int64_t row = dist.dims()[0];
    int64_t col = dist.dims()[1];
    auto* dist_data = dist.data<T>();
    // Test result: When row==130 the speed of these two methods almost the same
    if (row >= 130) {
      std::vector<std::tuple<int, int, T>> match_pair;

      for (int64_t i = 0; i < row; ++i) {
        for (int64_t j = 0; j < col; ++j) {
          match_pair.push_back(std::make_tuple(i, j, dist_data[i * col + j]));
        }
      }
      std::sort(match_pair.begin(), match_pair.end(), DistPairDescend<T>);
      std::vector<int> row_indices(row, -1);

      int64_t idx = 0;
      for (int64_t k = 0; k < row * col; ++k) {
        int64_t i = std::get<0>(match_pair[k]);
        int64_t j = std::get<1>(match_pair[k]);
        T dist = std::get<2>(match_pair[k]);

        if (idx >= row) {
          break;
        }
        if (match_indices[j] == -1 && row_indices[i] == -1 && dist > 0) {
          match_indices[j] = i;
          row_indices[i] = j;
          match_dist[j] = dist;
          idx += 1;
        }
      }
    } else {
      constexpr T kEPS = static_cast<T>(1e-6);
      std::vector<int> row_pool;
      for (int i = 0; i < row; ++i) {
        row_pool.push_back(i);
      }
      while (row_pool.size() > 0) {
        int max_idx = -1;
        int max_row_idx = -1;
        T max_dist = -1;
        for (int64_t j = 0; j < col; ++j) {
          if (match_indices[j] != -1) {
            continue;
          }
          for (size_t k = 0; k < row_pool.size(); ++k) {
            int m = row_pool[k];
            // distance is 0 between m-th row and j-th column
            if (dist_data[m * col + j] < kEPS) {
              continue;
            }
            if (dist_data[m * col + j] > max_dist) {
              max_idx = j;
              max_row_idx = m;
              max_dist = dist_data[m * col + j];
            }
          }
        }
        if (max_idx == -1) {
          // Cannot find good match.
          break;
        } else {
          PADDLE_ENFORCE_EQ(
              match_indices[max_idx], -1,
              platform::errors::InvalidArgument(
                  "The match_indices must be initialized to -1 at [%d].",
                  max_idx));
          match_indices[max_idx] = max_row_idx;
          match_dist[max_idx] = max_dist;
          // Erase the row index.
          row_pool.erase(
              std::find(row_pool.begin(), row_pool.end(), max_row_idx));
        }
      }
    }
  }

  void ArgMaxMatch(const Tensor& dist, int* match_indices, T* match_dist,
                   T overlap_threshold) const {
    constexpr T kEPS = static_cast<T>(1e-6);
    int64_t row = dist.dims()[0];
    int64_t col = dist.dims()[1];
    auto* dist_data = dist.data<T>();
    for (int64_t j = 0; j < col; ++j) {
      if (match_indices[j] != -1) {
        // the j-th column has been matched to one entity.
        continue;
      }
      int max_row_idx = -1;
      T max_dist = -1;
      for (int i = 0; i < row; ++i) {
        T dist = dist_data[i * col + j];
        if (dist < kEPS) {
          // distance is 0 between m-th row and j-th column
          continue;
        }
        if (dist >= overlap_threshold && dist > max_dist) {
          max_row_idx = i;
          max_dist = dist;
        }
      }
      if (max_row_idx != -1) {
        PADDLE_ENFORCE_EQ(
            match_indices[j], -1,
            platform::errors::InvalidArgument(
                "The match_indices must be initialized to -1 at [%d].", j));
        match_indices[j] = max_row_idx;
        match_dist[j] = max_dist;
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* dist_mat = context.Input<LoDTensor>("DistMat");
    auto* match_indices = context.Output<Tensor>("ColToRowMatchIndices");
    auto* match_dist = context.Output<Tensor>("ColToRowMatchDist");

    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();

    auto col = dist_mat->dims()[dist_mat->dims().size() - 1];

    if (dist_mat->dims().size() == 3UL) {
      // new version for DistMat Tensor [N, P, M]
      PADDLE_ENFORCE_EQ(
          dist_mat->lod().size(), 0UL,
          platform::errors::InvalidArgument(
              "When the input dimension is 3, "
              "The type of input should be Tensor, not LoDTensor. "));

      int64_t batch_size = dist_mat->dims()[0];

      match_indices->mutable_data<int>({batch_size, col}, context.GetPlace());
      match_dist->mutable_data<T>({batch_size, col}, context.GetPlace());

      math::SetConstant<platform::CPUDeviceContext, int> iset;
      iset(dev_ctx, match_indices, static_cast<int>(-1));
      math::SetConstant<platform::CPUDeviceContext, T> tset;
      tset(dev_ctx, match_dist, static_cast<T>(0));

      int* indices = match_indices->data<int>();
      T* dist = match_dist->data<T>();
      auto type = context.Attr<std::string>("match_type");
      auto threshold = context.Attr<float>("dist_threshold");

      framework::DDim one_ins_shape = {dist_mat->dims()[1],
                                       dist_mat->dims()[2]};
      for (int64_t i = 0; i < batch_size; ++i) {
        Tensor one_ins = dist_mat->Slice(i, i + 1).Resize(one_ins_shape);
        BipartiteMatch(one_ins, indices + i * col, dist + i * col);
        if (type == "per_prediction") {
          ArgMaxMatch(one_ins, indices + i * col, dist + i * col, threshold);
        }
      }
    } else {
      // old version for LoDTensor
      int64_t n = dist_mat->lod().size() == 0UL
                      ? 1
                      : static_cast<int64_t>(dist_mat->lod().back().size() - 1);
      if (dist_mat->lod().size()) {
        PADDLE_ENFORCE_EQ(
            dist_mat->lod().size(), 1UL,
            platform::errors::InvalidArgument("Only support 1 level of LoD."));
      }
      match_indices->mutable_data<int>({n, col}, context.GetPlace());
      match_dist->mutable_data<T>({n, col}, context.GetPlace());

      math::SetConstant<platform::CPUDeviceContext, int> iset;
      iset(dev_ctx, match_indices, static_cast<int>(-1));
      math::SetConstant<platform::CPUDeviceContext, T> tset;
      tset(dev_ctx, match_dist, static_cast<T>(0));

      int* indices = match_indices->data<int>();
      T* dist = match_dist->data<T>();
      auto type = context.Attr<std::string>("match_type");
      auto threshold = context.Attr<float>("dist_threshold");
      if (n == 1) {
        BipartiteMatch(*dist_mat, indices, dist);
        if (type == "per_prediction") {
          ArgMaxMatch(*dist_mat, indices, dist, threshold);
        }
      } else {
        auto lod = dist_mat->lod().back();
        for (size_t i = 0; i < lod.size() - 1; ++i) {
          if (lod[i + 1] > lod[i]) {
            Tensor one_ins = dist_mat->Slice(lod[i], lod[i + 1]);
            BipartiteMatch(one_ins, indices + i * col, dist + i * col);
            if (type == "per_prediction") {
              ArgMaxMatch(one_ins, indices + i * col, dist + i * col,
                          threshold);
            }
          }
        }
      }
    }
  }
};

class BipartiteMatchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "DistMat",
        "(LoDTensor or Tensor) this input is a 2-D LoDTensor with shape "
        "[K, M] or 3-D Tensor with shape [B, P, M], where K = B * P. "
        "It is pair-wise distance matrix between the entities "
        "represented by each row and each column. For example, assumed one "
        "entity is A with shape [K], another entity is B with shape [M]. The "
        "DistMat[i][j] is the distance between A[i] and B[j]. The bigger "
        "the distance is, the better macthing the pairs are. Please note, "
        "This tensor can contain LoD information to represent a batch of "
        "inputs. One instance of this batch can contain different numbers of "
        "entities.");
    AddAttr<std::string>(
        "match_type",
        "(string, default: bipartite) "
        "The type of matching method, should be 'bipartite' or "
        "'per_prediction', 'bipartite' by default.")
        .SetDefault("bipartite")
        .InEnum({"bipartite", "per_prediction"});
    AddAttr<float>(
        "dist_threshold",
        "(float, default: 0.5) "
        "If `match_type` is 'per_prediction', this threshold is to determine "
        "the extra matching bboxes based on the maximum distance.")
        .SetDefault(0.5);
    AddOutput("ColToRowMatchIndices",
              "(Tensor) A 2-D Tensor with shape [N, M] in int type. "
              "N is the batch size. If ColToRowMatchIndices[i][j] is -1, it "
              "means B[j] does not match any entity in i-th instance. "
              "Otherwise, it means B[j] is matched to row "
              "ColToRowMatchIndices[i][j] in i-th instance. The row number of "
              "i-th instance is saved in ColToRowMatchIndices[i][j].");
    AddOutput("ColToRowMatchDist",
              "(Tensor) A 2-D Tensor with shape [N, M] in float type. "
              "N is batch size. If ColToRowMatchIndices[i][j] is -1, "
              "ColToRowMatchDist[i][j] is also -1.0. Otherwise, assumed "
              "ColToRowMatchIndices[i][j] = d, and the row offsets of each "
              "instance are called LoD. Then "
              "ColToRowMatchDist[i][j] = DistMat[d+LoD[i]][j]");
    AddComment(R"DOC(
This operator is a greedy bipartite matching algorithm, which is used to
obtain the matching with the maximum distance based on the input
distance matrix. For input 2D matrix, the bipartite matching algorithm can
find the matched column for each row, also can find the matched row for
each column. And this operator only calculate matched indices from column
to row. For each instance, the number of matched indices is the number of
of columns of the input distance matrix.

There are two outputs to save matched indices and distance.
A simple description, this algorithm matched the best (maximum distance)
row entity to the column entity and the matched indices are not duplicated
in each row of ColToRowMatchIndices. If the column entity is not matched
any row entity, set -1 in ColToRowMatchIndices.

Please note that the input DistMat can be LoDTensor (with LoD) or Tensor.
If LoDTensor with LoD, the height of ColToRowMatchIndices is batch size.
If Tensor, the height of ColToRowMatchIndices is 1.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    bipartite_match, ops::BipartiteMatchOp, ops::BipartiteMatchOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(bipartite_match, ops::BipartiteMatchKernel<float>,
                       ops::BipartiteMatchKernel<double>);
