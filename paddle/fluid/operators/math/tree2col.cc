// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
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

#include "paddle/fluid/operators/math/tree2col.h"

#include <deque>
#include <stack>

namespace paddle {
namespace operators {
namespace math {
using Tensor = framework::Tensor;
std::vector<TreeNode> Tree2ColUtil::construct_patch(
    size_t root, int max_depth, const std::vector<std::vector<int>> &tr) {
  std::stack<TreeNode, std::deque<TreeNode>> stack;
  std::unordered_map<int, bool> visited;
  std::vector<TreeNode> patch;

  stack.push(TreeNode(root, 1, 1, 0));
  patch.emplace_back(TreeNode(root, 1, 1, 0));
  visited[root] = true;

  while (!stack.empty()) {
    TreeNode &u = stack.top();
    bool end = true;
    size_t node = u.get_node(), sz = tr[node].size();
    visited[node] = true;
    for (size_t i = 0; i < sz; i++) {
      size_t v = tr[node][i];
      if (!visited[v] && static_cast<int>(u.get_depth()) + 1 < max_depth) {
        visited[v] = true;
        stack.push(TreeNode(v, i, sz, u.get_depth() + 1));
        patch.emplace_back(v, i + 1, sz, u.get_depth() + 1);
        end = false;
      }
    }
    if (end) {
      stack.pop();
    }
  }
  return patch;
}

void Tree2ColUtil::construct_tree(const paddle::Tensor &EdgeSet,
                                  std::vector<std::vector<int>> *tr,
                                  size_t *node_count) {
  auto edge_set_dims = EdgeSet.dims();
  PADDLE_ENFORCE_EQ(edge_set_dims[1], 2,
                    platform::errors::InvalidArgument(
                        "The second dimension of the EdgeSet shall be 2, but "
                        "got %ld != 2. Please check the input value.",
                        edge_set_dims[1]));
  int64_t edge_count = EdgeSet.numel();

  const int *edge_data = EdgeSet.data<int>();

  for (int64_t i = 0; i < edge_count; i += 2) {
    int u = edge_data[i], v = edge_data[i + 1];
    if (u != 0 && v != 0) (*node_count)++;
  }
  (*node_count)++;

  tr->resize(static_cast<size_t>(*node_count + 1));

  for (int64_t i = 0; i < edge_count; i += 2) {
    int u = edge_data[i], v = edge_data[i + 1];
    if (u != 0 && v != 0) {
      tr->at(u).push_back(v);
    } else {
      break;
    }
  }
}

template <typename T>
class Tree2ColFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext &context,
                  const framework::Tensor &EdgeSet,
                  const framework::Tensor &node_features,
                  framework::Tensor *patch, int max_depth) {
    std::vector<std::vector<int>> tr;
    auto feature_dims = node_features.dims();
    auto cpu_place = BOOST_GET_CONST(platform::CPUPlace, context.GetPlace());
    math::SetConstant<platform::CPUDeviceContext, T> constant;
    int64_t feature_size = feature_dims[1];
    size_t patch_elem_size = 3 * static_cast<size_t>(feature_size);
    size_t node_count = 0, patch_count = 0, patch_size;
    Tree2ColUtil::construct_tree(EdgeSet, &tr, &node_count);
    std::vector<std::vector<TreeNode>> processing_list;
    for (size_t u = 1; u <= node_count; u++) {
      std::vector<TreeNode> temp_patch =
          Tree2ColUtil::construct_patch(u, max_depth, tr);
      if (!temp_patch.empty()) {
        processing_list.emplace_back(temp_patch);
      }
    }
    patch_size = processing_list.size();

    T *patch_data =
        patch->mutable_data<T>({static_cast<int64_t>(patch_size),
                                static_cast<int64_t>(patch_elem_size)},
                               cpu_place);
    constant(context, patch, 0);
    const T *features = node_features.data<T>();

    for (auto &patch_item : processing_list) {
      size_t pointer_base = patch_count * patch_elem_size;
      for (auto &v : patch_item) {
        T eta_l = v.eta_l<T>(max_depth), eta_r = v.eta_r<T>(max_depth),
          eta_t = v.eta_t<T>(max_depth);
        size_t id = v.get_node() - 1;
        for (int i = 0; i < feature_size; i++) {
          patch_data[pointer_base + i * 3] +=
              eta_l * features[id * feature_size + i];
          patch_data[pointer_base + i * 3 + 1] +=
              eta_r * features[id * feature_size + i];
          patch_data[pointer_base + i * 3 + 2] +=
              eta_t * features[id * feature_size + i];
        }
      }
      patch_count++;
    }
    patch->Resize({static_cast<int64_t>(patch_count),
                   static_cast<int64_t>(patch_elem_size)});
  }
};
template <typename T>
class Col2TreeFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext &context,
                  const framework::Tensor &EdgeSet,
                  const framework::Tensor &out_grad, framework::Tensor *in_grad,
                  int max_depth) {
    std::vector<std::vector<int>> tr;
    auto output_dims = out_grad.dims();
    auto cpu_place = BOOST_GET_CONST(platform::CPUPlace, context.GetPlace());
    math::SetConstant<platform::CPUDeviceContext, T> constant;
    int64_t output_size = output_dims[1];
    size_t grad_elem_size = 3 * static_cast<size_t>(output_size);
    size_t node_count = 0, grad_count = 0;
    Tree2ColUtil::construct_tree(EdgeSet, &tr, &node_count);
    std::vector<std::vector<TreeNode>> processing_list;
    std::vector<std::vector<TreeNode>> grad_list;
    grad_list.resize(node_count);
    for (size_t u = 1; u <= node_count; u++) {
      std::vector<TreeNode> tmp =
          Tree2ColUtil::construct_patch(u, max_depth, tr);
      if (!tmp.empty()) {
        processing_list.push_back(tmp);
      }
    }
    for (size_t patch_id = 0; patch_id < processing_list.size(); patch_id++) {
      for (auto v : processing_list[patch_id]) {
        grad_list[v.get_node() - 1].push_back(v.change_node(patch_id + 1));
      }
    }
    T *grad_data =
        in_grad->mutable_data<T>({static_cast<int64_t>(node_count),
                                  static_cast<int64_t>(grad_elem_size)},
                                 cpu_place);

    constant(context, in_grad, 0);
    const T *out_g = out_grad.data<T>();
    for (auto &patch_item : grad_list) {
      size_t pointer_base = grad_count * grad_elem_size;
      for (auto &v : patch_item) {
        T eta_l = v.eta_l<T>(max_depth), eta_r = v.eta_r<T>(max_depth),
          eta_t = v.eta_t<T>(max_depth);
        size_t id = v.get_node() - 1;
        for (int i = 0; i < output_size; i++) {
          grad_data[pointer_base + i * 3] +=
              eta_l * out_g[id * output_size + i];
          grad_data[pointer_base + i * 3 + 1] +=
              eta_r * out_g[id * output_size + i];
          grad_data[pointer_base + i * 3 + 2] +=
              eta_t * out_g[id * output_size + i];
        }
      }
      grad_count++;
    }
  }
};

template class Tree2ColFunctor<platform::CPUDeviceContext, float>;
template class Tree2ColFunctor<platform::CPUDeviceContext, double>;
template class Col2TreeFunctor<platform::CPUDeviceContext, float>;
template class Col2TreeFunctor<platform::CPUDeviceContext, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
