// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"
#include <gtest/gtest.h>
#include <ctime>
#include "paddle/fluid/distributed/index_dataset/index_dataset.pb.h"
using paddle::distributed::GraphIndex;

TEST(GRAPH_INDEX, RUN) {
  srand(time(0));
  GraphIndex graph1, graph2, graph3, graph4;
  int width = 3, height = 5, path_num = 9;
  graph1.set_width(width);
  graph1.set_height(height);
  graph1.set_item_path_nums(path_num);
  graph1.add_item(4, {3, 54, 12});
  graph1.add_item(6, {7, 2, 100});
  graph1.add_item(34, {7, 8, 12});
  graph1.save("paddle_graph_index.param");
  graph2.load("paddle_graph_index.param");
  ASSERT_EQ(graph2.width(), width);
  ASSERT_EQ(graph2.height(), height);
  auto map_2 = graph2.get_item_path_dict();
  auto map_1 = graph1.get_item_path_dict();
  ASSERT_EQ(map_2.size(), map_1.size());
  for (auto p1 : map_1) {
    ASSERT_EQ(map_2.find(p1.first) != map_2.end(), true);
    auto vec = map_2[p1.first];
    ASSERT_EQ(vec.size(), p1.second.size());
    for (int i = 0; i < (int)vec.size(); i++) {
      ASSERT_EQ(vec[i], p1.second[i]);
    }
  }
  std::vector<uint32_t> path_vec = {7, 12};
  std::vector<std::vector<uint64_t>> path2item =
      graph2.get_item_of_path(path_vec);
  ASSERT_EQ((int)path2item.size(), 2);
  for (auto item_vec : path2item) {
    ASSERT_EQ((int)item_vec.size(), 2);
  }
  graph3.set_width(width);
  graph3.set_height(height);
  graph3.set_item_path_nums(path_num);
  std::map<uint32_t, std::vector<uint64_t>> path_item_map;
  std::vector<uint64_t> item_vec1 = {7, 12, 1, 6};
  for (auto item : item_vec1) {
    auto vec = graph3.create_path(item);
    for (auto path : vec) {
      path_item_map[path].push_back(item);
    }
  }
  for (auto p : path_item_map) {
    sort(p.second.begin(), p.second.end());
    std::vector<uint32_t> t = {p.first};
    auto vec = graph3.get_item_of_path(t);
    sort(vec[0].begin(), vec[0].end());
    for (int i = 0; i < (int)vec[0].size(); i++) {
      ASSERT_EQ(vec[0][i], p.second[i]);
    }
    ASSERT_EQ(vec[0].size(), p.second.size());
  }
  graph4.set_height(3);
  graph4.set_width(5);
  graph4.set_item_path_nums(7);
  std::unordered_map<uint64_t, std::vector<uint32_t>> candidate_list;
  std::unordered_map<uint64_t, std::vector<double>> candidate_score;
  for (int i = 0; i < 100; i++) {
    std::vector<uint32_t> v;
    std::vector<double> s;
    for (int j = 0; j < 20; j++) {
      v.push_back(rand() % std::max(j + 1, i + 1));
      while (find(v.begin(), v.begin() + j, v.back()) != v.begin() + j) {
        v.pop_back();
        v.push_back(rand() % std::max(j + 1, i + 1));
      }
      s.push_back(1);
    }
    candidate_score[i] = s;
    candidate_list[i] = v;
  }
  graph4.update_Jpath_of_item(candidate_list, candidate_score, 10, 0.1, 4);
  auto x = graph4.get_item_path_dict();
  for (int i = 0; i < 100; i++) {
    std::unordered_set<uint32_t> xx(x[i].begin(), x[i].end());
    ASSERT_EQ(xx.size(), graph4.item_path_nums());
    for (auto path : xx) {
      ASSERT_EQ(std::find(candidate_list[i].begin(), candidate_list[i].end(),
                          path) != candidate_list[i].end(),
                true);
    }
  }
}