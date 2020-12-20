// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <fstream>
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/string/string_helper.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

TEST(TEST_GLOO, store_1) {
#ifdef _LINUX
#ifdef PADDLE_WITH_GLOO
#else
  auto store = gloo::rendezvous::HdfsStore("./test_gllo_store");
  store.set("1", std::vector<char>{'t', 'e', 's', 't'});
  store.get("1");
  try {
    store.get("2");
  } catch (...) {
    VLOG(3) << "catch expected error of not found";
  }
  store.wait(std::vector<std::string>{"test"});
  store.wait(std::vector<std::string>{"test"}, std::chrono::milliseconds(0));
  store.SetTimeoutSeconds(100000);
  store.EncodeName("1");
  store.TmpPath("1");
  store.ObjectPath("1");
  std::vector<bool> status(1, false);
  store.Check(std::vector<std::string>{"test"}, &status);

  auto gw = paddle::framework::GlooWrapper();
  gw.SetTimeoutSeconds(1000, 1000);
  gw.SetRank(0);
  gw.SetSize(1);
  gw.SetPrefix("");
  gw.SetIface("lo");
  gw.SetHdfsStore("", "", "");
  gw.Init();
  gw.SetHttpStore("", 8099, "");
  gw.Init();
  gw.Rank();
  gw.Size();
  gw.Barrier();
  std::vector<double> input;
  gw.AllReduce(input);
  int64_t t;
  gw.AllGather(t);
#endif
#endif
}

TEST(TEST_FLEET, fleet_1) {
  auto fleet = paddle::framework::FleetWrapper::GetInstance();
#ifdef PADDLE_WITH_PSLIB
#else
//  fleet->RunServer("", 0);
//  fleet->SaveModelOneTable(0, "", 0);
//  fleet->SaveModelOneTablePrefix(0, "", 0, "");
//  fleet->Confirm();
//  fleet->Revert();
//  paddle::string::erase_spaces("1 2");
#endif
}
