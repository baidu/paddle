# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# the steps of processing the results to get a statistics of
# running clang-tidy to paddle/fluid
# remove color output
# grep only path that has fluid keyword
# grep the files that triggered check rules
# supress third_party files
# supress protobuf generated files
# extract the last column (the check rule)
# count and sort by frequency

sed $'s/\e\\[[0-9;:]*[a-zA-Z]//g' "$@" \
| grep fluid \
| grep '\]$' \
| grep -v third_party  \
| grep -v -F ".pb.cc"  \
| rev | cut -d' ' -f1 | rev \
| sort | uniq -c | sort -k1 -r
