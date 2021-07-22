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

echo "Your username of the jupyter-docker is:  paddle"
echo "Please set your password: "
read -s passwd1
echo "Please enter the password again: "
read -s passwd2

if [ "${passwd1}" == "${passwd2}" ];then
    docker run -p 80:80 --rm --env USER_PASSWD="${passwd1}" -v $(pwd):/home/paddle pangyoki/paddle:2.1.0-jupyter
else
    echo "The two passwords are inconsistent!"
    exit
fi
