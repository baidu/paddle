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

echo "begin test elastic"

unset GREP_OPTIONS
rm -rf log

python -m pip install --no-cache-dir etcd3 -i https://mirror.baidu.com/pypi/simple

# common env
export PADDLE_ELASTIC_NP=2
export PADDLE_ELASTIC_SERVER=127.0.0.1:2379
export PADDLE_ELASTIC_JOB_ID=elastic-demo

# run node 0
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export DISTRIBUTED_TRAINER_ENDPOINTS=10.10.10.1:8001,10.10.10.2:8001
export PADDLE_TRAINERS=10.10.10.1,10.10.10.2
export TRAINER_PORTS_NUM=1
export POD_IP=10.10.10.1
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=2

python -m paddle.distributed.launch elastic_demo.py &> log_0.log &
p0=$!

for i in {1..10}
do
    if grep -q "INFO:ELASTIC:not ready" log_0.log; then
        echo "run node 0 ok"
        break
    else
        sleep 1
    fi
    if [ $i -eq 10 ]; then
        echo "run node 0 error"
        exit -1
    fi
done

# run node 1
export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=1
export DISTRIBUTED_TRAINER_ENDPOINTS=10.10.10.1:8001,10.10.10.2:8001
export PADDLE_TRAINERS=10.10.10.1,10.10.10.2
export TRAINER_PORTS_NUM=1
export POD_IP=10.10.10.2
export PADDLE_TRAINER_ID=1
export PADDLE_TRAINERS_NUM=2

python -m paddle.distributed.launch elastic_demo.py &> log_1.log &
p1=$!

for i in {1..10}
do
    if grep -q "INFO:ELASTIC:ready with hosts" log_1.log; then
        echo "run node 1 ok"
        break
    else
        sleep 1
    fi
    if [ $i -eq 10 ]; then
        echo "run node 1 error"
        exit -1
    fi
done

lw0="log/workerlog.0"

check_env() {
    sleep 3
    if grep -q "0-PADDLE_TRAINERS=$PADDLE_TRAINERS" $lw0 && grep -q "1-PADDLE_TRAINERS=$PADDLE_TRAINERS" $lw0; then
        echo "PADDLE_TRAINERS ok"
    else
        echo "PADDLE_TRAINERS error"
        exit -1
    fi
    
    if grep -q "0-DISTRIBUTED_TRAINER_ENDPOINTS=$DISTRIBUTED_TRAINER_ENDPOINTS" $lw0 && grep -q "1-DISTRIBUTED_TRAINER_ENDPOINTS=$DISTRIBUTED_TRAINER_ENDPOINTS" $lw0; then
        echo "DISTRIBUTED_TRAINER_ENDPOINTS ok"
    else
        echo "DISTRIBUTED_TRAINER_ENDPOINTS error"
        exit -1
    fi
}

check_env

for i in {1..10}
do
    kill $p1
    sleep 2
    if grep -q "INFO:ELASTIC:not ready" log_0.log; then
        echo "stop node 1 ok"
        break
    else
        sleep 1
    fi
    if [ $i -eq 10 ]; then
        echo "stop node 1 error"
        exit -1
    fi
done

# rerun node 1
export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=1
export DISTRIBUTED_TRAINER_ENDPOINTS=10.10.10.1:8001,10.10.10.3:8001
export PADDLE_TRAINERS=10.10.10.1,10.10.10.3
export TRAINER_PORTS_NUM=1
export POD_IP=10.10.10.3
export PADDLE_TRAINER_ID=1
export PADDLE_TRAINERS_NUM=2

python -m paddle.distributed.launch elastic_demo.py &> log_1.log &
p1=$!

for i in {1..10}
do
    if grep -q "INFO:ELASTIC:ready with hosts" log_1.log; then
        echo "rerun node 1 ok"
        break
    else
        sleep 1
    fi
    if [ $i -eq 10 ]; then
        echo "rerun node 1 error"
        exit -1
    fi
done

check_env

sleep 3
kill $p0 $p1
