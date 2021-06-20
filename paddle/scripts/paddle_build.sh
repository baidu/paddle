#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   Utils
#=================================================

set -ex

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

function print_usage() {
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}${SCRIPT_NAME}${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}build${NONE}: run build for x86 platform
    ${BLUE}test${NONE}: run all unit tests
    ${BLUE}single_test${NONE}: run a single unit test
    ${BLUE}bind_test${NONE}: parallel tests bind to different GPU
    ${BLUE}doc${NONE}: generate paddle documents
    ${BLUE}gen_doc_lib${NONE}: generate paddle documents library
    ${BLUE}html${NONE}: convert C++ source code into HTML
    ${BLUE}dockerfile${NONE}: generate paddle release dockerfile
    ${BLUE}fluid_inference_lib${NONE}: deploy fluid inference library
    ${BLUE}check_style${NONE}: run code style check
    ${BLUE}cicheck${NONE}: run CI tasks on Linux
    ${BLUE}maccheck${NONE}: run CI tasks on Mac
    "
}

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
    export PADDLE_ROOT
    if [ -z "${SCRIPT_NAME}" ]; then
        SCRIPT_NAME=$0
    fi

    ENABLE_MAKE_CLEAN=${ENABLE_MAKE_CLEAN:-ON}

    # NOTE(chenweihang): For easy debugging, CI displays the C++ error stacktrace by default 
    export FLAGS_call_stack_level=2

    # set CI_SKIP_CPP_TEST if only *.py changed
    # In order to avoid using in some CI(such as daily performance), the current
    # branch must not be `${BRANCH}` which is usually develop.
    if [ ${CI_SKIP_CPP_TEST:-ON} == "OFF"  ];then
        echo "CI_SKIP_CPP_TEST=OFF"
    else
        if [ "$(git branch | grep "^\*" | awk '{print $2}')" != "${BRANCH}" ]; then
            git diff --name-only ${BRANCH} | grep -v "\.py$" || export CI_SKIP_CPP_TEST=ON
        fi
    fi
}

function cmake_base() {
    # Build script will not fail if *.deb does not exist
    rm *.deb 2>/dev/null || true
    # Delete previous built whl packages
    rm -rf python/dist 2>/dev/null || true

    # Support build for all python versions, currently
    # including cp27-cp27m and cp27-cp27mu.
    PYTHON_FLAGS=""
    SYSTEM=`uname -s`
    if [ "$SYSTEM" == "Darwin" ]; then
        echo "Using python abi: $1"
        if [[ "$1" == "cp27-cp27m" ]] || [[ "$1" == "" ]]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/2.7" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7
                export PATH=/Library/Frameworks/Python.framework/Versions/2.7/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib"
                pip install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp35-cp35m" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.5" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.5/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.5/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5m.dylib"
                pip3.5 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp36-cp36m" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.6" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.6/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.6/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.6/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.6/include/python3.6m/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.6/lib/libpython3.6m.dylib"
                pip3.6 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp37-cp37m" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.7" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.7/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.7/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.7/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7m.dylib"
                pip3.7 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp38-cp38" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.8" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.8/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.8/include/python3.8/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.8/lib/libpython3.8.dylib"
                pip3.8 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp39-cp39" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.9" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.9/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.9/include/python3.9/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.9/lib/libpython3.9.dylib"
                pip3.9 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        fi
    else
        if [ "$1" != "" ]; then
            echo "using python abi: $1"
            if [ "$1" == "cp27-cp27m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs2/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.15-ucs4/lib:}
                export PATH=/opt/python/cp27-cp27m/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27m/bin/python
            -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27m/include/python2.7
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.15-ucs2/lib/libpython2.7.so"
                pip install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp27-cp27mu" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.15-ucs2/lib:}
                export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python
            -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27mu/include/python2.7
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.15-ucs4/lib/libpython2.7.so"
                pip install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp27-cp27m-gcc82" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs2/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.15-ucs4/lib:}
                export PATH=/opt/python/cp27-cp27m/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27m/bin/python
            -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27m/include/python2.7
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.15-ucs2/lib/libpython2.7.so"
                pip install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp27-cp27mu-gcc82" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.15-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.15-ucs2/lib:}
                export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python
            -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27mu/include/python2.7
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.15-ucs4/lib/libpython2.7.so"
                pip install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp35-cp35m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.5.1/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.5.1/include/python3.5m
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.5.1/lib/libpython3.so"
                pip3.5 install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp36-cp36m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.6.0/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.6.0/include/python3.6m
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.6.0/lib/libpython3.so"
                pip3.6 install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp37-cp37m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7m
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.so"
                pip3.7 install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp38-cp38" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.8.0/bin/python3.8
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.8.0/include/python3.8
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.8.0/lib/libpython3.so"
                pip3.8 install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp39-cp39" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.9.0/bin/python3.9
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.9.0/include/python3.9
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.9.0/lib/libpython3.so"
                pip3.9 install -r ${PADDLE_ROOT}/python/requirements.txt
           elif [ "$1" == "conda-python3.7" ]; then
                export LD_LIBRARY_PATH=/opt/conda/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/conda/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/bin/python
                                     -DPYTHON_INCLUDE_DIR:PATH=/opt/conda/include/python3.7m
                                     -DPYTHON_LIBRARIES:FILEPATH=/opt/conda/lib/libpython3.so"
                /opt/conda/bin/pip install -r ${PADDLE_ROOT}/python/requirements.txt
           fi
        else
            pip install -r ${PADDLE_ROOT}/python/requirements.txt
        fi
    fi

    if [ "$SYSTEM" == "Darwin" ]; then
        WITH_DISTRIBUTE="OFF"
        WITH_AVX=${WITH_AVX:-ON}
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-~/.cache/inference_demo}
    else
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-/root/.cache/inference_demo}
    fi

    distibuted_flag=${WITH_DISTRIBUTE:-OFF}
    gloo_flag=${distibuted_flag}

    if [ "$CMD" != "assert_file_approvals" ];then
      python -m pip install distro
      python ${PADDLE_ROOT}/tools/summary_env.py
      bash ${PADDLE_ROOT}/tools/get_cpu_info.sh
    fi

    cat <<EOF
    ========================================
    Configuring cmake in /paddle/build ...
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
        ${PYTHON_FLAGS}
        -DWITH_GPU=${WITH_GPU:-OFF}
        -DWITH_TENSORRT=${WITH_TENSORRT:-ON}
        -DWITH_ROCM=${WITH_ROCM:-OFF}
        -DWITH_DISTRIBUTE=${distibuted_flag}
        -DWITH_MKL=${WITH_MKL:-ON}
        -DWITH_AVX=${WITH_AVX:-OFF}
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All}
        -DWITH_PYTHON=${WITH_PYTHON:-ON}
        -DCUDNN_ROOT=/usr/
        -DWITH_TESTING=${WITH_TESTING:-ON}
        -DWITH_COVERAGE=${WITH_COVERAGE:-OFF}
        -DWITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF}
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DWITH_CONTRIB=${WITH_CONTRIB:-ON}
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON}
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR}
        -DPY_VERSION=${PY_VERSION:-2.7}
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build}
        -DWITH_PSCORE=${distibuted_flag}
        -DWITH_GLOO=${gloo_flag}
        -DWITH_LITE=${WITH_LITE:-OFF}
        -DWITH_XPU=${WITH_XPU:-OFF}
        -DLITE_GIT_TAG=release/v2.8
        -DWITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF}
        -DWITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF}
        -DWITH_STRIP=${WITH_STRIP:-ON}
    ========================================
EOF
    # Disable UNITTEST_USE_VIRTUALENV in docker because
    # docker environment is fully controlled by this script.
    # See /Paddle/CMakeLists.txt, UNITTEST_USE_VIRTUALENV option.
    set +e
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
        ${PYTHON_FLAGS} \
        -DWITH_GPU=${WITH_GPU:-OFF} \
        -DWITH_TENSORRT=${WITH_TENSORRT:-ON} \
        -DWITH_ROCM=${WITH_ROCM:-OFF} \
        -DWITH_DISTRIBUTE=${distibuted_flag} \
        -DWITH_MKL=${WITH_MKL:-ON} \
        -DWITH_AVX=${WITH_AVX:-OFF} \
        -DNOAVX_CORE_FILE=${NOAVX_CORE_FILE:-""} \
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} \
        -DWITH_PYTHON=${WITH_PYTHON:-ON} \
        -DCUDNN_ROOT=/usr/ \
        -DWITH_TESTING=${WITH_TESTING:-ON} \
        -DWITH_COVERAGE=${WITH_COVERAGE:-OFF} \
        -DWITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF} \
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_CONTRIB=${WITH_CONTRIB:-ON} \
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} \
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} \
        -DPY_VERSION=${PY_VERSION:-2.7} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build} \
        -DWITH_PSCORE=${distibuted_flag} \
        -DWITH_GLOO=${gloo_flag} \
        -DLITE_GIT_TAG=release/v2.8 \
        -DWITH_XPU=${WITH_XPU:-OFF} \
        -DXPU_SDK_ROOT=${XPU_SDK_ROOT:-""} \
        -DWITH_LITE=${WITH_LITE:-OFF} \
        -DWITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF} \
        -DWITH_STRIP=${WITH_STRIP:-ON} \
        -DWITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF};build_error=$?
    if [ "$build_error" != 0 ];then
        exit 7;
    fi
}

function cmake_gen() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cmake_base $1
}

function cmake_gen_in_current_dir() {
    cmake_base $1
}

function abort(){
    echo "Your change doesn't follow PaddlePaddle's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 4
}

function check_style() {
    trap 'abort' 0
    set -e

    if [ -x "$(command -v gimme)" ]; then
    	eval "$(GIMME_GO_VERSION=1.8.3 gimme)"
    fi


    pip install cpplint pylint pytest astroid isort
    # set up go environment for running gometalinter
    mkdir -p $GOPATH/src/github.com/PaddlePaddle/
    ln -sf ${PADDLE_ROOT} $GOPATH/src/github.com/PaddlePaddle/Paddle

    pre-commit install
    clang-format --version

    commit_files=on
    for file_name in `git diff --numstat upstream/$BRANCH |awk '{print $NF}'`;do
        if ! pre-commit run --files $file_name ; then
            commit_files=off
        fi
    done 
    
    if [ $commit_files == 'off' ];then
        echo "code format error"
        git diff 2>&1
        exit 4
    fi
    trap : 0
}

#=================================================
#              Build
#=================================================

function build_base() {
    set +e
    if [ "$SYSTEM" == "Linux" ];then
      if [ `nproc` -gt 16 ];then
          parallel_number=$(expr `nproc` - 8)
      else
          parallel_number=`nproc`
      fi
    else
      parallel_number=8
    fi
    if [ "$1" != "" ]; then
      parallel_number=$1
    fi

    if [[ "$ENABLE_MAKE_CLEAN" != "OFF" ]]; then
        make clean
    fi

    # reset ccache zero stats for collect PR's actual hit rate
    ccache -z

    make install -j ${parallel_number};build_error=$?

    # ci will collect ccache hit rate
    collect_ccache_hits

    if [ "$build_error" != 0 ];then
        exit 7;
    fi
}

function build_size() {
    cat <<EOF
    ============================================
    Calculate /paddle/build size and PR whl size
    ============================================
EOF
    if [ "$1" == "paddle_inference" ]; then
        cd ${PADDLE_ROOT}/build
        cp -r paddle_inference_install_dir paddle_inference
        tar -czf paddle_inference.tgz paddle_inference
        buildSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference.tgz |awk '{print $1}')
        echo "Paddle_Inference Size: $buildSize"
        echo "ipipe_log_param_Paddle_Inference_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    elif [ "$1" == "paddle_inference_c" ]; then
        cd ${PADDLE_ROOT}/build
        cp -r paddle_inference_c_install_dir paddle_inference_c
        tar -czf paddle_inference_c.tgz paddle_inference_c
        buildSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference_c.tgz |awk '{print $1}')
        echo "Paddle_Inference Capi Size: $buildSize"
        echo "ipipe_log_param_Paddle_Inference_capi_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    else
        SYSTEM=`uname -s`
        if [ "$SYSTEM" == "Darwin" ]; then
            com='du -h -d 0'
        else
            com='du -h --max-depth=0'
        fi
        buildSize=$($com ${PADDLE_ROOT}/build |awk '{print $1}')
        echo "Build Size: $buildSize"
        echo "ipipe_log_param_Build_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
        PR_whlSize=$($com ${PADDLE_ROOT}/build/python/dist |awk '{print $1}')
        echo "PR whl Size: $PR_whlSize"
        echo "ipipe_log_param_PR_whl_Size: $PR_whlSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    fi
}

function build() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ============================================
    Building in /paddle/build ...
    ============================================
EOF
    build_base $@
    current_branch=`git branch | grep \* | cut -d ' ' -f2`
    if [ "$current_branch" != "develop_base_pr" ];then
        build_size
    fi
}

function cmake_gen_and_build() {
    startTime_s=`date +%s`
    cmake_gen $1
    build $2
    endTime_s=`date +%s`
    echo "Build Time: $[ $endTime_s - $startTime_s ]s"
    echo "ipipe_log_param_Build_Time: $[ $endTime_s - $startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
}

function build_mac() {
    set +e
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ============================================
    Building in /paddle/build ...
    ============================================
EOF
    if [[ "$ENABLE_MAKE_CLEAN" != "OFF" ]]; then
        make clean
    fi

    # reset ccache zero stats for collect PR's actual hit rate
    ccache -z

    make install -j 8;build_error=$?

    # ci will collect ccache hit rate
    collect_ccache_hits

    if [ "$build_error" != 0 ];then
        exit 7;
    fi

    set -e
    build_size
}

function cmake_gen_and_build_mac() {
    startTime_s=`date +%s`
    cmake_gen $1
    build_mac
    endTime_s=`date +%s`
    echo "Build Time: $[ $endTime_s - $startTime_s ]s"
    echo "ipipe_log_param_Build_Time: $[ $endTime_s - $startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
}

function run_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
        if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
            ctest -V
        else
            ctest --output-on-failure
        fi
    fi
}


function combine_avx_noavx_build() {
    mkdir -p ${PADDLE_ROOT}/build.noavx
    cd ${PADDLE_ROOT}/build.noavx
    WITH_AVX=OFF
    cmake_base ${PYTHON_ABI:-""}
    build_base

    # build combined one
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    NOAVX_CORE_FILE=`find ${PADDLE_ROOT}/build.noavx/python/paddle/fluid/ -name "core_noavx.*"`
    WITH_AVX=ON

    cmake_base ${PYTHON_ABI:-""}
    build_base
}


function run_brpc_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [[ ${WITH_TESTING:-ON} == "ON" \
        && ${WITH_DISTRIBUTE:-OFF} == "ON" ]] ; then
    cat <<EOF
    ========================================
    Running brpc unit tests ...
    ========================================
EOF
        set +x
        declare -a other_tests=("test_listen_and_serv_op" "system_allocator_test" \
        "rpc_server_test" "varhandle_test" "collective_server_test" "brpc_serde_test")
        all_tests=`ctest -N`

        for t in "${other_tests[@]}"
        do
            if [[ ${all_tests} != *$t* ]]; then
                continue
            fi

            if [[ ${TESTING_DEBUG_MODE:-OFF} == "ON" ]] ; then
                ctest -V -R $t
            else
                ctest --output-on-failure -R $t
            fi
        done
        set -x

        if [[ ${TESTING_DEBUG_MODE:-OFF} == "ON" ]] ; then
            ctest -V -R test_dist_*
        else
            ctest --output-on-failure -R test_dist_*
        fi
    fi
}



function run_mac_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
        #remove proxy here to fix dist ut 'test_fl_listen_and_serv_op' error on mac. 
        #see details: https://github.com/PaddlePaddle/Paddle/issues/24738
        set +x
        my_proxy=$http_proxy
        export http_proxy=
        export https_proxy=
        set -x

        set +ex
        if [ "$1" == "cp27-cp27m" ]; then
            pip uninstall -y paddlepaddle
        elif [ "$1" == "cp35-cp35m" ]; then
            pip3.5 uninstall -y paddlepaddle
        elif [ "$1" == "cp36-cp36m" ]; then
            pip3.6 uninstall -y paddlepaddle
        elif [ "$1" == "cp37-cp37m" ]; then
            pip3.7 uninstall -y paddlepaddle
        elif [ "$1" == "cp38-cp38" ]; then
            pip3.8 uninstall -y paddlepaddle
        elif [ "$1" == "cp39-cp39" ]; then
            pip3.9 uninstall -y paddlepaddle
        fi
        set -ex

        if [ "$1" == "cp27-cp27m" ]; then
            set -e
            pip install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
            python ${PADDLE_ROOT}/paddle/scripts/installation_validate.py
        elif [ "$1" == "cp35-cp35m" ]; then
            pip3.5 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        elif [ "$1" == "cp36-cp36m" ]; then
            pip3.6 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        elif [ "$1" == "cp37-cp37m" ]; then
            pip3.7 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        elif [ "$1" == "cp38-cp38" ]; then
            pip3.8 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        elif [ "$1" == "cp39-cp39" ]; then
            pip3.9 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        fi
        tmpfile_rand=`date +%s%N`
        tmpfile=$tmp_dir/$tmpfile_rand
        set +ex
        ut_startTime_s=`date +%s`
        get_quickly_disable_ut||disable_ut_quickly='' # indicate whether the case was in quickly disable list 
        if [ ${NIGHTLY_MODE:-OFF} == "ON" ]; then
            nightly_label="(NIGHTLY_LABEL)"
        else
            nightly_label="(RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY)"
            echo "========================================="
            echo "Unittests with nightly labels  are only run at night"
            echo "========================================="
        fi
        bash $PADDLE_ROOT/tools/check_added_ut.sh
        get_precision_ut_mac
        if [[ "$on_precision" == "0" ]];then
            ctest -E "($disable_ut_quickly)" -LE ${nightly_label} --output-on-failure -j $2 | tee $tmpfile
        else
            ctest -R "($UT_list_prec)" -E "($disable_ut_quickly)" -LE ${nightly_label} --output-on-failure -j $2 | tee $tmpfile
            tmpfile_rand=`date +%s%N`
            tmpfile=$tmp_dir/$tmpfile_rand
            ctest -R "($UT_list_prec_1)" -E "($disable_ut_quickly)" -LE ${nightly_label} --output-on-failure -j $2 | tee $tmpfile
        fi
        failed_test_lists=''
        collect_failed_tests
        mactest_error=0
        retry_unittests_record=''
        retry_time=3
        exec_times=0
        exec_time_array=('first' 'second' 'third')
        exec_retry_threshold=10
        is_retry_execuate=0
        if [ -n "$failed_test_lists" ];then
            mactest_error=1
            read need_retry_ut_str <<< $(echo "$failed_test_lists" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
            need_retry_ut_arr=(${need_retry_ut_str})
            need_retry_ut_count=${#need_retry_ut_arr[@]}
            read retry_unittests <<< $(echo "$failed_test_lists" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
            if [ $need_retry_ut_count -lt $exec_retry_threshold ];then
                while ( [ $exec_times -lt $retry_time ] )
                    do
                        retry_unittests_record="$retry_unittests_record$failed_test_lists"
                        failed_test_lists_ult=`echo "${failed_test_lists}"`
                        if [[ "${exec_times}" == "1" ]];then
                            if [[ "${failed_test_lists}" == "" ]];then
                                break
                            else
                                read retry_unittests <<< $(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
                            fi
                        fi
                        echo "========================================="
                        echo "This is the ${exec_time_array[$exec_times]} time to re-run"
                        echo "========================================="
                        echo "The following unittest will be re-run:"
                        echo "${retry_unittests}"
                        echo "========================================="

                        retry_unittests_regular=''
                        for line in ${retry_unittests[@]} ;
                            do
                                if [[ "$retry_unittests_regular" == "" ]];then
                                    retry_unittests_regular="^$line$"
                                else
                                    retry_unittests_regular="$retry_unittests_regular|^$line$"
                                fi
                            done
                        rm -f $tmp_dir/*
                        failed_test_lists=''
                        ctest -R "($retry_unittests_regular)" --output-on-failure -j $2 | tee $tmpfile
                        collect_failed_tests
                        exec_times=$[$exec_times+1]
                    done
            else
                # There are more than 10 failed unit tests, so no unit test retry
                is_retry_execuate=1
            fi

        fi
        #mactest_error=$?
        ut_endTime_s=`date +%s`
        echo "Mac testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        echo "ipipe_log_param_Mac_TestCases_Time: $[ $ut_endTime_s - $ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
        paddle version
        # Recovery proxy to avoid failure in later steps
        export http_proxy=$my_proxy
        export https_proxy=$my_proxy
        if [ "$mactest_error" != 0 ];then
            show_ut_retry_result
        fi
    fi
}

function get_precision_ut_mac() {
    on_precision=0
    UT_list=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
    precison_cases=""
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python3.7 $PADDLE_ROOT/tools/get_pr_ut.py
        if [[ -f "ut_list" ]]; then
            echo "PREC length: "`wc -l ut_list`
            precision_cases=`cat ut_list`
        fi
    fi
    if [ ${PRECISION_TEST:-OFF} == "ON" ] && [[ "$precision_cases" != "" ]];then
        UT_list_re=''
        on_precision=1
        re=$(cat ut_list|awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
        UT_list_prec_1='ut_list_prec2'
        for case in $UT_list; do
            flag=$(echo $case|grep -oE $re)
            if [ -n "$flag" ];then
                if [ -z "$UT_list_prec" ];then
                    UT_list_prec="^$case$"
                elif [[ "${#UT_list_prec}" -gt 10000 ]];then
                    UT_list_prec_1="$UT_list_prec_1|^$case$"
                else
                    UT_list_prec="$UT_list_prec|^$case$"
                fi
            else
                echo ${case} "won't run in PRECISION_TEST mode."
            fi
        done
    fi
}

function fetch_upstream_develop_if_not_exist() {
    UPSTREAM_URL='https://github.com/PaddlePaddle/Paddle'
    origin_upstream_url=`git remote -v | awk '{print $1, $2}' | uniq | grep upstream | awk '{print $2}'` 
    if [ "$origin_upstream_url" == "" ]; then
        git remote add upstream $UPSTREAM_URL.git
    elif [ "$origin_upstream_url" != "$UPSTREAM_URL" ] \
            && [ "$origin_upstream_url" != "$UPSTREAM_URL.git" ]; then
        git remote remove upstream
        git remote add upstream $UPSTREAM_URL.git
    fi
    
    if [ ! -e "$PADDLE_ROOT/.git/refs/remotes/upstream/$BRANCH" ]; then 
        git fetch upstream # develop is not fetched
    fi
}

function generate_upstream_develop_api_spec() {
    fetch_upstream_develop_if_not_exist
    cur_branch=`git branch | grep \* | cut -d ' ' -f2`
    git checkout .
    git checkout -b develop_base_pr upstream/$BRANCH
    cmake_gen $1
    build $2
    cp ${PADDLE_ROOT}/python/requirements.txt /tmp

    git checkout $cur_branch
    generate_api_spec "$1" "DEV"
    git branch -D develop_base_pr
    ENABLE_MAKE_CLEAN="ON"
    rm -rf ${PADDLE_ROOT}/build/Makefile ${PADDLE_ROOT}/build/CMakeCache.txt
    cmake_change=`git diff --name-only upstream/$BRANCH | grep "cmake/external" || true`
    if [ ${cmake_change} ];then
        rm -rf ${PADDLE_ROOT}/build/third_party
    fi
}

function generate_api_spec() {
    set -e
    spec_kind=$2
    if [ "$spec_kind" != "PR" ] && [ "$spec_kind" != "DEV" ]; then
        echo "Not supported $2"
        exit 1
    fi

    mkdir -p ${PADDLE_ROOT}/build/.check_api_workspace
    cd ${PADDLE_ROOT}/build/.check_api_workspace
    virtualenv .${spec_kind}_env
    source .${spec_kind}_env/bin/activate

    if [ "$spec_kind" == "DEV" ]; then
        pip install -r /tmp/requirements.txt
    else
        pip install -r ${PADDLE_ROOT}/python/requirements.txt
    fi
    pip --no-cache-dir install ${PADDLE_ROOT}/build/python/dist/*whl
    spec_path=${PADDLE_ROOT}/paddle/fluid/API_${spec_kind}.spec
    python ${PADDLE_ROOT}/tools/print_signatures.py paddle > $spec_path

    # used to log op_register data_type
    op_type_path=${PADDLE_ROOT}/paddle/fluid/OP_TYPE_${spec_kind}.spec
    python ${PADDLE_ROOT}/tools/check_op_register_type.py > $op_type_path

    # print all ops desc in dict to op_desc_path
    op_desc_path=${PADDLE_ROOT}/paddle/fluid/OP_DESC_${spec_kind}.spec
    python ${PADDLE_ROOT}/tools/print_op_desc.py > $op_desc_path

    # print api and the md5 of source code of the api.
    api_source_md5_path=${PADDLE_ROOT}/paddle/fluid/API_${spec_kind}.source.md5
    python ${PADDLE_ROOT}/tools/count_api_without_core_ops.py -p paddle > $api_source_md5_path

    awk -F '(' '{print $NF}' $spec_path >${spec_path}.doc
    awk -F '(' '{$NF="";print $0}' $spec_path >${spec_path}.api
    if [ "$1" == "cp35-cp35m" ] || [ "$1" == "cp36-cp36m" ] || [ "$1" == "cp37-cp37m" ] || [ "$1" == "cp38-cp38" ] || [ "$1" == "cp39-cp39" ]; then
        # Use sed to make python2 and python3 sepc keeps the same
        sed -i 's/arg0: str/arg0: unicode/g' $spec_path
        sed -i "s/\(.*Transpiler.*\).__init__ (ArgSpec(args=\['self'].*/\1.__init__ /g" $spec_path
    fi   
    
    python ${PADDLE_ROOT}/tools/diff_use_default_grad_op_maker.py \
        ${PADDLE_ROOT}/paddle/fluid/op_use_default_grad_maker_${spec_kind}.spec

    deactivate
}

function check_approvals_of_unittest() {
    set +x
    if [ "$GITHUB_API_TOKEN" == "" ] || [ "$GIT_PR_ID" == "" ]; then
        return 0
    fi
    # approval_user_list: XiaoguangHu01 46782768,luotao1 6836917,phlrain 43953930,lanxianghit 47554610, zhouwei25 52485244, kolinwei 22165420
    check_times=$1
    if [ $check_times == 1 ]; then
        approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
        if [ "${approval_line}" != "" ]; then
            APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 22165420 52485244`
            echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
            if [ "${APPROVALS}" == "TRUE" ]; then
                echo "==================================="
                echo -e "\n current pr ${GIT_PR_ID} has got approvals. So, Pass CI directly!\n"
                echo "==================================="
                exit 0
            fi
        fi
    elif [ $check_times == 2 ]; then
        unittest_spec_diff=`python ${PADDLE_ROOT}/tools/diff_unittest.py ${PADDLE_ROOT}/paddle/fluid/UNITTEST_DEV.spec ${PADDLE_ROOT}/paddle/fluid/UNITTEST_PR.spec`
        if [ "$unittest_spec_diff" != "" ]; then
            approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
            APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 22165420 52485244 32428676 45041955`
            echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
            if [ "${APPROVALS}" == "FALSE" ]; then
                echo "************************************"
                echo -e "It is forbidden to disable or delete the unit-test.\n"
                echo -e "If you must delete it temporarily, please add it to[https://github.com/PaddlePaddle/Paddle/wiki/Temporarily-disabled-Unit-Test]."
                echo -e "Then you must have one RD (kolinwei(recommended), chalsliu, XieYunshen or zhouwei25) approval for the deletion of unit-test. \n"
                echo -e "If you have any problems about deleting unit-test, please read the specification [https://github.com/PaddlePaddle/Paddle/wiki/Deleting-unit-test-is-forbidden]. \n"
                echo -e "Following unit-tests are deleted in this PR: \n ${unittest_spec_diff} \n"
                echo "************************************"
                exit 6
            fi
        fi
    fi
    set -x
}

function check_diff_file_for_coverage() {
    diff_h_file=$(git diff --name-status test develop | awk '$1 != "D" {print $2}' | grep '\.h$' | awk -F "/" '{printf "%s,",$NF}')
    diff_cc_file=$(git diff --name-status test develop | awk '$1 != "D" {print $2}' | grep -E '\.(cc|c)$' | awk -F "/" '{printf "%s,",$NF}')
    diff_py_file=$(git diff --name-status test develop | grep '\.py$' | awk '$1 != "D" {printf "%s,",$2}')

    export PADDLE_GIT_DIFF_H_FILE=${diff_h_file%*,}
    export PADDLE_GIT_DIFF_CC_FILE=${diff_cc_file%*,}
    export PADDLE_GIT_DIFF_PY_FILE=${diff_py_file%*,}
}

function check_change_of_unittest() {
    generate_unittest_spec "PR"
    fetch_upstream_develop_if_not_exist
    git reset --hard upstream/$BRANCH
    cmake_gen $1
    generate_unittest_spec "DEV"
    check_approvals_of_unittest 2
}

function check_sequence_op_unittest(){
    /bin/bash ${PADDLE_ROOT}/tools/check_sequence_op.sh
}

function generate_unittest_spec() {
    spec_kind=$1
    if [ "$spec_kind" == "DEV" ]; then
        cat <<EOF
        ============================================
        Generate unit tests.spec of develop.
        ============================================
EOF
    elif [ "$spec_kind" == "PR" ]; then
        cat <<EOF
        ============================================
        Generate unit tests.spec of this PR.
        ============================================
EOF
    else
        echo "Not supported $1"
        exit 1
    fi
    spec_path=${PADDLE_ROOT}/paddle/fluid/UNITTEST_${spec_kind}.spec
    ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' > ${spec_path}
}


function assert_api_spec_approvals() {
    /bin/bash ${PADDLE_ROOT}/tools/check_api_approvals.sh;approval_error=$?
    if [ "$approval_error" != 0 ];then
       exit 6
    fi
}

function assert_file_diff_approvals() {
    /bin/bash ${PADDLE_ROOT}/tools/check_file_diff_approvals.sh;file_approval_error=$?
    if [ "$file_approval_error" != 0 ];then
       exit 6
    fi
}


function check_coverage() {
    /bin/bash ${PADDLE_ROOT}/tools/coverage/paddle_coverage.sh
}


function single_test() {
    TEST_NAME=$1
    if [ -z "${TEST_NAME}" ]; then
        echo -e "${RED}Usage:${NONE}"
        echo -e "${BOLD}${SCRIPT_NAME}${NONE} ${BLUE}single_test${NONE} [test_name]"
        exit 1
    fi
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running ${TEST_NAME} ...
    ========================================
EOF
        ctest --output-on-failure -R ${TEST_NAME}
    fi
}

function bind_test() {
    # the number of process to run tests
    NUM_PROC=6

    # calculate and set the memory usage for each process
    MEM_USAGE=$(printf "%.2f" `echo "scale=5; 1.0 / $NUM_PROC" | bc`)
    export FLAGS_fraction_of_gpu_memory_to_use=$MEM_USAGE

    # get the CUDA device count
    CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)

    for (( i = 0; i < $NUM_PROC; i++ )); do
        cuda_list=()
        for (( j = 0; j < $CUDA_DEVICE_COUNT; j++ )); do
            s=$[i+j]
            n=$[s%CUDA_DEVICE_COUNT]
            if [ $j -eq 0 ]; then
                cuda_list=("$n")
            else
                cuda_list="$cuda_list,$n"
            fi
        done
        echo $cuda_list
        # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
        # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
        env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC --output-on-failure &
    done
    wait
}

EXIT_CODE=0;
function caught_error() {
 for job in `jobs -p`; do
        # echo "PID => ${job}"
        if ! wait ${job} ; then
            echo "At least one test failed with exit code => $?" ;
            EXIT_CODE=1;
        fi
    done
}

function case_count(){
    cat <<EOF
    ============================================
    Generating TestCases Count ... 
    ============================================
EOF
    testcases=$1
    num=$(echo $testcases|grep -o '\^'|wc -l)
    if (( $2 == -1 )); then
        echo "exclusive TestCases count is $num"
        echo "ipipe_log_param_Exclusive_TestCases_Count: $num" >> ${PADDLE_ROOT}/build/build_summary.txt
    else
        echo "$2 card TestCases count is $num"
        echo "ipipe_log_param_${2}_Cards_TestCases_Count: $num" >> ${PADDLE_ROOT}/build/build_summary.txt
    fi
}

failed_test_lists=''
tmp_dir=`mktemp -d`

function collect_failed_tests() {
    for file in `ls $tmp_dir`; do
        exit_code=0
        grep -q 'The following tests FAILED:' $tmp_dir/$file||exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
}

# getting qucik disable ut list 
function get_quickly_disable_ut() {
    python -m pip install requests
    if disable_ut_quickly=$(python ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
        echo "========================================="
        echo "The following unittests have been disabled:"
        echo ${disable_ut_quickly}
        echo "========================================="
    else
        disable_ut_quickly=''
    fi
}

function card_test() {
    set -m
    case_count $1 $2
    ut_startTime_s=`date +%s` 

    testcases=$1
    cardnumber=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}

    # get the CUDA device count, XPU device count is one
    if [ "${WITH_XPU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    elif [ "${WITH_ROCM}" == "ON" ];then
        CUDA_DEVICE_COUNT=4
    else
        CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)
    fi

    if (( $cardnumber == -1 ));then
        cardnumber=$CUDA_DEVICE_COUNT
    fi

    if (( $# > 2 )); then
        parallel_job=`expr $3 \* $parallel_level_base`
    else
        parallel_job=$parallel_level_base
    fi

    if [[ "$testcases" == "" ]]; then
        return 0
    fi

    trap 'caught_error' CHLD
    tmpfile_rand=`date +%s%N`
    NUM_PROC=$[CUDA_DEVICE_COUNT/$cardnumber]
    echo "****************************************************************"
    echo "***These unittests run $parallel_job job each time with $cardnumber GPU***"
    echo "****************************************************************"
    for (( i = 0; i < $NUM_PROC; i++ )); do
        # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
        # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
        cuda_list=()
        for (( j = 0; j < cardnumber; j++ )); do
            if [ $j -eq 0 ]; then
                    cuda_list=("$[i*cardnumber]")
                else
                    cuda_list="$cuda_list,$[i*cardnumber+j]"
            fi
        done
        tmpfile=$tmp_dir/$tmpfile_rand"_"$i
        if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" -V --timeout 120 -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else  
                (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" --timeout 120 -V -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            fi
        else
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else
                (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            fi
        fi
    done
    wait; # wait for all subshells to finish
    ut_endTime_s=`date +%s`
    if (( $2 == -1 )); then
        echo "exclusive TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        echo "ipipe_log_param_Exclusive_TestCases_Total_Time: $[ $ut_endTime_s - $ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
    else
        echo "$2 card TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        echo "ipipe_log_param_${2}_Cards_TestCases_Total_Time: $[ $ut_endTime_s - $ut_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
    fi
    set +m
}

function parallel_test_base_gpu() {
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests in parallel way ...
    ========================================
EOF

set -x
        precison_cases=""
        bash $PADDLE_ROOT/tools/check_added_ut.sh
        if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
            python3.7 $PADDLE_ROOT/tools/get_pr_ut.py
            if [[ -f "ut_list" ]]; then
                set +x
                echo "PREC length: "`wc -l ut_list`
                precision_cases=`cat ut_list`
                set -x
            fi
        fi
        if [ -a "$PADDLE_ROOT/duplicate_ut" ];then
            duplicate_uts=$(cat $PADDLE_ROOT/duplicate_ut|sed -e 's/\r//g')
            if [[ "$duplicate_uts" != "" ]];then
                set +x
                echo "========================================"
                echo "The new unit test has the same name as the existing unit test"
                cat "$PADDLE_ROOT/duplicate_ut"
                echo "========================================"
                exit 102;
                set -x
            fi
        fi
        if [ -a "$PADDLE_ROOT/added_ut" ];then
            added_uts=^$(awk BEGIN{RS=EOF}'{gsub(/\n/,"$|^");print}' $PADDLE_ROOT/added_ut)$
            ctest -R "(${added_uts})" --output-on-failure --repeat-until-fail 3 --timeout 15;added_ut_error=$?
            if [ "$added_ut_error" != 0 ];then
                echo "========================================"
                echo "Added UT should not exceed 15 seconds"
                echo "========================================"
                exit 8;
            fi
        fi
set +x
        EXIT_CODE=0;
        test_cases=$(ctest -N -V) # get all test cases
        # Note(zhouwei): Parallel runs are relative to 'CTEST_PARALLEL_LEVEL', e.g: '4 job each time' means 4*CTEST_PARALLEL_LEVEL
        single_card_tests_high_parallel='^job$'     # cases list which would run the most job each time with single GPU
        single_card_tests_two_parallel='^job$'      # cases list which would run 2 job each time with single GPU
        single_card_tests_non_parallel='^job$'      # cases list which would run 1 job each time with single GPU
        single_card_tests='^job$'                   # all cases list which would take single GPU
        
        multiple_card_tests_two_parallel='^job$'    # cases list which would run 2 job each time with multiple GPUs, most cases would be two GPUs
        multiple_card_tests_non_parallel='^job$'    # cases list which would run 1 job each time with multiple GPUs, most cases would be two GPUs
        
        exclusive_tests_two_parallel='^job$'        # cases list which would run 2 job exclusively(with all GPUs)
        exclusive_tests_non_parallel='^job$'        # cases list which would run 1 job exclusively(with all GPUs)
        
        is_exclusive=''           # indicate whether the case is exclusive type
        is_multicard=''           # indicate whether the case is multiple GPUs type
        is_nightly=''             # indicate whether the case will only run at night
        get_quickly_disable_ut||disable_ut_quickly=''    # indicate whether the case was in quickly disable list

        UT_list=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
        output=$(python ${PADDLE_ROOT}/tools/parallel_UT_rule.py "${UT_list}")
        cpu_parallel_job=$(echo $output | cut -d ";" -f 1)
        tetrad_parallel_job=$(echo $output | cut -d ";" -f 2)
        two_parallel_job=$(echo $output | cut -d ";" -f 3)
        non_parallel_job=$(echo $output | cut -d ";" -f 4)
        while read -r line; do
            if [[ "$line" == "" ]]; then
                continue
            fi
                read matchstr <<< $(echo "$line"|grep -oEi 'Test[ \t]+#')
                if [[ "$matchstr" == "" ]]; then
                    # Any test case with LABELS property would be parse here
                    # RUN_TYPE=EXCLUSIVE mean the case would run exclusively
                    # RUN_TYPE=DIST mean the case would take two graph GPUs during runtime
                    # RUN_TYPE=NIGHTLY or RUN_TYPE=DIST:NIGHTLY or RUN_TYPE=EXCLUSIVE:NIGHTLY means the case will ONLY run at night
                    read is_exclusive <<< $(echo "$line"|grep -oEi "RUN_TYPE=EXCLUSIVE")
                    read is_multicard <<< $(echo "$line"|grep -oEi "RUN_TYPE=DIST")
                    read is_nightly <<< $(echo "$line"|grep -oEi "RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY")
                    continue
                fi
                read testcase <<< $(echo "$line"|grep -oEi "\w+$")

                if [[ "$is_nightly" != "" ]] && [ ${NIGHTLY_MODE:-OFF} == "OFF" ]; then
                    echo $testcase" will only run at night."
                    continue
                fi
                if [ ${PRECISION_TEST:-OFF} == "ON" ] && [[ "$precision_cases" != "" ]]; then
                    will_test="false"
                    for case in $precision_cases; do
                        if [[ $testcase == $case ]]; then
                            will_test="true"
                            break
                        fi
                    done
                    if [[ $will_test == "false" ]]; then
                        echo $testcase" won't run in PRECISION_TEST mode."
                        continue
                    fi
                fi

                if [[ "$is_multicard" == "" ]]; then
                  # trick: treat all test case with prefix "test_dist" as dist case, and would run on 2 GPUs
                  read is_multicard <<< $(echo "$testcase"|grep -oEi "test_dist_")
                fi

                if [[ "$is_exclusive" != "" ]]; then
                    if [[ $(echo $cpu_parallel_job$tetrad_parallel_job$two_parallel_job | grep -o "\^$testcase\\$") != "" ]]; then
                        exclusive_tests_two_parallel="$exclusive_tests_two_parallel|^$testcase$"
                    else
                        exclusive_tests_non_parallel="$exclusive_tests_non_parallel|^$testcase$"
                    fi
                elif [[ "$is_multicard" != "" ]]; then
                    if [[ $(echo $cpu_parallel_job$tetrad_parallel_job$two_parallel_job | grep -o "\^$testcase\\$") != "" ]]; then
                        multiple_card_tests_two_parallel="$multiple_card_tests_two_parallel|^$testcase$"
                    else
                        multiple_card_tests_non_parallel="$multiple_card_tests_non_parallel|^$testcase$"
                    fi
                else
                    if [[ $(echo $cpu_parallel_job | grep -o "\^$testcase\\$") != "" ]]; then
                        single_card_tests_high_parallel="$single_card_tests_high_parallel|^$testcase$"
                    elif [[ $(echo $tetrad_parallel_job$two_parallel_job | grep -o "\^$testcase\\$") != "" ]]; then
                        single_card_tests_two_parallel="$single_card_tests_two_parallel|^$testcase$"
                    else
                        single_card_tests_non_parallel="$single_card_tests_non_parallel|^$testcase$"
                    fi
                    single_card_tests="$single_card_tests|^$testcase$"
                fi
                is_exclusive=''
                is_multicard=''
                is_nightly=''
                matchstr=''
                testcase=''
        done <<< "$test_cases";

        card_test "$single_card_tests_high_parallel" 1 6        # run cases the most each time with single GPU
        card_test "$single_card_tests_two_parallel" 1 2         # run cases 2 job each time with single GPU
        card_test "$single_card_tests_non_parallel" 1           # run cases 1 job each time with single GPU
        
        card_test "$multiple_card_tests_two_parallel" 2 2       # run cases 2 job each time with two GPUs
        card_test "$multiple_card_tests_non_parallel" 2         # run cases 1 job each time with two GPUs
        
        card_test "$exclusive_tests_two_parallel" -1 2          # run cases exclusively, in this cases would be run with 2/4/8 GPUs
        card_test "$exclusive_tests_non_parallel" -1            # run cases exclusively, in this cases would be run with 2/4/8 GPUs
        collect_failed_tests
        rm -f $tmp_dir/*
        exec_times=0
        retry_unittests_record=''
        retry_time=3
        exec_time_array=('first' 'second' 'third')
        exec_retry_threshold=10
        is_retry_execuate=0
        if [ -n "$failed_test_lists" ];then
            if [ ${TIMEOUT_DEBUG_HELP:-OFF} == "ON" ];then
                bash $PADDLE_ROOT/tools/timeout_debug_help.sh "$failed_test_lists"    # cat logs for tiemout uts which killed by ctest
            fi
            read need_retry_ut_str <<< $(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
            need_retry_ut_arr=(${need_retry_ut_str})
            need_retry_ut_count=${#need_retry_ut_arr[@]}
            read retry_unittests <<< $(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
            if [ $need_retry_ut_count -lt $exec_retry_threshold ];then
                while ( [ $exec_times -lt $retry_time ] )
                    do
                        retry_unittests_record="$retry_unittests_record$failed_test_lists"
                        failed_test_lists_ult=`echo "${failed_test_lists}" |grep -Po '[^ ].*$'`
                        if [[ "${exec_times}" == "1" ]];then
                            if [[ "${failed_test_lists}" == "" ]];then
                                break
                            else
                                read retry_unittests <<< $(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/(.\+)//' | sed 's/- //' )
                            fi
                        fi
                        echo "========================================="
                        echo "This is the ${exec_time_array[$exec_times]} time to re-run"
                        echo "========================================="
                        echo "The following unittest will be re-run:"
                        echo "${retry_unittests}"
                            
                        for line in ${retry_unittests[@]} ;
                            do

                                read tmp_one_tmp <<< "$( echo $single_card_tests | grep -oEi $line )"
                                read tmp_mul_tmp <<< "$( echo $multiple_card_tests | grep -oEi $line )"
                                read exclusive_tmp <<< "$( echo $exclusive_tests | grep -oEi $line )"

                                if [[ "$tmp_one_tmp" != ""  ]]; then
                                    if [[ "$one_card_retry" == "" ]]; then
                                        one_card_retry="^$line$"
                                    else
                                        one_card_retry="$one_card_retry|^$line$"
                                    fi
                                elif [[ "$tmp_mul_tmp" != "" ]]; then
                                    if [[ "$multiple_card_retry" == "" ]]; then
                                        multiple_card_retry="^$line$"
                                    else
                                        multiple_card_retry="$multiple_card_retry|^$line$"
                                    fi
                                else
                                    if [[ "$exclusive_retry" == "" ]];then
                                        exclusive_retry="^$line$"
                                    else
                                        exclusive_retry="$exclusive_retry|^$line$"
                                    fi
                                fi

                            done

                        if [[ "$one_card_retry" != "" ]]; then
                            card_test "$one_card_retry" 1
                        fi

                        if [[ "$multiple_card_retry" != "" ]]; then
                            card_test "$multiple_card_retry" 2
                        fi

                        if [[ "$exclusive_retry" != "" ]]; then
                            card_test "$exclusive_retry" -1
                        fi
                        
                        exec_times=$[$exec_times+1]
                        failed_test_lists=''
                        collect_failed_tests
                        rm -f $tmp_dir/*
                        one_card_retry=''
                        multiple_card_retry=''
                        exclusive_retry=''
                    done
            else 
                # There are more than 10 failed unit tests, so no unit test retry
                is_retry_execuate=1
            fi
        fi

        if [[ "$EXIT_CODE" != "0" ]]; then
            show_ut_retry_result
        fi
set -ex
    fi
}

function show_ut_retry_result() {
    if [[ "$is_retry_execuate" != "0" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
        echo "========================================="
        echo "There are more than 10 failed unit tests, so no unit test retry!!!"
        echo "========================================="
        echo "The following tests FAILED: "
        echo "${failed_test_lists_ult}"
        exit 8;
    else
        read retry_unittests_ut_name <<< $(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
        retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=3) {print $2}}')
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-ran:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk BEGIN{RS=EOF}'{gsub(/\n/,"|");print}')
            echo "========================================"
            echo "There are failed tests, which have been executed re-run,but success rate is less than 50%:"
            echo "Summary Failed Tests... "
            echo "========================================"
            echo "The following tests FAILED: "
            echo "${retry_unittests_record}" | sort -u | grep -E "$failed_ut_re"
            exit 8;
        fi
    fi
}

function parallel_test_base_cpu() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit cpu tests ...
    ========================================
EOF
        ut_startTime_s=`date +%s`
        ctest --output-on-failure -j $1
        ut_endTime_s=`date +%s`
        echo "CPU testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        if [[ "$EXIT_CODE" != "0" ]]; then
            exit 8;
        fi
    fi
}

function insert_pile_to_h_cu_diff {
    # TODO get develop h/cu md5
    cd ${PADDLE_ROOT}
    find ${PADDLE_ROOT} -name '*.cu'| grep -v ${PADDLE_ROOT}/build >> ${PADDLE_ROOT}/tools/h_cu_files.log
    python ${PADDLE_ROOT}/tools/handle_h_cu_file.py 'get_h_file_md5' ${PADDLE_ROOT}
    
    # TODO insert pile to diff h/cu file 

    #insert pile to full h/cu file 
    python ${PADDLE_ROOT}/tools/handle_h_cu_file.py 'insert_pile_to_h_file' ${PADDLE_ROOT}
}



function precise_card_test_single {
    set +e
    set +x
    testcases=$1
    num=$2
    for case in $(echo $testcases | tr "$|^" "\n")
    do
        cd ${PADDLE_ROOT}/build
        precise_card_test "^${case}$" $num
        # c++ 
        if [ ! -d "${PADDLE_ROOT}/build/ut_map/$case" ];then
            mkdir ${PADDLE_ROOT}/build/ut_map/$case
        fi
        set -x
        find paddle/fluid -name '*.gcda'|xargs -I {} cp --path {} ut_map/$case
        find paddle/fluid -name '*.gcno'|xargs -I {} cp --path {} ut_map/$case
        python ${PADDLE_ROOT}/tools/get_single_test_cov.py ${PADDLE_ROOT} $case &
        
        # python
        ls python-coverage.data.*
        if [[ $? == 0 ]]
        then
            if [ ! -d "${PADDLE_ROOT}/build/pytest/$case" ];then
                mkdir -p ${PADDLE_ROOT}/build/pytest/$case
            fi
            mv python-coverage.data.* ${PADDLE_ROOT}/build/pytest/$case
        fi
        find paddle/fluid -name *.gcda | xargs rm -f #delete gcda
    done
}

function precise_card_test() {
    set -m
    testcases=$1
    if (( $# > 1 )); then
        cardnumber=$2
        cuda_list="0"
        if [ $cardnumber -eq 2 ]; then
            cuda_list=${CUDA_VISIBLE_DEVICES}
        else
            cuda_list="0"
        fi
    else
        cardnumber=2
        cuda_list=${CUDA_VISIBLE_DEVICES}
    fi

    if [[ "$testcases" == "" ]]; then
        return 0
    fi

    echo "****************************************************************"
    echo "***Running ut: $testcases***"
    echo "****************************************************************"
    
    tmpfile=$tmp_dir/$testcases".log"
    env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I 0,,1 -R "($testcases)" --timeout 500 --output-on-failure -V -j 1 > $tmpfile 
    set +m
}

function get_precise_tests_map_file {
    cd ${PADDLE_ROOT}/build
    pip install ${PADDLE_ROOT}/build/python/dist/*whl
    ut_total_startTime_s=`date +%s`
    EXIT_CODE=0;
    test_cases=$(ctest -N -V) # get all test cases
    single_card_tests='' # all cases list which would take one graph card
    exclusive_tests=''        # cases list which would be run exclusively
    multiple_card_tests=''    # cases list which would take multiple GPUs, most cases would be two GPUs
    is_exclusive=''           # indicate whether the case is exclusive type
    is_multicard=''           # indicate whether the case is multiple GPUs type
set +x

    while read -r line; do
        if [[ "$line" == "" ]]; then
            continue
        fi
            read matchstr <<< $(echo "$line"|grep -oEi 'Test[ \t]+#')
            if [[ "$matchstr" == "" ]]; then
                # Any test case with LABELS property would be parse here
                # RUN_TYPE=EXCLUSIVE mean the case would run exclusively
                # RUN_TYPE=DIST mean the case would take two graph GPUs during runtime
                read is_exclusive <<< $(echo "$line"|grep -oEi "RUN_TYPE=EXCLUSIVE")
                read is_multicard <<< $(echo "$line"|grep -oEi "RUN_TYPE=DIST")
                continue
            fi
            read testcase <<< $(echo "$line"|grep -oEi "\w+$")

            if [[ "$is_multicard" == "" ]]; then
                # trick: treat all test case with prefix "test_dist" as dist case, and would run on 2 GPUs
                read is_multicard <<< $(echo "$testcase"|grep -oEi "test_dist_")
            fi

            if [[ "$is_exclusive" != "" ]]; then
                if [[ "$exclusive_tests" == "" ]]; then
                    exclusive_tests="^$testcase$"
                else
                    exclusive_tests="$exclusive_tests|^$testcase$"
                fi
            elif [[ "$is_multicard" != "" ]]; then
                if [[ "$multiple_card_tests" == "" ]]; then
                    multiple_card_tests="^$testcase$"
                else
                    multiple_card_tests="$multiple_card_tests|^$testcase$"
                fi
            else
                if [[ "${single_card_tests}" -gt 3000 ]];then
                    if [[ "$single_card_tests_1" == "" ]]; then
                        single_card_tests_1="^$testcase$"
                    else
                        single_card_tests_1="$single_card_tests_1|^$testcase$"
                    fi
                    continue
                fi
                if [[ "$single_card_tests" == "" ]]; then
                    single_card_tests="^$testcase$"
                else
                    single_card_tests="$single_card_tests|^$testcase$"
                fi
            fi
            is_exclusive=''
            is_multicard=''
            is_nightly=''
            matchstr=''
            testcase=''
    done <<< "$test_cases";

set -x
    mkdir -p ${PADDLE_ROOT}/build/ut_map
    mkdir -p ${PADDLE_ROOT}/build/pytest

    precise_card_test_single "$single_card_tests" 1
    precise_card_test_single "$single_card_tests_1" 1
    precise_card_test_single "$multiple_card_tests" 2
    precise_card_test_single "$exclusive_tests"

    wait;
    python ${PADDLE_ROOT}/tools/get_ut_file_map.py 'get_not_success_ut' ${PADDLE_ROOT}
    #analy h/cu to Map file
    python ${PADDLE_ROOT}/tools/handle_h_cu_file.py 'analy_h_cu_file' $tmp_dir ${PADDLE_ROOT}
    wait;

    get_failedUts_precise_map_file

    #generate python coverage and generate python file to tests_map_file
    python ${PADDLE_ROOT}/tools/pyCov_multithreading.py ${PADDLE_ROOT}
    wait;

    #generate ut map
    python ${PADDLE_ROOT}/tools/get_ut_file_map.py 'get_ut_map' ${PADDLE_ROOT}
}

function get_failedUts_precise_map_file {
    if [[ -f "${PADDLE_ROOT}/build/utNotSuccess" ]]; then
        rerun_tests=`cat ${PADDLE_ROOT}/build/utNotSuccess`
        #remove pile to full h/cu file
        python ${PADDLE_ROOT}/tools/handle_h_cu_file.py 'remove_pile_from_h_file' ${PADDLE_ROOT}
        cd ${PADDLE_ROOT}/build
        cmake_base ${PYTHON_ABI:-""}
        build ${parallel_number}
        pip uninstall -y paddlepaddle-gpu
        pip install ${PADDLE_ROOT}/build/python/dist/*whl
        precise_card_test_single "$rerun_tests"
        wait;
    fi
}

function parallel_test_base_xpu() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit cpu tests ...
    ========================================
EOF

set +x
        ut_startTime_s=`date +%s`
        test_cases=$(ctest -N -V | grep "_xpu" )        # cases list which would be run exclusively
        get_quickly_disable_ut||disable_ut_quickly=''   # indicate whether the case was in quickly disable list
        while read -r line; do
            if [[ "$line" == "" ]]; then
                continue
            fi
            read testcase <<< $(echo "$line"|grep -oEi "\w+$")
            if [[ "$single_card_tests" == "" ]]; then
                single_card_tests="^$testcase$"
            else
                single_card_tests="$single_card_tests|^$testcase$"
            fi
        done <<< "$test_cases";
        card_test "$single_card_tests" 1
        collect_failed_tests
set -x
        ut_endTime_s=`date +%s`
        echo "XPU testCase Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
        if [[ "$EXIT_CODE" != "0" ]]; then
            exit 8;
        fi
    fi   
}

function parallel_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    pip install ${PADDLE_ROOT}/build/python/dist/*whl
    cp ${PADDLE_ROOT}/build/python/paddle/fluid/tests/unittests/op_test.py ${PADDLE_ROOT}/build/python
    ut_total_startTime_s=`date +%s`
    if [ "$WITH_GPU" == "ON" ] || [ "$WITH_ROCM" == "ON" ];then
        parallel_test_base_gpu
    else
        if [ "$WITH_XPU" == "ON" ];then
            parallel_test_base_xpu
        else
            parallel_test_base_cpu ${PROC_RUN:-1}
        fi
    fi
    ut_total_endTime_s=`date +%s`
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    echo "ipipe_log_param_TestCases_Total_Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
}

function enable_unused_var_check() {
    # NOTE(zhiqiu): Set FLAGS_enable_unused_var_check=1 here to enable unused_var_check,
    # which checks if an operator has unused input variable(s).
    # Currently, use it in coverage CI job.
    export FLAGS_enable_unused_var_check=1
}

function gen_doc_lib() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ========================================
    Building documentation library ...
    In /paddle/build
    ========================================
EOF
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \

    local LIB_TYPE=$1
    case $LIB_TYPE in
      full)
        # Build full Paddle Python module. Will timeout without caching 'copy_paddle_pybind' first
        make -j `nproc` framework_py_proto copy_paddle_pybind paddle_python
        ;;
      pybind)
        # Build paddle pybind library. Takes 49 minutes to build. Might timeout
        make -j `nproc` copy_paddle_pybind
        ;;
      proto)
        # Even smaller library.
        make -j `nproc` framework_py_proto
        ;;
      *)
        exit 0
        ;;
      esac
}


function gen_dockerfile() {
    # Set BASE_IMAGE according to env variables
    CUDA_MAJOR="$(echo $CUDA_VERSION | cut -d '.' -f 1).$(echo $CUDA_VERSION | cut -d '.' -f 2)"
    CUDNN_MAJOR=$(echo $CUDNN_VERSION | cut -d '.' -f 1)
    if [[ ${WITH_GPU} == "ON" ]]; then
        BASE_IMAGE="nvidia/cuda:${CUDA_MAJOR}-cudnn${CUDNN_MAJOR}-devel-ubuntu16.04"
    else
        BASE_IMAGE="ubuntu:16.04"
    fi

    DOCKERFILE_GPU_ENV=""
    DOCKERFILE_CUDNN_DSO=""
    DOCKERFILE_CUBLAS_DSO=""
    if [[ ${WITH_GPU:-OFF} == 'ON' ]]; then
        DOCKERFILE_GPU_ENV="ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}"
        DOCKERFILE_CUDNN_DSO="RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.${CUDNN_MAJOR} /usr/lib/x86_64-linux-gnu/libcudnn.so"
        DOCKERFILE_CUBLAS_DSO="RUN ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.${CUDA_MAJOR} /usr/lib/x86_64-linux-gnu/libcublas.so"
    fi

    cat <<EOF
    ========================================
    Generate ${PADDLE_ROOT}/build/Dockerfile ...
    ========================================
EOF
    
    ref_CUDA_MAJOR="$(echo $CUDA_VERSION | cut -d '.' -f 1)"
    if [[ ${WITH_GPU} == "ON"  ]]; then
        ref_gpu=gpu-cuda${ref_CUDA_MAJOR}-cudnn${CUDNN_MAJOR}
    else
        ref_gpu=cpu
    fi
    if [[ ${WITH_GPU} == "ON"  ]]; then
        install_gpu="_gpu"
    else
        install_gpu=""
    fi
    if [[ ${WITH_MKL} == "ON" ]]; then
        ref_mkl=mkl
    else
        ref_mkl=openblas
    fi

    ref_web=https://paddle-wheel.bj.bcebos.com/${PADDLE_BRANCH}-${ref_gpu}-${ref_mkl}

    ref_paddle2=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp27-cp27mu-linux_x86_64.whl
    ref_paddle35=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp35-cp35m-linux_x86_64.whl
    ref_paddle36=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp36-cp36m-linux_x86_64.whl
    ref_paddle37=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp37-cp37m-linux_x86_64.whl
    ref_paddle38=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp38-cp38-linux_x86_64.whl
    ref_paddle39=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp39-cp39-linux_x86_64.whl

    ref_paddle2_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp27-cp27mu-linux_x86_64.whl
    ref_paddle35_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp35-cp35m-linux_x86_64.whl
    ref_paddle36_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp36-cp36m-linux_x86_64.whl
    ref_paddle37_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp37-cp37m-linux_x86_64.whl
    ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp38-cp38-linux_x86_64.whl
    ref_paddle39_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}-cp39-cp39-linux_x86_64.whl

    if [[ ${PADDLE_BRANCH} != "0.0.0" && ${WITH_MKL} == "ON" && ${WITH_GPU} == "ON" ]]; then
        ref_paddle2=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp27-cp27mu-linux_x86_64.whl
        ref_paddle35=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp35-cp35m-linux_x86_64.whl
        ref_paddle36=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp36-cp36m-linux_x86_64.whl
        ref_paddle37=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp37-cp37m-linux_x86_64.whl
        ref_paddle38=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp38-cp38-linux_x86_64.whl
        ref_paddle39=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp39-cp39-linux_x86_64.whl
        ref_paddle2_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp27-cp27mu-linux_x86_64.whl
        ref_paddle35_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp35-cp35m-linux_x86_64.whl
        ref_paddle36_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp36-cp36m-linux_x86_64.whl
        ref_paddle37_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp37-cp37m-linux_x86_64.whl
        ref_paddle38_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp38-cp38-linux_x86_64.whl
        ref_paddle39_whl=paddlepaddle${install_gpu}-${PADDLE_BRANCH}.post${ref_CUDA_MAJOR}${CUDNN_MAJOR}-cp39-cp39-linux_x86_64.whl
    fi

    #ref_paddle2_mv1=""
    #ref_paddle2_mv2=""
    ref_paddle35_mv1=""
    ref_paddle35_mv2=""
    ref_paddle36_mv1=""
    ref_paddle36_mv2=""
    #ref_paddle37_mv1=""
    #ref_paddle37_mv2=""
    if [[ ${PADDLE_BRANCH} == "0.0.0" && ${WITH_GPU} == "ON" ]]; then
        #ref_paddle2_whl=paddlepaddle_gpu-1.5.1-cp27-cp27mu-linux_x86_64.whl
        ref_paddle35_whl=paddlepaddle_gpu-1.5.1-cp35-cp35m-linux_x86_64.whl
        ref_paddle36_whl=paddlepaddle_gpu-1.5.1-cp36-cp36m-linux_x86_64.whl
        #ref_paddle37_whl=paddlepaddle_gpu-1.5.1-cp37-cp37m-linux_x86_64.whl
        #ref_paddle2_mv1="mv ref_paddle2 paddlepaddle_gpu-1.5.1-cp27-cp27mu-linux_x86_64.whl &&"
        #ref_paddle2_mv2="&& mv paddlepaddle_gpu-1.5.1-cp27-cp27mu-linux_x86_64.whl ref_paddle2"
        ref_paddle35_mv1="mv ${ref_paddle35} ${ref_paddle35_whl} &&"
        ref_paddle35_mv2="&& mv ${ref_paddle35_whl} ${ref_paddle35}"
        ref_paddle36_mv1="mv ${ref_paddle36} ${ref_paddle36_whl} &&"
        ref_paddle36_mv2="&& mv ${ref_paddle36_whl} ${ref_paddle36}"
        #ref_paddle37_mv1="mv ref_paddle37 paddlepaddle_gpu-1.5.1-cp37-cp37m-linux_x86_64.whl &&"
        #ref_paddle37_mv2="&& mv paddlepaddle_gpu-1.5.1-cp37-cp37m-linux_x86_64.whl ref_paddle37"
    fi
    if [[ ${PADDLE_BRANCH} == "0.0.0" && ${WITH_GPU} != "ON" ]]; then
        #ref_paddle2_whl=paddlepaddle_gpu-1.5.1-cp27-cp27mu-linux_x86_64.whl
        ref_paddle35_whl=paddlepaddle-1.5.1-cp35-cp35m-linux_x86_64.whl
        ref_paddle36_whl=paddlepaddle-1.5.1-cp36-cp36m-linux_x86_64.whl
        #ref_paddle37_whl=paddlepaddle_gpu-1.5.1-cp37-cp37m-linux_x86_64.whl
        #ref_paddle2_mv1="mv ref_paddle2 paddlepaddle_gpu-1.5.1-cp27-cp27mu-linux_x86_64.whl &&"
        #ref_paddle2_mv2="&& mv paddlepaddle_gpu-1.5.1-cp27-cp27mu-linux_x86_64.whl ref_paddle2"
        ref_paddle35_mv1="mv ${ref_paddle35} ${ref_paddle35_whl} &&"
        ref_paddle35_mv2="&& mv ${ref_paddle35_whl} ${ref_paddle35}"
        ref_paddle36_mv1="mv ${ref_paddle36} ${ref_paddle36_whl} &&"
        ref_paddle36_mv2="&& mv ${ref_paddle36_whl} ${ref_paddle36}"
        #ref_paddle37_mv1="mv ref_paddle37 paddlepaddle_gpu-1.5.1-cp37-cp37m-linux_x86_64.whl &&"
        #ref_paddle37_mv2="&& mv paddlepaddle_gpu-1.5.1-cp37-cp37m-linux_x86_64.whl ref_paddle37"
    fi
    
    cat > ${PADDLE_ROOT}/build/Dockerfile <<EOF
    FROM ${BASE_IMAGE}
    MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
    ENV HOME /root
EOF

    if [[ ${WITH_GPU} == "ON"  ]]; then
        NCCL_DEPS="apt-get install -y --allow-downgrades --allow-change-held-packages libnccl2=2.4.7-1+cuda${CUDA_MAJOR} libnccl-dev=2.4.7-1+cuda${CUDA_MAJOR} || true"
    else
        NCCL_DEPS="true"
    fi

    if [[ ${WITH_GPU} == "ON" && ${CUDA_MAJOR} = "8.0" ]]; then 
        NCCL_DEPS="apt-get install -y --allow-downgrades --allow-change-held-packages libnccl2=2.2.13-1+cuda8.0 libnccl-dev=2.2.13-1+cuda8.0"
    fi

    PADDLE_VERSION="paddle version"
    CMD='"paddle", "version"'
    
    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y wget python3 python3-pip libgtk2.0-dev dmidecode python3-tk && \
        pip3 install py-cpuinfo==5.0.0 && wget ${ref_web}/${ref_paddle35} && ${ref_paddle35_mv1} pip3 install ${ref_paddle35_whl} ${ref_paddle35_mv2}; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f ${ref_paddle35} && \
        ldconfig
    ${DOCKERFILE_CUDNN_DSO}
    ${DOCKERFILE_CUBLAS_DSO}
    ${DOCKERFILE_GPU_ENV}
EOF
    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev
    RUN mkdir -p /root/python_build/ && wget -q https://www.sqlite.org/2018/sqlite-autoconf-3250300.tar.gz && \
        tar -zxf sqlite-autoconf-3250300.tar.gz && cd sqlite-autoconf-3250300 && \
        ./configure -prefix=/usr/local && make install -j8 && cd ../ && rm sqlite-autoconf-3250300.tar.gz && \
        wget -q https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz && \
        tar -xzf Python-3.6.0.tgz && cd Python-3.6.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/usr/local/ --enable-shared > /dev/null && \
        make -j8 > /dev/null && make altinstall > /dev/null && cd ../ && rm Python-3.6.0.tgz
    RUN apt-get install -y libgtk2.0-dev dmidecode python3-tk && ldconfig && \
        wget ${ref_web}/${ref_paddle36} && ${ref_paddle36_mv1} pip3.6 install ${ref_paddle36_whl} ${ref_paddle36_mv2}; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f ${ref_paddle36} && \
        ldconfig
EOF
    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev
    RUN wget -q https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz && \
        tar -xzf Python-3.7.0.tgz && cd Python-3.7.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/usr/local/ --enable-shared > /dev/null && \
        make -j8 > /dev/null && make altinstall > /dev/null && cd ../ && rm Python-3.7.0.tgz
    RUN apt-get install -y libgtk2.0-dev dmidecode python3-tk && ldconfig && \
        wget ${ref_web}/${ref_paddle37} && pip3.7 install ${ref_paddle37_whl}; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f ${ref_paddle37} && \
        ldconfig
EOF
    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev
    RUN wget -q https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz && \
        tar -xzf Python-3.8.0.tgz && cd Python-3.8.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/usr/local/ --enable-shared > /dev/null && \
        make -j8 > /dev/null && make altinstall > /dev/null && cd ../ && rm Python-3.8.0.tgz
    RUN apt-get install -y libgtk2.0-dev dmidecode python3-tk && ldconfig && \
        wget ${ref_web}/${ref_paddle38} && pip3.8 install ${ref_paddle38_whl}; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f ${ref_paddle38} && \
        ldconfig
EOF
    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev
    RUN wget -q https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz && \
        tar -xzf Python-3.9.0.tgz && cd Python-3.9.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/usr/local/ --enable-shared > /dev/null && \
        make -j8 > /dev/null && make altinstall > /dev/null && cd ../ && rm Python-3.9.0.tgz
    RUN apt-get install -y libgtk2.0-dev dmidecode python3-tk && ldconfig && \
        wget ${ref_web}/${ref_paddle39} && pip3.9 install ${ref_paddle39_whl}; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f ${ref_paddle39} && \
        ldconfig
EOF
    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y wget python-pip python-opencv libgtk2.0-dev dmidecode python-tk && easy_install -U pip && \
        wget ${ref_web}/${ref_paddle2} && pip install ${ref_paddle2_whl}; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f ${ref_paddle2} && \
        ${PADDLE_VERSION} && \
        ldconfig
EOF

    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # default command shows the paddle version and exit
    CMD [${CMD}]
EOF
}

function gen_fluid_lib() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ========================================
    Generating fluid library for train and inference ...
    ========================================
EOF
    parallel_number=`nproc`
    if [[ "$1" != "" ]]; then
      parallel_number=$1
    fi
    startTime_s=`date +%s`
    set +e
    cmake .. -DWITH_DISTRIBUTE=OFF -DON_INFER=ON -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-Auto};build_error=$?

    # reset ccache zero stats for collect PR's actual hit rate
    ccache -z

    make -j ${parallel_number} fluid_lib_dist;build_error=$?
    make -j ${parallel_number} inference_lib_dist;build_error=$?

    # ci will collect ccache hit rate
    collect_ccache_hits

    if [ "$build_error" != 0 ];then
        exit 7;
    fi
    endTime_s=`date +%s`
    echo "Build Time: $[ $endTime_s - $startTime_s ]s"
    echo "ipipe_log_param_Build_Time: $[ $endTime_s - $startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

    build_size "paddle_inference"
    build_size "paddle_inference_c"
}

function tar_fluid_lib() {
    cat <<EOF
    ========================================
    Taring fluid library for train and inference ...
    ========================================
EOF
    cd ${PADDLE_ROOT}/build
    cp -r paddle_install_dir fluid
    tar -czf fluid.tgz fluid
    cp -r paddle_inference_install_dir paddle_inference
    tar -czf paddle_inference.tgz paddle_inference
}

function test_fluid_lib() {
    cat <<EOF
    ========================================
    Testing fluid library for inference ...
    ========================================
EOF
    fluid_startTime_s=`date +%s`
    cd ${PADDLE_ROOT}/paddle/fluid/inference/api/demo_ci
    ./run.sh ${PADDLE_ROOT} ${WITH_MKL:-ON} ${WITH_GPU:-OFF} ${INFERENCE_DEMO_INSTALL_DIR} \
             ${TENSORRT_INCLUDE_DIR:-/usr/local/TensorRT/include} \
             ${TENSORRT_LIB_DIR:-/usr/local/TensorRT/lib}
    EXIT_CODE=$?
    fluid_endTime_s=`date +%s`
    echo "test_fluid_lib Total Time: $[ $fluid_endTime_s - $fluid_startTime_s ]s"
    echo "ipipe_log_param_Test_Fluid_Lib_Total_Time: $[ $fluid_endTime_s - $fluid_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt     
    ./clean.sh
    if [[ "$EXIT_CODE" != "0" ]]; then
        exit 8;
    fi
}

function test_go_inference_api() {
    cat <<EOF
    ========================================
    Testing go inference api ...
    ========================================
EOF

    # ln paddle_inference_c lib
    cd ${PADDLE_ROOT}/build
    ln -s ${PADDLE_ROOT}/build/paddle_inference_c_install_dir/ ${PADDLE_ROOT}/paddle/fluid/inference/goapi/paddle_inference_c

    # run go test
    cd ${PADDLE_ROOT}/paddle/fluid/inference/goapi
    bash test.sh
    EXIT_CODE=$?
    if [[ "$EXIT_CODE" != "0" ]]; then
        exit 8;
    fi
}

function test_fluid_lib_train() {
    cat <<EOF
    ========================================
    Testing fluid library for training ...
    ========================================
EOF
    fluid_train_startTime_s=`date +%s`
    cd ${PADDLE_ROOT}/paddle/fluid/train/demo
    ./run.sh ${PADDLE_ROOT} ${WITH_MKL:-ON}
    EXIT_CODE=$?
    fluid_train_endTime_s=`date +%s`
    echo "test_fluid_lib_train Total Time: $[ $fluid_train_endTime_s - $fluid_train_startTime_s ]s"
    echo "ipipe_log_param_Test_Fluid_Lib_Train_Total_Time: $[ $fluid_train_endTime_s - $fluid_train_startTime_s ]s"
    ./clean.sh
    if [[ "$EXIT_CODE" != "0" ]]; then
        exit 8;
    fi
}


function build_document_preview() {
    sh /paddle/tools/document_preview.sh ${PORT}
}

# origin name: example
function exec_samplecode_test() {
    pip install ${PADDLE_ROOT}/build/python/dist/*.whl
    paddle version
    cd ${PADDLE_ROOT}/tools
    if [ "$1" = "cpu" ] ; then
        python sampcd_processor.py cpu; example_error=$?
    elif [ "$1" = "gpu" ] ; then
        python sampcd_processor.py --threads=16 --full-test gpu; example_error=$?
    fi
    if [ "$example_error" != "0" ];then
      echo "Code instance execution failed" >&2
      exit 5
    fi
}


function collect_ccache_hits() {
    rate=$(ccache -s | grep 'cache hit rate' | awk '{print $4}')
    echo "ccache hit rate: ${rate}%"
    echo "ipipe_log_param_Ccache_Hit_Rate: ${rate}%" >> ${PADDLE_ROOT}/build/build_summary.txt
}


function test_op_benchmark() {
    # The PR will pass quickly when get approval from specific person.
    # Xreki 12538138, luotao1 6836917, Avin0323 16167147
    set +x
    approval_line=$(curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000)
    if [ "${approval_line}" != "" ]; then
        APPROVALS=$(echo ${approval_line} | python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 16167147 12538138 6836917)
        echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
        if [ "${APPROVALS}" == "TRUE" ]; then
            echo "==================================="
            echo -e "\n current pr ${GIT_PR_ID} has got approvals. So, Pass CI directly!\n"
            echo "==================================="
            exit 0
        fi
    fi
    set -x
    bash ${PADDLE_ROOT}/tools/test_op_benchmark.sh
}

function test_model_benchmark() {
    bash ${PADDLE_ROOT}/tools/test_model_benchmark.sh
}

function summary_check_problems() {
    set +x
    local check_style_code=$1
    local example_code=$2
    local check_style_info=$3
    local example_info=$4
    if [ $check_style_code -ne 0 -o $example_code -ne 0 ];then
      echo "========================================"
      echo "summary problems:"
      if [ $check_style_code -ne 0 -a $example_code -ne 0 ];then
        echo "There are 2 errors: Code format error and Example code error."
      else
        [ $check_style_code -ne 0 ] && echo "There is 1 error: Code format error."
        [ $example_code -ne 0 ] && echo "There is 1 error: Example code error."
      fi
      echo "========================================"
      if [ $check_style_code -ne 0 ];then
        echo "*****Code format error***** Please fix it according to the diff information:"
        echo "$check_style_info" | grep "code format error" -A $(echo "$check_style_info" | wc -l)
      fi
      if [ $example_code -ne 0 ];then
        echo "*****Example code error***** Please fix the error listed in the information:"
        echo "$example_info" | grep "API check -- Example Code" -A $(echo "$example_info" | wc -l)
      fi
      [ $check_style_code -ne 0 ] && exit $check_style_code
      [ $example_code -ne 0 ] && exit $example_code
    fi
    set -x
}


function reuse_so_cache() {
    get_html="https://api.github.com/repos/PaddlePaddle/Paddle"
    curl -X GET ${get_html}/commits -H "authorization: token ${GITHUB_API_TOKEN}" >tmp.txt
    merge_commit=`grep "sha" tmp.txt| awk -F \" 'NR==1{print $(NF-1)}'| sed 's# ##g'`
    curl -X GET ${get_html}/commits/${merge_commit} -H "authorization: token ${GITHUB_API_TOKEN}" >tmp.txt
    merge_pr=`grep -oP -m 1 '(#[0-9]*)' tmp.txt| sed 's/#//g'`
    curl -X GET ${get_html}/pulls/${merge_pr}/commits -H "authorization: token ${GITHUB_API_TOKEN}" >tmp.txt
    pr_commit=`grep "sha" tmp.txt |tail -3|head -1|awk -F : '{print $NF}'|sed 's#"##g'|sed 's#,##g'| sed 's# ##g'`
    set +e
    wget -q https://xly-devops.bj.bcebos.com/PR/Paddle/${merge_pr}/${pr_commit}/workspace/Paddle/build/proto_so.tar.gz
    down_proto_so=`echo $?`
    set -e
    if [ "${down_proto_so}" -eq 0 ];then
        export CI_SKIP_CPP_TEST=ON
        cd build && mv ../proto_so.tar.gz .
        tar --use-compress-program=pigz -xpf proto_so.tar.gz
        cmake_gen ${PYTHON_ABI:-""} ${parallel_number}
        cd python
        touch stub.cc
        alias cp=cp
        cp -r ../../python/paddle .
        python setup.py bdist_wheel
    else
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
    fi
}

function main() {
    local CMD=$1 
    local parallel_number=$2
    init
    case $CMD in
      build_only)
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        ;;
      build_and_check)
        set +e
        check_style_info=$(check_style)
        check_style_code=$?
        generate_upstream_develop_api_spec ${PYTHON_ABI:-""} ${parallel_number}
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        check_sequence_op_unittest
        generate_api_spec ${PYTHON_ABI:-""} "PR"
        set +e
        example_info=$(exec_samplecode_test cpu)
        example_code=$?
        summary_check_problems $check_style_code $example_code "$check_style_info" "$example_info"
        assert_api_spec_approvals
        ;;
      build)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        gen_dockerfile ${PYTHON_ABI:-""}
        assert_api_spec_approvals
        ;;
      combine_avx_noavx)
        combine_avx_noavx_build
        gen_dockerfile ${PYTHON_ABI:-""}
        ;;
      combine_avx_noavx_build_and_test)
        combine_avx_noavx_build
        gen_dockerfile ${PYTHON_ABI:-""}
        parallel_test_base
        ;;
      test)
        parallel_test
        ;;
      single_test)
        single_test $2
        ;;
      bind_test)
        bind_test
        ;;
      gen_doc_lib)
        gen_doc_lib $2
        ;;
      dockerfile)
        gen_dockerfile ${PYTHON_ABI:-""}
        ;;
      fluid_inference_lib)
        cmake_gen ${PYTHON_ABI:-""}
        gen_fluid_lib ${parallel_number}
        tar_fluid_lib
        test_fluid_lib
        ;;
      build_inference_lib)
        cmake_gen ${PYTHON_ABI:-""}
        gen_fluid_lib ${parallel_number}
        ;;
      check_style)
        check_style
        ;;
      cicheck)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        enable_unused_var_check
        parallel_test
        ;;
      cicheck_coverage)
        check_approvals_of_unittest 1
        check_diff_file_for_coverage
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        enable_unused_var_check
        parallel_test
        check_coverage
        check_change_of_unittest ${PYTHON_ABI:-""}
        ;;
      cpu_cicheck_coverage)
        check_approvals_of_unittest 1
        check_diff_file_for_coverage
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        enable_unused_var_check
        ;;
      gpu_cicheck_coverage)
        parallel_test
        check_coverage
        check_change_of_unittest ${PYTHON_ABI:-""}
        ;;
      ci_preciseTest)
        insert_pile_to_h_cu_diff 
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        enable_unused_var_check
        get_precise_tests_map_file
        ;;
      cicheck_brpc)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        run_brpc_test
        ;;
      assert_api)
        generate_upstream_develop_api_spec ${PYTHON_ABI:-""} ${parallel_number}
        assert_api_spec_approvals
        ;;
      test_inference)
        PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
        python ${PADDLE_ROOT}/tools/remove_grad_op_and_kernel.py
        gen_fluid_lib ${parallel_number}
        test_fluid_lib
        #test_fluid_lib_train
        #go inference test
        test_go_inference_api
        ;;
      test_train)
        gen_fluid_lib ${parallel_number}
        test_fluid_lib_train
        ;;
      assert_api_approvals)
        assert_api_spec_approvals
        ;;
      assert_file_approvals)
        assert_file_diff_approvals
        ;; 
      maccheck)
        cmake_gen_and_build_mac ${PYTHON_ABI:-""}
        run_mac_test ${PYTHON_ABI:-""} ${PROC_RUN:-1}
        ;;
      maccheck_py35)
        cmake_gen_and_build_mac ${PYTHON_ABI:-""}
        run_mac_test ${PYTHON_ABI:-""} ${PROC_RUN:-1}
        check_change_of_unittest ${PYTHON_ABI:-""}
        ;;
      macbuild)
        cmake_gen ${PYTHON_ABI:-""}
        build_mac
        ;;
      cicheck_py35)
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        parallel_test
        ;;
      check_xpu)
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        parallel_test
        ;;
      check_xpu_coverage)
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        parallel_test
        check_coverage
        ;;
      check_rocm_coverage)
        cmake_gen_and_build ${PYTHON_ABI:-""} ${parallel_number}
        parallel_test
        check_coverage
        ;;
      reuse_so_cicheck_py35)
        reuse_so_cache
        parallel_test
        ;;
      cmake_gen)
        cmake_gen ${PYTHON_ABI:-""}
        ;;
      cmake_gen_in_current_dir)
        cmake_gen_in_current_dir ${PYTHON_ABI:-""}
        ;;
      gen_fluid_lib)
        gen_fluid_lib ${parallel_number}
        ;;
      test_fluid_lib)
        test_fluid_lib
        ;;
      document)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        build_document_preview
        ;;
      api_example)
        example_info=$(exec_samplecode_test cpu)
        example_code=$?
        check_style_code=0
        check_style_info=
        summary_check_problems $check_style_code $example_code "$check_style_info" "$example_info"
        ;;
      test_op_benchmark)
        test_op_benchmark
        ;;
      test_model_benchmark)
        test_model_benchmark
        ;;
      *)
        print_usage
        exit 1
        ;;
      esac
      set +x
      if [[ -f ${PADDLE_ROOT}/build/build_summary.txt ]];then
        echo "=====================build summary======================"
        cat ${PADDLE_ROOT}/build/build_summary.txt
        echo "========================================================"
      fi
      echo "paddle_build script finished as expected"
}

main $@
