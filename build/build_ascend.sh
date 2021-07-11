#!/bin/bash

export CC=/usr/local/gcc-8.2/bin/gcc
export CXX=/usr/local/gcc-8.2/bin/g++

unset GREP_OPTIONS
branch=`git branch | grep \* | cut -d ' ' -f2`

place=ascend
rpc=none
testing=None
os=eluer
py_v=3.7.5
PADDLE_ROOT=/home/gongwb/go/src/github.com/Paddle/


if grep --quiet "Ubuntu" /etc/issue; then
  os=ubuntu
  build_type=debug
elif grep --quiet "CentOS" /etc/issue; then
  os=cent
  build_type=release
  if [[ $testing == 'None' ]]; then
      testing=y
  fi
else
  echo "Unrecognized OS, use ubuntu config"
  os=ubuntu
  build_type=debug
fi

build_type=release

if [[ $testing == 'None' ]]; then
  testing=y
fi

while true ; do
  case "$1" in
    -b) branch="$2" ; shift 2 ;;
    -p) place="$2" ; shift 2 ;;
    -t) testing="$2" ; shift 2 ;;
    -build_type) build_type="$2" ; shift 2 ;;
    -rpc) rpc="$2" ; shift 2 ;;
    -os) os="$2" ; shift 2 ;;
    -py_v) py_v="$2" ; shift 2 ;;
    *) 
       if [[ ${#1} > 0 ]]; then
          echo "not supported arugments ${1}" ; exit 1 ;
       else
           break
       fi 
       ;;
  esac
done

echo "user py_v:${py_v}"

echo "branch=${branch}"
echo "place=${place}"
echo "build_type=${build_type}"
echo "rpc=${rpc}"
echo "testing=${testing}"

#[[ $place == "gpu" || $place == "cpu" ]] || echo "not support ${place}" ; exit 1 ;
#[[ $testing == "yes" || $testing == "no" ]] || echo "not support ${testing}" ; exit 1 ;
#[[ $build_type == "release" || $build_type == "debug" ]] || echo "not support ${build_type}" ; exit 1 ;
#[[ $rpc == "brpc" || $rpc == "grpc" || $rpc == "brpcrdma" ]] || echo "not support ${rpc}" ; exit 1 ;

echo "last"

case "$place" in
    gpu) WITH_GPU=ON ;;
    cpu) WITH_GPU=OFF ;;
    ascend) WITH_GPU=OFF ;;
    *) echo "not support ${place}" ; exit 1 ;;
esac

case "$testing" in
    y) WITH_TESTING=ON ;;
    n) WITH_TESTING=OFF ;;
    *) echo "not support ${testing}" ; exit 1 ;;
esac

case "$build_type" in
    release) CMAKE_BUILD_TYPE=Release ;;
    debug) CMAKE_BUILD_TYPE=RelWithDebInfo ;;
    *) echo "not support ${build_type}" ; exit 1 ;;
esac

case "$rpc" in
    brpc) WITH_GRPC=OFF ;;
    grpc) WITH_GRPC=ON ;;
    rdma) WITH_GRPC=OFF ; WITH_BRPC_RDMA=ON ;;
    none) WITH_GRPC=OFF ; WITH_BRPC_RDMA=OFF ;;
    *) echo "not support ${rpc}" ; exit 1 ;;
esac

case "$os" in
    cent) 
        if [[ $py_v == "2.7" ]]; then
            export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
            export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
            PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python"
            pip install -r ${PADDLE_ROOT}/python/requirements.txt
        elif [[ $py_v == "3.5.1" ]]; then
		    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
            export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.5.1/bin/python3 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.5.1/include/python3.5m -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.5.1/lib/libpython3.so"
        elif [[ $py_v == "3.6.0" ]]; then
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
            export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.6.0/bin/python3 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.6.0/include/python3.6m -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.6.0/lib/libpython3.so"
        elif [[ $py_v == "3.7.0" ]]; then
		    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
            export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7m -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.so"
	    else
            echo "can't find the ${py_v} support"
            exit 1
	    fi
        ;;
    ubuntu) ;;
    euler) ;;
    *) echo "not support ${os}" ; exit 1 ;;
esac

build_dir=build_${os}_${branch}_${build_type}_${place}_${testing}_${rpc}_${py_v}
mkdir -p  ${build_dir}
cd ${build_dir}
third_party_dir=${os}_${build_type}_${place}_${py_v}

set -x
#cmake ../../  -DTHIRD_PARTY_PATH=/paddle/build/third_party/${third_party_dir}/ \
#         -DWITH_MKLML=ON \
#         -DWITH_MKLDNN=ON \
#         -DWITH_GPU=${WITH_GPU:-ON} \
#         -DWITH_C_API=OFF \
#         -DWITH_DISTRIBUTE=ON \
#         -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-RelWithDebInfo} \
#         -DWITH_TESTING=${WITH_TESTING:-ON} \
#         -DWITH_STYLE_CHECK=OFF \
#         -DWITH_CONTRIB=OFF \
#         -DWITH_GRPC=${WITH_GRPC:-ON} \
#         -DWITH_BRPC_RDMA=${WITH_BRPC_RDMA:-OFF} \
#         -DWITH_FLUID_ONLY=ON \
#         -DWITH_INFERENCE_API_TEST=OFF \
#         -DCMAKE_INSTALL_PREFIX=/root/paddlebuild/${third_party_dir}/install \
#         -DWITH_DOC=ON \
#         -DPY_VERSION=${py_v} \
#         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#         -DCUDA_ARCH_NAME=Volta
#         #-DWITH_BOX_PS=ON
#         #-DWITH_GLOO=True 

cmake  ../../  -DTHIRD_PARTY_PATH=${PADDLE_ROOT}/build/third_party/${third_party_dir}/ \
    -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` \
    -DWITH_ARM=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF \
    -DWITH_ASCEND=OFF -DWITH_ASCEND_CL=ON -DWITH_PSCORE=OFF -DWITH_DISTRIBUTE=ON -DWITH_TESTING=${WITH_TESTING:-ON}

set +x

#if [[ $os == "cent" ]]; then
    #cd $build_dir
    make TARGET=ARMV8 -j 30
#fi

