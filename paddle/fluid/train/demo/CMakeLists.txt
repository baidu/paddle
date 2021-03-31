cmake_minimum_required(VERSION 3.0)

project(cpp_train_demo CXX C)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (NOT DEFINED PADDLE_SRC)
  set(PADDLE_SRC "/home/yiak/WorkSpace/Github/Paddle")
  message(WARNING "please set PADDLE_SRC with -DPADDLE_SRC, defaults to ${PADDLE_SRC}")
endif()
if(NOT DEFINED PADDLE_LIB)
  set(PADDLE_LIB "${PADDLE_SRC}/build")
  message(WARNING "please set PADDLE_LIB with -DPADDLE_LIB, defaults to ${PADDLE_LIB}")
endif()

option(WITH_MKLDNN     "Compile PaddlePaddle with MKLDNN"                                   OFF)
option(WITH_MKL        "Compile PaddlePaddle with MKL support, default use openblas."       OFF)

execute_process(COMMAND find ${PADDLE_LIB}/third_party/threadpool
                             ${PADDLE_LIB}/third_party/dlpack
        -type f -name "*.h" -exec dirname {} \; OUTPUT_VARIABLE THIRD_INCR OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "THIRD_INCR : ${THIRD_INCR}")

string(REPLACE "\n" " " THIRD_INCR ${THIRD_INCR})
message(STATUS "THIRD_INCR : ${THIRD_INCR}")
include_directories("${THIRD_INCR}")
# include Eigen 3
include_directories("${PADDLE_SRC}/build/third_party/eigen3/src/extern_eigen3/")

include_directories("${PADDLE_SRC}")
include_directories("${PADDLE_LIB}")
message(STATUS "PADDLE_LIB ${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")

include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")
include_directories("${PADDLE_LIB}/third_party/threadpool")
include_directories("${PADDLE_LIB}/third_party/dlpack")

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

add_executable(demo_trainer demo_trainer.cc)

if(WITH_MKLDNN)
  add_definitions(-DPADDLE_WITH_MKLDNN)
  include_directories("${PADDLE_LIB}/third_party/install/mkldnn/include")
  if(WIN32)
    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.lib)
  else(WIN32)
    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/libmkldnn.so)
  endif(WIN32)
endif(WITH_MKLDNN)

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  if(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib)
  else(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel.so)
  endif(WIN32)
else()
  if(APPLE)
    set(MATH_LIB cblas)
  elseif(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.lib)
  else()
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.a)
  endif(APPLE)
endif()

if(APPLE)
  set(MACOS_LD_FLAGS "-undefined dynamic_lookup -Wl,-all_load -framework CoreFoundation -framework Security")
else(APPLE)
  set(ARCHIVE_START "-Wl,--whole-archive")
  set(ARCHIVE_END "-Wl,--no-whole-archive")
  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
endif(APPLE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

set_target_properties(demo_trainer PROPERTIES LINKER_LANGUAGE CXX)

# this line does not work as expected
# target_include_directories(demo_trainer PUBLIC "${THIRD_INCR}")

target_include_directories(demo_trainer PUBLIC ${PADDLE_LIB}/third_party/threadpool/src/extern_threadpool)
target_include_directories(demo_trainer PUBLIC ${PADDLE_LIB}/third_party/dlpack/src/extern_dlpack/include/)
target_link_libraries(demo_trainer
        ${MACOS_LD_FLAGS}
        ${ARCHIVE_START}
        ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_inference.so
        ${ARCHIVE_END}
        ${MATH_LIB}
        ${MKLDNN_LIB}
        gflags glog protobuf z xxhash
        ${EXTERNAL_LIB})
