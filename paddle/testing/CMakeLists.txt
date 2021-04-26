# for paddle test case

if(WITH_TESTING)
  cc_library(paddle_gtest_main SRCS paddle_gtest_main.cc DEPS device_context memory gtest gflags npu_instance)
endif()
