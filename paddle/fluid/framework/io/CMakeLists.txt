cc_library(fs SRCS fs.cc DEPS string_helper glog boost)
cc_library(shell SRCS shell.cc DEPS string_helper glog timer enforce)
