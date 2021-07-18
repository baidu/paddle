if(WITH_PSLIB)
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope pslib_brpc pslib)
    cc_library(tree_wrapper SRCS tree_wrapper.cc DEPS framework_proto
    variable_helper scope pslib_brpc pslib boost)
else()
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope)
endif(WITH_PSLIB)

if(WITH_NCCL)
    cc_library(nccl_wrapper SRCS nccl_wrapper.cc DEPS framework_proto variable_helper scope)
endif()
if(WITH_BOX_PS)
    nv_library(box_wrapper SRCS box_wrapper.cc box_wrapper.cu DEPS framework_proto lod_tensor box_ps)
else()
    cc_library(box_wrapper SRCS box_wrapper.cc DEPS framework_proto lod_tensor)
endif(WITH_BOX_PS)

if(WITH_GLOO)
    cc_library(gloo_wrapper SRCS gloo_wrapper.cc DEPS framework_proto variable_helper scope gloo)
else()
    cc_library(gloo_wrapper SRCS gloo_wrapper.cc DEPS framework_proto variable_helper scope)
endif(WITH_GLOO)

cc_test(test_fleet SRCS test_fleet.cc DEPS fleet_wrapper gloo_wrapper fs shell)
