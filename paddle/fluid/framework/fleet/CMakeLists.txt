if(WITH_PSLIB)
    if(WITH_PSLIB_BRPC)
        set(BRPC_DEPS pslib_brpc)
    else()
        set(BRPC_DEPS brpc)
    endif(WITH_PSLIB_BRPC)
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper layer scope ${BRPC_DEPS} pslib)
else()
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper layer scope)
endif(WITH_PSLIB)

if(WITH_HETERPS)
    if(WITH_NCCL)
        nv_library(ps_gpu_wrapper SRCS ps_gpu_wrapper.cu ps_gpu_wrapper.cc
        DEPS heter_ps ${BRPC_DEPS})
        add_subdirectory(heter_ps)
    elseif(WITH_RCCL)
        hip_library(ps_gpu_wrapper SRCS ps_gpu_wrapper.cu ps_gpu_wrapper.cc
        DEPS heter_ps ${BRPC_DEPS})
        add_subdirectory(heter_ps)
    endif(WITH_NCCL)
else()
    cc_library(ps_gpu_wrapper SRCS ps_gpu_wrapper.cc)
endif(WITH_HETERPS)

if(WITH_NCCL OR WITH_RCCL)
    cc_library(nccl_wrapper SRCS nccl_wrapper.cc DEPS framework_proto variable_helper scope)
endif()
if(WITH_BOX_PS)
    if(WITH_GPU)
        nv_library(box_wrapper SRCS box_wrapper.cc box_wrapper.cu DEPS framework_proto lod_tensor box_ps)
    endif()
    if(WITH_ROCM)
        hip_library(box_wrapper SRCS box_wrapper.cc box_wrapper.cu DEPS framework_proto lod_tensor box_ps)
    endif()
else()
    cc_library(box_wrapper SRCS box_wrapper.cc DEPS framework_proto lod_tensor)
endif(WITH_BOX_PS)


if(WITH_GLOO)
    cc_library(gloo_wrapper SRCS gloo_wrapper.cc DEPS framework_proto variable_helper scope gloo)
else()
    cc_library(gloo_wrapper SRCS gloo_wrapper.cc DEPS framework_proto variable_helper scope)
endif(WITH_GLOO)

if(WITH_PSLIB)
set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS
            "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()
set_source_files_properties(heter_wrapper.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()

cc_library(heter_wrapper SRCS heter_wrapper.cc DEPS framework_proto
device_context heter_service_proto ${BRPC_DEPS})

cc_test(test_fleet_cc SRCS test_fleet.cc DEPS fleet_wrapper gloo_wrapper fs shell)

if(WITH_ASCEND OR WITH_ASCEND_CL)
    cc_library(ascend_wrapper SRCS ascend_wrapper.cc DEPS framework_proto lod_tensor ascend_ge ascend_graph)
endif()
