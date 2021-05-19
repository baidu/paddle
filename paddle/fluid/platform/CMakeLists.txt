proto_library(profiler_proto SRCS profiler.proto DEPS framework_proto simple_threadpool)
proto_library(error_codes_proto SRCS error_codes.proto)
if(WITH_GPU)
  proto_library(cuda_error_proto SRCS cuda_error.proto)
endif(WITH_GPU)

if(WITH_XPU)
  set(XPU_CTX_DEPS xpulib ssl crypto rt z resolv dl)
ELSE()
  set(XPU_CTX_DEPS)
endif(WITH_XPU)

if(WITH_ASCEND)
    set(ASCEND_DEPS xpulib)
ELSE()
  set(ASCEND_DEPS)
endif(WITH_ASCEND)

if (WITH_PYTHON)
  py_proto_compile(profiler_py_proto SRCS profiler.proto)
  add_custom_target(profiler_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)
  add_dependencies(profiler_py_proto profiler_py_proto_init)

  if (NOT WIN32)
    add_custom_command(TARGET profiler_py_proto POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMAND cp *.py ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMENT "Copy generated python proto into directory paddle/fluid/proto/profiler."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else(NOT WIN32)
    string(REPLACE "/" "\\" proto_dstpath "${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler/")
    add_custom_command(TARGET profiler_py_proto POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMAND copy /Y *.py ${proto_dstpath}
        COMMENT "Copy generated python proto into directory paddle/fluid/proto/profiler."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif(NOT WIN32)
endif()

cc_library(flags SRCS flags.cc DEPS gflags)
cc_library(denormal SRCS denormal.cc DEPS)

cc_library(errors SRCS errors.cc DEPS error_codes_proto)
cc_test(errors_test SRCS errors_test.cc DEPS errors enforce)

set(enforce_deps flags errors boost)
if(WITH_GPU)
  set(enforce_deps ${enforce_deps} cuda_error_proto)
endif()
cc_library(enforce INTERFACE SRCS enforce.cc DEPS ${enforce_deps})
cc_library(monitor SRCS monitor.cc)
cc_test(enforce_test SRCS enforce_test.cc DEPS stringpiece enforce)

set(CPU_INFO_DEPS gflags glog enforce)
IF(WITH_XBYAK)
    list(APPEND CPU_INFO_DEPS xbyak)
ENDIF()
cc_library(cpu_info SRCS cpu_info.cc DEPS ${CPU_INFO_DEPS})
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info)

IF(WITH_GPU)
    nv_library(gpu_info SRCS gpu_info.cc DEPS gflags glog enforce monitor dynload_cuda)
ENDIF()
IF(WITH_ROCM)
    hip_library(gpu_info SRCS gpu_info.cc DEPS gflags glog enforce monitor dynload_cuda)
ENDIF()

cc_library(place SRCS place.cc DEPS enforce boost)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

if(WITH_XPU)
cc_library(xpu_info SRCS xpu_info.cc DEPS gflags glog enforce xpulib)
endif()

if(WITH_ASCEND)
    cc_library(ascend_npu_info SRCS ascend_npu_info.cc DEPS gflags glog enforce atlas_acl)
endif()

if(WITH_ASCEND_CL)
    cc_library(npu_info SRCS npu_info.cc DEPS gflags glog enforce monitor ascendcl acl_op_compiler)
endif()

add_subdirectory(dynload)
add_subdirectory(stream)

cc_library(cpu_helper SRCS cpu_helper.cc DEPS cblas enforce)
cc_test(cpu_helper_test SRCS cpu_helper_test.cc DEPS cpu_helper)

set(dgc_deps "")
IF(WITH_DGC)
    set(dgc_deps dgc)
ENDIF()

IF(WITH_GPU OR WITH_ROCM)
    set(GPU_CTX_DEPS dynload_cuda dynamic_loader cuda_stream)
ENDIF()

IF(WITH_ASCEND_CL)
    set(NPU_CTX_DEPS npu_stream npu_info)
ENDIF()

IF(WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
ELSE()
    set(MKLDNN_CTX_DEPS)
ENDIF()

IF(WITH_ASCEND_CL)
cc_library(stream_callback_manager SRCS stream_callback_manager.cc DEPS simple_threadpool enforce)
ENDIF()

IF(WITH_GPU)
    nv_library(stream_callback_manager SRCS stream_callback_manager.cc DEPS simple_threadpool enforce)
ENDIF()
IF(WITH_ROCM)
    hip_library(stream_callback_manager SRCS stream_callback_manager.cc DEPS simple_threadpool enforce)
ENDIF()

IF(WITH_GPU OR WITH_ROCM)
  set(STREAM_CALLBACK_DEPS stream_callback_manager)
ELSEIF(WITH_ASCEND_CL)
  set(STREAM_CALLBACK_DEPS stream_callback_manager)
ELSE()
  set(STREAM_CALLBACK_DEPS)
ENDIF()

if(WITH_GLOO)
    cc_library(gloo_context SRCS gloo_context.cc DEPS framework_proto gloo_wrapper enforce)
endif()

cc_library(cudnn_workspace_helper SRCS cudnn_workspace_helper.cc DEPS boost)

# memcpy depends on device_context, here add deps individually for
# avoiding cycle dependencies
cc_library(device_context SRCS device_context.cc init.cc DEPS simple_threadpool malloc xxhash ${STREAM_CALLBACK_DEPS}
    place eigen3 stringpiece cpu_helper cpu_info framework_proto ${GPU_CTX_DEPS} ${NPU_CTX_DEPS} ${MKLDNN_CTX_DEPS}
    ${dgc_deps} dlpack cudnn_workspace_helper ${XPU_CTX_DEPS})

cc_library(collective_helper SRCS collective_helper.cc collective_helper_npu.cc gen_comm_id_helper.cc DEPS framework_proto  device_context enforce)

if(WITH_GPU OR WITH_ROCM)
    cc_library(cuda_resource_pool SRCS cuda_resource_pool.cc DEPS gpu_info)
    target_link_libraries(device_context cuda_resource_pool)
endif()

if(WITH_ASCEND_CL)
    cc_library(npu_resource_pool SRCS npu_resource_pool.cc DEPS npu_info)
    target_link_libraries(device_context npu_resource_pool)
endif()

cc_test(init_test SRCS init_test.cc DEPS device_context)

if(WITH_GPU)
  nv_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info)
  nv_test(cudnn_helper_test SRCS cudnn_helper_test.cc DEPS dynload_cuda)
  nv_test(cudnn_desc_test SRCS cudnn_desc_test.cc DEPS dynload_cuda)
  nv_test(transform_test SRCS transform_test.cu DEPS memory place device_context)
endif()

if(WITH_ROCM)
  hip_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info)
  hip_test(miopen_helper_test SRCS miopen_helper_test.cc DEPS dynload_cuda)
  hip_test(cudnn_desc_test SRCS cudnn_desc_test.cc DEPS dynload_cuda tensor)
  hip_test(transform_test SRCS transform_test.cu DEPS memory place device_context)
endif()

cc_library(timer SRCS timer.cc)
cc_test(timer_test SRCS timer_test.cc DEPS timer)

cc_library(lodtensor_printer SRCS lodtensor_printer.cc DEPS ddim place tensor scope lod_tensor variable_helper framework_proto)
cc_test(lodtensor_printer_test SRCS lodtensor_printer_test.cc DEPS lodtensor_printer)

cc_library(device_tracer SRCS device_tracer.cc DEPS boost profiler_proto framework_proto ${GPU_CTX_DEPS})
if(WITH_GPU)
  nv_library(profiler SRCS profiler.cc profiler.cu DEPS device_tracer gpu_info enforce dynload_cuda)
  nv_test(cuda_helper_test SRCS cuda_helper_test.cu)
  nv_library(device_memory_aligment SRCS device_memory_aligment.cc DEPS cpu_info gpu_info place)
elseif(WITH_ROCM)
  hip_library(profiler SRCS profiler.cc profiler.cu DEPS device_tracer gpu_info enforce)
  hip_test(cuda_helper_test SRCS cuda_helper_test.cu)
  hip_library(device_memory_aligment SRCS device_memory_aligment.cc DEPS cpu_info gpu_info place)
else()
  cc_library(profiler SRCS profiler.cc DEPS device_tracer enforce)
  cc_library(device_memory_aligment SRCS device_memory_aligment.cc DEPS cpu_info place)
endif()

cc_test(profiler_test SRCS profiler_test.cc DEPS profiler)
cc_test(float16_test SRCS float16_test.cc DEPS lod_tensor)
cc_test(bfloat16_test SRCS bfloat16_test.cc DEPS lod_tensor)
cc_test(complex_test SRCS complex_test.cc DEPS lod_tensor)

IF(WITH_GPU)
  nv_test(float16_gpu_test SRCS float16_test.cu DEPS lod_tensor)
  nv_test(bfloat16_gpu_test SRCS bfloat16_test.cu DEPS lod_tensor)
  nv_test(complex_gpu_test SRCS complex_test.cu DEPS lod_tensor)
  nv_test(test_limit_gpu_memory SRCS test_limit_gpu_memory.cu DEPS gpu_info flags)
  nv_library(cuda_device_guard SRCS cuda_device_guard.cc DEPS gpu_info)
ENDIF()

IF(WITH_ROCM)
  hip_test(float16_gpu_test SRCS float16_test.cu DEPS lod_tensor)
  hip_test(test_limit_gpu_memory SRCS test_limit_gpu_memory.cu DEPS gpu_info flags)
  hip_library(cuda_device_guard SRCS cuda_device_guard.cc DEPS gpu_info)
ENDIF()

if(NOT APPLE AND NOT WIN32)
  cc_library(device_code SRCS device_code.cc DEPS device_context)
  if(WITH_GPU OR WITH_ROCM)
    cc_test(device_code_test SRCS device_code_test.cc DEPS device_code lod_tensor)
  endif()
endif()
