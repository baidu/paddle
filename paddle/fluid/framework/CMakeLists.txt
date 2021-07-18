
#windows treat symbolic file as a real file, which is different with unix
#We create a hidden file and compile it instead of origin source file.
function(windows_symbolic TARGET)
  set(oneValueArgs "")
  set(multiValueArgs SRCS PATH)
  cmake_parse_arguments(windows_symbolic "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(final_path ${CMAKE_CURRENT_SOURCE_DIR}/${windows_symbolic_PATH})
  foreach(src ${windows_symbolic_SRCS})
    get_filename_component(src ${src} NAME_WE)
    if (NOT EXISTS ${final_path}/${src}.cc OR NOT EXISTS ${final_path}/${src}.cu)
        message(FATAL " ${src}.cc and ${src}.cu must exsits, and ${src}.cu must be symbolic file.")
    endif()

    file(GENERATE OUTPUT ${final_path}/.${src}.cu INPUT ${final_path}/${src}.cc)

    add_custom_command(OUTPUT ${final_path}/.${src}.cu
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${final_path}/${src}.cc" "${final_path}/.${src}.cu"
            COMMENT "create hidden file of ${src}.cu")
    add_custom_target(${TARGET} ALL DEPENDS .${src}.cu)
  endforeach()
endfunction()

add_subdirectory(ir)
add_subdirectory(details)
add_subdirectory(fleet)
add_subdirectory(io)
#ddim lib
proto_library(framework_proto SRCS framework.proto)
proto_library(data_feed_proto SRCS data_feed.proto)
proto_library(trainer_desc_proto SRCS trainer_desc.proto DEPS framework_proto
  data_feed_proto)

cc_library(ddim SRCS ddim.cc DEPS eigen3 boost enforce)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)
cc_test(unroll_array_ops_test SRCS unroll_array_ops_test.cc)
cc_library(data_type SRCS data_type.cc DEPS framework_proto ddim device_context)
cc_test(data_type_test SRCS data_type_test.cc DEPS data_type place tensor)
if(WITH_GPU)
  if (WIN32)
    windows_symbolic(tensor_util SRCS tensor_util.cu)
    nv_library(tensor SRCS tensor.cc .tensor_util.cu DEPS place memory data_type device_context)
    add_dependencies(tensor tensor_util)
  else()
    nv_library(tensor SRCS tensor.cc tensor_util.cu DEPS place memory data_type device_context profiler)
  endif(WIN32)
else()
  cc_library(tensor SRCS tensor.cc tensor_util.cc DEPS place memory data_type device_context profiler)
endif()

cc_test(tensor_test SRCS tensor_test.cc DEPS tensor)
if(WITH_GPU)
  nv_test(tensor_util_test SRCS tensor_util_test.cc tensor_util_test.cu DEPS tensor dlpack_tensor)
else()
  cc_test(tensor_util_test SRCS tensor_util_test.cc DEPS tensor dlpack_tensor)
endif()

cc_test(copy_same_tensor_test SRCS copy_same_tensor_test.cc DEPS tensor)

cc_test(eigen_test SRCS eigen_test.cc DEPS tensor)

if(WITH_GPU)
  nv_test(mixed_vector_test SRCS mixed_vector_test.cc mixed_vector_test.cu DEPS place memory device_context tensor)
else()
  cc_test(mixed_vector_test SRCS mixed_vector_test.cc DEPS place memory device_context tensor)
endif()
cc_library(lod_tensor SRCS lod_tensor.cc DEPS ddim place tensor framework_proto version)
cc_library(device_worker SRCS device_worker.cc DEPS trainer_desc_proto lod_tensor)

cc_test(lod_tensor_test SRCS lod_tensor_test.cc DEPS lod_tensor memory)
nv_test(lod_tensor_gpu_test SRCS lod_tensor_test.cu DEPS lod_tensor)
cc_test(device_worker_test SRCS device_worker_test.cc DEPS device_worker)

cc_library(garbage_collector SRCS garbage_collector.cc DEPS device_context memory gflags glog)

cc_library(reader SRCS reader.cc DEPS lod_tensor ddim)
cc_test(reader_test SRCS reader_test.cc DEPS reader)

cc_library(threadpool SRCS threadpool.cc DEPS enforce)
cc_test(threadpool_test SRCS threadpool_test.cc DEPS threadpool)

cc_library(var_type_traits SRCS var_type_traits DEPS lod_tensor selected_rows framework_proto)
if (WITH_GPU)
  target_link_libraries(var_type_traits dynload_cuda)
endif()
cc_test(var_type_traits_test SRCS var_type_traits_test.cc DEPS var_type_traits)

cc_library(scope SRCS scope.cc DEPS glog threadpool xxhash var_type_traits)

cc_library(scope_pool SRCS scope_pool.cc DEPS scope)
cc_test(scope_test SRCS scope_test.cc DEPS scope)
cc_test(variable_test SRCS variable_test.cc DEPS tensor var_type_traits)

cc_library(data_device_transform SRCS data_device_transform.cc DEPS tensor)
nv_test(data_device_transform_test SRCS data_device_transform_test.cu
        DEPS operator op_registry device_context math_function scope)

if(WITH_GPU)
  if (WIN32)
#windows treat symbolic file as a real file, which is different with unix
#We create a hidden file and compile it instead of origin source file.
      windows_symbolic(hidden_file SRCS data_type_transform.cu)
      nv_library(data_type_transform SRCS .data_type_transform.cu DEPS tensor)
      add_dependencies(data_type_transform hidden_file)
  else()
      nv_library(data_type_transform SRCS data_type_transform.cu DEPS tensor)
  endif(WIN32)
  nv_test(data_type_transform_test SRCS data_type_transform_test.cc data_type_transform_test.cu DEPS data_type_transform)
else()
  cc_library(data_type_transform SRCS data_type_transform.cc DEPS tensor)
  cc_test(data_type_transform_test SRCS data_type_transform_test.cc DEPS data_type_transform)
endif()

cc_library(data_layout_transform SRCS data_layout_transform.cc DEPS tensor math_function)
cc_test(data_layout_transform_test SRCS data_layout_transform_test.cc DEPS data_layout_transform)

cc_library(data_transform SRCS data_transform.cc DEPS math_function tensor
        framework_proto selected_rows data_device_transform data_type_transform data_layout_transform)

cc_library(attribute SRCS attribute.cc DEPS framework_proto boost)
cc_test(program_desc_test SRCS program_desc_test.cc DEPS proto_desc
device_context)
cc_library(op_proto_maker SRCS op_proto_maker.cc DEPS framework_proto attribute glog)
cc_test(op_proto_maker_test SRCS op_proto_maker_test.cc DEPS op_proto_maker)
cc_library(no_need_buffer_vars_inference SRCS no_need_buffer_vars_inference.cc DEPS attribute device_context)
cc_library(op_info SRCS op_info.cc DEPS attribute framework_proto no_need_buffer_vars_inference)
cc_library(shape_inference SRCS shape_inference.cc DEPS ddim attribute device_context)

cc_test(no_need_buffer_vars_inference_test SRCS no_need_buffer_vars_inference_test.cc DEPS no_need_buffer_vars_inference layer)

cc_library(transfer_scope_cache SRCS transfer_scope_cache.cc DEPS scope framework_proto device_context)
cc_library(op_kernel_type SRCS op_kernel_type.cc DEPS device_context place)

cc_library(unused_var_check SRCS unused_var_check.cc DEPS glog no_need_buffer_vars_inference)

cc_library(operator SRCS operator.cc DEPS op_info device_context tensor scope glog trainer_desc_proto data_feed_proto
    shape_inference data_transform lod_tensor profiler transfer_scope_cache op_kernel_type op_call_stack unused_var_check nan_inf_utils)

cc_test(operator_test SRCS operator_test.cc DEPS operator op_registry device_context)
cc_test(operator_exception_test SRCS operator_exception_test.cc DEPS operator op_registry device_context)

cc_library(version SRCS version.cc)
cc_test(version_test SRCS version_test.cc DEPS version)

cc_library(proto_desc SRCS var_desc.cc op_desc.cc block_desc.cc program_desc.cc DEPS shape_inference op_info operator glog version)

cc_library(op_registry SRCS op_registry.cc DEPS op_proto_maker op_info operator glog proto_desc)

cc_library(op_call_stack SRCS op_call_stack.cc DEPS op_proto_maker enforce)
cc_test(op_call_stack_test SRCS op_call_stack_test.cc DEPS op_call_stack)

nv_test(op_registry_test SRCS op_registry_test.cc DEPS op_registry)

if(WITH_PYTHON)
  py_proto_compile(framework_py_proto SRCS framework.proto data_feed.proto)
  py_proto_compile(trainer_py_proto SRCS trainer_desc.proto data_feed.proto)
#Generate an empty \
    #__init__.py to make framework_py_proto as a valid python module.
  add_custom_target(framework_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)
  add_dependencies(framework_py_proto framework_py_proto_init trainer_py_proto)
  if (NOT WIN32)
    add_custom_command(TARGET framework_py_proto POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto
      COMMAND cp *.py ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/
      COMMENT "Copy generated python proto into directory paddle/fluid/proto."
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else(NOT WIN32)
    string(REPLACE "/" "\\" proto_dstpath "${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/")
    add_custom_command(TARGET framework_py_proto POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto
          COMMAND copy /Y *.py ${proto_dstpath}
          COMMENT "Copy generated python proto into directory paddle/fluid/proto."
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif(NOT WIN32)
endif()

cc_library(lod_rank_table SRCS lod_rank_table.cc DEPS lod_tensor)

cc_library(feed_fetch_method SRCS feed_fetch_method.cc DEPS lod_tensor scope glog)
cc_library(variable_helper SRCS variable_helper.cc DEPS lod_tensor)

cc_library(naive_executor SRCS naive_executor.cc DEPS op_registry device_context scope framework_proto glog lod_rank_table feed_fetch_method graph_to_program_pass variable_helper)

cc_library(executor_gc_helper SRCS executor_gc_helper.cc DEPS scope proto_desc operator garbage_collector)
if(WITH_DISTRIBUTE)
  cc_library(executor SRCS executor.cc multi_trainer.cc pipeline_trainer.cc dataset_factory.cc
  dist_multi_trainer.cc trainer_factory.cc trainer.cc data_feed_factory.cc
  data_feed.cc device_worker.cc hogwild_worker.cc downpour_worker.cc downpour_worker_opt.cc
  pull_dense_worker.cc section_worker.cc device_worker_factory.cc data_set.cc DEPS op_registry
  device_context scope framework_proto trainer_desc_proto glog fs shell
  fleet_wrapper tree_wrapper box_wrapper lodtensor_printer
  lod_rank_table feed_fetch_method sendrecvop_rpc communicator collective_helper ${GLOB_DISTRIBUTE_DEPS}
  graph_to_program_pass variable_helper data_feed_proto timer)
set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
set_source_files_properties(executor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
else()
  cc_library(executor SRCS executor.cc multi_trainer.cc pipeline_trainer.cc dataset_factory.cc
  dist_multi_trainer.cc trainer_factory.cc trainer.cc data_feed_factory.cc
  data_feed.cc device_worker.cc hogwild_worker.cc downpour_worker.cc downpour_worker_opt.cc
  pull_dense_worker.cc section_worker.cc device_worker_factory.cc data_set.cc DEPS op_registry
  device_context scope framework_proto data_feed_proto trainer_desc_proto glog
  lod_rank_table fs shell fleet_wrapper box_wrapper lodtensor_printer feed_fetch_method
  graph_to_program_pass variable_helper timer tree_wrapper)
  cc_test(test_naive_executor SRCS naive_executor_test.cc DEPS naive_executor elementwise_add_op)
endif()

target_link_libraries(executor while_op_helper executor_gc_helper recurrent_op_helper conditional_block_op_helper)

cc_library(parallel_executor SRCS parallel_executor.cc DEPS
        threaded_ssa_graph_executor scope_buffered_ssa_graph_executor parallel_ssa_graph_executor async_ssa_graph_executor
        graph build_strategy collective_helper
        fast_threaded_ssa_graph_executor variable_helper)

cc_test(dist_multi_trainer_test SRCS dist_multi_trainer_test.cc DEPS executor)
cc_library(prune SRCS prune.cc DEPS framework_proto boost)
cc_test(prune_test SRCS prune_test.cc DEPS op_info prune recurrent_op device_context)
cc_test(var_type_inference_test SRCS var_type_inference_test.cc DEPS op_registry
        proto_desc)
cc_library(selected_rows SRCS selected_rows.cc DEPS tensor)
cc_test(selected_rows_test SRCS selected_rows_test.cc DEPS selected_rows)

cc_test(op_kernel_type_test SRCS op_kernel_type_test.cc DEPS place device_context framework_proto op_kernel_type)
cc_test(cow_ptr_tests SRCS details/cow_ptr_test.cc)

cc_test(tuple_test SRCS tuple_test.cc )

cc_test(inlined_vector_test SRCS inlined_vector_test.cc)

if (NOT WIN32)
cc_test(rw_lock_test SRCS rw_lock_test.cc)
endif (NOT WIN32)

cc_library(dlpack_tensor SRCS dlpack_tensor.cc DEPS tensor dlpack)
cc_test(dlpack_tensor_test SRCS dlpack_tensor_test.cc DEPS dlpack_tensor glog)

cc_library(op_compatible_info SRCS op_compatible_info DEPS string_helper proto_desc)
cc_test(op_compatible_info_test SRCS op_compatible_info_test.cc DEPS op_compatible_info proto_desc string_helper glog)

cc_library(save_load_util SRCS save_load_util DEPS tensor scope layer)
cc_test(save_load_util_test SRCS save_load_util_test.cc DEPS save_load_util tensor scope layer)

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE PADDLE_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
	)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE PADDLE_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
	)

message(STATUS "commit: ${PADDLE_COMMIT}")
message(STATUS "branch: ${PADDLE_BRANCH}")

configure_file(commit.h.in commit.h)

set(FLUID_FRAMEWORK_MODULES proto_desc memory lod_tensor executor data_feed_proto layer)

cc_library(paddle_framework DEPS ${FLUID_FRAMEWORK_MODULES})
cc_library(paddle_framework_shared
    SHARED SRCS executor.cc operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/c/c_api.cc
    ${CMAKE_SOURCE_DIR}/paddle/fluid/imperative/layer.cc
    DEPS ${FLUID_FRAMEWORK_MODULES})
get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
set_target_properties(paddle_framework_shared PROPERTIES OUTPUT_NAME paddle_framework)
target_link_libraries(paddle_framework_shared ${os_dependency_modules})

if (LINUX)
  set(FLUID_FRAMEWORK_SHARED_LIB
      ${PADDLE_BINARY_DIR}/paddle/fluid/framework/libpaddle_framework.so
      CACHE INTERNAL "Fluid framework lib")
endif()

if (WIN32)
  set(FLUID_FRAMEWORK_SHARED_LIB
      ${PADDLE_BINARY_DIR}/paddle/fluid/framework/libpaddle_framework.dll
      CACHE INTERNAL "Fluid framework lib")
endif()

if(APPLE)
  set(FLUID_FRAMEWORK_SHARED_LIB
      ${PADDLE_BINARY_DIR}/paddle/fluid/framework/libpaddle_framework.dylib
      CACHE INTERNAL "Fluid framework lib")
endif()
