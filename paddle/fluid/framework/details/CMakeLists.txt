cc_library(var_handle SRCS var_handle.cc DEPS place framework_proto node)
cc_library(op_handle_base SRCS op_handle_base.cc DEPS var_handle device_context lod_tensor)

cc_library(scale_loss_grad_op_handle SRCS scale_loss_grad_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(fetch_op_handle SRCS fetch_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)
cc_library(fetch_async_op_handle SRCS fetch_async_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory)

cc_library(share_tensor_buffer_functor SRCS share_tensor_buffer_functor.cc DEPS framework_proto scope place operator op_registry) 
cc_library(computation_op_handle SRCS computation_op_handle.cc DEPS framework_proto scope place operator op_registry)
cc_library(share_tensor_buffer_op_handle SRCS share_tensor_buffer_op_handle.cc DEPS op_handle_base scope computation_op_handle share_tensor_buffer_functor)
cc_library(rpc_op_handle SRCS rpc_op_handle.cc DEPS framework_proto scope place operator op_registry)
cc_library(fetch_barrier_op_handle SRCS fetch_barrier_op_handle.cc DEPS framework_proto scope place operator op_registry)
cc_library(multi_devices_helper SRCS multi_devices_helper.cc DEPS graph graph_helper)

cc_library(variable_visitor SRCS variable_visitor.cc DEPS lod_tensor selected_rows)

if(WITH_PSCORE)
    set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
    set_source_files_properties(reduce_op_handle.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
    set_source_files_properties(threaded_ssa_graph_executor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
    set_source_files_properties(async_ssa_graph_executor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()


if(WITH_GPU)
    nv_library(nan_inf_utils SRCS nan_inf_utils_detail.cc nan_inf_utils_detail.cu DEPS framework_proto scope place)
    nv_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor)
    nv_library(fused_all_reduce_op_handle SRCS fused_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor place device_memory_aligment)
    nv_library(grad_merge_all_reduce_op_handle SRCS grad_merge_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor
            ddim memory dynload_cuda variable_visitor place device_memory_aligment all_reduce_op_handle fused_all_reduce_op_handle)

    if(WITH_DGC)
        nv_library(sparse_all_reduce_op_handle SRCS sparse_all_reduce_op_handle.cc DEPS op_handle_base scope 
            lod_tensor ddim memory dynload_cuda variable_visitor dgc all_reduce_op_handle)
    endif()

    if(WITH_DISTRIBUTE)
        nv_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
                ddim dynload_cuda selected_rows_functor)
    else()
        nv_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
            ddim dynload_cuda selected_rows_functor)
    endif()
    nv_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor dynload_cuda)
    nv_library(fused_broadcast_op_handle SRCS fused_broadcast_op_handle.cc DEPS broadcast_op_handle)
elseif(WITH_ROCM)
    hip_library(nan_inf_utils SRCS nan_inf_utils_detail.cc nan_inf_utils_detail.cu DEPS framework_proto scope place)
    hip_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor)
    hip_library(fused_all_reduce_op_handle SRCS fused_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            dynload_cuda variable_visitor place device_memory_aligment)
    hip_library(grad_merge_all_reduce_op_handle SRCS grad_merge_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor
            ddim memory dynload_cuda variable_visitor place device_memory_aligment all_reduce_op_handle fused_all_reduce_op_handle)

    if(WITH_DISTRIBUTE)
        hip_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
                    ddim dynload_cuda selected_rows_functor)
    else()
        hip_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
                    ddim dynload_cuda selected_rows_functor)
    endif()
    hip_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor dynload_cuda)
    hip_library(fused_broadcast_op_handle SRCS fused_broadcast_op_handle.cc DEPS broadcast_op_handle)
else()
    cc_library(nan_inf_utils SRCS nan_inf_utils_detail.cc DEPS framework_proto scope place)
    cc_library(all_reduce_op_handle SRCS all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
             variable_visitor)
    cc_library(fused_all_reduce_op_handle SRCS fused_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor ddim memory
            variable_visitor place device_memory_aligment)
    cc_library(grad_merge_all_reduce_op_handle SRCS grad_merge_all_reduce_op_handle.cc DEPS op_handle_base scope lod_tensor
            ddim memory variable_visitor place device_memory_aligment all_reduce_op_handle fused_all_reduce_op_handle)
    if(WITH_DISTRIBUTE)
        cc_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
                ddim selected_rows_functor)
    else()
        cc_library(reduce_op_handle SRCS reduce_op_handle.cc DEPS op_handle_base variable_visitor scope
            ddim selected_rows_functor)
    endif()
    cc_library(broadcast_op_handle SRCS broadcast_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)
    cc_library(fused_broadcast_op_handle SRCS fused_broadcast_op_handle.cc DEPS broadcast_op_handle)
endif()

cc_library(gather_op_handle SRCS gather_op_handle.cc DEPS op_handle_base scope ddim memory variable_visitor)

cc_library(eager_deletion_op_handle SRCS eager_deletion_op_handle.cc DEPS lod_tensor selected_rows reference_count_pass_helper)

set(SSA_GRAPH_EXECUTOR_DEPS graph framework_proto
    multi_devices_helper
    sequential_execution_pass
    modify_op_lock_and_record_event_pass
    all_reduce_deps_pass
    reference_count_pass
    eager_deletion_pass
    buffer_shared_inplace_op_pass
    buffer_shared_cross_op_memory_reuse_pass
    inplace_addto_op_pass
    set_reader_device_info_utils
    add_reader_dependency_pass)
cc_library(ssa_graph_executor SRCS ssa_graph_executor.cc DEPS ${SSA_GRAPH_EXECUTOR_DEPS})

cc_library(threaded_ssa_graph_executor SRCS threaded_ssa_graph_executor.cc DEPS fetch_op_handle ssa_graph_executor scope
        simple_threadpool device_context)

cc_library(parallel_ssa_graph_executor SRCS parallel_ssa_graph_executor.cc DEPS threaded_ssa_graph_executor)

set(ASYNC_SSA_GRAPH_EXECUTOR_DEPS threaded_ssa_graph_executor)

cc_library(async_ssa_graph_executor SRCS async_ssa_graph_executor.cc DEPS ${ASYNC_SSA_GRAPH_EXECUTOR_DEPS})

cc_test(broadcast_op_test SRCS broadcast_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context broadcast_op_handle)
cc_test(gather_op_test SRCS gather_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
        device_context gather_op_handle)

cc_library(scope_buffered_monitor SRCS scope_buffered_monitor.cc DEPS scope profiler selected_rows)
cc_library(scope_buffered_ssa_graph_executor SRCS scope_buffered_ssa_graph_executor.cc DEPS ssa_graph_executor scope_buffered_monitor)
#cc_test(reduce_op_handle_test SRCS reduce_op_handle_test.cc DEPS var_handle op_handle_base scope ddim memory
#        device_context reduce_op_handle )
cc_library(bind_threaded_ssa_graph_executor SRCS bind_threaded_ssa_graph_executor.cc
        DEPS fetch_op_handle gflags ssa_graph_executor scope simple_threadpool device_context)
cc_library(fast_threaded_ssa_graph_executor SRCS fast_threaded_ssa_graph_executor.cc
        DEPS fetch_async_op_handle ssa_graph_executor scope simple_threadpool device_context async_variable)
cc_test(fused_broadcast_op_test SRCS fused_broadcast_op_handle_test.cc DEPS fused_broadcast_op_handle)

cc_test(exception_holder_test SRCS exception_holder_test.cc )

set(IR_PASS_DEPS graph_viz_pass multi_devices_graph_pass
    multi_devices_graph_print_pass multi_devices_graph_check_pass
    fuse_elewise_add_act_pass fuse_bn_act_pass fuse_bn_add_act_pass 
    multi_batch_merge_pass 
    fuse_relu_depthwise_conv_pass
    lock_free_optimize_pass
    coalesce_grad_tensor_pass fuse_all_reduce_op_pass backward_optimizer_op_deps_pass
    fuse_adam_op_pass fuse_sgd_op_pass fuse_momentum_op_pass
    sync_batch_norm_pass runtime_context_cache_pass)
if(NOT APPLE AND NOT WIN32 AND (WITH_GPU OR WITH_ROCM))
  set(IR_PASS_DEPS ${IR_PASS_DEPS} fusion_group_pass)
endif()
cc_library(build_strategy SRCS build_strategy.cc DEPS pass_builder ${IR_PASS_DEPS})

if (WITH_MKLDNN)
  target_link_libraries(build_strategy mkldnn_placement_pass)
endif()
