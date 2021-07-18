file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0 FLAGS_fast_eager_deletion_mode=1 FLAGS_memory_fraction_of_eager_deletion=1.0)
set(dist_ENVS http_proxy="" https_proxy="")

file(GLOB DIST_TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_dist_*.py")
if(NOT WITH_NCCL)
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_dgc_nccl")
endif()
string(REPLACE ".py" "" DIST_TEST_OPS "${DIST_TEST_OPS}")
list(APPEND DIST_TEST_OPS test_parallel_dygraph_mnist)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_se_resnext)
list(APPEND DIST_TEST_OPS test_listen_and_serv_op)
set(MIXED_DIST_TEST_OPS ${DIST_TEST_OPS})
#remove distribute unittests.
list(APPEND MIXED_DIST_TEST_OPS test_dgc_op)
list(APPEND MIXED_DIST_TEST_OPS test_dgc_momentum_op)
list(APPEND MIXED_DIST_TEST_OPS test_dgc_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_simple_dist_transpiler)
list(APPEND MIXED_DIST_TEST_OPS test_nce_remote_table_op)
list(APPEND MIXED_DIST_TEST_OPS test_recv_save_op)
list(APPEND MIXED_DIST_TEST_OPS test_transpiler_ops)
list(APPEND MIXED_DIST_TEST_OPS test_lookup_remote_table_op)
list(APPEND MIXED_DIST_TEST_OPS test_launch)
list(APPEND MIXED_DIST_TEST_OPS test_launch_ps)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_async)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_geo)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_half_async)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_sync)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_api_input)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_checkpoint)
list(APPEND MIXED_DIST_TEST_OPS test_collective_optimizer)
foreach(TEST_OP ${MIXED_DIST_TEST_OPS})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
endforeach()

if(NOT WITH_GPU OR WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_c_comm_init_all_op)
    LIST(REMOVE_ITEM TEST_OPS test_allgather)
    LIST(REMOVE_ITEM TEST_OPS test_allreduce)
    LIST(REMOVE_ITEM TEST_OPS test_broadcast)
    LIST(REMOVE_ITEM TEST_OPS test_reducescatter)
    LIST(REMOVE_ITEM TEST_OPS test_reducescatter_api)
endif()

if(WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_boxps)
    LIST(REMOVE_ITEM TEST_OPS test_paddlebox_datafeed)
    LIST(REMOVE_ITEM TEST_OPS test_trainer_desc)
    LIST(REMOVE_ITEM TEST_OPS test_multiprocess_reader_exception)
    LIST(REMOVE_ITEM TEST_OPS test_avoid_twice_initialization)
    LIST(REMOVE_ITEM TEST_OPS test_checkpoint_notify_op)
endif()

if (NOT ${WITH_GPU})
    LIST(REMOVE_ITEM TEST_OPS test_conv2d_fusion_op)
    LIST(REMOVE_ITEM TEST_OPS test_rank_attention_op) # TODO(shenliang03): rank_attention_op support CPU device in future
    LIST(REMOVE_ITEM TEST_OPS test_parallel_dygraph_mnist) # TODO(Yancey1989): parallel dygraph support CPU device in future
elseif(${CUDNN_VERSION} VERSION_LESS 7100)
    LIST(REMOVE_ITEM TEST_OPS test_conv2d_fusion_op)
endif()

if(NOT WITH_GPU OR WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_pipeline)
    LIST(REMOVE_ITEM TEST_OPS test_boxps)
    LIST(REMOVE_ITEM TEST_OPS test_paddlebox_datafeed)
endif()
list(REMOVE_ITEM TEST_OPS test_seq_concat_op) # FIXME(helin): https://github.com/PaddlePaddle/Paddle/issues/8290
list(REMOVE_ITEM TEST_OPS test_lstm_unit_op) # # FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5185
list(REMOVE_ITEM TEST_OPS test_cond_op) # FIXME(qijun): https://github.com/PaddlePaddle/Paddle/issues/5101#issuecomment-339814957

list(REMOVE_ITEM TEST_OPS op_test) # op_test is a helper python file, not a test
list(REMOVE_ITEM TEST_OPS decorator_helper) # decorator_helper is a helper python file, not a test
if(APPLE)
    if(NOT WITH_DISTRIBUTE)
        list(REMOVE_ITEM TEST_OPS test_desc_clone)
        list(REMOVE_ITEM TEST_OPS test_program_code)
    endif(NOT WITH_DISTRIBUTE)
    message(WARNING "These tests has been disabled in OSX before being fixed:\n test_fuse_elewise_add_act_pass \n test_detection_map_op \n test_dist_se_resnext_*")
    # this op is not support on mac
    list(REMOVE_ITEM TEST_OPS test_fusion_seqexpand_concat_fc_op)
    list(REMOVE_ITEM TEST_OPS test_detection_map_op)
    list(REMOVE_ITEM TEST_OPS test_fuse_elewise_add_act_pass)
endif()
if(NOT WITH_MKLML)
    # this op is not support on openblas
    list(REMOVE_ITEM TEST_OPS test_fusion_seqexpand_concat_fc_op)
endif()

if(NOT WITH_MKL OR NOT WITH_AVX)
  list(REMOVE_ITEM TEST_OPS test_match_matrix_tensor_op)
  list(REMOVE_ITEM TEST_OPS test_var_conv_2d)
endif()
if(WITH_COVERAGE OR NOT WITH_AVX OR WIN32 OR WITH_NV_JETSON)
  list(REMOVE_ITEM TEST_OPS test_pyramid_hash_op)
  list(REMOVE_ITEM TEST_OPS test_fleet_pyramid_hash)
endif()

if(WITH_GPU OR NOT WITH_MKLML)
    # matmul with multiple heads need MKL support
    LIST(REMOVE_ITEM TEST_OPS test_matmul_op_with_head)
endif()

if(NOT WITH_CRYPTO)
    LIST(REMOVE_ITEM TEST_OPS test_crypto)
endif()

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(WITH_COVERAGE)
      add_test(NAME ${TARGET_NAME}
          COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_modules_ENVS}
          COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
          ${PYTHON_EXECUTABLE} -m coverage run --branch -p ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
      add_test(NAME ${TARGET_NAME}
          COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_modules_ENVS}
          ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()

    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 350)
  endif()
endfunction()

function(bash_test_modules TARGET_NAME)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS LABELS)
    cmake_parse_arguments(bash_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(timeout 350)
    if(${bash_test_modules_TIMEOUT})
        set(timeout ${bash_test_modules_TIMEOUT})
    endif()

    if(WITH_COVERAGE)
        add_test(NAME ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python 
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout} ${bash_test_modules_ENVS}
            WITH_COVERAGE=ON COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
            bash ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
        add_test(NAME ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python 
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout} ${bash_test_modules_ENVS}
            bash ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_MODULES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (bash_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()

    if(bash_test_modules_LABELS)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT ${timeout} LABELS ${bash_test_modules_LABELS})
    else()
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT ${timeout})
    endif()
endfunction()

list(REMOVE_ITEM TEST_OPS test_warpctc_op)

list(REMOVE_ITEM TEST_OPS test_parallel_executor_crf)
list(REMOVE_ITEM TEST_OPS test_data_norm_op)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_fetch_feed)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_transformer)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_transformer_auto_growth)
list(REMOVE_ITEM TEST_OPS test_bilinear_interp_op)
list(REMOVE_ITEM TEST_OPS test_nearest_interp_op)
list(REMOVE_ITEM TEST_OPS test_imperative_resnet)
list(REMOVE_ITEM TEST_OPS test_imperative_resnet_sorted_gradient)
list(REMOVE_ITEM TEST_OPS test_imperative_mnist_sorted_gradient)
list(REMOVE_ITEM TEST_OPS test_imperative_se_resnext)
list(REMOVE_ITEM TEST_OPS test_imperative_mnist)
list(REMOVE_ITEM TEST_OPS test_ir_memory_optimize_transformer)
list(REMOVE_ITEM TEST_OPS test_layers)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_seresnext_base_cpu)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_seresnext_with_reduce_cpu)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_seresnext_with_fuse_all_reduce_cpu)
list(REMOVE_ITEM TEST_OPS test_imperative_ocr_attention_model)
list(REMOVE_ITEM TEST_OPS test_async_ssa_graph_executor_mnist)
list(REMOVE_ITEM TEST_OPS test_install_check)
list(REMOVE_ITEM TEST_OPS test_basic_gru_api)
list(REMOVE_ITEM TEST_OPS test_basic_gru_unit_op)
list(REMOVE_ITEM TEST_OPS test_basic_lstm_api)
list(REMOVE_ITEM TEST_OPS test_basic_lstm_unit_op)
list(REMOVE_ITEM TEST_OPS test_imperative_debug_string)
list(REMOVE_ITEM TEST_OPS test_fuse_bn_act_pass)
list(REMOVE_ITEM TEST_OPS test_imperative_static_runner_mnist)
list(REMOVE_ITEM TEST_OPS test_imperative_static_runner_while)
list(REMOVE_ITEM TEST_OPS test_conv3d_transpose_op)



list(REMOVE_ITEM TEST_OPS test_sampling_id_op)

if (APPLE OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_dataset)
  list(REMOVE_ITEM TEST_OPS test_dataset_dataloader)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_base)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exception)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_process)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_fds_clear)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exit_func)
  list(REMOVE_ITEM TEST_OPS test_imperative_signal_handler)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_exception)
endif()

if(NOT WITH_GPU OR WIN32 OR APPLE)
  list(REMOVE_ITEM TEST_OPS test_build_strategy_fusion_group_pass)
endif()

# Some ops need to check results when gc is enabled
# Currently, only ops that register NoNeedBufferVarsInference need to do this test
set(TEST_OPS_WITH_GC
  test_affine_channel_op
  test_concat_op
  test_elementwise_add_op
  test_elementwise_sub_op
  test_fill_zeros_like2_op
  test_gather_op
  test_gather_nd_op
  test_linear_chain_crf_op
  test_lod_reset_op
  test_lookup_table_op
  test_mean_op
  test_pad2d_op
  test_scatter_op
  test_sequence_concat
  test_sequence_conv
  test_sequence_pool
  test_sequence_expand_as
  test_sequence_expand
  test_sequence_pad_op
  test_sequence_unpad_op
  test_sequence_scatter_op
  test_sequence_slice_op
  test_slice_op
  test_space_to_depth_op
  test_squared_l2_distance_op)

foreach(TEST_OP ${TEST_OPS_WITH_GC})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP} ENVS ${GC_ENVS})
endforeach()

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
py_test_modules(test_adam_op_multi_thread MODULES test_adam_op ENVS FLAGS_inner_op_parallelism=4)
py_test_modules(test_warpctc_op MODULES test_warpctc_op)
py_test_modules(test_bilinear_interp_op MODULES test_bilinear_interp_op ENVS ${GC_ENVS})
py_test_modules(test_nearest_interp_op MODULES test_nearest_interp_op ENVS ${GC_ENVS})
py_test_modules(test_imperative_resnet MODULES test_imperative_resnet ENVS
    FLAGS_cudnn_deterministic=1 SERIAL)
set_tests_properties(test_imperative_resnet PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" RUN_SERIAL TRUE)
py_test_modules(test_imperative_resnet_sorted_gradient MODULES test_imperative_resnet_sorted_gradient ENVS
        FLAGS_cudnn_deterministic=1 SERIAL)
set_tests_properties(test_imperative_resnet_sorted_gradient PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" RUN_SERIAL TRUE)
py_test_modules(test_imperative_mnist MODULES test_imperative_mnist ENVS
    FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_mnist_sorted_gradient MODULES test_imperative_mnist_sorted_gradient ENVS
        FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_se_resnext MODULES test_imperative_se_resnext ENVS
    FLAGS_cudnn_deterministic=1 SERIAL)
set_tests_properties(test_imperative_se_resnext PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE" RUN_SERIAL TRUE)
py_test_modules(test_imperative_ocr_attention_model MODULES test_imperative_ocr_attention_model ENVS
        FLAGS_cudnn_deterministic=1 SERIAL)
py_test_modules(test_install_check MODULES test_install_check ENVS
        FLAGS_cudnn_deterministic=1 SERIAL)
set_tests_properties(test_install_check PROPERTIES LABELS "RUN_TYPE=DIST")
py_test_modules(test_imperative_debug_string MODULES test_imperative_debug_string ENVS FLAGS_dygraph_debug=1)
py_test_modules(test_imperative_static_runner_mnist MODULES test_imperative_static_runner_mnist ENVS
    FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_static_runner_while MODULES test_imperative_static_runner_while ENVS
    FLAGS_cudnn_deterministic=1)
if(WITH_DISTRIBUTE)
    # FIXME(typhoonzero): add these tests back
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transformer")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transpiler")

    #not need
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_base")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_base")

    py_test_modules(test_lookup_remote_table_op MODULES test_lookup_remote_table_op ENVS ${dist_ENVS})
    py_test_modules(test_nce_remote_table_op MODULES test_nce_remote_table_op ENVS ${dist_ENVS})
    py_test_modules(test_recv_save_op MODULES test_recv_save_op ENVS ${dist_ENVS})
    py_test_modules(test_transpiler_ops MODULES test_transpiler_ops ENVS ${dist_ENVS})
    py_test_modules(test_communicator_async MODULES test_communicator_async ENVS ${dist_ENVS})
    py_test_modules(test_communicator_geo MODULES test_communicator_geo ENVS ${dist_ENVS})
    py_test_modules(test_communicator_half_async MODULES test_communicator_half_async ENVS ${dist_ENVS} FLAGS_communicator_send_queue_size=1 FLAGS_communicator_max_merge_var_num=1)
    py_test_modules(test_communicator_sync MODULES test_communicator_sync ENVS ${dist_ENVS} FLAGS_communicator_send_queue_size=1 FLAGS_communicator_max_merge_var_num=1)
    py_test_modules(test_collective_optimizer MODULES test_collective_optimizer)
    if(WITH_DGC)
        # if with dgc, test all dgc tests.
        # NOTE. dist dgc tests is already in DIST_TEST_OPS
        py_test_modules(test_dgc_op MODULES test_dgc_op)
        py_test_modules(test_dgc_momentum_op MODULES test_dgc_momentum_op)
        py_test_modules(test_dgc_optimizer MODULES test_dgc_optimizer)
    else()
        # if not with dgc, must close all dgc tests
        list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_dgc_nccl")
        list(REMOVE_ITEM DIST_TEST_OPS "test_dist_se_resnext_dgc")
    endif()
    if(NOT APPLE)
        if(WITH_GPU)
            # NOTE. test_launch only work in gpu collective mode
            bash_test_modules(test_launch MODULES test_launch.sh  ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
            py_test_modules(test_fleet_checkpoint MODULES test_fleet_checkpoint)
        endif()
        bash_test_modules(test_launch_ps MODULES test_launch_ps.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})

        set(dist_ut_port 1000)
        foreach(TEST_OP ${DIST_TEST_OPS})
            bash_test_modules(${TEST_OP} MODULES dist_test.sh SERIAL LABELS "RUN_TYPE=EXCLUSIVE" ENVS "PADDLE_DIST_UT_PORT=${dist_ut_port}")
            MATH(EXPR dist_ut_port "${dist_ut_port}+50")
        endforeach(TEST_OP)
    endif(NOT APPLE)
endif()

py_test_modules(test_parallel_executor_crf MODULES test_parallel_executor_crf)
py_test_modules(test_parallel_executor_transformer MODULES test_parallel_executor_transformer)
py_test_modules(test_parallel_executor_transformer_auto_growth MODULES test_parallel_executor_transformer_auto_growth ENVS FLAGS_allocator_strategy=auto_growth)
py_test_modules(test_layers MODULES test_layers ENVS FLAGS_cudnn_deterministic=1)
py_test_modules(test_parallel_executor_seresnext_base_cpu MODULES test_parallel_executor_seresnext_base_cpu)
py_test_modules(test_parallel_executor_seresnext_with_reduce_cpu MODULES test_parallel_executor_seresnext_with_reduce_cpu)
py_test_modules(test_parallel_executor_seresnext_with_fuse_all_reduce_cpu MODULES test_parallel_executor_seresnext_with_fuse_all_reduce_cpu)
py_test_modules(test_data_norm_op MODULES test_data_norm_op)
py_test_modules(test_fuse_bn_act_pass MODULES test_fuse_bn_act_pass ENVS FLAGS_cudnn_deterministic=1 FLAGS_cudnn_batchnorm_spatial_persistent=1 FLAGS_conv_workspace_size_limit=1000)

if(NOT WIN32)
    py_test_modules(test_ir_memory_optimize_transformer MODULES test_ir_memory_optimize_transformer)
    # FIXME(zcd): temporally disable test_parallel_executor_fetch_feed in Windows CI because of the random failure.
    py_test_modules(test_parallel_executor_fetch_feed MODULES test_parallel_executor_fetch_feed)
    set_tests_properties(test_parallel_executor_fetch_feed PROPERTIES TIMEOUT 450)
endif()

set_tests_properties(test_parallel_executor_seresnext_base_cpu PROPERTIES TIMEOUT 900)
set_tests_properties(test_parallel_executor_seresnext_with_reduce_cpu PROPERTIES TIMEOUT 750)
set_tests_properties(test_parallel_executor_seresnext_with_fuse_all_reduce_cpu PROPERTIES TIMEOUT 750)

add_subdirectory(sequence)
add_subdirectory(dygraph_to_static)

if (WITH_MKLDNN)
    add_subdirectory(mkldnn)
endif()

add_subdirectory(ir)

if (WITH_TESTING)
    set_property(TEST test_parallel_executor_mnist PROPERTY ENVIRONMENT GLOG_vmodule=all_reduce_deps_pass=10)
endif()

set_tests_properties(test_parallel_executor_test_while_train test_parallel_executor_mnist
        test_parallel_executor_feed_persistable_var
        test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass
        test_data_norm_op test_imperative_using_non_zero_gpu test_fuse_bn_act_pass
        test_optimizer_in_control_flow test_dataloader_keep_order
        test_dataloader_unkeep_order
        test_parallel_executor_fetch_isolated_var
        test_parallel_executor_inference_feed_partial_data
        test_parallel_ssa_graph_inference_feed_partial_data
        test_fetch_unmerged
        test_buffer_shared_memory_reuse_pass PROPERTIES LABELS "RUN_TYPE=DIST")

set_tests_properties(test_parallel_executor_crf test_sync_batch_norm_op test_inplace_abn_op
        test_parallel_executor_seresnext_base_gpu
        test_parallel_executor_seresnext_with_reduce_gpu
        test_parallel_executor_seresnext_with_fuse_all_reduce_gpu
        PROPERTIES LABELS "RUN_TYPE=DIST" RUN_SERIAL TRUE)

if(NOT WIN32 AND NOT APPLE)
    set_tests_properties(test_imperative_data_loader_base PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_imperative_data_loader_fds_clear PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_imperative_data_loader_exception PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_exception PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
endif()
