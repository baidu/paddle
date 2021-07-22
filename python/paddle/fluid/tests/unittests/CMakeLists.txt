file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")
set(GC_ENVS FLAGS_eager_delete_tensor_gb=0.0 FLAGS_fast_eager_deletion_mode=1 FLAGS_memory_fraction_of_eager_deletion=1.0)
set(dist_ENVS http_proxy="" https_proxy="")

file(GLOB DIST_TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_dist_*.py")
list(REMOVE_ITEM DIST_TEST_OPS "test_dist_op")
if ((NOT WITH_NCCL) AND (NOT WITH_RCCL))
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_dgc_nccl")
endif()
string(REPLACE ".py" "" DIST_TEST_OPS "${DIST_TEST_OPS}")
list(APPEND DIST_TEST_OPS test_parallel_dygraph_mnist)
list(APPEND DIST_TEST_OPS test_pipeline)
list(APPEND DIST_TEST_OPS test_static_model_parallel)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_se_resnext)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_sparse_embedding)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_sparse_embedding_over_height)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_transformer)
list(APPEND DIST_TEST_OPS test_fleet_pipeline_meta_optimizer)
list(APPEND DIST_TEST_OPS test_fleet_pipeline_meta_optimizer_with_recompute)
list(APPEND DIST_TEST_OPS test_fleet_raw_program_meta_optimizer)
list(APPEND DIST_TEST_OPS test_fleet_graph_execution_meta_optimizer)
list(APPEND DIST_TEST_OPS test_gen_nccl_id_op)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_unused_variables)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_control_flow)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_dataparallel)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_pipeline_parallel)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_tensor_parallel)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_sharding_parallel)
list(APPEND DIST_TEST_OPS test_parallel_dygraph_mp_layers)
list(APPEND DIST_TEST_OPS test_parallel_class_center_sample)
set(MIXED_DIST_TEST_OPS ${DIST_TEST_OPS})
#remove distribute unittests.
list(APPEND MIXED_DIST_TEST_OPS test_dgc_op)
list(APPEND MIXED_DIST_TEST_OPS test_dgc_momentum_op)
list(APPEND MIXED_DIST_TEST_OPS test_dgc_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_simple_dist_transpiler)
list(APPEND MIXED_DIST_TEST_OPS test_recv_save_op)
list(APPEND MIXED_DIST_TEST_OPS test_c_comm_init_op)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_async)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_ps_gpu)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_geo)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_half_async)
list(APPEND MIXED_DIST_TEST_OPS test_communicator_sync)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_ps)
list(APPEND MIXED_DIST_TEST_OPS test_launch_coverage)
list(APPEND MIXED_DIST_TEST_OPS test_fleetrun)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_run_random_port)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_async)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_cloud)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_ascend)
list(APPEND MIXED_DIST_TEST_OPS test_ascend_group)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_launch_nproc)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_api_input)
list(APPEND MIXED_DIST_TEST_OPS test_collective_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_base)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_base_2)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_base_3)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_recompute_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_pipeline_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_pipeline_meta_optimizer_with_recompute)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_raw_program_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_amp_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_amp_init)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_gradient_merge_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_sharding_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_localsgd_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_lars_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_lamb_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_dgc_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_fp16_allreduce_meta_optimizer)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_private_function)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_graph_executor)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_meta_optimizer_base)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_distributed_strategy)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_auto)
list(APPEND MIXED_DIST_TEST_OPS test_fleet_static_mp_layers)
foreach(TEST_OP ${MIXED_DIST_TEST_OPS})
  list(REMOVE_ITEM TEST_OPS ${TEST_OP})
endforeach()

if(((NOT WITH_ROCM) AND (NOT WITH_GPU)) OR WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_c_comm_init_all_op)
    LIST(REMOVE_ITEM TEST_OPS test_c_concat)
    LIST(REMOVE_ITEM TEST_OPS test_c_split)
    LIST(REMOVE_ITEM TEST_OPS test_allgather)
    LIST(REMOVE_ITEM TEST_OPS test_c_identity)
    LIST(REMOVE_ITEM TEST_OPS test_c_embedding_op)
    LIST(REMOVE_ITEM TEST_OPS test_allreduce)
    LIST(REMOVE_ITEM TEST_OPS test_broadcast)
    LIST(REMOVE_ITEM TEST_OPS test_collective_reduce)
    LIST(REMOVE_ITEM TEST_OPS test_pipeline_parallel)
    LIST(REMOVE_ITEM TEST_OPS test_collective_scatter)
    LIST(REMOVE_ITEM TEST_OPS test_collective_sendrecv)
    LIST(REMOVE_ITEM TEST_OPS test_reducescatter)
    LIST(REMOVE_ITEM TEST_OPS test_reducescatter_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_split_embedding)
    LIST(REMOVE_ITEM TEST_OPS test_collective_split_embedding_none_divisible)
    LIST(REMOVE_ITEM TEST_OPS test_collective_split_row_linear)
    LIST(REMOVE_ITEM TEST_OPS test_collective_split_col_linear)
    LIST(REMOVE_ITEM TEST_OPS test_collective_reduce_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_scatter_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_barrier_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_allreduce_api)
    LIST(REMOVE_ITEM TEST_OPS test_new_group_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_broadcast_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_allgather_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_alltoall_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_sendrecv_api)
    LIST(REMOVE_ITEM TEST_OPS test_collective_wait)
    LIST(REMOVE_ITEM TEST_OPS test_memcpy_op)
    LIST(REMOVE_ITEM TEST_OPS test_raw_program_optimizer)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_gradient_scale)
endif()

if(WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_multiprocess_reader_exception)
    LIST(REMOVE_ITEM TEST_OPS test_trainer_desc)
    LIST(REMOVE_ITEM TEST_OPS test_checkpoint_notify_op)
    LIST(REMOVE_ITEM TEST_OPS test_downpoursgd)
    LIST(REMOVE_ITEM TEST_OPS test_fleet)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_nocvm_1)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_rolemaker)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_rolemaker_3)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_unitaccessor)
    LIST(REMOVE_ITEM TEST_OPS test_ps_dispatcher)
    LIST(REMOVE_ITEM TEST_OPS test_ir_memory_optimize_nlp)
    LIST(REMOVE_ITEM TEST_OPS test_nvprof)

    # TODO: Fix these unittests failed on Windows
    LIST(REMOVE_ITEM TEST_OPS test_debugger)
    if (WITH_GPU)
        LIST(REMOVE_ITEM TEST_OPS test_update_loss_scaling_op)
    endif()
endif()

if(NOT WITH_DISTRIBUTE OR WIN32)
    # DISTRIBUTE related
    LIST(REMOVE_ITEM TEST_OPS test_avoid_twice_initialization)
    LIST(REMOVE_ITEM TEST_OPS test_distributed_strategy)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_metric)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_ps)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_rolemaker_2)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_utils)

    # TODO: Fix these unittests failed on Windows
    list(REMOVE_ITEM TEST_OPS test_fake_init_op)
endif()

if(NOT WITH_DISTRIBUTE)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_rolemaker_new)
    LIST(REMOVE_ITEM TEST_OPS test_desc_clone_dist)
endif()

if(WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_complex_matmul)
endif()

LIST(REMOVE_ITEM TEST_OPS test_fleet_checkpoint)
LIST(REMOVE_ITEM TEST_OPS test_auto_checkpoint)
LIST(REMOVE_ITEM TEST_OPS test_auto_checkpoint1)
LIST(REMOVE_ITEM TEST_OPS test_auto_checkpoint2)
LIST(REMOVE_ITEM TEST_OPS test_auto_checkpoint3)
LIST(REMOVE_ITEM TEST_OPS test_auto_checkpoint_multiple)
LIST(REMOVE_ITEM TEST_OPS test_auto_checkpoint_dist_basic)
LIST(REMOVE_ITEM TEST_OPS test_hdfs1)
LIST(REMOVE_ITEM TEST_OPS test_hdfs2)
LIST(REMOVE_ITEM TEST_OPS test_hdfs3)
LIST(REMOVE_ITEM TEST_OPS test_checkpoint_saver)

if(APPLE OR WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_fs_interface)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_metric)
endif()

list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_hybrid_parallel)

if ((NOT WITH_GPU) AND (NOT WITH_ROCM))
    LIST(REMOVE_ITEM TEST_OPS test_conv2d_fusion_op)
    LIST(REMOVE_ITEM TEST_OPS test_rank_attention_op) # TODO(shenliang03): rank_attention_op support CPU device in future
    LIST(REMOVE_ITEM TEST_OPS test_batch_fc_op) # TODO(shenliang03): batch_fc_op support CPU device in future
    LIST(REMOVE_ITEM TEST_OPS test_parallel_dygraph_mnist) # TODO(Yancey1989): parallel dygraph support CPU device in future
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_unused_variables)
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_se_resnext)
    LIST(REMOVE_ITEM TEST_OPS test_parallel_dygraph_sparse_embedding)
    LIST(REMOVE_ITEM TEST_OPS test_parallel_dygraph_sparse_embedding_over_height)
    LIST(REMOVE_ITEM TEST_OPS test_parallel_dygraph_transformer)
    LIST(REMOVE_ITEM TEST_OPS test_parallel_dygraph_sync_batch_norm)
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_control_flow)
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_dataparallel)
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_pipeline_parallel)
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_tensor_parallel)
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_sharding_parallel)
    list(REMOVE_ITEM TEST_OPS test_parallel_dygraph_mp_layers)
    LIST(REMOVE_ITEM TEST_OPS test_imperative_auto_mixed_precision)
    LIST(REMOVE_ITEM TEST_OPS test_mixed_precision)
    LIST(REMOVE_ITEM TEST_OPS test_fleet_base_single)
    LIST(REMOVE_ITEM TEST_OPS test_dygraph_recompute)
    list(REMOVE_ITEM TEST_OPS test_parallel_class_center_sample)
elseif(WITH_GPU)
    if (${CUDNN_VERSION} VERSION_LESS 7100)
        LIST(REMOVE_ITEM TEST_OPS test_conv2d_fusion_op)
    endif()
endif()

if (WITH_NCCL)
    if (${NCCL_VERSION} VERSION_LESS 2212)
        LIST(REMOVE_ITEM DIST_TEST_OPS test_parallel_dygraph_sparse_embedding)
        LIST(REMOVE_ITEM DIST_TEST_OPS test_parallel_dygraph_sparse_embedding_over_height)
        LIST(REMOVE_ITEM DIST_TEST_OPS test_parallel_dygraph_transformer)
    endif()
endif()

if ((NOT WITH_NCCL) AND (NOT WITH_RCCL))
    list(REMOVE_ITEM TEST_OPS test_imperative_group)
    LIST(REMOVE_ITEM TEST_OPS test_new_group_api)
endif()

if(((NOT WITH_ROCM) AND (NOT WITH_GPU)) OR WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_boxps)
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

if(WITH_COVERAGE OR WIN32 OR WITH_NV_JETSON)
  list(REMOVE_ITEM TEST_OPS test_pyramid_hash_op)
endif()

list(REMOVE_ITEM TEST_OPS test_fleet_pyramid_hash)

if((WITH_ROCM OR WITH_GPU) OR NOT WITH_MKLML)
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

    if(WITH_COVERAGE AND NOT (WITH_INCREMENTAL_COVERAGE AND "$ENV{PADDLE_GIT_DIFF_PY_FILE}" STREQUAL ""))
        add_test(NAME ${TARGET_NAME}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_modules_ENVS}
                COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
                ${PYTHON_EXECUTABLE} -m coverage run --branch -p ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    elseif(WITH_ASCEND_CL)
        # AscendCL need to include ascend toolkit python path, or ACL error will be thrown when running ctest
        add_test(NAME ${TARGET_NAME}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python:$ENV{PYTHONPATH} ${py_test_modules_ENVS}
                ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
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
    if(WIN32)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 150)
    endif()
  endif()
endfunction()


function(bash_test_modules TARGET_NAME)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options SERIAL)
    set(oneValueArgs TIMEOUT START_BASH)
    set(multiValueArgs DEPS ENVS LABELS)
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
            bash ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
        add_test(NAME ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python 
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout} ${bash_test_modules_ENVS}
            bash ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (bash_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()

    if(bash_test_modules_LABELS)
        set_tests_properties(${TARGET_NAME} PROPERTIES LABELS ${bash_test_modules_LABELS})
    endif()
endfunction()

function(parallel_bash_test_modules TARGET_NAME)
    if(NOT WITH_TESTING)
        return()
    endif()

    set(options SERIAL)
    set(oneValueArgs TIMEOUT START_BASH)
    set(multiValueArgs DEPS ENVS LABELS UnitTests)
    cmake_parse_arguments(parallel_bash_test_modules "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})


    set(timeout 120)
    if(${parallel_bash_test_modules_TIMEOUT})
        set(timeout ${parallel_bash_test_modules_TIMEOUT})
    endif()

    list(JOIN  parallel_bash_test_modules_UnitTests " " uts_string)

    if(WITH_COVERAGE)
        add_test(NAME ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python 
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout} ${parallel_bash_test_modules_ENVS} UnitTests=${uts_string}
            WITH_COVERAGE=ON COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
            bash ${CMAKE_CURRENT_BINARY_DIR}/${parallel_bash_test_modules_START_BASH}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
        add_test(NAME ${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python 
            TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout} ${parallel_bash_test_modules_ENVS} UnitTests=${uts_string}
            bash ${CMAKE_CURRENT_BINARY_DIR}/${parallel_bash_test_modules_START_BASH}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (parallel_bash_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()

    if(parallel_bash_test_modules_LABELS)
        set_tests_properties(${TARGET_NAME} PROPERTIES LABELS ${parallel_bash_test_modules_LABELS})
    endif()
endfunction()

list(REMOVE_ITEM TEST_OPS test_feed_data_check_shape_type)
list(REMOVE_ITEM TEST_OPS test_fetch_lod_tensor_array)
list(REMOVE_ITEM TEST_OPS test_warpctc_op)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_crf)
list(REMOVE_ITEM TEST_OPS test_parallel_executor_profiler)
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
list(REMOVE_ITEM TEST_OPS test_fuse_all_reduce_pass)
list(REMOVE_ITEM TEST_OPS test_fuse_bn_act_pass)
list(REMOVE_ITEM TEST_OPS test_fuse_bn_add_act_pass)
list(REMOVE_ITEM TEST_OPS test_imperative_static_runner_mnist)
list(REMOVE_ITEM TEST_OPS test_imperative_static_runner_while)
# disable test_cumsum_op temporaily
# list(REMOVE_ITEM TEST_OPS test_cumsum_op)

# disable this unittest temporarily
list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exception)

if (APPLE OR WIN32)
  list(REMOVE_ITEM TEST_OPS test_dataset)
  list(REMOVE_ITEM TEST_OPS test_dataset_dataloader)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_base)
  # list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exception)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_process)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_fds_clear)
  list(REMOVE_ITEM TEST_OPS test_imperative_data_loader_exit_func)
  list(REMOVE_ITEM TEST_OPS test_imperative_signal_handler)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_static)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_dynamic)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_exception)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_iterable_dataset)
  list(REMOVE_ITEM TEST_OPS test_multiprocess_dataloader_dataset)
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
set_tests_properties(test_imperative_resnet PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
py_test_modules(test_imperative_resnet_sorted_gradient MODULES test_imperative_resnet_sorted_gradient ENVS
        FLAGS_cudnn_deterministic=1 SERIAL)
set_tests_properties(test_imperative_resnet_sorted_gradient PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
py_test_modules(test_imperative_mnist MODULES test_imperative_mnist ENVS
    FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_mnist_sorted_gradient MODULES test_imperative_mnist_sorted_gradient ENVS
        FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_se_resnext MODULES test_imperative_se_resnext ENVS
    FLAGS_cudnn_deterministic=1 SERIAL)
set_tests_properties(test_imperative_se_resnext PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
py_test_modules(test_imperative_ocr_attention_model MODULES test_imperative_ocr_attention_model ENVS
        FLAGS_cudnn_deterministic=1 SERIAL)
py_test_modules(test_install_check MODULES test_install_check ENVS
        FLAGS_cudnn_deterministic=1 SERIAL)
set_tests_properties(test_install_check PROPERTIES LABELS "RUN_TYPE=DIST")
py_test_modules(test_imperative_static_runner_mnist MODULES test_imperative_static_runner_mnist ENVS
    FLAGS_cudnn_deterministic=1)
py_test_modules(test_imperative_static_runner_while MODULES test_imperative_static_runner_while ENVS
    FLAGS_cudnn_deterministic=1)
set_tests_properties(test_conv2d_op PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_conv2d_op_depthwise_conv PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_conv2d_api PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_conv_nn_grad PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_norm_nn_grad PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_nn_grad PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
if(WITH_DISTRIBUTE)
    # FIXME(typhoonzero): add these tests back
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transformer")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_transpiler")

    # TODO(sandyhouse): fix and add the ut back
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_hallreduce")

    #not need
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_base")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_base")


    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_ctr")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_lars")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_train")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_save_load")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_text_classification")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_train")
    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_word2vec")

    list(REMOVE_ITEM DIST_TEST_OPS "test_dist_fleet_gloo")

    py_test_modules(test_recv_save_op MODULES test_recv_save_op ENVS ${dist_ENVS})
    py_test_modules(test_communicator_async MODULES test_communicator_async ENVS ${dist_ENVS})
    py_test_modules(test_communicator_ps_gpu MODULES test_communicator_ps_gpu ENVS ${dist_ENVS})
    py_test_modules(test_communicator_geo MODULES test_communicator_geo ENVS ${dist_ENVS})
    py_test_modules(test_communicator_half_async MODULES test_communicator_half_async ENVS ${dist_ENVS} FLAGS_communicator_send_queue_size=1 FLAGS_communicator_max_merge_var_num=1)
    py_test_modules(test_communicator_sync MODULES test_communicator_sync ENVS ${dist_ENVS} FLAGS_communicator_send_queue_size=1 FLAGS_communicator_max_merge_var_num=1)
    py_test_modules(test_collective_optimizer MODULES test_collective_optimizer)
    if(NOT APPLE)
    	   py_test_modules(test_fleet_base MODULES test_fleet_base ENVS ${dist_ENVS})
    	   py_test_modules(test_fleet_base_2 MODULES test_fleet_base_2 ENVS ${dist_ENVS})
    	   py_test_modules(test_fleet_base_3 MODULES test_fleet_base_3 ENVS ${dist_ENVS})
    	   py_test_modules(test_fleet_recompute_meta_optimizer MODULES test_fleet_recompute_meta_optimizer ENVS ${dist_ENVS})
	       py_test_modules(test_fleet_graph_executor MODULES test_fleet_graph_executor ENVS ${dist_ENVS})
           py_test_modules(test_fleet_gradient_merge_meta_optimizer MODULES test_fleet_gradient_merge_meta_optimizer ENVS ${dist_ENVS})
           py_test_modules(test_fleet_sharding_meta_optimizer MODULES test_fleet_sharding_meta_optimizer ENVS ${dist_ENVS})
           py_test_modules(test_fleet_amp_meta_optimizer MODULES test_fleet_amp_meta_optimizer ENVS ${dist_ENVS})
           py_test_modules(test_fleet_amp_init MODULES test_fleet_amp_init ENVS ${dist_ENVS})
           py_test_modules(test_fleet_fp16_allreduce_meta_optimizer MODULES test_fleet_fp16_allreduce_meta_optimizer ENVS ${dist_ENVS})
    	   py_test_modules(test_fleet_private_function MODULES test_fleet_private_function ENVS ${dist_ENVS})
	   py_test_modules(test_fleet_meta_optimizer_base MODULES test_fleet_meta_optimizer_base ENVS ${dist_ENVS})
	   py_test_modules(test_fleet_distributed_strategy MODULES test_fleet_distributed_strategy)
	   py_test_modules(test_fleet_static_mp_layers MODULES test_fleet_static_mp_layers)
	   #py_test_modules(test_fleet_auto MODULES test_fleet_auto ENVS ${dist_ENVS})
        if(NOT WIN32)
            py_test_modules(test_fleet_localsgd_meta_optimizer MODULES test_fleet_localsgd_meta_optimizer ENVS ${dist_ENVS})
            py_test_modules(test_fleet_lars_meta_optimizer MODULES test_fleet_lars_meta_optimizer ENVS ${dist_ENVS})
            py_test_modules(test_fleet_lamb_meta_optimizer MODULES test_fleet_lamb_meta_optimizer ENVS ${dist_ENVS})
        endif(NOT WIN32)
    endif(NOT APPLE)
    if(WITH_DGC)
        # if with dgc, test all dgc tests.
        # NOTE. dist dgc tests is already in DIST_TEST_OPS
        py_test_modules(test_dgc_op MODULES test_dgc_op)
        py_test_modules(test_dgc_momentum_op MODULES test_dgc_momentum_op)
        py_test_modules(test_dgc_optimizer MODULES test_dgc_optimizer)
        py_test_modules(test_fleet_dgc_meta_optimizer MODULES test_fleet_dgc_meta_optimizer)
    else()
        # if not with dgc, must close all dgc tests
        list(REMOVE_ITEM DIST_TEST_OPS "test_dist_mnist_dgc_nccl")
        list(REMOVE_ITEM DIST_TEST_OPS "test_dist_se_resnext_dgc")
    endif()
    if(NOT APPLE)
        if(WITH_GPU OR WITH_ROCM)
            bash_test_modules(test_c_comm_init_op START_BASH test_c_comm_init_op.sh  ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
            py_test_modules(test_launch_coverage MODULES test_launch_coverage)
        endif()

        bash_test_modules(test_fleetrun START_BASH test_fleetrun.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
        bash_test_modules(test_fleet_run_random_port START_BASH test_fleet_run_random_port.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
        bash_test_modules(test_fleet_launch_async START_BASH test_fleet_launch_async.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
        bash_test_modules(test_fleet_launch_cloud START_BASH test_fleet_launch_cloud.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
        bash_test_modules(test_fleet_launch_nproc START_BASH test_fleet_launch_nproc.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
        if(WITH_ASCEND OR WITH_ASCEND_CL)
            bash_test_modules(test_fleet_launch_ascend START_BASH test_fleet_launch_ascend.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
            bash_test_modules(test_ascend_group START_BASH test_ascend_group.sh ENVS PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR})
        endif()

        # port range (20000, 23000) is reserved for dist-ops
        set(dist_ut_port 20001)
        foreach(TEST_OP ${DIST_TEST_OPS})
            bash_test_modules(${TEST_OP} START_BASH dist_test.sh SERIAL LABELS "RUN_TYPE=EXCLUSIVE" ENVS "PADDLE_DIST_UT_PORT=${dist_ut_port}")
            MATH(EXPR dist_ut_port "${dist_ut_port}+35")
            if(dist_ut_port GREATER_EQUAL 22998)
                message(FATAL_ERROR "available ports have been exhausted:${dist_ut_port}")
            endif()
        endforeach(TEST_OP)
        # solve it later.
        bash_test_modules(test_fleet_launch_ps START_BASH test_fleet_launch_ps.sh SERIAL LABELS "RUN_TYPE=EXCLUSIVE" ENVS "PADDLE_DIST_UT_PORT=${dist_ut_port}" PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR} )
        bash_test_modules(test_new_group START_BASH test_new_group.sh SERIAL LABELS "RUN_TYPE=EXCLUSIVE" ENVS "PADDLE_DIST_UT_PORT=${dist_ut_port}+20" PADDLE_BINARY_DIR=${PADDLE_BINARY_DIR} )
    endif(NOT APPLE)
endif()

py_test_modules(test_parallel_executor_crf MODULES test_parallel_executor_crf)
# Coverage pipeline use cuda 10.1 now, profiler will random hang in cuda 10.1,
# see https://github.com/PaddlePaddle/Paddle/issues/29082 for details.
# We guess there are some bugs in cuda 10.1 or 10.2,
# since this unittest is stable in cuda 11 (py3 pipeline) now.
if(NOT WITH_COVERAGE)
  py_test_modules(test_parallel_executor_profiler MODULES test_parallel_executor_profiler)
  set_tests_properties(test_parallel_executor_profiler PROPERTIES LABELS "RUN_TYPE=DIST")
  set_tests_properties(test_parallel_executor_profiler PROPERTIES TIMEOUT 120)
endif()
py_test_modules(test_parallel_executor_transformer MODULES test_parallel_executor_transformer)
if(WIN32)
    py_test_modules(test_parallel_executor_transformer_auto_growth MODULES test_parallel_executor_transformer_auto_growth ENVS FLAGS_allocator_strategy=auto_growth CUDA_VISIBLE_DEVICES=0)
    py_test_modules(test_fuse_all_reduce_pass MODULES test_fuse_all_reduce_pass ENVS CUDA_VISIBLE_DEVICES=0)
    py_test_modules(test_feed_data_check_shape_type MODULES test_feed_data_check_shape_type ENVS CUDA_VISIBLE_DEVICES=0)
    py_test_modules(test_fetch_lod_tensor_array MODULES test_fetch_lod_tensor_array ENVS CUDA_VISIBLE_DEVICES=0)
else()
    py_test_modules(test_parallel_executor_transformer_auto_growth MODULES test_parallel_executor_transformer_auto_growth ENVS FLAGS_allocator_strategy=auto_growth)
    py_test_modules(test_fuse_all_reduce_pass MODULES test_fuse_all_reduce_pass)
    py_test_modules(test_feed_data_check_shape_type MODULES test_feed_data_check_shape_type)
    py_test_modules(test_fetch_lod_tensor_array MODULES test_fetch_lod_tensor_array)
endif()

py_test_modules(test_data_norm_op MODULES test_data_norm_op)
py_test_modules(test_fuse_bn_act_pass MODULES test_fuse_bn_act_pass ENVS FLAGS_cudnn_deterministic=1 FLAGS_cudnn_batchnorm_spatial_persistent=1 FLAGS_conv_workspace_size_limit=1000)
py_test_modules(test_fuse_bn_add_act_pass MODULES test_fuse_bn_add_act_pass ENVS FLAGS_cudnn_deterministic=1 FLAGS_cudnn_batchnorm_spatial_persistent=1 FLAGS_conv_workspace_size_limit=1000)

# NOTE: These unittests will appear NaN steadily in windows CI. After analysis,
# it is found that windows CI will run all the training unittests with the ON_INFER option turned on,
# which will not appear in other CIs. The calculation behavior of some ops in inference mode is
# inconsistent with that in non-inference mode.
if(NOT ON_INFER)
    py_test_modules(test_parallel_executor_seresnext_base_cpu MODULES test_parallel_executor_seresnext_base_cpu)
    py_test_modules(test_parallel_executor_seresnext_with_reduce_cpu MODULES test_parallel_executor_seresnext_with_reduce_cpu)
    py_test_modules(test_parallel_executor_seresnext_with_fuse_all_reduce_cpu MODULES test_parallel_executor_seresnext_with_fuse_all_reduce_cpu)
    set_tests_properties(test_parallel_executor_seresnext_base_cpu PROPERTIES TIMEOUT 900)
    set_tests_properties(test_parallel_executor_seresnext_with_reduce_cpu PROPERTIES TIMEOUT 750)
    set_tests_properties(test_parallel_executor_seresnext_with_fuse_all_reduce_cpu PROPERTIES TIMEOUT 750)
endif()

if(NOT WIN32)
    # TODO: fix these unittests failure on Windows
    py_test_modules(test_layers MODULES test_layers ENVS FLAGS_cudnn_deterministic=1)
    py_test_modules(test_ir_memory_optimize_transformer MODULES test_ir_memory_optimize_transformer)
    # FIXME(zcd): temporally disable test_parallel_executor_fetch_feed in Windows CI because of the random failure.
    py_test_modules(test_parallel_executor_fetch_feed MODULES test_parallel_executor_fetch_feed)
    set_tests_properties(test_parallel_executor_fetch_feed PROPERTIES TIMEOUT 450)
endif()

if(WITH_DISTRIBUTE AND NOT APPLE AND NOT WIN32)
    py_test_modules(test_fleet_checkpoint MODULES test_fleet_checkpoint)
    set_tests_properties(test_fleet_checkpoint PROPERTIES TIMEOUT 200)
    set_tests_properties(test_fleet_checkpoint PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_auto_checkpoint START_BASH dist_test.sh TIMEOUT 200 LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_auto_checkpoint1 START_BASH dist_test.sh TIMEOUT 200  LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_auto_checkpoint2 START_BASH dist_test.sh TIMEOUT 200  LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_auto_checkpoint3 START_BASH dist_test.sh TIMEOUT 200  LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_auto_checkpoint_multiple START_BASH dist_test.sh TIMEOUT 200  LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_auto_checkpoint_dist_basic START_BASH dist_test.sh TIMEOUT 200  LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_hdfs1 START_BASH dist_test.sh TIMEOUT 200  LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_hdfs2 START_BASH dist_test.sh TIMEOUT 200   LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
    bash_test_modules(test_hdfs3 START_BASH dist_test.sh TIMEOUT 200  LABELS "RUN_TYPE=EXCLUSIVE:NIGHTLY")
endif()

add_subdirectory(sequence)
add_subdirectory(dygraph_to_static)
add_subdirectory(rnn)

if (WITH_XPU)
    add_subdirectory(xpu)
endif()

# dist xpu tests:
if (WITH_XPU_BKCL)
    py_test(test_collective_reduce_api_xpu SRCS "test_collective_reduce_api.py")
    py_test(test_collective_allreduce_api_xpu SRCS "test_collective_allreduce_api.py")
endif()

if (WITH_ASCEND_CL)
    add_subdirectory(npu)
endif()

if (WITH_MKLDNN)
    add_subdirectory(mkldnn)
endif()

add_subdirectory(asp)

add_subdirectory(ir)

if (WITH_TESTING)
    set_property(TEST test_parallel_executor_mnist PROPERTY ENVIRONMENT GLOG_vmodule=all_reduce_deps_pass=10)
endif()

set_tests_properties(test_parallel_executor_test_while_train test_parallel_executor_mnist
        test_parallel_executor_feed_persistable_var
        test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass
        test_data_norm_op
        test_dataloader_keep_order
        test_dataloader_unkeep_order
        test_parallel_executor_inference_feed_partial_data
        test_parallel_ssa_graph_inference_feed_partial_data
        test_fetch_unmerged
        test_buffer_shared_memory_reuse_pass PROPERTIES LABELS "RUN_TYPE=DIST")
# disable test_parallel_executor_fetch_isolated_var
# set_tests_properties(test_parallel_executor_fetch_isolated_var PROPERTIES LABELS "RUN_TYPE=DIST")
set_tests_properties(test_parallel_executor_crf test_sync_batch_norm_op test_inplace_abn_op
        test_parallel_executor_seresnext_base_gpu
        test_parallel_executor_seresnext_with_reduce_gpu
        test_parallel_executor_seresnext_with_fuse_all_reduce_gpu
        test_parallel_executor_fetch_isolated_var
        PROPERTIES LABELS "RUN_TYPE=DIST")

if(NOT WIN32 AND NOT APPLE)
    set_tests_properties(test_imperative_signal_handler PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_imperative_data_loader_base PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_imperative_data_loader_fds_clear PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    # set_tests_properties(test_imperative_data_loader_exception PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_static PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_dynamic PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_exception PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_iterable_dataset_static PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_iterable_dataset_dynamic PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_dataset PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_multiprocess_dataloader_static PROPERTIES TIMEOUT 120)
endif()

if (NOT WIN32)
    set_tests_properties(test_multiprocess_reader_exception PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
    set_tests_properties(test_layers PROPERTIES TIMEOUT 120)
    set_tests_properties(test_ir_memory_optimize_transformer PROPERTIES TIMEOUT 120)
endif()

if (WITH_DISTRIBUTE AND NOT WIN32)
    set_tests_properties(test_fleet_utils PROPERTIES TIMEOUT 120)
endif()

if (WITH_DISTRIBUTE)
    set_tests_properties(test_communicator_half_async PROPERTIES TIMEOUT 120)
    set_tests_properties(test_dist_fleet_ctr2 PROPERTIES TIMEOUT 200)
    set_tests_properties(test_dist_fleet_sparse_embedding_ctr PROPERTIES TIMEOUT 200)
    set_tests_properties(test_dist_fleet_infer PROPERTIES TIMEOUT 200)
    set_tests_properties(test_dist_fleet_raw_program_optimizer PROPERTIES TIMEOUT 120)
    set_tests_properties(test_dist_fleet_raw_program_optimizer_fuse_allreduce PROPERTIES TIMEOUT 60)
endif()

if (WITH_DISTRIBUTE AND NOT APPLE)
    if(WITH_GPU OR WITH_ROCM)
        set_tests_properties(test_c_comm_init_op PROPERTIES TIMEOUT 120)
        set_tests_properties(test_dist_mnist_gradient_merge PROPERTIES TIMEOUT 120)
    endif()
endif()

# setting timeout value as 15S
set_tests_properties(test_sync_batch_norm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cross_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_lod_tensor_to_selected_rows PROPERTIES TIMEOUT 120)
set_tests_properties(test_lstm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_star_gan_with_gradient_penalty PROPERTIES TIMEOUT 120)
set_tests_properties(test_warpctc_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bicubic_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_deformable_conv_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nearest_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_profiler PROPERTIES TIMEOUT 120)
set_tests_properties(test_inplace_softmax_with_cross_entropy PROPERTIES TIMEOUT 120)
set_tests_properties(test_cross_entropy2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_fetch_unmerged PROPERTIES TIMEOUT 120)
set_tests_properties(test_gru_unit_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_activation_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(test_empty_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_transformer PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_div_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_regularizer_api PROPERTIES TIMEOUT 150)
set_tests_properties(test_multiclass_nms_op PROPERTIES TIMEOUT 120)
if(NOT WIN32)
    set_tests_properties(test_ir_memory_optimize_nlp PROPERTIES TIMEOUT 120)
endif()
set_tests_properties(test_add_reader_dependency PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilateral_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_buffer_shared_memory_reuse_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_relu_depthwise_conv_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_fleet_util PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_transformer_sorted_gradient PROPERTIES TIMEOUT 120)
set_tests_properties(test_matmul_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nearest_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_trilinear_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bicubic_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_gather_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_static_save_load PROPERTIES TIMEOUT 250)
set_tests_properties(test_pylayer_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_paddle_save_load_binary PROPERTIES TIMEOUT 120)
if (WIN32)
    set_tests_properties(test_static_save_load_large PROPERTIES TIMEOUT 900)
    set_tests_properties(test_paddle_save_load PROPERTIES TIMEOUT 250)
else()
    set_tests_properties(test_static_save_load_large PROPERTIES TIMEOUT 600)
    set_tests_properties(test_paddle_save_load PROPERTIES TIMEOUT 250)
endif()
set_tests_properties(test_imperative_selected_rows_to_lod_tensor PROPERTIES TIMEOUT 120)
set_tests_properties(test_index_select_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_ssa_graph_inference_feed_partial_data PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_crf PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_save_load PROPERTIES TIMEOUT 120)
set_tests_properties(test_concat_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_partial_eager_deletion_transformer PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_seresnext_with_reduce_gpu PROPERTIES TIMEOUT 120)
set_tests_properties(test_dropout_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_argsort_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_gather_nd_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_sub_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_row_conv_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_seresnext_with_fuse_all_reduce_gpu PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_min_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_nan_inf PROPERTIES TIMEOUT 120)
set_tests_properties(test_deformable_conv_v1_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_transformer_auto_growth PROPERTIES TIMEOUT 120)
set_tests_properties(test_py_reader_using_executor PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_add_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_weight_decay PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ptb_rnn_sorted_gradient PROPERTIES TIMEOUT 120)
set_tests_properties(test_crop_tensor_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_eager_deletion_lstm_net PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_mnist PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ptb_rnn PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_save_load_v2 PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_transpose_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv3d_transpose_part2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv3d_transpose_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_prroi_pool_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_multiprocess_dataloader_iterable_dataset_static PROPERTIES TIMEOUT 120)
set_tests_properties(test_lstm_cudnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_stack_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilinear_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_deformable_psroi_pooling PROPERTIES TIMEOUT 120)
set_tests_properties(test_trilinear_interp_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_static_runner_mnist PROPERTIES TIMEOUT 120)
set_tests_properties(test_masked_select_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_sigmoid_cross_entropy_with_logits_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_optimizer_v2 PROPERTIES TIMEOUT 120)
set_tests_properties(test_partial_sum_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cond PROPERTIES TIMEOUT 120)
set_tests_properties(test_space_to_depth_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_dyn_rnn PROPERTIES TIMEOUT 120)
set_tests_properties(test_sgd_op PROPERTIES TIMEOUT 250)
set_tests_properties(test_parallel_executor_seresnext_base_gpu PROPERTIES TIMEOUT 120)
set_tests_properties(test_norm_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(test_matrix_nms_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_generator_dataloader PROPERTIES TIMEOUT 120)
set_tests_properties(test_partial_concat_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_optimizer_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_softmax_with_cross_entropy_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_reduce_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_adam_optimizer_fp32_fp64 PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv_nn_grad PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv3d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_program_prune_backward PROPERTIES TIMEOUT 120)
set_tests_properties(test_group_norm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_optimizer PROPERTIES TIMEOUT 120)
set_tests_properties(test_pool2d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_transpose_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_eager_deletion_gru_net PROPERTIES TIMEOUT 120)
set_tests_properties(test_activation_op PROPERTIES TIMEOUT 270)
set_tests_properties(test_normal PROPERTIES TIMEOUT 120)
set_tests_properties(test_lstmp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_bilinear_interp_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_decoupled_py_reader PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_bn_act_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_op_depthwise_conv PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv2d_api PROPERTIES TIMEOUT 120)
set_tests_properties(test_elementwise_mul_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_cyclic_cifar_dataset PROPERTIES TIMEOUT 120)
set_tests_properties(test_fuse_all_reduce_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_dygraph_multi_forward PROPERTIES TIMEOUT 120)
set_tests_properties(test_norm_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_ocr_attention_model PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_mnist PROPERTIES TIMEOUT 120)
set_tests_properties(test_fused_elemwise_activation_op PROPERTIES TIMEOUT 270)
set_tests_properties(test_gru_op PROPERTIES TIMEOUT 200)
set_tests_properties(test_layer_norm_op PROPERTIES TIMEOUT 150)
set_tests_properties(test_pool3d_op PROPERTIES TIMEOUT 150)
set_tests_properties(test_regularizer PROPERTIES TIMEOUT 150)
set_tests_properties(test_imperative_resnet PROPERTIES TIMEOUT 200)
set_tests_properties(test_imperative_resnet_sorted_gradient PROPERTIES TIMEOUT 200)
set_tests_properties(test_imperative_se_resnext PROPERTIES TIMEOUT 200)
set_tests_properties(test_matmul_v2_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_strided_slice_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_translated_layer PROPERTIES TIMEOUT 120)
set_tests_properties(test_parallel_executor_inference_feed_partial_data PROPERTIES TIMEOUT 120)
set_tests_properties(test_pad3d_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataloader_keep_order PROPERTIES TIMEOUT 120)
set_tests_properties(test_mean_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_dataloader_unkeep_order PROPERTIES TIMEOUT 120)
set_tests_properties(test_reader_reset PROPERTIES TIMEOUT 120)
set_tests_properties(test_pool3d_api PROPERTIES TIMEOUT 120)
if(WITH_DISTRIBUTE AND WITH_GPU AND WITH_NCCL)
    set_tests_properties(test_parallel_dygraph_dataparallel PROPERTIES TIMEOUT 120)
    set_tests_properties(test_parallel_dygraph_unused_variables PROPERTIES TIMEOUT 120)
    set_tests_properties(test_parallel_dygraph_control_flow PROPERTIES TIMEOUT 120)
    set_tests_properties(test_parallel_dygraph_pipeline_parallel PROPERTIES TIMEOUT 120)
    set_tests_properties(test_parallel_dygraph_tensor_parallel PROPERTIES TIMEOUT 200)
    set_tests_properties(test_parallel_dygraph_sharding_parallel PROPERTIES TIMEOUT 120)
    set_tests_properties(test_parallel_dygraph_mp_layers PROPERTIES TIMEOUT 120)
    set_tests_properties(test_parallel_class_center_sample PROPERTIES TIMEOUT 120)
    if(${NCCL_VERSION} VERSION_GREATER_EQUAL 2212)
        set_tests_properties(test_parallel_dygraph_sparse_embedding PROPERTIES TIMEOUT 120)
        set_tests_properties(test_parallel_dygraph_transformer PROPERTIES TIMEOUT 120)
    endif()
endif()
if((WITH_ROCM OR WITH_GPU) AND NOT WIN32)
    set_tests_properties(test_collective_allgather_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_alltoall_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_sendrecv_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_broadcast_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_allreduce_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_new_group_api PROPERTIES TIMEOUT 120)
    if(WITH_DISTRIBUTE)
        set_tests_properties(test_pipeline PROPERTIES TIMEOUT 120)
        set_tests_properties(test_static_model_parallel PROPERTIES TIMEOUT 240)
    endif()
    set_tests_properties(test_reducescatter_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_broadcast PROPERTIES TIMEOUT 120)
    set_tests_properties(test_reducescatter PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_reduce_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_pipeline_parallel PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_reduce PROPERTIES TIMEOUT 120)
    set_tests_properties(test_allreduce PROPERTIES TIMEOUT 120)
    set_tests_properties(test_c_concat PROPERTIES TIMEOUT 120)
    set_tests_properties(test_c_split PROPERTIES TIMEOUT 120)
    set_tests_properties(test_allgather PROPERTIES TIMEOUT 120)
    set_tests_properties(test_c_identity PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_scatter_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_barrier_api PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_scatter PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_sendrecv PROPERTIES TIMEOUT 120)
    set_tests_properties(test_collective_split_embedding
        test_collective_split_embedding_none_divisible
        test_collective_split_row_linear
        test_collective_split_col_linear
        test_collective_scatter_api
        test_collective_barrier_api
        test_collective_reduce_api
        test_pipeline_parallel
        test_collective_allreduce_api
        test_new_group_api
        test_collective_broadcast_api
        test_collective_allgather_api
        test_collective_alltoall_api
        PROPERTIES LABELS "RUN_TYPE=DIST")
endif()
if(WITH_GPU OR WITH_ROCM)
    set_tests_properties(test_imperative_auto_mixed_precision PROPERTIES TIMEOUT 120)
    set_tests_properties(test_parallel_dygraph_sync_batch_norm PROPERTIES TIMEOUT 120)
    set_tests_properties(test_rank_attention_op PROPERTIES TIMEOUT 120)
endif()
set_tests_properties(test_inplace_addto_strategy PROPERTIES TIMEOUT 120)
