include(operators)

set(COLLECTIVE_DEPS "")

set(COLLECTIVE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach(src ${OPS})
    set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${COLLECTIVE_COMPILE_FLAGS})
endforeach()

register_operators(EXCLUDES c_gen_bkcl_id_op gen_bkcl_id_op c_gen_nccl_id_op gen_nccl_id_op c_gen_hccl_id_op gen_hccl_id_op DEPS ${COLLECTIVE_DEPS})

if(WITH_NCCL OR WITH_RCCL)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} nccl_common collective_helper)
    op_library(c_gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
    op_library(gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

if(WITH_GLOO)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} gloo_wrapper)
endif()

if(WITH_XPU_BKCL)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} collective_helper)
    op_library(c_gen_bkcl_id_op DEPS ${COLLECTIVE_DEPS})
    op_library(gen_bkcl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

if(WITH_ASCEND_CL)
    cc_library(gen_hccl_id_op_helper SRCS gen_hccl_id_op_helper.cc DEPS dynload_warpctc dynamic_loader scope)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} collective_helper gen_hccl_id_op_helper npu_utils)
    op_library(c_gen_hccl_id_op DEPS ${COLLECTIVE_DEPS})
    op_library(gen_hccl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${COLLECTIVE_DEPS} PARENT_SCOPE)
set(GLOB_COLLECTIVE_DEPS ${COLLECTIVE_DEPS} CACHE INTERNAL "collective dependency")

if(WITH_ASCEND_CL)
    set(COMMON_TEST_DEPS_FOR_HCOM c_comm_init_hccl_op c_gen_hccl_id_op gen_hccl_id_op_helper
        gen_hccl_id_op op_registry ascend_hccl flags
        dynamic_loader dynload_warpctc scope device_context enforce executor)
    cc_test(c_broadcast_op_npu_test SRCS c_broadcast_op_npu_test.cc
        DEPS c_broadcast_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM})
    cc_test(c_allreduce_sum_op_npu_test SRCS c_allreduce_sum_op_npu_test.cc
        DEPS c_allreduce_sum_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM}
        alloc_float_status_op elementwise_div_op)
    cc_test(c_reducescatter_op_npu_test SRCS c_reducescatter_op_npu_test.cc
        DEPS c_reducescatter_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM})
    cc_test(c_allgather_op_npu_test SRCS c_allgather_op_npu_test.cc
        DEPS c_allgather_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM})
    cc_test(c_reduce_sum_op_npu_test SRCS c_reduce_sum_op_npu_test.cc
            DEPS c_reduce_sum_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM})
    cc_test(c_allreduce_max_op_npu_test SRCS c_allreduce_max_op_npu_test.cc
        DEPS c_allreduce_max_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM})
    cc_test(send_v2_op_npu_test SRCS send_v2_op_npu_test.cc
        DEPS send_v2_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM})
    cc_test(recv_v2_op_npu_test SRCS recv_v2_op_npu_test.cc
        DEPS recv_v2_op ${COLLECTIVE_DEPS} ${COMMON_TEST_DEPS_FOR_HCOM})
    cc_test(c_sync_comm_stream_op_npu_test SRCS c_sync_comm_stream_op_npu_test.cc
        DEPS op_registry c_broadcast_op c_comm_init_hccl_op c_sync_comm_stream_op c_gen_hccl_id_op gen_hccl_id_op_helper ${COLLECTIVE_DEPS} ascend_hccl dynamic_loader dynload_warpctc scope device_context enforce executor)
    cc_test(c_sync_calc_stream_op_npu_test SRCS c_sync_calc_stream_op_npu_test.cc
        DEPS op_registry elementwise_add_op c_sync_calc_stream_op c_gen_hccl_id_op gen_hccl_id_op_helper ${COLLECTIVE_DEPS} ascend_hccl dynamic_loader dynload_warpctc scope device_context enforce executor)
endif()
