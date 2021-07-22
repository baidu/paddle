set_source_files_properties(table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(table_test SRCS table_test.cc DEPS common_table table tensor_accessor
ps_framework_proto ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(dense_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(dense_table_test SRCS dense_table_test.cc DEPS common_table table
tensor_accessor ps_framework_proto ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(barrier_table_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(barrier_table_test SRCS barrier_table_test.cc DEPS common_table table tensor_accessor ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(brpc_service_dense_sgd_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(brpc_service_dense_sgd_test SRCS brpc_service_dense_sgd_test.cc DEPS scope server client communicator ps_service boost table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(brpc_service_sparse_sgd_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(brpc_service_sparse_sgd_test SRCS brpc_service_sparse_sgd_test.cc DEPS scope server client communicator ps_service boost table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(brpc_utils_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(brpc_utils_test SRCS brpc_utils_test.cc DEPS brpc_utils scope math_function ${COMMON_DEPS} ${RPC_DEPS})

set_source_files_properties(graph_node_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(graph_node_test SRCS graph_node_test.cc DEPS graph_py_service scope server client communicator ps_service boost table ps_framework_proto ${COMMON_DEPS})

set_source_files_properties(index_wrapper_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_test(index_wrapper_test SRCS index_wrapper_test.cc DEPS index_wrapper graph_py_service scope server client communicator ps_service boost table ps_framework_proto ${COMMON_DEPS})
