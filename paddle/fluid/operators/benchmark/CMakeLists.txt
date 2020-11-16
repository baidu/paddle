cc_test(op_tester SRCS op_tester.cc op_tester_config.cc
        DEPS memory timer framework_proto proto_desc lod_tensor op_registry
        device_context scope ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})
if(WITH_TESTING AND WITH_GPU)
        set_target_properties(op_tester PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
