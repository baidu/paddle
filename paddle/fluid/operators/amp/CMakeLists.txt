include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/amp.
    include(unity_build_rule.cmake)
endif()
register_operators()

if(WITH_ASCEND_CL)
    cc_test(check_finite_and_unscale_op_npu_test SRCS check_finite_and_unscale_op_npu_test.cc DEPS op_registry check_finite_and_unscale_op scope device_context enforce executor)
endif()
