include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/elementwise.
    include(unity_build_rule.cmake)
endif()
register_operators(DEPS op_version_registry)

cc_test(test_elementwise_add_op_inplace SRCS test_elementwise_add_op_inplace.cc DEPS op_registry elementwise_add_op scope device_context enforce executor)
cc_test(test_elementwise_div_grad_grad SRCS test_elementwise_div_grad_grad.cc DEPS op_registry elementwise_div_op scope device_context enforce executor)
cc_test(test_elementwise_add_grad_grad SRCS test_elementwise_add_grad_grad.cc DEPS op_registry elementwise_add_op scope device_context enforce executor)

if(WITH_ASCEND_CL)
cc_test(elementwise_op_npu_test SRCS elementwise_op_npu_test.cc DEPS op_registry elementwise_add_op elementwise_sub_op elementwise_mul_op elementwise_div_op elementwise_max_op elementwise_min_op elementwise_pow_op scope device_context enforce executor)
endif()
