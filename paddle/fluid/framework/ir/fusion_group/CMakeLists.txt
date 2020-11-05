cc_library(code_generator
    SRCS operation.cc code_generator.cc code_generator_helper.cc
    DEPS graph subgraph_detector)
if(WITH_GPU)
    cc_test(test_code_generator SRCS code_generator_tester.cc DEPS code_generator device_code lod_tensor graph_viz_pass)
endif()

cc_library(fusion_group_pass
    SRCS fusion_group_pass.cc elementwise_group_detector.cc
    DEPS subgraph_detector fuse_pass_base code_generator device_code)
cc_test(test_fusion_group_pass SRCS fusion_group_pass_tester.cc DEPS fusion_group_pass graph_viz_pass)
if(NOT ON_INFER)
set_tests_properties(test_code_generator PROPERTIES TIMEOUT 120)
endif()
