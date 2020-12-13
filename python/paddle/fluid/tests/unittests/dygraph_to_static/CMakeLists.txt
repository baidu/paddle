file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_MUSL)
    # TODO: In the musl docker environment provided by SEC, 
    # the test_yolov3 will randomly calculate the result of 
    # nan, error like:
    # AssertionError: 
    #   dygraph_loss: [15742.11914062 9392.61047363]
    #   static_loss: [nan, nan]
    # SEC needs to follow up on this issue, and need to be 
    # resolved before CI requared
    LIST(REMOVE_ITEM TEST_OPS test_yolov3)
endif()

if(WIN32)
    LIST(REMOVE_ITEM TEST_OPS test_convert_operators)
endif()
    

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)

set_tests_properties(test_se_resnet PROPERTIES TIMEOUT 900)
set_tests_properties(test_tsm PROPERTIES TIMEOUT 900)
if(NOT WITH_MUSL)
    set_tests_properties(test_yolov3 PROPERTIES TIMEOUT 900 LABELS "RUN_TYPE=EXCLUSIVE")
endif()
set_tests_properties(test_mobile_net PROPERTIES TIMEOUT 120)
set_tests_properties(test_seq2seq PROPERTIES TIMEOUT 120)
set_tests_properties(test_cycle_gan PROPERTIES TIMEOUT 120)
set_tests_properties(test_lac PROPERTIES TIMEOUT 120)
set_tests_properties(test_bert PROPERTIES TIMEOUT 120)
set_tests_properties(test_basic_api_transformation PROPERTIES TIMEOUT 120)
set_tests_properties(test_reinforcement_learning PROPERTIES TIMEOUT 120)
set_tests_properties(test_bmn PROPERTIES TIMEOUT 120)
#set_tests_properties(test_mnist PROPERTIES TIMEOUT 120)

if(NOT WIN32)
    set_tests_properties(test_resnet_v2 PROPERTIES TIMEOUT 120)
    #set_tests_properties(test_resnet PROPERTIES TIMEOUT 120)
endif()
if(WIN32)
    set_tests_properties(test_resnet PROPERTIES TIMEOUT 600)
    set_tests_properties(test_transformer PROPERTIES TIMEOUT 600)
endif()
