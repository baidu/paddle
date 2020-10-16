file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
        py_test(${src} SRCS ${src}.py)
endforeach()
set_tests_properties(test_image_classification_fp16 PROPERTIES TIMEOUT 120)
set_tests_properties(test_weight_decay_extend PROPERTIES TIMEOUT 120)
