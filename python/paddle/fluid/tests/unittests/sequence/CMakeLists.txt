file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
set_tests_properties(test_sequence_conv PROPERTIES TIMEOUT 120)
set_tests_properties(test_sequence_concat PROPERTIES TIMEOUT 120)
set_tests_properties(test_sequence_pool PROPERTIES TIMEOUT 120)
