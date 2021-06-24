file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
