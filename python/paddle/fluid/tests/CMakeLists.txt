file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")


foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

add_subdirectory(unittests)
add_subdirectory(book)

if(NOT APPLE AND NOT WIN32)
  add_subdirectory(custom_op)
endif()
set_tests_properties(test_beam_search_decoder PROPERTIES TIMEOUT 120)
