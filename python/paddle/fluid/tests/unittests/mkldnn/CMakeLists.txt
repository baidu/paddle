file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

get_filename_component(PARNET_PATH ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)

function(py_test_modules_mkdnn TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules_mkdnn "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(WITH_COVERAGE AND NOT (WITH_INCREMENTAL_COVERAGE AND "$ENV{PADDLE_GIT_DIFF_PY_FILE}" STREQUAL ""))
        add_test(NAME ${TARGET_NAME}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} ${py_test_modules_ENVS}
                COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
                ${PYTHON_EXECUTABLE} -m coverage run --branch -p ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
        add_test(NAME ${TARGET_NAME}
                COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHONPATH} ${py_test_modules_ENVS}
                ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if (py_test_modules_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    if(WIN32)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 150)
    endif()
  endif()
endfunction()

foreach(TEST_OP ${TEST_OPS})
    py_test_modules_mkdnn(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)
set_tests_properties(test_concat_mkldnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_conv3d_mkldnn_op PROPERTIES TIMEOUT 120)
set_tests_properties(test_flags_mkldnn_ops_on_off PROPERTIES TIMEOUT 120)
