file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

file(GLOB DIST_TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_dist_*.py")
string(REPLACE ".py" "" DIST_TEST_OPS "${DIST_TEST_OPS}")

foreach(TEST_OP ${DIST_TEST_OPS})
    list(REMOVE_ITEM TEST_OPS ${TEST_OP})
endforeach()

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()

function(py_dist_test TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_dist_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(WITH_COVERAGE AND WITH_GPU AND WITH_NCCL AND NOT WIN32)
      add_test(NAME ${TARGET_NAME}
               COMMAND ${CMAKE_COMMAND} -E env FLAGS_init_allocated_mem=true FLAGS_cudnn_deterministic=true
               FLAGS_cpu_deterministic=true NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
               PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_dist_test_ENVS}
               COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
               ${PYTHON_EXECUTABLE} -u ${py_dist_test_SRCS} ${py_dist_test_ARGS}
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      # No unit test should exceed 10 minutes.
      set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600 LABELS "RUN_TYPE=DIST")
    endif()

    
  endif()
endfunction()

foreach(src ${DIST_TEST_OPS})
    message(STATUS ${src})
    py_dist_test(${src} SRCS ${src}.py)
endforeach()

set_tests_properties(test_pretrained_model PROPERTIES TIMEOUT 600)
set_tests_properties(test_dataset_imdb PROPERTIES TIMEOUT 150)
