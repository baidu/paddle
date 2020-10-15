function(train_test TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs ARGS)
    cmake_parse_arguments(train_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT APPLE AND NOT WIN32)
        cc_test(test_train_${TARGET_NAME}
                SRCS test_train_${TARGET_NAME}.cc
                DEPS paddle_fluid_shared
                ARGS --dirname=${PYTHON_TESTS_DIR}/book/)
    else()
        cc_test(test_train_${TARGET_NAME}${arg}
                SRCS test_train_${TARGET_NAME}.cc
                DEPS paddle_fluid_api
                ARGS --dirname=${PYTHON_TESTS_DIR}/book/)
    endif()
    set_tests_properties(test_train_${TARGET_NAME}
            PROPERTIES FIXTURES_REQUIRED test_${TARGET_NAME}_infer_model)
    if(NOT WIN32 AND NOT APPLE)
        set_tests_properties(test_train_${TARGET_NAME}
                PROPERTIES TIMEOUT 150)
    endif()
endfunction(train_test)


if(WITH_TESTING)
  train_test(recognize_digits)
endif()
