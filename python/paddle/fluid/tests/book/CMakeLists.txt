file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# default test
foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
    set_tests_properties(${src} PROPERTIES FIXTURES_SETUP ${src}_infer_model)
endforeach()
set_tests_properties(test_word2vec PROPERTIES TIMEOUT 120)
set_tests_properties(test_recognize_digits PROPERTIES TIMEOUT 120)
set_tests_properties(test_image_classification PROPERTIES TIMEOUT 120)
set_tests_properties(test_label_semantic_roles PROPERTIES TIMEOUT 120)
set_tests_properties(test_machine_translation PROPERTIES TIMEOUT 120)
set_tests_properties(test_rnn_encoder_decoder PROPERTIES TIMEOUT 120)
