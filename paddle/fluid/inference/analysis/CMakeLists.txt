unset(analysis_deps CACHE)
set(analysis_deps # analysis_deps can be extended accross the project
        framework_proto proto_desc graph pass paddle_inference_io executor pretty_log
        ir_pass_manager
        CACHE INTERNAL "")

add_subdirectory(ir_passes)
add_subdirectory(passes)

cc_library(analysis_helper SRCS helper.cc DEPS framework_proto proto_desc graph paddle_inference_io)

cc_library(ir_pass_manager SRCS ir_pass_manager.cc DEPS graph pass ${INFER_IR_PASSES} analysis_helper)

cc_library(argument INTERFACE SRCS argument.cc DEPS scope proto_desc)
cc_library(analysis_pass INTERFACE SRCS analysis_pass.cc DEPS proto_desc)

cc_library(analysis SRCS analyzer.cc
  DEPS ${analysis_deps} analysis_helper
  analysis_pass ${INFER_IR_PASSES}
  )

function(inference_analysis_test_build TARGET)
  if(WITH_TESTING)
     set(options "")
     set(oneValueArgs "")
     set(multiValueArgs SRCS EXTRA_DEPS)
     cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
     inference_base_test_build(${TARGET}
             SRCS ${analysis_test_SRCS}
             DEPS analysis pass ${GLOB_PASS_LIB} ${analysis_test_EXTRA_DEPS})
  endif()
endfunction()

function(inference_analysis_test_run TARGET)
  if(WITH_TESTING)
     set(options "")
     set(oneValueArgs "")
     set(multiValueArgs COMMAND ARGS)
     cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
     inference_base_test_run(${TARGET}
	     COMMAND ${analysis_test_COMMAND}
             ARGS ${analysis_test_ARGS})
  endif()
endfunction()

function(inference_analysis_test TARGET)
  if(WITH_TESTING)
     set(options "")
     set(oneValueArgs "")
     set(multiValueArgs SRCS ARGS EXTRA_DEPS)
     cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
     inference_base_test_build(${TARGET}
             SRCS ${analysis_test_SRCS}
             DEPS analysis pass ${GLOB_PASS_LIB} ${analysis_test_EXTRA_DEPS})
     inference_base_test_run(${TARGET}
	     COMMAND ${TARGET}
             ARGS ${analysis_test_ARGS})
  endif()
endfunction(inference_analysis_test)


inference_analysis_test(test_analyzer
  SRCS analyzer_tester.cc
  EXTRA_DEPS reset_tensor_array paddle_inference_api
  ARGS --inference_model_dir=${WORD2VEC_MODEL_DIR})
