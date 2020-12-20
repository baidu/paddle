set(PYBIND_DEPS pybind python proto_desc memory executor fleet_wrapper box_wrapper prune
  feed_fetch_method pass_builder parallel_executor profiler layer tracer engine scope_pool
  analysis_predictor imperative_profiler imperative_flag save_load_util dlpack_tensor device_context
  gloo_wrapper infer_io_utils heter_wrapper generator op_version_registry ps_gpu_wrapper)

if (WITH_NCCL)
  set(PYBIND_DEPS ${PYBIND_DEPS} nccl_wrapper)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
endif()

if(NOT WIN32)
  set(PYBIND_DEPS ${PYBIND_DEPS} data_loader)
  set(PYBIND_DEPS ${PYBIND_DEPS} mmap_allocator)
  if (WITH_NCCL)
    set(PYBIND_DEPS ${PYBIND_DEPS} nccl_context)
  endif()
endif(NOT WIN32)

if(WITH_PYTHON)
  list(APPEND PYBIND_DEPS py_func_op)
endif()

if (WITH_DISTRIBUTE)
  list(APPEND PYBIND_DEPS communicator)
endif()

set(PYBIND_SRCS
  pybind.cc
  exception.cc
  protobuf.cc
  const_value.cc
  global_value_getter_setter.cc
  reader_py.cc
  fleet_wrapper_py.cc
  heter_wrapper_py.cc
  ps_gpu_wrapper_py.cc
  gloo_wrapper_py.cc
  box_helper_py.cc
  data_set_py.cc
  imperative.cc
  ir.cc
  inference_api.cc
  compatible.cc
  generator_py.cc)

if(WITH_GLOO)
  set(PYBIND_DEPS ${PYBIND_DEPS} gloo_context)
  set(PYBIND_SRCS ${PYBIND_SRCS} gloo_context_py.cc)
endif(WITH_GLOO)

if (WITH_CRYPTO)
  set(PYBIND_DEPS ${PYBIND_DEPS} paddle_crypto)
  set(PYBIND_SRCS ${PYBIND_SRCS} crypto.cc)
endif (WITH_CRYPTO)

if (WITH_DISTRIBUTE)
  list(APPEND PYBIND_SRCS communicator_py.cc)
endif()

if (WITH_NCCL)
  list(APPEND PYBIND_SRCS nccl_wrapper_py.cc)
endif()

if(WITH_PYTHON)
  # generate op pybind functions automatically for dygraph.
  set(OP_FUNCTION_GENERETOR_DEPS pybind proto_desc executor layer tracer engine imperative_profiler imperative_flag)
  list(APPEND OP_FUNCTION_GENERETOR_DEPS ${GLOB_OP_LIB})
  list(APPEND OP_FUNCTION_GENERETOR_DEPS ${GLOB_OPERATOR_DEPS})

  if(WITH_NCCL)
    list(APPEND OP_FUNCTION_GENERETOR_DEPS nccl_context)
  endif(WITH_NCCL)

  add_executable(op_function_generator op_function_generator.cc)
  target_link_libraries(op_function_generator ${OP_FUNCTION_GENERETOR_DEPS})
  get_property (os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(op_function_generator ${os_dependency_modules})

  set(impl_file ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function_impl.h)
  set(tmp_impl_file ${impl_file}.tmp)

  if(WIN32)
    file(WRITE ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/op_function_generator_retry.bat ""
    "set build_times=1\n"
    ":retry\n"
    "ECHO op_function_generator run %build_times% time\n"
    "${CMAKE_BINARY_DIR}/paddle/fluid/pybind/${CMAKE_BUILD_TYPE}/op_function_generator ${impl_file}\n"
    "if %ERRORLEVEL% NEQ 0 (\n"
    "    set /a build_times=%build_times%+1\n"
    "    if %build_times% GTR 100 (\n"
    "        exit /b 1\n"
    "    ) else (\n"
    "        goto :retry\n"
    "    )\n"
    ")\n"
    "exit /b 0")

    add_custom_command(TARGET op_function_generator POST_BUILD
          COMMAND ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/op_function_generator_retry.bat
    )

    if(${CBLAS_PROVIDER} STREQUAL MKLML)
      add_custom_command(TARGET op_function_generator
            PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_LIB} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
            COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
          )
    else(${CBLAS_PROVIDER} STREQUAL EXTERN_OPENBLAS)
      add_custom_command(TARGET op_function_generator
            PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_SHARED_LIB} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
          )
    endif()
    if(WITH_MKLDNN)
      add_custom_command(TARGET op_function_generator
          PRE_LINK
          COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_SHARED_LIB} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
          )
    endif()
  else(WIN32)
    # If there are no *.so in /usr/lib or LD_LIBRARY_PATH, 
    # copy these *.so to current directory and append current directory to
    # LD_LIBRARY_PATH. This is different with Windows platformm, which search 
    # *.dll in current directory automatically.
    add_custom_command(TARGET op_function_generator
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:."
              "${CMAKE_CURRENT_BINARY_DIR}/op_function_generator"
              "${tmp_impl_file}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file} ${impl_file}
          COMMENT "copy_if_different ${impl_file}"
          VERBATIM
    )
    if(WITH_MKL)
      add_custom_command(TARGET op_function_generator
            PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_LIB} ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB} ${CMAKE_CURRENT_BINARY_DIR}
            )
    endif(WITH_MKL)
    if(WITH_MKLDNN)
      add_custom_command(TARGET op_function_generator
          PRE_LINK
          COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_SHARED_LIB} ${CMAKE_CURRENT_BINARY_DIR}
          )
    endif(WITH_MKLDNN)
  endif(WIN32)

  if(WITH_ROCM_PLATFORM)
    cc_library(paddle_pybind SHARED
      SRCS ${PYBIND_SRCS}
      DEPS ${PYBIND_DEPS}
      ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})
  else()
    cc_library(paddle_pybind SHARED
      SRCS ${PYBIND_SRCS}
      DEPS ${PYBIND_DEPS}
      ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})
    if(NOT APPLE AND NOT WIN32)
      target_link_libraries(paddle_pybind rt)
    endif(NOT APPLE AND NOT WIN32)
  endif(WITH_ROCM_PLATFORM)

  get_property (os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(paddle_pybind ${os_dependency_modules})
  add_dependencies(paddle_pybind op_function_generator)
endif(WITH_PYTHON)
