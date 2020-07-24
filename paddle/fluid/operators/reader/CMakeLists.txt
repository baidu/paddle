include(operators)

cc_library(reader_op_registry SRCS reader_op_registry.cc DEPS operator op_registry reader)
set(LOCAL_READER_LIBS)

function(reader_library TARGET_NAME)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    set(options "")
    set(common_deps reader_op_registry)
    cmake_parse_arguments(reader_library "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})
    op_library(${TARGET_NAME} SRCS ${reader_library_SRCS} DEPS ${common_deps} ${reader_library_DEPS})
    set(LOCAL_READER_LIBS
            ${TARGET_NAME}
            ${LOCAL_READER_LIBS}
        PARENT_SCOPE)
endfunction()

cc_library(py_reader SRCS py_reader.cc DEPS reader)
cc_library(buffered_reader SRCS buffered_reader.cc DEPS reader simple_threadpool)

reader_library(create_double_buffer_reader_op SRCS create_double_buffer_reader_op.cc DEPS buffered_reader)
reader_library(create_py_reader_op SRCS create_py_reader_op.cc DEPS py_reader)

op_library(read_op DEPS py_reader buffered_reader)

cc_test(reader_blocking_queue_test SRCS reader_blocking_queue_test.cc DEPS timer)
# Export local libraries to parent
# set(READER_LIBRARY ${LOCAL_READER_LIBS} PARENT_SCOPE)
