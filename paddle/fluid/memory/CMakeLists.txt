add_subdirectory(detail)
add_subdirectory(allocation)

if (WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
else ()
    set(MKLDNN_CTX_DEPS)
endif()

cc_library(malloc SRCS malloc.cc DEPS
    place enforce allocator_facade profiler ${MKLDNN_CTX_DEPS})
cc_library(memcpy SRCS memcpy.cc DEPS place device_context)

cc_library(memory DEPS malloc memcpy)

if (WITH_GPU)
    nv_test(malloc_test
            SRCS malloc_test.cu
            DEPS device_context malloc)
endif()
if (WITH_ROCM_PLATFORM)
    hip_test(malloc_test
            SRCS malloc_test.hip.cu
            DEPS device_context malloc)
endif()

#if (WITH_GPU)
#   nv_test(pinned_memory_test SRCS pinned_memory_test.cu  DEPS place memory)
#endif()
