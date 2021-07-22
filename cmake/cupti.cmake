if(NOT WITH_GPU)
    return()
endif()


set(CUPTI_ROOT "/usr" CACHE PATH "CUPTI ROOT")
find_path(CUPTI_INCLUDE_DIR cupti.h
        PATHS ${CUPTI_ROOT} ${CUPTI_ROOT}/include
        $ENV{CUPTI_ROOT} $ENV{CUPTI_ROOT}/include
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include
        NO_DEFAULT_PATH
        )

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
    set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(APPEND CUPTI_CHECK_LIBRARY_DIRS
        ${CUPTI_ROOT}
        ${CUPTI_ROOT}/lib64
        ${CUPTI_ROOT}/lib
        ${CUPTI_ROOT}/lib/${TARGET_ARCH}-linux-gnu
        $ENV{CUPTI_ROOT}
        $ENV{CUPTI_ROOT}/lib64
        $ENV{CUPTI_ROOT}/lib
        /usr/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64)
find_library(CUPTI_LIBRARY NAMES libcupti.so libcupti.dylib # libcupti_static.a
       PATHS ${CUPTI_CHECK_LIBRARY_DIRS} ${CUPTI_INCLUDE_DIR} ${__libpath_hist}
       NO_DEFAULT_PATH
       DOC "Path to cuPTI library.")

get_filename_component(CUPTI_LIBRARY_PATH ${CUPTI_LIBRARY} DIRECTORY)
if(CUPTI_INCLUDE_DIR AND CUPTI_LIBRARY)
    set(CUPTI_FOUND ON)
else()
    set(CUPTI_FOUND OFF)
endif()
