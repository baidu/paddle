if(NOT WITH_GPU)
    return()
endif()

if(WIN32)
    set(CUDNN_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
else(WIN32)
    set(CUDNN_ROOT "/usr" CACHE PATH "CUDNN ROOT")
endif(WIN32)

find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS ${CUDNN_ROOT} ${CUDNN_ROOT}/include
    $ENV{CUDNN_ROOT} $ENV{CUDNN_ROOT}/include ${CUDA_TOOLKIT_INCLUDE}
    NO_DEFAULT_PATH
)

get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
    set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(APPEND CUDNN_CHECK_LIBRARY_DIRS
    ${CUDNN_ROOT}
    ${CUDNN_ROOT}/lib64
    ${CUDNN_ROOT}/lib
    ${CUDNN_ROOT}/lib/${TARGET_ARCH}-linux-gnu
    ${CUDNN_ROOT}/local/cuda-${CUDA_VERSION}/targets/${TARGET_ARCH}-linux/lib/
    $ENV{CUDNN_ROOT}
    $ENV{CUDNN_ROOT}/lib64
    $ENV{CUDNN_ROOT}/lib
    /usr/lib
	${CUDA_TOOLKIT_ROOT_DIR}
	${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
	)
set(CUDNN_LIB_NAME "")

if (LINUX)
    set(CUDNN_LIB_NAME "libcudnn.so")
endif(LINUX)

if(WIN32)
    # only support cudnn7
    set(CUDNN_LIB_NAME "cudnn.lib" "cudnn64_7.dll")
endif(WIN32)

if(APPLE)
    set(CUDNN_LIB_NAME "libcudnn.dylib" "libcudnn.so")
endif(APPLE)

find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME} # libcudnn_static.a
    PATHS ${CUDNN_CHECK_LIBRARY_DIRS} ${CUDNN_INCLUDE_DIR} ${__libpath_hist}
          NO_DEFAULT_PATH
    DOC "Path to cuDNN library.")


if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    set(CUDNN_FOUND ON)
else()
    set(CUDNN_FOUND OFF)
endif()

macro(find_cudnn_version cudnn_header_file) 
    file(READ ${cudnn_header_file} CUDNN_VERSION_FILE_CONTENTS)
    get_filename_component(CUDNN_LIB_PATH ${CUDNN_LIBRARY} DIRECTORY)

    string(REGEX MATCH "define CUDNN_VERSION +([0-9]+)"
        CUDNN_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_VERSION +([0-9]+)" "\\1"
        CUDNN_VERSION "${CUDNN_VERSION}")

    if("${CUDNN_VERSION}" STREQUAL "2000")
        message(STATUS "Current cuDNN version is v2. ")
    else()
        string(REGEX MATCH "define CUDNN_MAJOR +([0-9]+)" CUDNN_MAJOR_VERSION
            "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR +([0-9]+)" "\\1"
            CUDNN_MAJOR_VERSION "${CUDNN_MAJOR_VERSION}")
        string(REGEX MATCH "define CUDNN_MINOR +([0-9]+)" CUDNN_MINOR_VERSION
            "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR +([0-9]+)" "\\1"
            CUDNN_MINOR_VERSION "${CUDNN_MINOR_VERSION}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL +([0-9]+)"
            CUDNN_PATCHLEVEL_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL +([0-9]+)" "\\1"
            CUDNN_PATCHLEVEL_VERSION "${CUDNN_PATCHLEVEL_VERSION}")

        if(NOT CUDNN_MAJOR_VERSION)
            set(CUDNN_VERSION "???")
        else()
            add_definitions("-DCUDNN_MAJOR_VERSION=\"${CUDNN_MAJOR_VERSION}\"")
            math(EXPR CUDNN_VERSION
                "${CUDNN_MAJOR_VERSION} * 1000 +
                 ${CUDNN_MINOR_VERSION} * 100 + ${CUDNN_PATCHLEVEL_VERSION}")
            message(STATUS "Current cuDNN header is ${cudnn_header_file} "
              "Current cuDNN version is v${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCHLEVEL_VERSION}. ")
        endif()
    endif()
endmacro()

if(CUDNN_FOUND)
  find_cudnn_version(${CUDNN_INCLUDE_DIR}/cudnn.h) 
  if (NOT CUDNN_MAJOR_VERSION) 
    find_cudnn_version(${CUDNN_INCLUDE_DIR}/cudnn_version.h) 
  endif()
endif()
