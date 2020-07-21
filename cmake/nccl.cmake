if(NOT WITH_GPU)
    return()
endif()

# Now we don't support NCCL on windows
if(WIN32)
    return()
endif()

set(NCCL_ROOT "/usr" CACHE PATH "NCCL ROOT")
find_path(NCCL_INCLUDE_DIR nccl.h
    PATHS ${NCCL_ROOT} ${NCCL_ROOT}/include ${NCCL_ROOT}/local/include
    $ENV{NCCL_ROOT} $ENV{NCCL_ROOT}/include $ENV{NCCL_ROOT}/local/include
    NO_DEFAULT_PATH
)

if(WITH_NCCL)
    if(NOT NCCL_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find nccl.h. "
            "You can set WITH_NCCL=OFF if you don't want to use nccl, or install nccl with \"apt install\". "
            "Please refer to https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#down for detail information.")
    endif()

    file(READ ${NCCL_INCLUDE_DIR}/nccl.h NCCL_VERSION_FILE_CONTENTS)

    string(REGEX MATCH "define NCCL_VERSION_CODE +([0-9]+)"
        NCCL_VERSION "${NCCL_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define NCCL_VERSION_CODE +([0-9]+)" "\\1"
        NCCL_VERSION "${NCCL_VERSION}")

    if("${NCCL_VERSION}" GREATER "2000")
        message(STATUS "Current NCCL header is ${NCCL_INCLUDE_DIR}/nccl.h. "
            "Current NCCL version is v${NCCL_VERSION}. ")
    else()
        # in old version nccl, it may not define NCCL_VERSION_CODE
        string(REGEX MATCH "define NCCL_MAJOR +([0-9]+)" NCCL_MAJOR_VERSION
            "${NCCL_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define NCCL_MAJOR +([0-9]+)" "\\1"
            NCCL_MAJOR_VERSION "${NCCL_MAJOR_VERSION}")
        string(REGEX MATCH "define NCCL_MINOR +([0-9]+)" NCCL_MINOR_VERSION
            "${NCCL_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define NCCL_MINOR +([0-9]+)" "\\1"
            NCCL_MINOR_VERSION "${NCCL_MINOR_VERSION}")
        string(REGEX MATCH "define NCCL_PATCH +([0-9]+)"
            NCCL_PATCH_VERSION "${NCCL_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define NCCL_PATCH +([0-9]+)" "\\1"
            NCCL_PATCH_VERSION "${NCCL_PATCH_VERSION}")

        if(NOT NCCL_MAJOR_VERSION)
            set(NCCL_VERSION "0")
        else()
            math(EXPR NCCL_VERSION
                "${NCCL_MAJOR_VERSION} * 1000 +
                 ${NCCL_MINOR_VERSION} * 100 + ${NCCL_PATCH_VERSION}")
        endif()
        add_definitions("-DNCCL_VERSION_CODE=$NCCL_VERSION")

        message(STATUS "Current NCCL header is ${NCCL_INCLUDE_DIR}/nccl.h. "
            "Current NCCL version is v${NCCL_MAJOR_VERSION}.${NCCL_MINOR_VERSION}.${NCCL_PATCH_VERSION} ")
    endif()
endif()
