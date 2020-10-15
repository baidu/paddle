# Attention: cmake will append these flags to compile command automatically.
# So if you want to add global option, change this file rather than flags.cmake

# NOT WIN32
# DEBUG:  default: "-g"
# RELEASE:  default: "-O3 -DNDEBUG"
# RELWITHDEBINFO: default: "-O2 -g -DNDEBUG"
# MINSIZEREL: default: "-O2 -g -DNDEBUG"

if(NOT WIN32)
    set(CMAKE_C_FLAGS_DEBUG "-g")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")

    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
endif()

if(WITH_GPU)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")
endif()

if(WIN32)
    set(WIN_PROPS ${CMAKE_SOURCE_DIR}/cmake/paddle_win.props)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Os -DNDEBUG")
endif()

