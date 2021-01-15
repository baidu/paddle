# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)
# Always invoke `FIND_PACKAGE(Protobuf)` for importing function protobuf_generate_cpp
IF(NOT WIN32)
    FIND_PACKAGE(Protobuf QUIET)
ENDIF(NOT WIN32)

UNSET_VAR(PROTOBUF_INCLUDE_DIR)
UNSET_VAR(PROTOBUF_FOUND)
UNSET_VAR(PROTOBUF_PROTOC_EXECUTABLE)
UNSET_VAR(PROTOBUF_PROTOC_LIBRARY)
UNSET_VAR(PROTOBUF_LITE_LIBRARY)
UNSET_VAR(PROTOBUF_LIBRARY)
UNSET_VAR(PROTOBUF_INCLUDE_DIR)
UNSET_VAR(Protobuf_PROTOC_EXECUTABLE)
function(protobuf_generate_python SRCS)
    # shameless copy from https://github.com/Kitware/CMake/blob/master/Modules/FindProtobuf.cmake
    if(NOT ARGN)
        message(SEND_ERROR "Error: PROTOBUF_GENERATE_PYTHON() called without any proto files")
        return()
    endif()

    if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
        # Create an include path for each file specified
        foreach(FIL ${ARGN})
            get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
            get_filename_component(ABS_PATH ${ABS_FIL} PATH)
            list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
            if(${_contains_already} EQUAL -1)
                list(APPEND _protobuf_include_path -I ${ABS_PATH})
            endif()
        endforeach()
    else()
        set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
        set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
    endif()

    if(DEFINED Protobuf_IMPORT_DIRS)
        foreach(DIR ${Protobuf_IMPORT_DIRS})
            get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
            list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
            if(${_contains_already} EQUAL -1)
                list(APPEND _protobuf_include_path -I ${ABS_PATH})
            endif()
        endforeach()
    endif()

    set(${SRCS})
    foreach(FIL ${ARGN})
        get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
        get_filename_component(FIL_WE ${FIL} NAME_WE)
        if(NOT PROTOBUF_GENERATE_CPP_APPEND_PATH)
            get_filename_component(FIL_DIR ${FIL} DIRECTORY)
            if(FIL_DIR)
                set(FIL_WE "${FIL_DIR}/${FIL_WE}")
            endif()
        endif()
        list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py")
        add_custom_command(
                OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py"
                COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${ABS_FIL}
                DEPENDS ${ABS_FIL} ${PROTOBUF_PROTOC_EXECUTABLE}
                COMMENT "Running Python protocol buffer compiler on ${FIL}"
                VERBATIM )
    endforeach()

    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()

# Print and set the protobuf library information,
# finish this cmake process and exit from this file.
macro(PROMPT_PROTOBUF_LIB)
    SET(protobuf_DEPS ${ARGN})

    MESSAGE(STATUS "Protobuf protoc executable: ${PROTOBUF_PROTOC_EXECUTABLE}")
    MESSAGE(STATUS "Protobuf-lite library: ${PROTOBUF_LITE_LIBRARY}")
    MESSAGE(STATUS "Protobuf library: ${PROTOBUF_LIBRARY}")
    MESSAGE(STATUS "Protoc library: ${PROTOBUF_PROTOC_LIBRARY}")
    MESSAGE(STATUS "Protobuf version: ${PROTOBUF_VERSION}")
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

    # Assuming that all the protobuf libraries are of the same type.
    IF(${PROTOBUF_LIBRARY} MATCHES ${CMAKE_STATIC_LIBRARY_SUFFIX})
        SET(protobuf_LIBTYPE STATIC)
    ELSEIF(${PROTOBUF_LIBRARY} MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$")
        SET(protobuf_LIBTYPE SHARED)
    ELSE()
        MESSAGE(FATAL_ERROR "Unknown library type: ${PROTOBUF_LIBRARY}")
    ENDIF()

    ADD_LIBRARY(protobuf ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protobuf PROPERTY IMPORTED_LOCATION ${PROTOBUF_LIBRARY})

    ADD_LIBRARY(protobuf_lite ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protobuf_lite PROPERTY IMPORTED_LOCATION ${PROTOBUF_LITE_LIBRARY})

    ADD_LIBRARY(libprotoc ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET libprotoc PROPERTY IMPORTED_LOCATION ${PROTOC_LIBRARY})

    ADD_EXECUTABLE(protoc IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protoc PROPERTY IMPORTED_LOCATION ${PROTOBUF_PROTOC_EXECUTABLE})
    # FIND_Protobuf.cmake uses `Protobuf_PROTOC_EXECUTABLE`.
    # make `protobuf_generate_cpp` happy.
    SET(Protobuf_PROTOC_EXECUTABLE ${PROTOBUF_PROTOC_EXECUTABLE})

    FOREACH(dep ${protobuf_DEPS})
        ADD_DEPENDENCIES(protobuf ${dep})
        ADD_DEPENDENCIES(protobuf_lite ${dep})
        ADD_DEPENDENCIES(libprotoc ${dep})
        ADD_DEPENDENCIES(protoc ${dep})
    ENDFOREACH()

    RETURN()
endmacro()
macro(SET_PROTOBUF_VERSION)
    EXEC_PROGRAM(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --version OUTPUT_VARIABLE PROTOBUF_VERSION)
    STRING(REGEX MATCH "[0-9]+.[0-9]+" PROTOBUF_VERSION "${PROTOBUF_VERSION}")
endmacro()

set(PROTOBUF_ROOT "" CACHE PATH "Folder contains protobuf")
IF (WIN32)
    SET(PROTOBUF_ROOT ${THIRD_PARTY_PATH}/install/protobuf)
ENDIF(WIN32)

if (NOT "${PROTOBUF_ROOT}" STREQUAL "")
    find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h PATHS ${PROTOBUF_ROOT}/include NO_DEFAULT_PATH)
    find_library(PROTOBUF_LIBRARY protobuf libprotobuf.lib PATHS ${PROTOBUF_ROOT}/lib NO_DEFAULT_PATH)
    find_library(PROTOBUF_LITE_LIBRARY protobuf-lite libprotobuf-lite.lib PATHS ${PROTOBUF_ROOT}/lib NO_DEFAULT_PATH)
    find_library(PROTOBUF_PROTOC_LIBRARY protoc libprotoc.lib PATHS ${PROTOBUF_ROOT}/lib NO_DEFAULT_PATH)
    find_program(PROTOBUF_PROTOC_EXECUTABLE protoc PATHS ${PROTOBUF_ROOT}/bin NO_DEFAULT_PATH)
    if (PROTOBUF_INCLUDE_DIR AND PROTOBUF_LIBRARY AND PROTOBUF_LITE_LIBRARY AND PROTOBUF_PROTOC_LIBRARY AND PROTOBUF_PROTOC_EXECUTABLE)
        SET(PROTOBUF_FOUND true)
        message(STATUS "Using custom protobuf library in ${PROTOBUF_ROOT}.")
        SET_PROTOBUF_VERSION()
        PROMPT_PROTOBUF_LIB()
    endif()
endif()

FUNCTION(build_protobuf TARGET_NAME BUILD_FOR_HOST)
    STRING(REPLACE "extern_" "" TARGET_DIR_NAME "${TARGET_NAME}")
    SET(PROTOBUF_PREFIX_DIR  ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME})
    SET(PROTOBUF_SOURCE_DIR  ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME}/src/${TARGET_NAME})
    SET(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${TARGET_DIR_NAME})

    SET(${TARGET_NAME}_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(${TARGET_NAME}_LITE_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_EXECUTABLE
        "${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
         PARENT_SCOPE)

    SET(OPTIONAL_CACHE_ARGS "")
    SET(OPTIONAL_ARGS "")
    IF(BUILD_FOR_HOST)
        SET(OPTIONAL_ARGS "-Dprotobuf_WITH_ZLIB=OFF")
    ELSE()
        SET(OPTIONAL_ARGS
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
            "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
            "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}"
	    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"
            "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
            "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
            "-Dprotobuf_WITH_ZLIB=ON"
            "-DZLIB_ROOT:FILEPATH=${ZLIB_ROOT}"
            ${EXTERNAL_OPTIONAL_ARGS})
        SET(OPTIONAL_CACHE_ARGS "-DZLIB_ROOT:STRING=${ZLIB_ROOT}")
    ENDIF()
    IF(WIN32)
        SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} 
            "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
            "-DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-Dprotobuf_MSVC_STATIC_RUNTIME=${MSVC_STATIC_CRT}")
    ENDIF()

    SET(PROTOBUF_REPOSITORY  https://gitee.com/tianjianhe/protobuf.git)
    SET(PROTOBUF_TAG         v3.8.0)

    cache_third_party(${TARGET_NAME}
        REPOSITORY    ${PROTOBUF_REPOSITORY}
        TAG           ${PROTOBUF_TAG}
        DIR           PROTOBUF_SOURCE_DIR)

    ExternalProject_Add(
        ${TARGET_NAME}
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        "${PROTOBUF_DOWNLOAD_CMD}"
        PREFIX          ${PROTOBUF_PREFIX_DIR}
        SOURCE_DIR      ${PROTOBUF_SOURCE_DIR}
        UPDATE_COMMAND  ""
        DEPENDS         zlib
        CONFIGURE_COMMAND
                        ${CMAKE_COMMAND} ${PROTOBUF_SOURCE_DIR}/cmake
                         ${OPTIONAL_ARGS}
                        -Dprotobuf_BUILD_TESTS=OFF
                        -DCMAKE_SKIP_RPATH=ON
                        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                        -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
                        -DCMAKE_INSTALL_LIBDIR=lib
                        -DBUILD_SHARED_LIBS=OFF
        CMAKE_CACHE_ARGS
                        -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
                        -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
                        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
                        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                        ${OPTIONAL_CACHE_ARGS}
    )
ENDFUNCTION()

# SET(PROTOBUF_VERSION 3.1.0)

IF(NOT PROTOBUF_FOUND)
    build_protobuf(extern_protobuf FALSE)

    SET(PROTOBUF_INCLUDE_DIR ${extern_protobuf_INCLUDE_DIR}
        CACHE PATH "protobuf include directory." FORCE)
    SET(PROTOBUF_LITE_LIBRARY ${extern_protobuf_LITE_LIBRARY}
        CACHE FILEPATH "protobuf lite library." FORCE)
    SET(PROTOBUF_LIBRARY ${extern_protobuf_LIBRARY}
        CACHE FILEPATH "protobuf library." FORCE)
    SET(PROTOBUF_PROTOC_LIBRARY ${extern_protobuf_PROTOC_LIBRARY}
        CACHE FILEPATH "protoc library." FORCE)

    SET(PROTOBUF_PROTOC_EXECUTABLE ${extern_protobuf_PROTOC_EXECUTABLE}
        CACHE FILEPATH "protobuf executable." FORCE)
    PROMPT_PROTOBUF_LIB(extern_protobuf)
ENDIF(NOT PROTOBUF_FOUND)
