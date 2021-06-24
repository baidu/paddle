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

# Find the CBlas and lapack libraries
#
# It will search MKLML, OpenBlas, reference-cblas, extern-openblas in order.
#
# If any cblas implementation found, the following variable will be set.
#    CBLAS_PROVIDER  # one of MKLML, OPENBLAS, REFERENCE
#    CBLAS_INC_DIR   # the include directory for cblas.
#    CBLAS_LIBS      # a list of libraries should be linked by paddle.
#                    # Each library should be full path to object file.

generate_dummy_static_lib(LIB_NAME "cblas" GENERATOR "cblas.cmake")

if(WITH_LIBXSMM)
  target_link_libraries(cblas ${LIBXSMM_LIBS})
  add_dependencies(cblas extern_libxsmm)
endif()

## Find MKLML First.
if(WITH_MKLML)
  include(external/mklml)       # download, install mklml package
  set(CBLAS_PROVIDER MKLML)
  set(CBLAS_INC_DIR  ${MKLML_INC_DIR})
  set(CBLAS_LIBRARIES ${MKLML_LIB})

  add_definitions(-DPADDLE_WITH_MKLML)
  add_definitions(-DLAPACK_FOUND)

  add_dependencies(cblas mklml)
  target_link_libraries(cblas dynload_mklml)

  message(STATUS "Found cblas and lapack in MKLML "
    "(include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
endif()

## Then find openblas.
if(NOT DEFINED CBLAS_PROVIDER)
  set(OPENBLAS_ROOT $ENV{OPENBLAS_ROOT} CACHE PATH "Folder contains Openblas")
  set(OPENBLAS_INCLUDE_SEARCH_PATHS
          ${OPENBLAS_ROOT}/include
          /usr/include
          /usr/include/openblas
          /usr/local/opt/openblas/include)
  set(OPENBLAS_LIB_SEARCH_PATHS
          ${OPENBLAS_ROOT}/lib
          /usr/lib
          /usr/lib/blas/openblas
          /usr/lib/openblas
          /usr/local/opt/openblas/lib)

  find_path(OPENBLAS_INC_DIR NAMES cblas.h
    PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS} NO_DEFAULT_PATH)
  find_path(OPENBLAS_LAPACKE_INC_DIR NAMES lapacke.h
    PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
  find_library(OPENBLAS_LIB NAMES openblas
    PATHS ${OPENBLAS_LIB_SEARCH_PATHS})

  if(OPENBLAS_LAPACKE_INC_DIR AND OPENBLAS_INC_DIR AND OPENBLAS_LIB)
    file(READ "${OPENBLAS_INC_DIR}/openblas_config.h" config_file)
    string(REGEX MATCH "OpenBLAS ([0-9]+\.[0-9]+\.[0-9]+)" tmp ${config_file})
    string(REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)" ver ${tmp})
    
    if (${ver} VERSION_GREATER_EQUAL "0.3.7")
      set(CBLAS_PROVIDER OPENBLAS)
      set(CBLAS_INC_DIR ${OPENBLAS_INC_DIR} ${OPENBLAS_LAPACKE_INC_DIR})
      set(CBLAS_LIBRARIES ${OPENBLAS_LIB})

      add_definitions(-DPADDLE_USE_OPENBLAS)
      add_definitions(-DLAPACK_FOUND)

      message(STATUS "Found OpenBLAS (include: ${OPENBLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
      message(STATUS "Found lapack in OpenBLAS (include: ${OPENBLAS_LAPACKE_INC_DIR})")
    endif()
  endif()
endif()

## Then find the reference-cblas if WITH_SYSTEM_BLAS.  www.netlib.org/blas/
if(NOT DEFINED CBLAS_PROVIDER AND WITH_SYSTEM_BLAS)
  set(REFERENCE_CBLAS_ROOT $ENV{REFERENCE_CBLAS_ROOT} CACHE PATH
    "Folder contains reference-cblas")
  set(REFERENCE_CBLAS_INCLUDE_SEARCH_PATHS
    ${REFERENCE_CBLAS_ROOT}/include
    /usr/include
    /usr/include/cblas
  )
  set(REFERENCE_CBLAS_LIB_SEARCH_PATHS
    ${REFERENCE_CBLAS_ROOT}/lib
    /usr/lib
    /usr/lib/blas/reference/
    /usr/lib/reference/
  )

  find_path(REFERENCE_CBLAS_INCLUDE_DIR NAMES cblas.h PATHS
        ${REFERENCE_CBLAS_INCLUDE_SEARCH_PATHS})
  find_library(REFERENCE_CBLAS_LIBRARY NAMES cblas PATHS
        ${REFERENCE_CBLAS_LIB_SEARCH_PATHS})
  find_library(REFERENCE_BLAS_LIBRARY NAMES blas PATHS
        ${REFERENCE_CBLAS_LIB_SEARCH_PATHS})

  if(REFERENCE_CBLAS_INCLUDE_DIR AND REFERENCE_CBLAS_LIBRARY)
    set(CBLAS_PROVIDER REFERENCE_CBLAS)
    set(CBLAS_INC_DIR ${REFERENCE_CBLAS_INCLUDE_DIR})
    set(CBLAS_LIBRARIES ${REFERENCE_CBLAS_LIBRARY})
    add_definitions(-DPADDLE_USE_REFERENCE_CBLAS)
    message(STATUS "Found reference-cblas (include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
  endif()
endif()

## Then build openblas by external_project
if(NOT DEFINED CBLAS_PROVIDER)
  include(external/openblas)          # download, build, install openblas
  set(CBLAS_PROVIDER EXTERN_OPENBLAS)
  add_dependencies(cblas extern_openblas)
  add_definitions(-DPADDLE_USE_OPENBLAS)
  message(STATUS "Build OpenBLAS by External Project "
    "(include: ${CBLAS_INC_DIR}, library: ${CBLAS_LIBRARIES})")
endif()

# FIXME(gangliao): generate cblas target to track all high performance
# linear algebra libraries for cc_library(xxx SRCS xxx.c DEPS cblas)

include_directories(${CBLAS_INC_DIR})
if(${CBLAS_PROVIDER} STREQUAL REFERENCE_CBLAS)
  target_link_libraries(cblas gfortran ${CBLAS_LIBRARIES} ${REFERENCE_BLAS_LIBRARY})
elseif(NOT ${CBLAS_PROVIDER} STREQUAL MKLML)
  target_link_libraries(cblas ${CBLAS_LIBRARIES})
endif()

