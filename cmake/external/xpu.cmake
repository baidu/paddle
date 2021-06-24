if (NOT WITH_XPU)
    return()
endif()

INCLUDE(ExternalProject)
SET(XPU_PROJECT                 "extern_xpu")
SET(XPU_API_LIB_NAME            "libxpuapi.so")
SET(XPU_RT_LIB_NAME             "libxpurt.so")

IF(WITH_AARCH64)
  SET(XPU_XRE_DIR_NAME "xre-kylin_aarch64")
  SET(XPU_XDNN_DIR_NAME "xdnn-kylin_aarch64")
  SET(XPU_XCCL_DIR_NAME "xccl-kylin_aarch64")
ELSEIF(WITH_SUNWAY)
  SET(XPU_XRE_DIR_NAME "xre-deepin_sw6_64")
  SET(XPU_XDNN_DIR_NAME "xdnn-deepin_sw6_64")
  SET(XPU_XCCL_DIR_NAME "xccl-deepin_sw6_64")
ELSEIF(WITH_BDCENTOS)
  SET(XPU_XRE_DIR_NAME "xre-bdcentos_x86_64")
  SET(XPU_XDNN_DIR_NAME "xdnn-bdcentos_x86_64")
  SET(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
ELSEIF(WITH_UBUNTU)
  SET(XPU_XRE_DIR_NAME "xre-ubuntu_x86_64")
  SET(XPU_XDNN_DIR_NAME "xdnn-ubuntu_x86_64")
  SET(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
ELSEIF(WITH_CENTOS)
  SET(XPU_XRE_DIR_NAME "xre-centos7_x86_64")
  SET(XPU_XDNN_DIR_NAME "xdnn-centos7_x86_64")
  SET(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
ELSE ()
  SET(XPU_XRE_DIR_NAME "xre-ubuntu_x86_64")
  SET(XPU_XDNN_DIR_NAME "xdnn-ubuntu_x86_64")
  SET(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
ENDIF()

IF(NOT XPU_BASE_URL)
  SET(XPU_BASE_URL "https://baidu-kunlun-product.cdn.bcebos.com/KL-SDK/klsdk-dev/20210527")
ENDIF()

SET(XPU_XRE_URL  "${XPU_BASE_URL}/${XPU_XRE_DIR_NAME}.tar.gz" CACHE STRING "" FORCE)
SET(XPU_XDNN_URL "${XPU_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz" CACHE STRING "" FORCE)
SET(XPU_XCCL_URL "${XPU_BASE_URL}/${XPU_XCCL_DIR_NAME}.tar.gz" CACHE STRING "" FORCE)
SET(XPU_PACK_DEPENCE_URL "https://baidu-kunlun-public.su.bcebos.com/paddle_depence/pack_paddle_depence.sh" CACHE STRING "" FORCE)

SET(XPU_SOURCE_DIR              "${THIRD_PARTY_PATH}/xpu")
SET(XPU_DOWNLOAD_DIR            "${XPU_SOURCE_DIR}/src/${XPU_PROJECT}")
SET(XPU_INSTALL_DIR             "${THIRD_PARTY_PATH}/install/xpu")
SET(XPU_INC_DIR                 "${THIRD_PARTY_PATH}/install/xpu/include")
SET(XPU_LIB_DIR                 "${THIRD_PARTY_PATH}/install/xpu/lib")

SET(XPU_API_LIB                 "${XPU_LIB_DIR}/${XPU_API_LIB_NAME}")
SET(XPU_RT_LIB                  "${XPU_LIB_DIR}/${XPU_RT_LIB_NAME}")

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${XPU_INSTALL_DIR}/lib")

FILE(WRITE ${XPU_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(XPU)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY xpu/include xpu/lib \n"
  "        DESTINATION ${XPU_INSTALL_DIR})\n")

ExternalProject_Add(
    ${XPU_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${XPU_SOURCE_DIR}
    DOWNLOAD_DIR          ${XPU_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget ${XPU_PACK_DEPENCE_URL}
                          && bash pack_paddle_depence.sh ${XPU_XRE_URL} ${XPU_XRE_DIR_NAME} ${XPU_XDNN_URL} ${XPU_XDNN_DIR_NAME} ${XPU_XCCL_URL} ${XPU_XCCL_DIR_NAME}

    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${XPU_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${XPU_INSTALL_ROOT}
)

INCLUDE_DIRECTORIES(${XPU_INC_DIR})
ADD_LIBRARY(shared_xpuapi SHARED IMPORTED GLOBAL)
set_property(TARGET shared_xpuapi PROPERTY IMPORTED_LOCATION "${XPU_API_LIB}")

# generate a static dummy target to track xpulib dependencies
# for cc_library(xxx SRCS xxx.c DEPS xpulib)
generate_dummy_static_lib(LIB_NAME "xpulib" GENERATOR "xpu.cmake")

TARGET_LINK_LIBRARIES(xpulib ${XPU_API_LIB} ${XPU_RT_LIB})

IF(WITH_XPU_BKCL)
  MESSAGE(STATUS "Compile with XPU BKCL!")
  ADD_DEFINITIONS(-DPADDLE_WITH_XPU_BKCL)

  SET(XPU_BKCL_LIB_NAME         "libbkcl.so")
  SET(XPU_BKCL_LIB              "${XPU_LIB_DIR}/${XPU_BKCL_LIB_NAME}")
  SET(XPU_BKCL_INC_DIR          "${THIRD_PARTY_PATH}/install/xpu/include")
  INCLUDE_DIRECTORIES(${XPU_BKCL_INC_DIR})
  TARGET_LINK_LIBRARIES(xpulib ${XPU_API_LIB} ${XPU_RT_LIB} ${XPU_BKCL_LIB})
ELSE(WITH_XPU_BKCL)
  TARGET_LINK_LIBRARIES(xpulib ${XPU_API_LIB} ${XPU_RT_LIB})
ENDIF(WITH_XPU_BKCL)

ADD_DEPENDENCIES(xpulib ${XPU_PROJECT})

# Ensure that xpu/api.h can be included without dependency errors.
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc CONTENT "")
add_library(xpu_headers_dummy STATIC ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc)
add_dependencies(xpu_headers_dummy extern_xpu)
link_libraries(xpu_headers_dummy)
