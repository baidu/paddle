include(operators)

set(COLLECTIVE_DEPS "")

set(COLLECTIVE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach(src ${OPS})
    set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${COLLECTIVE_COMPILE_FLAGS})
endforeach()

register_operators(EXCLUDES c_gen_bkcl_id_op gen_bkcl_id_op c_gen_nccl_id_op gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})

# if(WITH_ASCEND_CL)
#     set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} hccl collective_helper)
#     op_library(c_gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
#     op_library(gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})

# 	# add_definitions(-DENABLE_DVPP_INTERFACE)

# 	# # Compile options
# 	# add_compile_options(-std=c++11 -fPIE -fstack-protector-all -Werror -Wno-deprecated-declarations)

# 	# # Skip build rpath
# 	# set(CMAKE_SKIP_BUILD_RPATH True)

# 	# # Set output directory
# 	# # set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/code/hccl_c_plus)
# 	# # set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)

# 	# # Set include directory and library directory
# 	# # /usr/local/Ascend/ascend-toolkit/latest/acllib/include/
# 	# set(ACL_INC_DIR $ENV{ASCEND_HOME}/ascend-toolkit/latest/acllib/include)
# 	# set(ACL_LIB_DIR $ENV{ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64/stub)

# 	# # hccl
# 	# #/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/
# 	# set(HCCL_INC_DIR $ENV{ASCEND_HOME}/ascend-toolkit/latest/fwkacllib/include)
# 	# set(HCCL_LIB_DIR $ENV{ASCEND_HOME}/ascend-toolkit/latest/fwkacllib/lib64)

# 	# # Header path
# 	# include_directories("./")
# 	# include_directories(${ACL_INC_DIR})
# 	# include_directories(${HCCL_INC_DIR})
# 	# include_directories(${ASCEND_BASE_DIR})
# 	# include_directories(${ASCEND_BASE_DIR}/Framework)

# 	# # add host lib path
# 	# link_directories(${ACL_LIB_DIR})
# 	# link_directories(${HCCL_LIB_DIR})

# 	# add_executable(main ${SRC_FILES} ${PROJECT_SRC_ROOT}/example.cpp)
# 	# add_executable(main ${SRC_FILES} ${ASCEND_BASE_SRC_FILES} ${PROJECT_SRC_ROOT}/main.cpp)

# 	# target_link_libraries(main ascendcl pthread hccl -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)
# 	# target_link_libraries(main ascendcl acl_dvpp pthread hccl -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)
# endif()

if(WITH_NCCL)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} nccl_common collective_helper)
    op_library(c_gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
    op_library(gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

if(WITH_GLOO)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} gloo_wrapper)
endif()

if(WITH_XPU_BKCL)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} collective_helper)
    op_library(c_gen_bkcl_id_op DEPS ${COLLECTIVE_DEPS})
    op_library(gen_bkcl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${COLLECTIVE_DEPS} PARENT_SCOPE)
set(GLOB_COLLECTIVE_DEPS ${COLLECTIVE_DEPS} CACHE INTERNAL "collective dependency")
