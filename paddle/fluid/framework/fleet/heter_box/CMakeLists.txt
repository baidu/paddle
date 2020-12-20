#add_subdirectory(cudf)
add_subdirectory(hashtable)
nv_library(heter_box SRCS heter_box.cu DEPS gpu_ps)
#include_directories("PADDLE_SOURCE_DIR/paddle/fluid/framework/fleet/heter_box")
