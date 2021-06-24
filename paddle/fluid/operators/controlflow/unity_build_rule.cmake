# This file records the Unity Build compilation rules.
# The source files in a `register_unity_group` called are compiled in a unity
# file.
# Generally, the combination rules in this file do not need to be modified.
# If there are some redefined error in compiling with the source file which
# in combination rule, you can remove the source file from the following rules.
register_unity_group(cc
    compare_all_op.cc
    compare_op.cc
    conditional_block_infer_op.cc
    feed_op.cc
    fetch_op.cc
    get_places_op.cc
    logical_op.cc
    bitwise_op.cc
    tensor_array_read_write_op.cc
    while_op.cc)
register_unity_group(cu
    logical_op.cu
    bitwise_op.cu
    compare_op.cu
    compare_all_op.cu)
