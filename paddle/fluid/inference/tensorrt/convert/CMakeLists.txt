# Add TRT tests
nv_library(tensorrt_converter
           SRCS matmul_op.cc conv2d_op.cc fc_op.cc pool2d_op.cc elementwise_op.cc
                batch_norm_op.cc activation_op.cc softmax_op.cc concat_op.cc dropout_op.cc group_norm_op.cc
                pad_op.cc split_op.cc prelu_op.cc leaky_relu_op.cc gelu_op.cc layer_norm_op.cc multihead_matmul_op.cc
                shuffle_channel_op.cc swish_op.cc instance_norm_op.cc stack_op.cc transpose_op.cc flatten_op.cc
                emb_eltwise_layernorm.cc skip_layernorm.cc scale_op.cc slice_op.cc hard_sigmoid_op.cc hard_swish_op.cc clip_op.cc
                gather_op.cc
                anchor_generator_op.cc
                yolo_box_op.cc
                roi_align_op.cc
                affine_channel_op.cc
                multiclass_nms_op.cc
                nearest_interp_op.cc
                reshape_op.cc
                reduce_op.cc
                gather_nd_op.cc
                nearest_interp_v2_op.cc
           DEPS tensorrt_engine tensorrt_plugin operator scope framework_proto op_registry)

nv_test(test_op_converter SRCS test_op_converter.cc DEPS
  paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_engine tensorrt_converter)
