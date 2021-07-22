include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/fused.
    include(unity_build_rule.cmake)
endif()
register_operators(EXCLUDES
    fused_bn_activation_op
    conv_fusion_op
    fusion_transpose_flatten_concat_op
    fusion_conv_inception_op
    fused_fc_elementwise_layernorm_op
    multihead_matmul_op
    skip_layernorm_op
    fused_embedding_eltwise_layernorm_op
    fusion_group_op
    fusion_gru_op
    fusion_lstm_op
    fused_bn_add_activation_op)

# fusion_gru_op does not have CUDA kernel
op_library(fusion_gru_op)
op_library(fusion_lstm_op)
file(APPEND ${pybind_file} "USE_CPU_ONLY_OP(fusion_gru);\n")
file(APPEND ${pybind_file} "USE_CPU_ONLY_OP(fusion_lstm);\n")


if (WITH_GPU OR WITH_ROCM)
    # fused_bn_activation_op needs cudnn 7.4.1 above
    # HIP not support bn act fuse in MIOPEN
    if ((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 7401))
        op_library(fused_bn_activation_op)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fused_batch_norm_act);\n")
    endif()
    # conv_fusion_op needs cudnn 7 above
    if (NOT ${CUDNN_VERSION} VERSION_LESS 7100)
        op_library(conv_fusion_op)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(conv2d_fusion);\n")
    endif()
    # fusion_transpose_flatten_concat_op
    op_library(fusion_transpose_flatten_concat_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fusion_transpose_flatten_concat);\n")
    # fusion_conv_inception_op needs cudnn 7 above
    # HIP not support cudnnConvolutionBiasActivationForward
    if ((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 7100))
        op_library(fusion_conv_inception_op)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(conv2d_inception_fusion);\n")
    endif()
    # fused_fc_elementwise_layernorm_op
    op_library(fused_fc_elementwise_layernorm_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fused_fc_elementwise_layernorm);\n")
    # multihead_matmul_op
    op_library(multihead_matmul_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(multihead_matmul);\n")
    op_library(skip_layernorm_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(skip_layernorm);\n")
    op_library(fused_embedding_eltwise_layernorm_op)
    file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fused_embedding_eltwise_layernorm);\n")
    # fusion_group
    if(NOT APPLE AND NOT WIN32)
        op_library(fusion_group_op DEPS device_code)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fusion_group);\n")
        cc_test(test_fusion_group_op SRCS fusion_group_op_test.cc DEPS fusion_group_op)
    endif()
    # fused_bn_add_activation
    # HIP not support bn act fuse in MIOPEN
    if ((NOT WITH_ROCM) AND (NOT ${CUDNN_VERSION} VERSION_LESS 7401))
        op_library(fused_bn_add_activation_op)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(fused_bn_add_activation);\n")
    endif()
endif()
