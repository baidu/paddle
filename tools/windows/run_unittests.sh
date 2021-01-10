# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set +x
NIGHTLY_MODE=$1

PADDLE_ROOT="$(cd "$PWD/../" && pwd )"
if [ ${NIGHTLY_MODE:-OFF} == "ON" ]; then
    nightly_label=""
else
    nightly_label="(RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY)"
    echo "========================================="
    echo "Unittests with nightly labels  are only run at night"
    echo "========================================="
fi

if disable_ut_quickly=$(python ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
    echo "========================================="
    echo "The following unittests have been disabled:"
    echo ${disable_ut_quickly}
    echo "========================================="
else
    disable_ut_quickly=''
fi

# /*==================Fixed Disabled Windows unittests==============================*/
# TODO: fix these unittest that is bound to fail
diable_wingpu_test="^test_gradient_clip$|\
^test_translated_layer$|\
^test_imperative_resnet$|\
^test_imperative_resnet_sorted_gradient$|\
^test_model$|\
^test_decoupled_py_reader$|\
^test_generator_dataloader$|\
^test_multiprocess_dataloader_iterable_dataset_static$|\
^test_py_reader_using_executor$|\
^test_parallel_executor_feed_persistable_var$|\
^test_parallel_executor_fetch_isolated_var$|\
^test_parallel_executor_inference_feed_partial_data$|\
^test_parallel_executor_seresnext_base_gpu$|\
^test_parallel_executor_seresnext_with_fuse_all_reduce_gpu$|\
^test_parallel_executor_seresnext_with_reduce_gpu$|\
^test_parallel_ssa_graph_inference_feed_partial_data$|\
^test_sync_batch_norm_op$|\
^test_fuse_relu_depthwise_conv_pass$|\
^test_buffer_shared_memory_reuse_pass$|\
^test_buffer_shared_memory_reuse_pass_and_fuse_optimization_op_pass$|\
^test_dataloader_keep_order$|\
^test_dataloader_unkeep_order$|\
^test_add_reader_dependency$|\
^test_cholesky_op$|\
^test_dataloader_early_reset$|\
^test_decoupled_py_reader_data_check$|\
^test_fleet_base_single$|\
^test_fuse_elewise_add_act_pass$|\
^test_fuse_optimizer_pass$|\
^test_multiprocess_dataloader_iterable_dataset_dynamic$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_partial_eager_deletion_transformer$|\
^test_rnn_nets$|\
^test_py_reader_combination$|\
^test_py_reader_pin_memory$|\
^test_py_reader_push_pop$|\
^test_reader_reset$|\
^test_imperative_se_resnext$|\
^test_imperative_static_runner_while$|\
^test_fuse_bn_act_pass$|\
^test_fuse_bn_add_act_pass$|\
^test_gru_rnn_op$|\
^test_rnn_op$|\
^test_simple_rnn_op$|\
^test_lstm_cudnn_op$|\
^test_conv2d_int8_mkldnn_op$|\
^test_crypto$|\
^test_program_prune_backward$|\
^test_imperative_ocr_attention_model$|\
^test_sentiment$|\
^test_imperative_basic$|\
^test_jit_save_load$|\
^test_imperative_mnist$|\
^test_imperative_mnist_sorted_gradient$|\
^test_imperative_static_runner_mnist$|\
^test_fuse_all_reduce_pass$|\
^test_bert$|\
^test_lac$|\
^test_mnist$|\
^test_mobile_net$|\
^test_ptb_lm$|\
^test_ptb_lm_v2$|\
^test_se_resnet$|\
^test_imperative_qat_channelwise$|\
^test_imperative_qat$|\
^test_imperative_out_scale$|\
^diable_wingpu_test$"
# /*============================================================================*/

# these unittest that cost long time, diabled temporarily, Maybe moved to the night
long_time_test="^best_fit_allocator_test$|\
^test_image_classification$|\
^decorator_test$|\
^test_dataset_cifar$|\
^test_dataset_imdb$|\
^test_dataset_movielens$|\
^test_datasets$|\
^test_pretrained_model$|\
^test_concat_op$|\
^test_elementwise_add_op$|\
^test_elementwise_sub_op$|\
^test_gather_op$|\
^test_gather_nd_op$|\
^test_sequence_concat$|\
^test_sequence_conv$|\
^test_sequence_pool$|\
^test_sequence_slice_op$|\
^test_space_to_depth_op$|\
^test_activation_nn_grad$|\
^test_activation_op$|\
^test_auto_growth_gpu_memory_limit$|\
^test_bicubic_interp_op$|\
^test_bicubic_interp_v2_op$|\
^test_bilinear_interp_v2_op$|\
^test_conv2d_op$|\
^test_conv3d_op$|
^test_conv3d_transpose_part2_op$|\
^test_conv_nn_grad$|\
^test_crop_tensor_op$|\
^test_cross_entropy2_op$|\
^test_cross_op$|\
^test_deformable_conv_v1_op$|\
^test_dropout_op$|\
^test_dygraph_multi_forward$|\
^test_elementwise_div_op$|\
^test_elementwise_nn_grad$|\
^test_empty_op$|\
^test_fused_elemwise_activation_op$|\
^test_group_norm_op$|\
^test_gru_op$|\
^test_gru_unit_op$|\
^test_imperative_lod_tensor_to_selected_rows$|\
^test_imperative_optimizer$|\
^test_imperative_ptb_rnn$|\
^test_imperative_save_load$|\
^test_imperative_selected_rows_to_lod_tensor$|\
^test_imperative_star_gan_with_gradient_penalty$|\
^test_imperative_transformer_sorted_gradient$|\
^test_layer_norm_op$|\
^test_masked_select_op$|\
^test_multiclass_nms_op$|\
^test_naive_best_fit_gpu_memory_limit$|\
^test_nearest_interp_v2_op$|\
^test_nn_grad$|\
^test_norm_nn_grad$|\
^test_normal$|\
^test_pool3d_op$|\
^test_pool2d_op$|\
^test_prroi_pool_op$|\
^test_regularizer$|\
^test_regularizer_api$|\
^test_softmax_with_cross_entropy_op$|\
^test_static_save_load$|\
^test_trilinear_interp_op$|\
^test_trilinear_interp_v2_op$|\
^test_bilinear_interp_op$|\
^test_nearest_interp_op$|\
^test_sequence_conv$|\
^test_sgd_op$|\
^test_transformer$|\
^test_beam_search_decoder$|\
^test_argsort_op$|\
^test_eager_deletion_gru_net$|\
^test_lstmp_op$|\
^test_label_semantic_roles$|\
^test_machine_translation$|\
^test_row_conv_op$|\
^test_deformable_conv_op$|\
^test_inplace_softmax_with_cross_entropy$|\
^test_conv2d_transpose_op$|\
^test_conv3d_transpose_op$|\
^test_cyclic_cifar_dataset$|\
^test_deformable_psroi_pooling$|\
^test_elementwise_mul_op$|\
^test_imperative_auto_mixed_precision$|\
^test_imperative_optimizer_v2$|\
^test_imperative_ptb_rnn_sorted_gradient$|\
^test_imperative_save_load_v2$|\
^test_nan_inf$|\
^test_norm_op$|\
^test_reduce_op$|\
^test_sigmoid_cross_entropy_with_logits_op$|\
^test_stack_op$|\
^test_strided_slice_op$|\
^test_transpose_op$"

export FLAGS_call_stack_level=2
export FLAGS_fraction_of_gpu_memory_to_use=0.92
export CUDA_VISIBLE_DEVICES=0
ctest -E "$disable_ut_quickly|$diable_wingpu_test|$long_time_test" -R test_kron_op -LE "${nightly_label}" --output-on-failure -C Release --repeat until-pass:4 after-timeout:4
