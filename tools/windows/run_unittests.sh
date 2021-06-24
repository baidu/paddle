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
PRECISION_TEST=$2
WITH_GPU=$3

export PADDLE_ROOT="$(cd "$PWD/../" && pwd )"
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

# check added ut
if [ ${WITH_GPU:-OFF} == "ON" ];then
    set +e
    cp $PADDLE_ROOT/tools/check_added_ut.sh $PADDLE_ROOT/tools/check_added_ut_win.sh
    bash $PADDLE_ROOT/tools/check_added_ut_win.sh
    rm -rf $PADDLE_ROOT/tools/check_added_ut_win.sh
    set -e
fi

# /*==================Fixed Disabled Windows GPU MKL unittests==============================*/
# TODO: fix these unittest that is bound to fail
disable_wingpu_test="^test_model$|\
^test_dataloader_early_reset$|\
^test_add_reader_dependency$|\
^test_decoupled_py_reader$|\
^test_generator_dataloader$|\
^test_parallel_dygraph_sync_batch_norm$|\
^test_py_reader_using_executor$|\
^test_parallel_executor_seresnext_base_gpu$|\
^test_parallel_executor_seresnext_with_fuse_all_reduce_gpu$|\
^test_parallel_executor_seresnext_with_reduce_gpu$|\
^test_program_prune_backward$|\
^test_decoupled_py_reader_data_check$|\
^test_fleet_base_single$|\
^test_multiprocess_dataloader_iterable_dataset_dynamic$|\
^test_parallel_executor_feed_persistable_var$|\
^test_parallel_executor_inference_feed_partial_data$|\
^test_parallel_ssa_graph_inference_feed_partial_data$|\
^test_py_reader_combination$|\
^test_py_reader_pin_memory$|\
^test_py_reader_push_pop$|\
^test_reader_reset$|\
^test_imperative_se_resnext$|\
^test_sync_batch_norm_op$|\
^test_imperative_static_runner_while$|\
^test_dataloader_keep_order$|\
^test_dataloader_unkeep_order$|\
^test_multiprocess_dataloader_iterable_dataset_static$|\
^test_fuse_bn_act_pass$|\
^test_fuse_bn_add_act_pass$|\
^disable_wingpu_test$"


# /*============================================================================*/

# /*==================Fixed Disabled Windows CPU OPENBLAS unittests==============================*/
# TODO: fix these unittest that is bound to fail
disable_wincpu_test="^jit_kernel_test$|\
^test_analyzer_transformer$|\
^test_vision_models$|\
^test_dygraph_multi_forward$|\
^test_imperative_transformer_sorted_gradient$|\
^test_program_prune_backward$|\
^test_imperative_resnet$|\
^test_imperative_resnet_sorted_gradient$|\
^test_imperative_se_resnext$|\
^test_imperative_static_runner_mnist$|\
^test_bmn$|\
^test_mobile_net$|\
^test_resnet_v2$|\
^test_se_resnet$|\
^disable_wincpu_test$"

# these unittest that cost long time, diabled temporarily, Maybe moved to the night
long_time_test="^best_fit_allocator_test$|\
^test_gru_op$|\
^decorator_test$|\
^test_dataset_cifar$|\
^test_dataset_imdb$|\
^test_dataset_movielens$|\
^test_datasets$|\
^test_pretrained_model$|\
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
^test_sigmoid_cross_entropy_with_logits_op$|\
^test_strided_slice_op$"

if [ ${WITH_GPU:-OFF} == "ON" ];then
    export FLAGS_fraction_of_gpu_memory_to_use=0.92
    export CUDA_VISIBLE_DEVICES=0

    UT_list=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
    num=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d' | wc -l)
    echo "Windows 1 card TestCases count is $num"
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python ${PADDLE_ROOT}/tools/get_pr_ut.py || echo "Failed to obtain ut_list !"
        if [[ -f "ut_list" ]]; then
            echo "PREC length: "`wc -l ut_list`
            precision_cases=`cat ut_list`
        fi
    fi

    set +e
    if [ ${PRECISION_TEST:-OFF} == "ON" ] && [[ "$precision_cases" != "" ]];then
        UT_list_res=$(python ${PADDLE_ROOT}/tools/windows/get_prec_ut_list.py "$UT_list" )
        UT_list_prec=$(echo "${UT_list_res}" | grep -v 'PRECISION_TEST')
        echo "${UT_list_res}" | grep 'PRECISION_TEST'
        UT_list=$UT_list_prec
    fi
    set -e

    output=$(python ${PADDLE_ROOT}/tools/parallel_UT_rule.py "${UT_list}")
    cpu_parallel_job=$(echo $output | cut -d ";" -f 1)
    tetrad_parallel_job=$(echo $output | cut -d ";" -f 2)
    two_parallel_job=$(echo $output | cut -d ";" -f 3)
    non_parallel_job=$(echo $output | cut -d ";" -f 4)
fi

failed_test_lists=''
tmp_dir=`mktemp -d`
function collect_failed_tests() {
    set +e
    for file in `ls $tmp_dir`; do
        grep -q 'The following tests FAILED:' $tmp_dir/$file
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
    set -e
}

function run_unittest_cpu() {
    tmpfile=$tmp_dir/$RANDOM
    (ctest -E "$disable_ut_quickly|$disable_wincpu_test" -LE "${nightly_label}" --output-on-failure -C Release -j 8 | tee $tmpfile) &
    wait;
}

function run_unittest_gpu() {
    test_case=$1
    parallel_job=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}
    if [ "$2" == "" ]; then
        parallel_job=$parallel_level_base
    else
        parallel_job=`expr $2 \* $parallel_level_base`
    fi
    echo "************************************************************************"
    echo "********These unittests run $parallel_job job each time with 1 GPU**********"
    echo "************************************************************************"
    export CUDA_VISIBLE_DEVICES=0
    tmpfile=$tmp_dir/$RANDOM
    (ctest -R "$test_case" -E "$disable_ut_quickly|$disable_wingpu_test|$long_time_test" -LE "${nightly_label}" --output-on-failure -C Release -j $parallel_job | tee $tmpfile ) &
    wait;
}

function unittests_retry(){
    is_retry_execuate=0
    wintest_error=1
    retry_time=3
    exec_times=0
    exec_retry_threshold=10
    retry_unittests=$(echo "${failed_test_lists}" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
    need_retry_ut_counts=$(echo "$retry_unittests" |awk -F ' ' '{print }'| sed '/^$/d' | wc -l)
    retry_unittests_regular=$(echo "$retry_unittests" |awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
    tmpfile=$tmp_dir/$RANDOM

    if [ $need_retry_ut_counts -lt $exec_retry_threshold ];then
            retry_unittests_record=''
            while ( [ $exec_times -lt $retry_time ] )
                do
                    retry_unittests_record="$retry_unittests_record$failed_test_lists"
                    if ( [[ "$exec_times" == "0" ]] );then
                        cur_order='first'
                    elif ( [[ "$exec_times" == "1" ]] );then
                        cur_order='second'
                        if [[ "$failed_test_lists" == "" ]]; then
                            break
                        else
                            retry_unittests=$(echo "${failed_test_lists}" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
                            retry_unittests_regular=$(echo "$retry_unittests" |awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
                        fi
                    elif ( [[ "$exec_times" == "2" ]] );then
                        cur_order='third'
                    fi
                    echo "========================================="
                    echo "This is the ${cur_order} time to re-run"
                    echo "========================================="
                    echo "The following unittest will be re-run:"
                    echo "${retry_unittests}"
                    echo "========================================="
                    rm -f $tmp_dir/*
                    failed_test_lists=''
                    (ctest -R "($retry_unittests_regular)" --output-on-failure -C Release -j 1 | tee $tmpfile ) &
                    wait;
                    collect_failed_tests
                    exec_times=$(echo $exec_times | awk '{print $0+1}')
                done
    else
        # There are more than 10 failed unit tests, so no unit test retry
        is_retry_execuate=1
    fi
    rm -f $tmp_dir/*
}

function show_ut_retry_result() {
    if [[ "$is_retry_execuate" != "0" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}"`
        echo "========================================="
        echo "There are more than 10 failed unit tests, so no unit test retry!!!"
        echo "========================================="
        echo "${failed_test_lists_ult}"
        exit 8;
    else
        retry_unittests_ut_name=$(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
        retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=3) {print $2}}')
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-run:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"|"$1}} END{print all_str}')
            echo "========================================"
            echo "There are failed tests, which have been executed re-run,but success rate is less than 50%:"
            echo "Summary Failed Tests... "
            echo "========================================"
            echo "The following tests FAILED: "
            echo "${retry_unittests_record}" | grep -E "$failed_ut_re"
            exit 8;
        fi
    fi
}

set +e

export FLAGS_call_stack_level=2
if [ "${WITH_GPU:-OFF}" == "ON" ];then
    if [ -f "$PADDLE_ROOT/added_ut" ];then
        added_uts=^$(awk BEGIN{RS=EOF}'{gsub(/\n/,"$|^");print}' $PADDLE_ROOT/added_ut)$
        ctest -R "(${added_uts})" --output-on-failure -C Release --repeat-until-fail 3;added_ut_error=$?
        rm -f $PADDLE_ROOT/added_ut
        if [ "$added_ut_error" != 0 ];then
            echo "========================================"
            echo "Added UT should pass three additional executions"
            echo "========================================"
            exit 8;
        fi
    fi
    run_unittest_gpu $cpu_parallel_job 12
    run_unittest_gpu $tetrad_parallel_job 4
    run_unittest_gpu $two_parallel_job 2
    run_unittest_gpu $non_parallel_job
else
    run_unittest_cpu
fi
collect_failed_tests
set -e
rm -f $tmp_dir/*
if [[ "$failed_test_lists" != "" ]]; then
    unittests_retry
    show_ut_retry_result
fi
