file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

function(_inference_analysis_python_api_int8_test target model_dir data_path filename use_mkldnn)
    py_test(${target} SRCS ${filename}
        ENVS CPU_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
             FLAGS_use_mkldnn=${use_mkldnn}
        ARGS --infer_model ${model_dir}/model
             --infer_data ${data_path}
             --int8_model_save_path int8_models/${target}
             --warmup_batch_size ${WARMUP_BATCH_SIZE}
             --batch_size 50)
endfunction()

function(inference_analysis_python_api_int8_test target model_dir data_path filename)
    _inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_path} ${filename} False)
endfunction()

function(inference_analysis_python_api_int8_test_custom_warmup_batch_size target model_dir data_dir filename warmup_batch_size)
    set(WARMUP_BATCH_SIZE ${warmup_batch_size})
    inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_dir} ${filename})
endfunction()

function(inference_analysis_python_api_int8_test_mkldnn target model_dir data_path filename)
    _inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_path} ${filename} True)
endfunction()

function(download_quant_data install_dir data_file check_sum)
    if (NOT EXISTS ${install_dir}/${data_file})
	    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8 ${data_file} ${check_sum})
    endif()
endfunction()

function(download_quant_model install_dir data_file check_sum)
    if (NOT EXISTS ${install_dir}/${data_file})
	    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file} ${check_sum})
    endif()
endfunction()

function(download_quant_fp32_model install_dir data_file check_sum)
    if (NOT EXISTS ${install_dir}/${data_file})
	    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8/QAT_models/fp32 ${data_file} ${check_sum})
    endif()
endfunction()

function(inference_quant_int8_image_classification_test target quant_model_dir dataset_path)
    py_test(${target} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quant_int8_image_classification_comparison.py"
            ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 FLAGS_use_mkldnn=true
            ARGS --quant_model ${quant_model_dir}
                 --infer_data ${dataset_path}
                 --batch_size 25
                 --batch_num 2
                 --acc_diff_threshold 0.1)
endfunction()


# set batch_size 10 for UT only (avoid OOM). For whole dataset, use batch_size 25 
function(inference_quant2_int8_image_classification_test target quant_model_dir fp32_model_dir dataset_path)
    py_test(${target} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quant2_int8_image_classification_comparison.py"
            ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 FLAGS_use_mkldnn=true
            ARGS --quant_model ${quant_model_dir}
                 --fp32_model ${fp32_model_dir}
                 --infer_data ${dataset_path}
                 --batch_size 10
                 --batch_num 2
                 --acc_diff_threshold 0.1)
endfunction()

# set batch_size 10 for UT only (avoid OOM). For whole dataset, use batch_size 20 
function(inference_quant2_int8_nlp_test target quant_model_dir fp32_model_dir dataset_path labels_path ops_to_quantize)
    py_test(${target} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quant2_int8_nlp_comparison.py"
            ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 FLAGS_use_mkldnn=true
            ARGS --quant_model ${quant_model_dir}
		 --fp32_model ${fp32_model_dir}
                 --infer_data ${dataset_path}
		 --labels ${labels_path}
                 --batch_size 10
                 --batch_num 2
                 --acc_diff_threshold 0.1
		 --ops_to_quantize ${ops_to_quantize})
endfunction()

function(download_quant_data install_dir data_file check_sum)
    if (NOT EXISTS ${install_dir}/${data_file})
           inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8 ${data_file} ${check_sum})
    endif()
endfunction()

function(download_quant_model install_dir data_file check_sum)
    if (NOT EXISTS ${install_dir}/${data_file})
           inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file} ${check_sum})
    endif()
endfunction()

function(save_quant_ic_model_test target quant_model_dir int8_model_save_path)
    py_test(${target} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/save_quant_model.py
            ARGS --quant_model_path ${quant_model_dir}
	         --int8_model_save_path ${int8_model_save_path}
		 --debug)
endfunction()

function(save_quant_nlp_model_test target quant_model_dir int8_model_save_path ops_to_quantize)
    py_test(${target} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/save_quant_model.py
            ARGS --quant_model_path ${quant_model_dir}
	         --int8_model_save_path ${int8_model_save_path}
		 --ops_to_quantize ${ops_to_quantize})
endfunction()

function(convert_model2dot_test target model_path save_graph_dir save_graph_name)
    py_test(${target} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/convert_model2dot.py
            ARGS --model_path ${model_path}
	         --save_graph_dir ${save_graph_dir}
	         --save_graph_name ${save_graph_name})
endfunction()

if(WIN32)
	list(REMOVE_ITEM TEST_OPS test_light_nas)
	list(REMOVE_ITEM TEST_OPS test_post_training_quantization_mnist)
	list(REMOVE_ITEM TEST_OPS test_post_training_quantization_mobilenetv1)
	list(REMOVE_ITEM TEST_OPS test_post_training_quantization_resnet50)
	list(REMOVE_ITEM TEST_OPS test_post_training_quantization_lstm_model)
	list(REMOVE_ITEM TEST_OPS test_imperative_ptq)
	list(REMOVE_ITEM TEST_OPS test_weight_quantization_mobilenetv1)
	list(REMOVE_ITEM TEST_OPS test_quantize_transpiler_v2)
	list(REMOVE_ITEM TEST_OPS test_imperative_qat_amp)
endif()

if(LINUX AND WITH_MKLDNN)

	#### Image classification dataset: ImageNet (small)
	# The dataset should already be downloaded for INT8v2 unit tests
	set(IMAGENET_DATA_PATH "${INFERENCE_DEMO_INSTALL_DIR}/imagenet/data.bin")

	#### INT8 image classification python api test
	# Models should be already downloaded for INT8v2 unit tests

	set(INT8_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")

	#### QUANT & INT8 comparison python api tests

	set(QUANT_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/quant")

	### Quant1 for image classification

	# Quant ResNet50
	set(QUANT_RESNET50_MODEL_DIR "${QUANT_INSTALL_DIR}/ResNet50_quant")
	set(QUANT_RESNET50_MODEL_ARCHIVE "ResNet50_qat_model.tar.gz")
	download_quant_model(${QUANT_RESNET50_MODEL_DIR} ${QUANT_RESNET50_MODEL_ARCHIVE} ff89b934ab961c3a4a844193ece2e8a7)
	inference_quant_int8_image_classification_test(test_quant_int8_resnet50_mkldnn ${QUANT_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant ResNet101
	set(QUANT_RESNET101_MODEL_DIR "${QUANT_INSTALL_DIR}/ResNet101_quant")
	set(QUANT_RESNET101_MODEL_ARCHIVE "ResNet101_qat_model.tar.gz")
	download_quant_model(${QUANT_RESNET101_MODEL_DIR} ${QUANT_RESNET101_MODEL_ARCHIVE} 95c6d01e3aeba31c13efb2ba8057d558)
	# inference_quant_int8_image_classification_test(test_quant_int8_resnet101_mkldnn ${QUANT_RESNET101_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant GoogleNet
	set(QUANT_GOOGLENET_MODEL_DIR "${QUANT_INSTALL_DIR}/GoogleNet_quant")
	set(QUANT_GOOGLENET_MODEL_ARCHIVE "GoogleNet_qat_model.tar.gz")
	download_quant_model(${QUANT_GOOGLENET_MODEL_DIR} ${QUANT_GOOGLENET_MODEL_ARCHIVE} 1d4a7383baa63e7d1c423e8db2b791d5)
	inference_quant_int8_image_classification_test(test_quant_int8_googlenet_mkldnn ${QUANT_GOOGLENET_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant MobileNetV1
	set(QUANT_MOBILENETV1_MODEL_DIR "${QUANT_INSTALL_DIR}/MobileNetV1_quant")
	set(QUANT_MOBILENETV1_MODEL_ARCHIVE "MobileNetV1_qat_model.tar.gz")
	download_quant_model(${QUANT_MOBILENETV1_MODEL_DIR} ${QUANT_MOBILENETV1_MODEL_ARCHIVE} 3b774d94a9fcbb604d09bdb731fc1162)
	inference_quant_int8_image_classification_test(test_quant_int8_mobilenetv1_mkldnn ${QUANT_MOBILENETV1_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant MobileNetV2
	set(QUANT_MOBILENETV2_MODEL_DIR "${QUANT_INSTALL_DIR}/MobileNetV2_quant")
	set(QUANT_MOBILENETV2_MODEL_ARCHIVE "MobileNetV2_qat_model.tar.gz")
	download_quant_model(${QUANT_MOBILENETV2_MODEL_DIR} ${QUANT_MOBILENETV2_MODEL_ARCHIVE} 758a99d9225d8b73e1a8765883f96cdd)
	inference_quant_int8_image_classification_test(test_quant_int8_mobilenetv2_mkldnn ${QUANT_MOBILENETV2_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant VGG16
	set(QUANT_VGG16_MODEL_DIR "${QUANT_INSTALL_DIR}/VGG16_quant")
	set(QUANT_VGG16_MODEL_ARCHIVE "VGG16_qat_model.tar.gz")
	download_quant_model(${QUANT_VGG16_MODEL_DIR} ${QUANT_VGG16_MODEL_ARCHIVE} c37e63ca82a102f47be266f8068b0b55)
	# inference_quant_int8_image_classification_test(test_quant_int8_vgg16_mkldnn ${QUANT_VGG16_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant VGG19
	set(QUANT_VGG19_MODEL_DIR "${QUANT_INSTALL_DIR}/VGG19_quant")
	set(QUANT_VGG19_MODEL_ARCHIVE "VGG19_qat_model.tar.gz")
	download_quant_model(${QUANT_VGG19_MODEL_DIR} ${QUANT_VGG19_MODEL_ARCHIVE} 62bcd4b6c3ca2af67e8251d1c96ea18f)
	# inference_quant_int8_image_classification_test(test_quant_int8_vgg19_mkldnn ${QUANT_VGG19_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	### Quant2 for image classification

	# Quant2 ResNet50 with input/output scales in `fake_quantize_moving_average_abs_max` operators,
	# with weight scales in `fake_dequantize_max_abs` operators
        set(QUANT2_RESNET50_MODEL_DIR "${QUANT_INSTALL_DIR}/ResNet50_quant2")
	set(QUANT2_RESNET50_MODEL_ARCHIVE "ResNet50_qat_perf.tar.gz")
	download_quant_model(${QUANT2_RESNET50_MODEL_DIR} ${QUANT2_RESNET50_MODEL_ARCHIVE} e87309457e8c462a579340607f064d66)
	set(FP32_RESNET50_MODEL_DIR "${INT8_INSTALL_DIR}/resnet50")
	inference_quant2_int8_image_classification_test(test_quant2_int8_resnet50_mkldnn ${QUANT2_RESNET50_MODEL_DIR}/ResNet50_qat_perf/float ${FP32_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant2 ResNet50 with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes,
	# with weight scales in `fake_dequantize_max_abs` operators
	set(QUANT2_RESNET50_RANGE_MODEL_DIR "${QUANT_INSTALL_DIR}/ResNet50_quant2_range")
	set(QUANT2_RESNET50_RANGE_MODEL_ARCHIVE "ResNet50_qat_range.tar.gz")
	download_quant_model(${QUANT2_RESNET50_RANGE_MODEL_DIR} ${QUANT2_RESNET50_RANGE_MODEL_ARCHIVE} 2fdc8a139f041c0d270abec826b2d304)
	inference_quant2_int8_image_classification_test(test_quant2_int8_resnet50_range_mkldnn ${QUANT2_RESNET50_RANGE_MODEL_DIR}/ResNet50_qat_range ${FP32_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant2 ResNet50 with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes,
	# with weight scales in `fake_channel_wise_dequantize_max_abs` operators
	set(QUANT2_RESNET50_CHANNELWISE_MODEL_DIR "${QUANT_INSTALL_DIR}/ResNet50_quant2_channelwise")
	set(QUANT2_RESNET50_CHANNELWISE_MODEL_ARCHIVE "ResNet50_qat_channelwise.tar.gz")
	download_quant_model(${QUANT2_RESNET50_CHANNELWISE_MODEL_DIR} ${QUANT2_RESNET50_CHANNELWISE_MODEL_ARCHIVE} 887a1b1b0e9a4efd10f263a43764db26)
	inference_quant2_int8_image_classification_test(test_quant2_int8_resnet50_channelwise_mkldnn ${QUANT2_RESNET50_CHANNELWISE_MODEL_DIR}/ResNet50_qat_channelwise ${FP32_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# Quant2 MobileNetV1
        set(QUANT2_MOBILENETV1_MODEL_DIR "${QUANT_INSTALL_DIR}/MobileNetV1_quant2")
	set(QUANT2_MOBILENETV1_MODEL_ARCHIVE "MobileNet_qat_perf.tar.gz")
	download_quant_model(${QUANT2_MOBILENETV1_MODEL_DIR} ${QUANT2_MOBILENETV1_MODEL_ARCHIVE} 7f626e453db2d56fed6c2538621ffacf)
	set(FP32_MOBILENETV1_MODEL_DIR "${INT8_INSTALL_DIR}/mobilenetv1")
	inference_quant2_int8_image_classification_test(test_quant2_int8_mobilenetv1_mkldnn ${QUANT2_MOBILENETV1_MODEL_DIR}/MobileNet_qat_perf/float ${FP32_MOBILENETV1_MODEL_DIR}/model ${IMAGENET_DATA_PATH})
	
	### Quant2 for NLP

	set(NLP_DATA_ARCHIVE "Ernie_dataset.tar.gz")
	set(NLP_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/Ernie_dataset")
	set(NLP_DATA_PATH "${NLP_DATA_DIR}/Ernie_dataset/1.8w.bs1")
	set(NLP_LABLES_PATH "${NLP_DATA_DIR}/Ernie_dataset/label.xnli.dev")
	download_quant_data(${NLP_DATA_DIR} ${NLP_DATA_ARCHIVE} e650ce0cbc1fadbed5cc2c01d4e734dc)

	# Quant2 Ernie
	set(QUANT2_ERNIE_MODEL_ARCHIVE "ernie_qat.tar.gz")
	set(QUANT2_ERNIE_MODEL_DIR "${QUANT_INSTALL_DIR}/Ernie_quant2")
	download_quant_model(${QUANT2_ERNIE_MODEL_DIR} ${QUANT2_ERNIE_MODEL_ARCHIVE} f7cdf4720755ecf66efbc8044e9922d9)
	set(FP32_ERNIE_MODEL_ARCHIVE "ernie_fp32_model.tar.gz")
	set(FP32_ERNIE_MODEL_DIR "${QUANT_INSTALL_DIR}/Ernie_float")
	download_quant_fp32_model(${FP32_ERNIE_MODEL_DIR} ${FP32_ERNIE_MODEL_ARCHIVE} 114f38804a3ef8c45e7259e68bbd838b)
	set(QUANT2_ERNIE_OPS_TO_QUANTIZE "fc,reshape2,transpose2,matmul,elementwise_add")
	inference_quant2_int8_nlp_test(test_quant2_int8_ernie_mkldnn ${QUANT2_ERNIE_MODEL_DIR}/Ernie_qat/float ${FP32_ERNIE_MODEL_DIR}/ernie_fp32_model ${NLP_DATA_PATH} ${NLP_LABLES_PATH} ${QUANT2_ERNIE_OPS_TO_QUANTIZE})

	# Quant2 GRU
	set(QUANT2_GRU_MODEL_ARCHIVE "GRU_quant_acc.tar.gz")
	set(QUANT2_GRU_MODEL_DIR "${QUANT_INSTALL_DIR}/GRU_quant2")
	download_quant_model(${QUANT2_GRU_MODEL_DIR} ${QUANT2_GRU_MODEL_ARCHIVE} cf207f8076dcfb8b74d8b6bdddf9090c)
	set(QUANT2_GRU_OPS_TO_QUANTIZE "multi_gru")

	### Save FP32 model or INT8 model from Quant model
        
	set(QUANT2_INT8_RESNET50_SAVE_PATH "${QUANT_INSTALL_DIR}/ResNet50_quant2_int8")
	save_quant_ic_model_test(save_quant2_model_resnet50 ${QUANT2_RESNET50_MODEL_DIR}/ResNet50_qat_perf/float ${QUANT2_INT8_RESNET50_SAVE_PATH})

	set(QUANT2_INT8_ERNIE_SAVE_PATH "${QUANT_INSTALL_DIR}/Ernie_quant2_int8")
	save_quant_nlp_model_test(save_quant2_model_ernie ${QUANT2_ERNIE_MODEL_DIR}/Ernie_qat/float ${QUANT2_INT8_ERNIE_SAVE_PATH} ${QUANT2_ERNIE_OPS_TO_QUANTIZE})

	set(QUANT2_INT8_GRU_SAVE_PATH "${QUANT_INSTALL_DIR}/GRU_quant2_int8")
	save_quant_nlp_model_test(save_quant2_model_gru ${QUANT2_GRU_MODEL_DIR}/GRU_quant_acc ${QUANT2_INT8_GRU_SAVE_PATH} ${QUANT2_GRU_OPS_TO_QUANTIZE})

	# Convert Quant2 model to dot and pdf files 
	set(QUANT2_INT8_ERNIE_DOT_SAVE_PATH "${QUANT_INSTALL_DIR}/Ernie_quant2_int8_dot_file")
	convert_model2dot_test(convert_model2dot_ernie ${QUANT2_ERNIE_MODEL_DIR}/Ernie_qat/float ${QUANT2_INT8_ERNIE_DOT_SAVE_PATH} "Ernie_quant2_int8")

endif()

# Since the tests for Quant & INT8 comparison support only testing on Linux 
# with MKL-DNN, we remove it here to not test it on other systems.
list(REMOVE_ITEM TEST_OPS
	test_mkldnn_int8_quantization_strategy
	quant_int8_image_classification_comparison
	quant_int8_nlp_comparison)

#TODO(wanghaoshuang): Fix this unitest failed on GCC8.
LIST(REMOVE_ITEM TEST_OPS test_auto_pruning)
LIST(REMOVE_ITEM TEST_OPS test_filter_pruning)
	
# fix
if(WIN32)
    SET(SINGLE_CARD_TEST_OPS
        test_user_defined_quantization
        test_quantization_scale_pass
        test_quantization_pass
        test_moving_average_abs_max_scale_op
        test_imperative_qat_channelwise
        test_imperative_qat
        test_imperative_out_scale
        test_graph)
    LIST(REMOVE_ITEM TEST_OPS ${SINGLE_CARD_TEST_OPS})
    foreach(src ${SINGLE_CARD_TEST_OPS})
        py_test(${src} SRCS ${src}.py ENVS CUDA_VISIBLE_DEVICES=0)
    endforeach()
endif()


foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()

# setting timeout value for old unittests
if(NOT WIN32)
    set_tests_properties(test_post_training_quantization_lstm_model PROPERTIES TIMEOUT 120)
    set_tests_properties(test_post_training_quantization_mobilenetv1 PROPERTIES TIMEOUT 600 LABELS "RUN_TYPE=NIGHTLY")
    set_tests_properties(test_post_training_quantization_resnet50 PROPERTIES TIMEOUT 600 LABELS "RUN_TYPE=NIGHTLY")
    set_tests_properties(test_post_training_quantization_mnist PROPERTIES TIMEOUT 120)
    set_tests_properties(test_imperative_ptq PROPERTIES TIMEOUT 120)
    set_tests_properties(test_weight_quantization_mobilenetv1 PROPERTIES TIMEOUT 120)
endif()

set_tests_properties(test_graph PROPERTIES TIMEOUT 120)
set_tests_properties(test_quantization_pass PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_qat_channelwise PROPERTIES TIMEOUT 120)
set_tests_properties(test_user_defined_quantization PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_qat PROPERTIES TIMEOUT 120)
set_tests_properties(test_imperative_out_scale PROPERTIES TIMEOUT 120)
if(LINUX AND WITH_MKLDNN)
    set_tests_properties(test_quant2_int8_mobilenetv1_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(convert_model2dot_ernie PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant2_int8_resnet50_channelwise_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant_int8_mobilenetv2_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant2_int8_resnet50_range_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(save_quant2_model_resnet50 PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant_int8_resnet50_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant_int8_mobilenetv1_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant2_int8_ernie_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant_int8_googlenet_mkldnn PROPERTIES TIMEOUT 120)
    set_tests_properties(test_quant2_int8_resnet50_mkldnn PROPERTIES TIMEOUT 120)
endif()
