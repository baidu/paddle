set(INFERENCE_EXTRA_DEPS paddle_inference_api paddle_inference_io ir_pass_manager analysis_predictor benchmark)

if(WITH_GPU AND TENSORRT_FOUND)
    set(INFERENCE_EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} analysis ${analysis_deps})
endif()

function(download_data install_dir data_file)
    string(REGEX MATCH "[^/\\]+$" file_name ${data_file})
    if (NOT EXISTS ${install_dir}/${file_name})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL} ${data_file})
    endif()
endfunction()

function(download_int8_data install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8 ${data_file})
    endif()
endfunction()

function(download_bfloat16_data install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/bfloat16 ${data_file})
    endif()
endfunction()

function(download_GRU_data install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/gru ${data_file})
    endif()
endfunction()

function(download_quant_data install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
	    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file})
    endif()
endfunction()

function(download_model_and_data install_dir model_name data_name)
    download_data(${install_dir} ${model_name}) 
    download_data(${install_dir} ${data_name})
endfunction()

function(download_result install_dir result_name)
    download_data(${install_dir} ${result_name})
endfunction()

function(inference_analysis_api_test target install_dir filename)
    inference_analysis_test(${target} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${install_dir}/model --infer_data=${install_dir}/data.txt --refer_result=${install_dir}/result.txt)
endfunction()

function(inference_multiple_models_analysis_api_test target install_dir filename)
    inference_analysis_test(${target} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${install_dir}/mobilenet_v2_models/1 --infer_model2=${install_dir}/mobilenet_v2_models/xx --infer_model3=${install_dir}/mobilenet_v2_models/3)
endfunction()

function(inference_analysis_api_test_build TARGET_NAME filename)
	inference_analysis_test_build(${TARGET_NAME} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS})
endfunction()

function(inference_analysis_api_int8_test_run TARGET_NAME test_binary model_dir data_path)
	inference_analysis_test_run(${TARGET_NAME}
	COMMAND ${test_binary}
        ARGS --infer_model=${model_dir}/model
             --infer_data=${data_path}
             --warmup_batch_size=${WARMUP_BATCH_SIZE}
             --batch_size=50
             --enable_int8=true
             --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
	     --iterations=2)
endfunction()

function(inference_analysis_api_int8_test_run_custom_warmup_batch_size TARGET_NAME test_binary model_dir data_path warmup_batch_size)
    set(WARMUP_BATCH_SIZE ${warmup_batch_size})
    inference_analysis_api_int8_test_run(${TARGET_NAME} ${test_binary} ${model_dir} ${data_path})
endfunction()

function(inference_analysis_api_bfloat16_test_run TARGET_NAME test_binary model_dir data_path)
	inference_analysis_test_run(${TARGET_NAME}
	COMMAND ${test_binary}
        ARGS --infer_model=${model_dir}/model
             --infer_data=${data_path}
             --batch_size=50
             --enable_bf16=true
             --paddle_num_threads=${CPU_NUM_THREADS_ON_CI}
	     --iterations=2)
endfunction()

function(inference_analysis_api_object_dection_int8_test_run TARGET_NAME test_binary model_dir data_path)
	inference_analysis_test_run(${TARGET_NAME}
	COMMAND ${test_binary}
        ARGS --infer_model=${model_dir}/model
             --infer_data=${data_path}
             --warmup_batch_size=10
             --batch_size=300
             --enable_int8=true
             --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
	     --iterations=1)
endfunction()

function(inference_analysis_api_test_with_fake_data_build TARGET_NAME filename)
	inference_analysis_test_build(${TARGET_NAME} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS})
endfunction()

function(inference_analysis_api_test_with_fake_data_run TARGET_NAME test_binary model_dir disable_fc)
    inference_analysis_test_run(${TARGET_NAME}
	COMMAND ${test_binary}
        ARGS --infer_model=${model_dir}/model
             --disable_mkldnn_fc=${disable_fc}) 
endfunction()

function(inference_analysis_api_quant_test_run TARGET_NAME test_binary fp32_model_dir int8_model_dir data_path)
    inference_analysis_test_run(${TARGET_NAME}
    COMMAND ${test_binary}
        ARGS --fp32_model=${fp32_model_dir}
             --int8_model=${int8_model_dir}
             --infer_data=${data_path}
             --batch_size=50
             --enable_int8=true
             --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
             --with_accuracy_layer=false
             --iterations=2)
endfunction()

function(inference_analysis_api_lexical_test_run TARGET_NAME test_binary infer_model data_path)
    inference_analysis_test_run(${TARGET_NAME}
    COMMAND ${test_binary}
        ARGS --infer_model=${infer_model}
             --infer_data=${data_path}
             --batch_size=50
             --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
             --with_accuracy_layer=true
             --use_analysis=true
             --iterations=2)
endfunction()

function(inference_analysis_api_lexical_bfloat16_test_run TARGET_NAME test_binary infer_model data_path)
    inference_analysis_test_run(${TARGET_NAME}
    COMMAND ${test_binary}
        ARGS --infer_model=${infer_model}
             --infer_data=${data_path}
             --batch_size=50
             --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
             --with_accuracy_layer=true
             --use_analysis=true
             --enable_bf16=true
             --iterations=2)
endfunction()

function(preprocess_data2bin_test_run target py_script_source data_dir output_file)
	py_test(${target} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${py_script_source}
	        ARGS --data_dir=${data_dir}
		     --output_file=${output_file}
		     --local)
endfunction()

if(NOT APPLE AND WITH_MKLML)
    # RNN1
    set(RNN1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn1")
    download_model_and_data(${RNN1_INSTALL_DIR} "rnn1/model.tar.gz" "rnn1/data.txt.tar.gz")
    inference_analysis_api_test(test_analyzer_rnn1 ${RNN1_INSTALL_DIR} analyzer_rnn1_tester.cc)
    
    # seq_pool1
    set(SEQ_POOL1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/seq_pool")
    download_model_and_data(${SEQ_POOL1_INSTALL_DIR} "seq_pool1_model_.tar.gz" "seq_pool1_data.txt.tar.gz")
    inference_analysis_api_test(test_analyzer_seq_pool1_compare_determine ${SEQ_POOL1_INSTALL_DIR} analyzer_seq_pool1_compare_determine_tester.cc)
    inference_analysis_api_test(test_analyzer_seq_pool1 ${SEQ_POOL1_INSTALL_DIR} analyzer_seq_pool1_compare_tester.cc)
    inference_analysis_api_test(test_analyzer_seq_pool1_fuse_compare_zero_copy ${SEQ_POOL1_INSTALL_DIR} analyzer_seq_pool1_fuse_compare_zero_copy_tester.cc)
    inference_analysis_api_test(test_analyzer_seq_pool1_fuse_statis ${SEQ_POOL1_INSTALL_DIR} analyzer_seq_pool1_fuse_statis_tester.cc)
    inference_analysis_api_test(test_analyzer_seq_pool1_profile ${SEQ_POOL1_INSTALL_DIR} analyzer_seq_pool1_profile_tester.cc)
    if(NOT WIN32)
        set_tests_properties(test_analyzer_seq_pool1_compare_determine PROPERTIES TIMEOUT 120)
        set_tests_properties(test_analyzer_seq_pool1 PROPERTIES TIMEOUT 120)
        set_tests_properties(test_analyzer_seq_pool1_fuse_compare_zero_copy PROPERTIES TIMEOUT 120)
        set_tests_properties(test_analyzer_seq_pool1_fuse_statis PROPERTIES TIMEOUT 120)
        set_tests_properties(test_analyzer_seq_pool1_profile PROPERTIES TIMEOUT 120)
    endif()
else()
    # TODO: fix this test on MACOS and OPENBLAS, the reason is that
    # fusion_seqexpand_concat_fc_op is not supported on MACOS and OPENBLAS
    message(WARNING "These tests has been disabled in OSX or WITH_MKL=OFF before being fixed: \n test_analyzer_rnn1")
    message(WARNING "These tests has been disabled in OSX or WITH_MKL=OFF before being fixed: \n test_analyzer_seq_pool1")
endif()


# RNN2
set(RNN2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn2")
download_model_and_data(${RNN2_INSTALL_DIR} "rnn2_model.tar.gz" "rnn2_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_rnn2 ${RNN2_INSTALL_DIR} analyzer_rnn2_tester.cc)

# TODO(luotao, Superjom) Disable DAM test, temporarily fix
# https://github.com/PaddlePaddle/Paddle/issues/15032#issuecomment-455990914.
# After inference framework refactor, will reopen it.
# normal DAM
set(DAM_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/dam")
download_model_and_data(${DAM_INSTALL_DIR} "DAM_model.tar.gz" "DAM_data.txt.tar.gz")
#inference_analysis_api_test(test_analyzer_dam ${DAM_INSTALL_DIR} analyzer_dam_tester.cc EXTRA_DEPS legacy_allocator)

# small DAM
set(DAM_SMALL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/small_dam")
download_model_and_data(${DAM_SMALL_INSTALL_DIR} "dam_small_model.tar.gz" "dam_small_data.txt.tar.gz")
inference_analysis_test(test_analyzer_small_dam SRCS analyzer_dam_tester.cc
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${DAM_SMALL_INSTALL_DIR}/model --infer_data=${DAM_SMALL_INSTALL_DIR}/data.txt)

#save model 
inference_analysis_api_test(test_analyzer_save_model ${DAM_SMALL_INSTALL_DIR} analyzer_save_model_tester.cc)

# chinese_ner
set(CHINESE_NER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/chinese_ner")
download_model_and_data(${CHINESE_NER_INSTALL_DIR} "chinese_ner_model.tar.gz" "chinese_ner-data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_ner ${CHINESE_NER_INSTALL_DIR} analyzer_ner_tester.cc)

# lac
set(LAC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lac")
download_model_and_data(${LAC_INSTALL_DIR} "lac_model.tar.gz" "lac_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_lac ${LAC_INSTALL_DIR} analyzer_lac_tester.cc)

# Pyramid DNN
set(PYRAMID_DNN_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/pyramid_dnn")
download_model_and_data(${PYRAMID_DNN_INSTALL_DIR} "PyramidDNN_model.tar.gz" "PyramidDNN_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_pyramid_dnn ${PYRAMID_DNN_INSTALL_DIR} analyzer_pyramid_dnn_tester.cc)

#Ernie
set(ERNIE_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/Ernie")
download_model_and_data(${ERNIE_INSTALL_DIR} "Ernie_model.tar.gz" "Ernie_data.txt.tar.gz" "Ernie_result.txt.tar.gz")
download_result(${ERNIE_INSTALL_DIR} "Ernie_result.txt.tar.gz")
inference_analysis_api_test(test_analyzer_ernie ${ERNIE_INSTALL_DIR} analyzer_ernie_tester.cc)

#Ernie large
set(ERNIE_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/Ernie_Large")
download_model_and_data(${ERNIE_INSTALL_DIR} "Ernie_large_model.tar.gz" "Ernie_large_data.txt.tar.gz" "Ernie_large_result.txt.tar.gz")
download_result(${ERNIE_INSTALL_DIR} "Ernie_large_result.txt.tar.gz")
inference_analysis_test(test_analyzer_ernie_large SRCS analyzer_ernie_tester.cc
    EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
    ARGS --infer_model=${ERNIE_INSTALL_DIR}/model --infer_data=${ERNIE_INSTALL_DIR}/data.txt --refer_result=${ERNIE_INSTALL_DIR}/result.txt --ernie_large=true)
if(NOT WIN32 AND NOT APPLE)
    set_tests_properties(test_analyzer_ernie_large PROPERTIES TIMEOUT 150 LABELS "RUN_TYPE=NIGHTLY")
endif()
if (WIN32)
    set_tests_properties(test_analyzer_ernie_large PROPERTIES TIMEOUT 200)
endif()

# text_classification
set(TEXT_CLASSIFICATION_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/text_classification")
download_model_and_data(${TEXT_CLASSIFICATION_INSTALL_DIR} "text-classification-Senta.tar.gz" "text_classification_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_text_classification ${TEXT_CLASSIFICATION_INSTALL_DIR} analyzer_text_classification_tester.cc)

# seq_conv1
set(SEQ_CONV1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/seq_conv1")
download_model_and_data(${SEQ_CONV1_INSTALL_DIR} "seq_conv1_model.tar.gz" "seq_conv1_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_seq_conv1 ${SEQ_CONV1_INSTALL_DIR} analyzer_seq_conv1_tester.cc)

# transformer, the dataset only works on batch_size=8 now
set(TRANSFORMER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/transformer")
download_model_and_data(${TRANSFORMER_INSTALL_DIR} "temp/transformer_model.tar.gz" "temp/transformer_data.txt.tar.gz")
inference_analysis_test(test_analyzer_transformer SRCS analyzer_transformer_compare_tester.cc 
  EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
  ARGS --infer_model=${TRANSFORMER_INSTALL_DIR}/model --infer_data=${TRANSFORMER_INSTALL_DIR}/data.txt --batch_size=8 
       --cpu_num_threads=${CPU_NUM_THREADS_ON_CI})
inference_analysis_test(test_analyzer_transformer_fuse SRCS analyzer_transformer_fuse_tester.cc 
  EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
  ARGS --infer_model=${TRANSFORMER_INSTALL_DIR}/model --infer_data=${TRANSFORMER_INSTALL_DIR}/data.txt --batch_size=8 
       --cpu_num_threads=${CPU_NUM_THREADS_ON_CI})
inference_analysis_test(test_analyzer_transformer_profile SRCS analyzer_transformer_profile_tester.cc 
  EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
  ARGS --infer_model=${TRANSFORMER_INSTALL_DIR}/model --infer_data=${TRANSFORMER_INSTALL_DIR}/data.txt --batch_size=8 
       --cpu_num_threads=${CPU_NUM_THREADS_ON_CI})

# ocr
set(OCR_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/ocr")
if (NOT EXISTS ${OCR_INSTALL_DIR}/ocr.tar.gz)
    inference_download_and_uncompress(${OCR_INSTALL_DIR} "http://paddlemodels.bj.bcebos.com/" "inference-vis-demos/ocr.tar.gz")
endif()
inference_analysis_api_test(test_analyzer_ocr ${OCR_INSTALL_DIR} analyzer_vis_tester.cc)

# densebox
set(DENSEBOX_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/densebox")
download_data(${DENSEBOX_INSTALL_DIR} "densebox.tar.gz")
#inference_analysis_test(test_analyzer_detect SRCS analyzer_detect_tester.cc 
#  EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
#  ARGS --infer_model=${DENSEBOX_INSTALL_DIR}/model --infer_data=${DENSEBOX_INSTALL_DIR}/detect_input_50.txt 
#       --infer_shape=${DENSEBOX_INSTALL_DIR}/shape_50.txt)
#set_property(TEST test_analyzer_detect PROPERTY ENVIRONMENT GLOG_vmodule=analysis_predictor=2)

# mobilenet with transpose op
set(MOBILENET_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet")
if (NOT EXISTS ${MOBILENET_INSTALL_DIR}/mobilenet.tar.gz)
    inference_download_and_uncompress(${MOBILENET_INSTALL_DIR} "http://paddlemodels.bj.bcebos.com/" "inference-vis-demos/mobilenet.tar.gz")
endif()
inference_analysis_api_test(test_analyzer_mobilenet_transpose ${MOBILENET_INSTALL_DIR} analyzer_vis_tester.cc)

### Image classification tests with fake data
set(IMG_CLASS_TEST_APP "test_analyzer_image_classification")
set(IMG_CLASS_TEST_APP_SRC "analyzer_image_classification_tester.cc")

# build test binary to be used in subsequent tests
inference_analysis_api_test_with_fake_data_build(${IMG_CLASS_TEST_APP} ${IMG_CLASS_TEST_APP_SRC})

# googlenet
set(GOOGLENET_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/googlenet")
download_data(${GOOGLENET_MODEL_DIR} "googlenet.tar.gz")
inference_analysis_api_test_with_fake_data_run(test_analyzer_googlenet ${IMG_CLASS_TEST_APP}
	${GOOGLENET_MODEL_DIR} false)

# resnet50
set(RESNET50_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/resnet50")
download_data(${RESNET50_MODEL_DIR} "resnet50_model.tar.gz")
inference_analysis_api_test_with_fake_data_run(test_analyzer_resnet50 ${IMG_CLASS_TEST_APP}
	${RESNET50_MODEL_DIR} true)
if (WIN32)
    set_tests_properties(test_analyzer_resnet50 PROPERTIES TIMEOUT 200)
endif()


# mobilenet with depthwise_conv op
set(MOBILENET_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet_depthwise_conv")
download_data(${MOBILENET_MODEL_DIR} "mobilenet_model.tar.gz")
inference_analysis_api_test_with_fake_data_run(test_analyzer_mobilenet_depthwise_conv ${IMG_CLASS_TEST_APP}
	${MOBILENET_MODEL_DIR} false)

if(WITH_MKLDNN)

  ### INT8 tests

  set(INT8_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")

  ## Image classification models

  # ImageNet small dataset
  # It may be already downloaded for Quant & INT8 unit tests
  set(IMAGENET_DATA_ARCHIVE "imagenet_val_100_tail.tar.gz")
  set(IMAGENET_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/imagenet")
  set(IMAGENET_DATA_PATH "${IMAGENET_DATA_DIR}/data.bin")
  download_int8_data(${IMAGENET_DATA_DIR} ${IMAGENET_DATA_ARCHIVE})

  # build test binary to be used in subsequent tests
  set(INT8_IMG_CLASS_TEST_APP "test_analyzer_int8_image_classification")
  set(INT8_IMG_CLASS_TEST_APP_SRC "analyzer_int8_image_classification_tester.cc")
  inference_analysis_api_test_build(${INT8_IMG_CLASS_TEST_APP} ${INT8_IMG_CLASS_TEST_APP_SRC})

  # resnet50 int8
  set(INT8_RESNET50_MODEL_DIR "${INT8_DATA_DIR}/resnet50")
  download_int8_data(${INT8_RESNET50_MODEL_DIR} "resnet50_int8_model.tar.gz" )
  inference_analysis_api_int8_test_run(test_analyzer_int8_resnet50 ${INT8_IMG_CLASS_TEST_APP} ${INT8_RESNET50_MODEL_DIR} ${IMAGENET_DATA_PATH})

  # mobilenetv1 int8
  set(INT8_MOBILENETV1_MODEL_DIR "${INT8_DATA_DIR}/mobilenetv1")
  download_int8_data(${INT8_MOBILENETV1_MODEL_DIR} "mobilenetv1_int8_model.tar.gz" )
  inference_analysis_api_int8_test_run(test_analyzer_int8_mobilenetv1 ${INT8_IMG_CLASS_TEST_APP} ${INT8_MOBILENETV1_MODEL_DIR} ${IMAGENET_DATA_PATH})
 
  # mobilenetv2 int8
  set(INT8_MOBILENETV2_MODEL_DIR "${INT8_DATA_DIR}/mobilenetv2")
  download_int8_data(${INT8_MOBILENETV2_MODEL_DIR} "mobilenet_v2_int8_model.tar.gz" )
  inference_analysis_api_int8_test_run(test_analyzer_int8_mobilenetv2 ${INT8_IMG_CLASS_TEST_APP} ${INT8_MOBILENETV2_MODEL_DIR} ${IMAGENET_DATA_PATH})
 
  # resnet101 int8
  # TODO(grygielski) Enable after MKL-DNN 1.0 merge
  set(INT8_RESNET101_MODEL_DIR "${INT8_DATA_DIR}/resnet101")
  download_int8_data(${INT8_RESNET101_MODEL_DIR} "Res101_int8_model.tar.gz" )
#   inference_analysis_api_int8_test_run(test_analyzer_int8_resnet101 ${INT8_IMG_CLASS_TEST_APP} ${INT8_RESNET101_MODEL_DIR} ${IMAGENET_DATA_PATH})
 
  # vgg16 int8
  # TODO(grygielski) Enable after MKL-DNN 1.0 merge
  set(INT8_VGG16_MODEL_DIR "${INT8_DATA_DIR}/vgg16")
  download_int8_data(${INT8_VGG16_MODEL_DIR} "VGG16_int8_model.tar.gz" )
#   inference_analysis_api_int8_test_run(test_analyzer_int8_vgg16 ${INT8_IMG_CLASS_TEST_APP} ${INT8_VGG16_MODEL_DIR} ${IMAGENET_DATA_PATH})
 
  # vgg19 int8
  # TODO(grygielski) Enable after MKL-DNN 1.0 merge
  set(INT8_VGG19_MODEL_DIR "${INT8_DATA_DIR}/vgg19")
  download_int8_data(${INT8_VGG19_MODEL_DIR} "VGG19_int8_model.tar.gz" )
#   inference_analysis_api_int8_test_run(test_analyzer_int8_vgg19 ${INT8_IMG_CLASS_TEST_APP} ${INT8_VGG19_MODEL_DIR} ${IMAGENET_DATA_PATH})

  # googlenet int8
  set(INT8_GOOGLENET_MODEL_DIR "${INT8_DATA_DIR}/googlenet")
  download_int8_data(${INT8_GOOGLENET_MODEL_DIR} "GoogleNet_int8_model.tar.gz" )
  inference_analysis_api_int8_test_run_custom_warmup_batch_size(test_analyzer_int8_googlenet ${INT8_IMG_CLASS_TEST_APP} ${INT8_GOOGLENET_MODEL_DIR} ${IMAGENET_DATA_PATH} 10)

  ### BFLOAT16 tests

  # build test binary to be used in subsequent tests
  set(BF16_IMG_CLASS_TEST_APP "test_analyzer_bfloat16_image_classification")
  set(BF16_IMG_CLASS_TEST_APP_SRC "analyzer_bfloat16_image_classification_tester.cc")
  inference_analysis_api_test_build(${BF16_IMG_CLASS_TEST_APP} ${BF16_IMG_CLASS_TEST_APP_SRC})

  # resnet50 bfloat16
  inference_analysis_api_bfloat16_test_run(test_analyzer_bfloat16_resnet50 ${BF16_IMG_CLASS_TEST_APP} ${INT8_RESNET50_MODEL_DIR} ${IMAGENET_DATA_PATH})
  
  # googlenet bfloat16
  inference_analysis_api_bfloat16_test_run(test_analyzer_bfloat16_googlenet ${BF16_IMG_CLASS_TEST_APP} ${INT8_GOOGLENET_MODEL_DIR} ${IMAGENET_DATA_PATH})

  # mobilenetv1 bfloat16
  inference_analysis_api_bfloat16_test_run(test_analyzer_bfloat16_mobilenetv1 ${BF16_IMG_CLASS_TEST_APP} ${INT8_MOBILENETV1_MODEL_DIR} ${IMAGENET_DATA_PATH})

  # mobilenetv2 bfloat16
  inference_analysis_api_bfloat16_test_run(test_analyzer_bfloat16_mobilenetv2 ${BF16_IMG_CLASS_TEST_APP} ${INT8_MOBILENETV2_MODEL_DIR} ${IMAGENET_DATA_PATH})

  ### Object detection models
  set(PASCALVOC_DATA_PATH "${INT8_DATA_DIR}/pascalvoc_val_head_300.bin")
  set(INT8_OBJ_DETECT_TEST_APP "test_analyzer_int8_object_detection")
  set(INT8_OBJ_DETECT_TEST_APP_SRC "analyzer_int8_object_detection_tester.cc")

  # download dataset if necessary
  download_int8_data(${INT8_DATA_DIR} "pascalvoc_val_head_300.tar.gz")


  # build test binary to be used in subsequent tests
  inference_analysis_api_test_build(${INT8_OBJ_DETECT_TEST_APP} ${INT8_OBJ_DETECT_TEST_APP_SRC})

  # mobilenet-ssd int8
  set(INT8_MOBILENET_SSD_MODEL_DIR "${INT8_DATA_DIR}/mobilenet-ssd")
  download_int8_data(${INT8_MOBILENET_SSD_MODEL_DIR} "mobilenet_ssd_int8_model.tar.gz" )
  inference_analysis_api_object_dection_int8_test_run(test_analyzer_int8_mobilenet_ssd ${INT8_OBJ_DETECT_TEST_APP} ${INT8_MOBILENET_SSD_MODEL_DIR} ${PASCALVOC_DATA_PATH})

  ### Lexcial analysis GRU model
  set(GRU_PATH "${INFERENCE_DEMO_INSTALL_DIR}/gru")
  download_GRU_data("${GRU_PATH}" "GRU_eval_data.tar.gz")
  download_GRU_data("${GRU_PATH}" "GRU_eval_model_v2.tar.gz")
  set(GRU_DATA_PATH "${GRU_PATH}/GRU_eval_data.bin")
  set(GRU_MODEL_PATH "${GRU_PATH}/GRU_eval_model_v2")
  set(LEXICAL_TEST_APP "test_analyzer_lexical_analysis")
  set(LEXICAL_TEST_APP_SRC "analyzer_lexical_analysis_gru_tester.cc")

  # build test binary to be used in subsequent tests
  inference_analysis_api_test_build(${LEXICAL_TEST_APP} ${LEXICAL_TEST_APP_SRC})
  # run lexcial analysis test
  inference_analysis_api_lexical_test_run(test_analyzer_lexical_gru ${LEXICAL_TEST_APP} ${GRU_MODEL_PATH} ${GRU_DATA_PATH})
  # run bfloat16 lexical analysis test
  inference_analysis_api_lexical_bfloat16_test_run(test_analyzer_lexical_gru_bfloat16 ${LEXICAL_TEST_APP} ${GRU_MODEL_PATH} ${GRU_DATA_PATH})

  ### optimized FP32 vs. Quant INT8 tests
  
  set(QUANT_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/quant")
  set(QUANT_IMG_CLASS_TEST_APP "test_analyzer_quant_image_classification")
  set(QUANT_IMG_CLASS_TEST_APP_SRC "analyzer_quant_image_classification_tester.cc")

  # build test binary to be used in subsequent tests
  inference_analysis_api_test_build(${QUANT_IMG_CLASS_TEST_APP} ${QUANT_IMG_CLASS_TEST_APP_SRC})

  # MobileNetV1 FP32 vs. Quant INT8
  # The FP32 model should already be downloaded for slim Quant unit tests on Linux
  set(QUANT2_MobileNetV1_MODEL_DIR "${QUANT_DATA_DIR}/MobileNetV1_quant2")
  set(QUANT2_INT8_MobileNetV1_MODEL_DIR "${QUANT_DATA_DIR}/MobileNetV1_quant2_int8")
  if(NOT LINUX)
      download_quant_data(${QUANT2_MobileNetV1_MODEL_DIR} "MobileNet_qat_perf.tar.gz")
  endif(NOT LINUX)
  download_quant_data(${QUANT2_INT8_MobileNetV1_MODEL_DIR} "MobileNet_qat_perf_int8.tar.gz")
  inference_analysis_api_quant_test_run(test_analyzer_quant_performance_benchmark ${QUANT_IMG_CLASS_TEST_APP} ${QUANT2_MobileNetV1_MODEL_DIR}/MobileNet_qat_perf/float ${QUANT2_INT8_MobileNetV1_MODEL_DIR}/MobileNet_qat_perf_int8 ${IMAGENET_DATA_PATH})

  ### Other tests
 
  # MKLDNN quantizer config
  set(MKLDNN_QUANTIZER_CONFIG_TEST_APP "test_mkldnn_quantizer_config")
  set(MKLDNN_QUANTIZER_CONFIG_TEST_APP_SRC "mkldnn_quantizer_config_tester.cc")
  inference_analysis_api_test_build(${MKLDNN_QUANTIZER_CONFIG_TEST_APP} ${MKLDNN_QUANTIZER_CONFIG_TEST_APP_SRC})
  inference_analysis_test_run(test_mkldnn_quantizer_config COMMAND ${MKLDNN_QUANTIZER_CONFIG_TEST_APP})

  # preprocess data2bin imagenet
    download_int8_data(${INT8_DATA_DIR} "imagenet_small.tar.gz")
    set(IMAGENET_SMALL_DATA_DIR "${INT8_DATA_DIR}/imagenet_small")
    set(IMAGENET_SMALL_OUTPUT_FILE "imagenet_small.bin")
    preprocess_data2bin_test_run(preprocess_local_imagenet "full_ILSVRC2012_val_preprocess.py" ${IMAGENET_SMALL_DATA_DIR} ${IMAGENET_SMALL_OUTPUT_FILE})
    
  # preprocess data2bin pascalvoc
  download_int8_data(${INT8_DATA_DIR} "pascalvoc_small.tar.gz")
  set(PASCALVOC_SMALL_DATA_DIR "${INT8_DATA_DIR}/pascalvoc_small")
  set(PASCALVOC_SMALL_OUTPUT_FILE "pascalvoc_small.bin")
  preprocess_data2bin_test_run(preprocess_local_pascalvoc "full_pascalvoc_test_preprocess.py" ${PASCALVOC_SMALL_DATA_DIR} ${PASCALVOC_SMALL_OUTPUT_FILE})

endif()

# bert, max_len=20, embedding_dim=128
set(BERT_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/bert_emb128")
download_model_and_data(${BERT_INSTALL_DIR} "bert_emb128_model.tar.gz" "bert_data_len20.txt.tar.gz")
inference_analysis_api_test(test_analyzer_bert ${BERT_INSTALL_DIR} analyzer_bert_tester.cc)

# multiple models prediction
set(MMP_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/multi_model_prediction")
download_data(${MMP_INSTALL_DIR} PaddleInference/mobilenet_v2_models.tar.gz) 
inference_multiple_models_analysis_api_test(test_analyzer_multi_model_prediction ${MMP_INSTALL_DIR} analyzer_mmp_tester.cc)

if(WITH_GPU AND TENSORRT_FOUND)
    set(TRT_MODEL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/trt_models")
    if (NOT EXISTS ${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models.tar.gz)
        inference_download_and_uncompress(${TRT_MODEL_INSTALL_DIR} ${INFERENCE_URL}/tensorrt_test "trt_inference_test_models.tar.gz")
    endif()
    set(TEST_SPLIT_CONVERTER_MODEL "${TRT_MODEL_INSTALL_DIR}/trt_split_op_converter_test")
    if (NOT EXISTS ${TEST_SPLIT_CONVERTER_MODEL}/split_converter.tgz)
        inference_download_and_uncompress(${TEST_SPLIT_CONVERTER_MODEL} ${INFERENCE_URL}/tensorrt_test "split_converter.tgz")
    endif()
    set(TEST_INSTANCE_NORM_MODEL "${TRT_MODEL_INSTALL_DIR}/trt_instance_norm_test")
    if (NOT EXISTS ${TEST_INSTANCE_NORM_MODEL}/instance_norm.tgz)
        inference_download_and_uncompress(${TEST_INSTANCE_NORM_MODEL} ${INFERENCE_URL}/tensorrt_test "instance_norm.tgz")
    endif()
    inference_analysis_test(trt_mobilenet_test SRCS trt_mobilenet_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(trt_resnet50_test SRCS trt_resnet50_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(trt_resnext_test SRCS trt_resnext_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(trt_fc_prelu_test SRCS trt_fc_prelu_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(trt_cascade_rcnn_test SRCS trt_cascade_rcnn_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(trt_split_converter_test SRCS trt_split_converter_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${TEST_SPLIT_CONVERTER_MODEL}/)
    inference_analysis_test(trt_instance_norm_test SRCS trt_instance_norm_converter_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${TEST_INSTANCE_NORM_MODEL}/)
    inference_analysis_test(test_analyzer_capi_exp_gpu SRCS analyzer_capi_exp_gpu_tester.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
    inference_analysis_test(test_analyzer_capi_exp_xpu SRCS analyzer_capi_exp_xpu_tester.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_inference_test_models)
            
    set(TRT_MODEL_QUANT_RESNET_DIR "${INFERENCE_DEMO_INSTALL_DIR}/small_quant_model")
    if (NOT EXISTS ${INFERENCE_DEMO_INSTALL_DIR}/small_quant_model.tgz)
        inference_download_and_uncompress(${INFERENCE_DEMO_INSTALL_DIR} ${INFERENCE_URL}/tensorrt_test "small_quant_model.tgz")
    endif()
    inference_analysis_test(trt_quant_int8_test SRCS trt_quant_int8_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_QUANT_RESNET_DIR})

    set(TRT_MODEL_QUANT_YOLOV3_DIR "${INFERENCE_DEMO_INSTALL_DIR}/yolov3_r50_quant_aware")
    if (NOT EXISTS ${INFERENCE_DEMO_INSTALL_DIR}/yolov3_r50_quant_aware.tgz)
        inference_download_and_uncompress(${INFERENCE_DEMO_INSTALL_DIR} ${INFERENCE_URL}/tensorrt_test "yolov3_r50_quant_aware.tgz")
    endif()
    inference_analysis_test(trt_quant_int8_yolov3_r50_test SRCS trt_quant_int8_yolov3_r50_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_QUANT_YOLOV3_DIR})

    set(TEST_TRT_DYNAMIC_MODEL2 "${TRT_MODEL_INSTALL_DIR}/complex_model_dynamic")
    if (NOT EXISTS ${TEST_TRT_DYNAMIC_MODEL2}/complex_model_dynamic2.tar.gz)
        inference_download_and_uncompress(${TEST_TRT_DYNAMIC_MODEL2} ${INFERENCE_URL}/tensorrt_test "complex_model_dynamic2.tar.gz")
    endif()

    set(TEST_TRT_DYNAMIC_MODEL "${TRT_MODEL_INSTALL_DIR}/conv_bn_swish_split_gelu")
    if (NOT EXISTS ${TEST_TRT_DYNAMIC_MODEL}/conv_bn_swish_split_gelu.tar.gz)
        inference_download_and_uncompress(${TEST_TRT_DYNAMIC_MODEL} ${INFERENCE_URL}/tensorrt_test "conv_bn_swish_split_gelu.tar.gz")
    endif()
    inference_analysis_test(trt_dynamic_shape_test SRCS trt_dynamic_shape_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
            ARGS --infer_model=${TRT_MODEL_INSTALL_DIR})

    set(TEST_TRT_ERNIE_MODEL "${TRT_MODEL_INSTALL_DIR}/ernie_test")
    if (NOT EXISTS ${TEST_TRT_ERNIE_MODEL}/ernie_model_4.tar.gz)
        inference_download_and_uncompress(${TEST_TRT_ERNIE_MODEL} ${INFERENCE_URL}/tensorrt_test "ernie_model_4.tar.gz")
    endif()

    inference_analysis_test(test_trt_dynamic_shape_ernie SRCS trt_dynamic_shape_ernie_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${TEST_TRT_ERNIE_MODEL}/ernie_model_4)

    set(TEST_TRT_TRANSFORMER_PRUNE_MODEL "${TRT_MODEL_INSTALL_DIR}/transformer_prune")
    if (NOT EXISTS ${TEST_TRT_TRANSFORMER_PRUNE_MODEL}/transformer_prune.tar.gz)
        inference_download_and_uncompress(${TEST_TRT_TRANSFORMER_PRUNE_MODEL} ${INFERENCE_URL}/tensorrt_test "transformer_prune.tar.gz")
    endif()

    inference_analysis_test(test_trt_dynamic_shape_transformer_prune SRCS trt_dynamic_shape_transformer_prune_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${TEST_TRT_TRANSFORMER_PRUNE_MODEL}/transformer_prune)

    if (NOT EXISTS ${TEST_TRT_ERNIE_MODEL}/ernie_model_4_unserialized.tgz)
        inference_download_and_uncompress(${TEST_TRT_ERNIE_MODEL} ${INFERENCE_URL}/tensorrt_test "ernie_model_4_unserialized.tgz")
    endif()

    inference_analysis_test(test_trt_dynamic_shape_ernie_ser_deser SRCS trt_dynamic_shape_ernie_serialize_deserialize_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${TEST_TRT_ERNIE_MODEL}/ernie_model_4_unserialized)

    if (NOT EXISTS ${TEST_TRT_ERNIE_MODEL}/ernie_model_4_fp16_unserialized.tgz)
        inference_download_and_uncompress(${TEST_TRT_ERNIE_MODEL} ${INFERENCE_URL}/tensorrt_test "ernie_model_4_fp16_unserialized.tgz")
    endif()

    inference_analysis_test(test_trt_dynamic_shape_ernie_fp16_ser_deser SRCS trt_dynamic_shape_ernie_fp16_serialize_deserialize_test.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${TEST_TRT_ERNIE_MODEL}/ernie_model_4_fp16_unserialized)

endif()

set(LITE_MODEL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lite")
download_data(${LITE_MODEL_INSTALL_DIR} "mul_model_fp32.tgz")

inference_analysis_test(lite_mul_model_test SRCS lite_mul_model_test.cc
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${LITE_MODEL_INSTALL_DIR})
inference_analysis_test(lite_resnet50_test SRCS lite_resnet50_test.cc
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${RESNET50_MODEL_DIR})

inference_analysis_test(test_analyzer_capi_exp SRCS analyzer_capi_exp_tester.cc
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
        ARGS --infer_model=${RESNET50_MODEL_DIR}/model)

inference_analysis_test(test_analyzer_capi_exp_pd_config SRCS analyzer_capi_exp_pd_config_tester.cc
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
        ARGS --infer_model=${MOBILENET_INSTALL_DIR}/model)

inference_analysis_test(test_analyzer_capi_exp_pd_tensor SRCS analyzer_capi_exp_pd_tensor_tester.cc
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
        ARGS --infer_model=${MOBILENET_INSTALL_DIR}/model)

if (NOT APPLE AND NOT WIN32)
    inference_analysis_test(test_analyzer_capi_exp_pd_threads SRCS analyzer_capi_exp_pd_threads_tester.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
            ARGS --infer_model=${MOBILENET_INSTALL_DIR}/model)
endif()
inference_analysis_test(test_analyzer_zerocopytensor_tensor SRCS analyzer_zerocopy_tensor_tester.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${OCR_INSTALL_DIR}/model)        

inference_analysis_test(test_analyzer_paddletensor_tensor SRCS analyzer_paddle_tensor_tester.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} 
            ARGS --infer_model=${OCR_INSTALL_DIR}/model --infer_data=${OCR_INSTALL_DIR}/data.txt --refer_result=${OCR_INSTALL_DIR}/result.txt)    
            
if(WITH_MKLDNN)
  inference_analysis_test(test_analyzer_capi_exp_int SRCS analyzer_capi_exp_int_tester.cc
            EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
            ARGS --infer_model=${INT8_DATA_DIR}/resnet50/model)
endif()

inference_analysis_test(test_analyzer_capi_exp_ner SRCS analyzer_capi_exp_ner_tester.cc 
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} paddle_inference_c
        ARGS --infer_model=${CHINESE_NER_INSTALL_DIR}/model)

if(WITH_GPU)
    inference_analysis_test(paddle_infer_api_test SRCS paddle_infer_api_test.cc
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${RESNET50_MODEL_DIR})
endif()

if(WITH_GPU AND TENSORRT_FOUND)
    set_tests_properties(trt_resnext_test PROPERTIES TIMEOUT 300)
    set_tests_properties(trt_quant_int8_yolov3_r50_test PROPERTIES TIMEOUT 300)
    set_tests_properties(trt_resnet50_test PROPERTIES TIMEOUT 300)
    set_tests_properties(trt_cascade_rcnn_test PROPERTIES TIMEOUT 300)
    set_tests_properties(test_trt_dynamic_shape_ernie_ser_deser PROPERTIES TIMEOUT 300)
    set_tests_properties(test_trt_dynamic_shape_ernie_fp16_ser_deser PROPERTIES TIMEOUT 300)
    set_tests_properties(test_trt_dynamic_shape_ernie PROPERTIES TIMEOUT 300)
endif()

if(WITH_MKLDNN)
    set_tests_properties(test_analyzer_int8_resnet50 PROPERTIES TIMEOUT 120)
    set_tests_properties(test_analyzer_int8_mobilenet_ssd PROPERTIES TIMEOUT 120)
    set_tests_properties(test_analyzer_quant_performance_benchmark PROPERTIES TIMEOUT 120)
    set_tests_properties(test_analyzer_int8_mobilenetv2 PROPERTIES TIMEOUT 120)
    set_tests_properties(test_analyzer_int8_mobilenetv1 PROPERTIES TIMEOUT 120)
endif()

set_tests_properties(lite_resnet50_test PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_mobilenet_transpose PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_resnet50 PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_ner PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_ernie PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_googlenet PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_small_dam PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_transformer PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_bert PROPERTIES TIMEOUT 120)
set_tests_properties(test_analyzer_mobilenet_depthwise_conv PROPERTIES TIMEOUT 120)
if(WITH_GPU AND TENSORRT_FOUND)
    set_tests_properties(trt_mobilenet_test PROPERTIES TIMEOUT 120)
    if(WITH_MKLDNN)
        set_tests_properties(test_analyzer_bfloat16_resnet50 PROPERTIES TIMEOUT 120)
    endif()
endif()
if(ON_INFER OR WITH_GPU)
    set_tests_properties(test_analyzer_transformer_profile PROPERTIES TIMEOUT 120)
endif()
