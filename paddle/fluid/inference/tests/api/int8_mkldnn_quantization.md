# INT8 MKL-DNN post-training quantization

This document describes how to use Paddle inference Engine to convert the FP32 models to INT8 models using INT8 MKL-DNN post-training quantization. We provide the instructions on enabling INT8 MKL-DNN quantization in Paddle inference and show the accuracy and performance results of the quantized models, including 7 image classification models: GoogleNet, MobileNet-V1, MobileNet-V2, ResNet-101, ResNet-50, VGG16, VGG19, and 1 object detection model Mobilenet-SSD.

## 0. Install PaddlePaddle

Follow PaddlePaddle [installation instruction](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html) to install PaddlePaddle. If you build PaddlePaddle yourself, please use the following cmake arguments.

```bash
cmake ..  -DWITH_TESTING=ON -WITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_MKLDNN=ON -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON
```

Note: MKL-DNN and MKL are required.

## 1. Enable INT8 MKL-DNN quantization

For reference, please examine the code of unit test enclosed in [analyzer_int8_image_classification_tester.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/analyzer_int8_image_classification_tester.cc) and [analyzer_int8_object_detection_tester.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/analyzer_int8_object_detection_tester.cc).

* ### Create Analysis config

INT8 quantization is one of the optimizations in analysis config. More information about analysis config can be found [here](https://www.paddlepaddle.org.cn/documentation/docs/en/advanced_guide/inference_deployment/inference/native_infer_en.html#a-name-use-analysisconfig-to-manage-inference-configurations-use-analysisconfig-to-manage-inference-configurations-a)

* ### Create quantize config by analysis config

We enable the MKL-DNN quantization procedure by calling an appropriate method from analysis config. Afterwards, all the required quantization parameters (quantization op names, quantization strategies etc.) can be set through quantizer config which is present in the analysis config. It is also necessary to specify a pre-processed warmup dataset and desired batch size.

```cpp
//Enable MKL-DNN quantization
cfg.EnableMkldnnQuantizer();

//use analysis config to call the MKL-DNN quantization config
cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(100);
```

## 2. Accuracy and Performance benchmark for Image Classification models

We provide the results of accuracy and performance measured on Intel(R) Xeon(R) Gold 6271 on single core.

>**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 Accuracy | INT8 Accuracy | Accuracy Diff(INT8-FP32) |
|:------------:|:-------------:|:-------------:|:------------------------:|
|  GoogleNet   |    70.50%     |    70.08%     |          -0.42%          |
| MobileNet-V1 |    70.78%     |    70.41%     |          -0.37%          |
| MobileNet-V2 |    71.90%     |    71.34%     |          -0.56%          |
|  ResNet-101  |    77.50%     |    77.43%     |          -0.07%          |
|  ResNet-50   |    76.63%     |    76.57%     |          -0.06%          |
|    VGG16     |    72.08%     |    72.05%     |          -0.03%          |
|    VGG19     |    72.57%     |    72.57%     |          0.00%           |

>**II. Throughput on Intel(R) Xeon(R) Gold 6271 (batch size 1 on single core)**

|    Model     | FP32 Throughput(images/s) | INT8 Throughput(images/s) | Ratio(INT8/FP32) |
|:------------:|:-------------------------:|:-------------------------:|:----------------:|
|  GoogleNet   |           32.53           |           68.32           |       2.13       |
| MobileNet-V1 |           73.98           |          224.91           |       3.04       |
| MobileNet-V2 |           86.59           |          204.91           |       2.37       |
|  ResNet-101  |           7.15            |           26.73           |       3.74       |
|  ResNet-50   |           13.15           |           49.48           |       3.76       |
|    VGG16     |           3.34            |           10.11           |       3.03       |
|    VGG19     |           2.83            |           8.68            |       3.07       |

* ## Prepare dataset

* Download and preprocess the full ILSVRC2012 Validation dataset.

```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```

Then the ILSVRC2012 Validation dataset binary file is saved by default in `$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin`

* Prepare user local dataset.

```bash
cd /PATH/TO/PADDLE/
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py --local --data_dir=/PATH/TO/USER/DATASET --output_file=/PATH/TO/OUTPUT/BINARY
```

Available options in the above command and their descriptions are as follows:
- **No parameters set:** The script will download the ILSVRC2012_img_val data from server and convert it into a binary file.
- **local:** Once set, the script will process user local data.
- **data_dir:** Path to user local dataset. Default value: None.
- **label_list:** Path to image_label list file. Default value: `val_list.txt`.
- **output_file:** Path to the generated binary file. Default value: `imagenet_small.bin`.
- **data_dim:** The length and width of the preprocessed image. The default value: 224.

The user dataset preprocessed binary file by default is saved in `imagenet_small.bin`.


* ## Commands to reproduce image classification benchmark

You can run `test_analyzer_int8_imagenet_classification` with the following arguments to reproduce the accuracy result on Resnet50.

```bash
cd /PATH/TO/PADDLE/build
./paddle/fluid/inference/tests/api/test_analyzer_int8_image_classification --infer_model=third_party/inference_demo/int8v2/resnet50/model --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
```

To verify all the 7 models, you need to set the parameter of `--infer_model` to one of the following values in command line:

```bash
--infer_model /PATH/TO/PADDLE/build/third_party/inference_demo/int8v2/MODEL_NAME/model
```

```text
MODEL_NAME=googlenet, mobilenetv1, mobilenetv2, resnet101, resnet50, vgg16, vgg19
```

## 3. Accuracy and Performance benchmark for Object Detection models

>**I. mAP on Intel(R) Xeon(R) Gold 6271 (batch size 100 on single core):**

|     Model     | FP32 Accuracy | INT8 Accuracy | Accuracy Diff(INT8-FP32) |
|:-------------:|:-------------:|:-------------:|:------------------------:|
| Mobilenet-SSD |    73.80%     |    73.17%     |          -0.63           |

>**II. Throughput on Intel(R) Xeon(R) Gold 6271 (batch size 100 on single core)**

|     Model     | FP32 Throughput(images/s) | INT8 Throughput(images/s) | Ratio(INT8/FP32) |
|:-------------:|:-------------------------:|:-------------------------:|:----------------:|
| Mobilenet-ssd |           37.94           |          114.94           |       3.03       |

* ## Prepare dataset

* Download and preprocess the full Pascal VOC2007 test set.
  
```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_pascalvoc_test_preprocess.py
```

The Pascal VOC2007 test set binary file is saved by default in `$HOME/.cache/paddle/dataset/pascalvoc/pascalvoc_full.bin`

* Prepare user local dataset.

```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_pascalvoc_test_preprocess.py --local --data_dir=/PATH/TO/USER/DATASET --img_annotation_list=/PATH/TO/ANNOTATION/LIST --label_file=/PATH/TO/LABEL/FILE --output_file=/PATH/TO/OUTPUT/FILE
```
Available options in the above command and their descriptions are as follows:
- **No parameters set:** The script will download the full pascalvoc test dataset and preprocess and convert it into a binary file.
- **local:** Once set, the script will process user local data.
- **data_dir:** Path to user local dataset. Default value: None.
- **img_annotation_list:** Path to img_annotation list file. Default value: `test_100.txt`.
- **label_file:** Path to labels list. Default value: `label_list`.
- **output_file:** Path to generated binary file. Default value: `pascalvoc_small.bin`.

The user dataset preprocessed binary file by default is saved in `pascalvoc_small.bin`.

* ## Commands to reproduce object detection benchmark

You can run `test_analyzer_int8_object_detection` with the following arguments to reproduce the benchmark results for Mobilenet-SSD.

```bash
cd /PATH/TO/PADDLE/build
./paddle/fluid/inference/tests/api/test_analyzer_int8_object_detection --infer_model=third_party/inference_demo/int8v2/mobilenet-ssd/model --infer_data=$HOME/.cache/paddle/dataset/pascalvoc/pascalvoc_full.bin --warmup_batch_size=10 --batch_size=100 --paddle_num_threads=1
```

## 4. Notes

* Measurement of accuracy requires a model which accepts two inputs: data and labels.
* Different sampling batch size data may cause slight difference on INT8 accuracy.
* CAPI performance data is better than python API performance data because of the python overhead. Especially for the small computational model, python overhead will be more obvious.
