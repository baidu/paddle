# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import six
import numpy as np
import platform
import os
from .... import Executor
from .... import io
from .... import core
from ....compiler import CompiledProgram
from ....framework import IrGraph
from ..core.strategy import Strategy
import logging
import sys

__all__ = ['MKLDNNPostTrainingQuantStrategy']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class MKLDNNPostTrainingQuantStrategy(Strategy):
    """
    The strategy for Post Training quantization strategy.
    """

    def __init__(self, int8_model_save_path=None, fp32_model_path=None):
        """
        Args:
            int8_model_save_path(str): The path to save int8 model with fp32 weights.
                            None means it doesn't save int8 model. defalut: None.
            fp32_model_path(str): The path to model with fp32 weight. defalut: None.
        """

        super(Strategy, self).__init__()
        self.start_epoch = 0
        self.end_epoch = 0
        self.int8_model_save_path = int8_model_save_path
        self.fp32_model_path = fp32_model_path

    def on_compression_begin(self, context):
        """
	Prepare the data and quantify the model
        """
        logger.info('InferQuantStrategy::on_compression_begin')

        #Prepare the Analysis Config
        infer_config = core.AnalysisConfig("AnalysisConfig")
        infer_config.switch_ir_optim(True)
        infer_config.disable_gpu
        infer_config.set_model(self.fp32_model_path)
        infer_config.enable_mkldnn()

        #Prepare the data for calculating the quantization scales 
        warmup_reader = context.eval_reader()
        if six.PY2:
            data = warmup_reader.next()

        if six.PY3:
            data = warmup_reader.__next__()

        num_images = len(data)
        images = core.PaddleTensor()
        images.name = "x"
        images.shape = [num_images, ] + list(data[0][0].shape)
        images.dtype = core.PaddleDType.FLOAT32
        image_data = [img.tolist() for (img, _) in data]

        image_data = np.array(image_data).astype("float32")
        image_data = image_data.ravel()
        images.data = core.PaddleBuf(image_data.tolist())

        labels = core.PaddleTensor()
        labels.name = "y"
        labels.shape = [num_images, 1]
        labels.dtype = core.PaddleDType.INT64
        label_data = [[label] for (_, label) in data]
        label_data = np.array(label_data)
        labels.data = core.PaddleBuf(label_data)

        warmup_data = [images, labels]

        #Enable the int8 quantization
        infer_config.enable_quantizer()
        infer_config.quantizer_config().set_quant_data(warmup_data)
        infer_config.quantizer_config().set_quant_batch_size(num_images)
        #Run INT8 MKL-DNN Quantization
        predictor = core.create_paddle_predictor(infer_config)
        if self.int8_model_save_path and self.int8_model_save_path != '':
            if not os.path.exists(self.int8_model_save_path):
                os.makedirs(self.int8_model_save_path)
            predictor.SaveOptimModel(self.int8_model_save_path)

        logger.info('Finish InferQuantStrategy::on_compresseion_begin')
