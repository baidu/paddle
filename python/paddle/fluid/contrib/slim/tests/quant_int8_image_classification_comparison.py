#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import unittest
import os
import sys
import argparse
import logging
import struct
import six
import numpy as np
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantInt8MkldnnPass
from paddle.fluid import core

paddle.enable_static()

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=0,
        help='Number of the first minibatches to skip in performance statistics.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='If used, the graph of Quant model is drawn.')
    parser.add_argument(
        '--quant_model', type=str, default='', help='A path to a Quant model.')
    parser.add_argument('--infer_data', type=str, default='', help='Data file.')
    parser.add_argument(
        '--batch_num',
        type=int,
        default=0,
        help='Number of batches to process. 0 or less means whole dataset. Default: 0.'
    )
    parser.add_argument(
        '--acc_diff_threshold',
        type=float,
        default=0.01,
        help='Accepted accuracy difference threshold.')

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


class QuantInt8ImageClassificationComparisonTest(unittest.TestCase):
    """
    Test for accuracy comparison of Quant FP32 and INT8 Image Classification inference.
    """

    def _reader_creator(self, data_file='data.bin'):
        def reader():
            with open(data_file, 'rb') as fp:
                num = fp.read(8)
                num = struct.unpack('q', num)[0]
                imgs_offset = 8
                img_ch = 3
                img_w = 224
                img_h = 224
                img_pixel_size = 4
                img_size = img_ch * img_h * img_w * img_pixel_size
                label_size = 8
                labels_offset = imgs_offset + num * img_size

                step = 0
                while step < num:
                    fp.seek(imgs_offset + img_size * step)
                    img = fp.read(img_size)
                    img = struct.unpack_from(
                        '{}f'.format(img_ch * img_w * img_h), img)
                    img = np.array(img)
                    img.shape = (img_ch, img_w, img_h)
                    fp.seek(labels_offset + label_size * step)
                    label = fp.read(label_size)
                    label = struct.unpack('q', label)[0]
                    yield img, int(label)
                    step += 1

        return reader

    def _get_batch_accuracy(self, batch_output=None, labels=None):
        total = 0
        correct = 0
        correct_5 = 0
        for n, result in enumerate(batch_output):
            index = result.argsort()
            top_1_index = index[-1]
            top_5_index = index[-5:]
            total += 1
            if top_1_index == labels[n]:
                correct += 1
            if labels[n] in top_5_index:
                correct_5 += 1
        acc1 = float(correct) / float(total)
        acc5 = float(correct_5) / float(total)
        return acc1, acc5

    def _prepare_for_fp32_mkldnn(self, graph):
        ops = graph.all_op_nodes()
        for op_node in ops:
            name = op_node.name()
            if name in ['depthwise_conv2d']:
                input_var_node = graph._find_node_by_name(
                    op_node.inputs, op_node.input("Input")[0])
                weight_var_node = graph._find_node_by_name(
                    op_node.inputs, op_node.input("Filter")[0])
                output_var_node = graph._find_node_by_name(
                    graph.all_var_nodes(), op_node.output("Output")[0])
                attrs = {
                    name: op_node.op().attr(name)
                    for name in op_node.op().attr_names()
                }

                conv_op_node = graph.create_op_node(
                    op_type='conv2d',
                    attrs=attrs,
                    inputs={
                        'Input': input_var_node,
                        'Filter': weight_var_node
                    },
                    outputs={'Output': output_var_node})

                graph.link_to(input_var_node, conv_op_node)
                graph.link_to(weight_var_node, conv_op_node)
                graph.link_to(conv_op_node, output_var_node)
                graph.safe_remove_nodes(op_node)

        return graph

    def _predict(self,
                 test_reader=None,
                 model_path=None,
                 batch_size=1,
                 batch_num=1,
                 skip_batch_num=0,
                 transform_to_int8=False):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        inference_scope = fluid.executor.global_scope()
        with fluid.scope_guard(inference_scope):
            if os.path.exists(os.path.join(model_path, '__model__')):
                [inference_program, feed_target_names,
                 fetch_targets] = fluid.io.load_inference_model(model_path, exe)
            else:
                [inference_program, feed_target_names,
                 fetch_targets] = fluid.io.load_inference_model(
                     model_path, exe, 'model', 'params')

            graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
            if (self._debug):
                graph.draw('.', 'quant_orig', graph.all_op_nodes())
            if (transform_to_int8):
                mkldnn_int8_pass = QuantInt8MkldnnPass(
                    _scope=inference_scope, _place=place)
                graph = mkldnn_int8_pass.apply(graph)
            else:
                graph = self._prepare_for_fp32_mkldnn(graph)

            inference_program = graph.to_program()

            dshape = [3, 224, 224]
            outputs = []
            infer_accs1 = []
            infer_accs5 = []
            fpses = []
            batch_times = []
            total_samples = 0
            iters = 0
            infer_start_time = time.time()
            for data in test_reader():
                if batch_num > 0 and iters >= batch_num:
                    break
                if iters == skip_batch_num:
                    total_samples = 0
                    infer_start_time = time.time()
                images = list(map(lambda x: x[0].reshape(dshape), data))
                images = np.array(images).astype('float32')
                labels = np.array([x[1] for x in data]).astype('int64')

                start = time.time()
                out = exe.run(inference_program,
                              feed={feed_target_names[0]: images},
                              fetch_list=fetch_targets)
                batch_time = (time.time() - start) * 1000  # in miliseconds
                outputs.append(out[0])
                batch_acc1, batch_acc5 = self._get_batch_accuracy(out[0],
                                                                  labels)
                infer_accs1.append(batch_acc1)
                infer_accs5.append(batch_acc5)
                samples = len(data)
                total_samples += samples
                batch_times.append(batch_time)
                fps = samples / batch_time * 1000
                fpses.append(fps)
                iters += 1
                appx = ' (warm-up)' if iters <= skip_batch_num else ''
                _logger.info('batch {0}{5}, acc1: {1:.4f}, acc5: {2:.4f}, '
                             'latency: {3:.4f} ms, fps: {4:.2f}'.format(
                                 iters, batch_acc1, batch_acc5, batch_time /
                                 batch_size, fps, appx))

            # Postprocess benchmark data
            batch_latencies = batch_times[skip_batch_num:]
            batch_latency_avg = np.average(batch_latencies)
            latency_avg = batch_latency_avg / batch_size
            fpses = fpses[skip_batch_num:]
            fps_avg = np.average(fpses)
            infer_total_time = time.time() - infer_start_time
            acc1_avg = np.mean(infer_accs1)
            acc5_avg = np.mean(infer_accs5)
            _logger.info('Total inference run time: {:.2f} s'.format(
                infer_total_time))

            return outputs, acc1_avg, acc5_avg, fps_avg, latency_avg

    def _summarize_performance(self, fp32_fps, fp32_lat, int8_fps, int8_lat):
        _logger.info('--- Performance summary ---')
        _logger.info('FP32: avg fps: {0:.2f}, avg latency: {1:.4f} ms'.format(
            fp32_fps, fp32_lat))
        _logger.info('INT8: avg fps: {0:.2f}, avg latency: {1:.4f} ms'.format(
            int8_fps, int8_lat))

    def _compare_accuracy(self, fp32_acc1, fp32_acc5, int8_acc1, int8_acc5,
                          threshold):
        _logger.info('--- Accuracy summary ---')
        _logger.info(
            'Accepted top1 accuracy drop threshold: {0}. (condition: (FP32_top1_acc - IN8_top1_acc) <= threshold)'
            .format(threshold))
        _logger.info(
            'FP32: avg top1 accuracy: {0:.4f}, avg top5 accuracy: {1:.4f}'.
            format(fp32_acc1, fp32_acc5))
        _logger.info(
            'INT8: avg top1 accuracy: {0:.4f}, avg top5 accuracy: {1:.4f}'.
            format(int8_acc1, int8_acc5))
        assert fp32_acc1 > 0.0
        assert int8_acc1 > 0.0
        assert fp32_acc1 - int8_acc1 <= threshold

    def test_graph_transformation(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        quant_model_path = test_case_args.quant_model
        assert quant_model_path, 'The Quant model path cannot be empty. Please, use the --quant_model option.'
        data_path = test_case_args.infer_data
        assert data_path, 'The dataset path cannot be empty. Please, use the --infer_data option.'
        batch_size = test_case_args.batch_size
        batch_num = test_case_args.batch_num
        skip_batch_num = test_case_args.skip_batch_num
        acc_diff_threshold = test_case_args.acc_diff_threshold
        self._debug = test_case_args.debug

        _logger.info('Quant FP32 & INT8 prediction run.')
        _logger.info('Quant model: {0}'.format(quant_model_path))
        _logger.info('Dataset: {0}'.format(data_path))
        _logger.info('Batch size: {0}'.format(batch_size))
        _logger.info('Batch number: {0}'.format(batch_num))
        _logger.info('Accuracy drop threshold: {0}.'.format(acc_diff_threshold))

        _logger.info('--- Quant FP32 prediction start ---')
        val_reader = paddle.batch(
            self._reader_creator(data_path), batch_size=batch_size)
        fp32_output, fp32_acc1, fp32_acc5, fp32_fps, fp32_lat = self._predict(
            val_reader,
            quant_model_path,
            batch_size,
            batch_num,
            skip_batch_num,
            transform_to_int8=False)
        _logger.info('--- Quant INT8 prediction start ---')
        val_reader = paddle.batch(
            self._reader_creator(data_path), batch_size=batch_size)
        int8_output, int8_acc1, int8_acc5, int8_fps, int8_lat = self._predict(
            val_reader,
            quant_model_path,
            batch_size,
            batch_num,
            skip_batch_num,
            transform_to_int8=True)

        self._summarize_performance(fp32_fps, fp32_lat, int8_fps, int8_lat)
        self._compare_accuracy(fp32_acc1, fp32_acc5, int8_acc1, int8_acc5,
                               acc_diff_threshold)


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
