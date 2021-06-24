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
from paddle.fluid.contrib.slim.quantization import Quant2Int8MkldnnPass
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
        '--quant_model', type=str, default='', help='A path to a Quant model.')
    parser.add_argument(
        '--fp32_model', type=str, default='', help='A path to an FP32 model.')
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
    parser.add_argument(
        '--ops_to_quantize',
        type=str,
        default='',
        help='A comma separated list of operators to quantize. Only quantizable operators are taken into account. If the option is not used, an attempt to quantize all quantizable operators will be made.'
    )
    parser.add_argument(
        '--op_ids_to_skip',
        type=str,
        default='',
        help='A comma separated list of operator ids to skip in quantization.')
    parser.add_argument(
        '--targets',
        type=str,
        default='quant,int8,fp32',
        help='A comma separated list of inference types to run ("int8", "fp32", "quant"). Default: "quant,int8,fp32"'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='If used, the graph of Quant model is drawn.')

    test_args, args = parser.parse_known_args(namespace=unittest)
    return test_args, sys.argv[:1] + args


class Quant2Int8ImageClassificationComparisonTest(unittest.TestCase):
    """
    Test for accuracy comparison of FP32 and Quant2 INT8 Image Classification inference.
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
                 target='quant'):
        assert target in ['quant', 'int8', 'fp32']
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
            quant_transform_pass = Quant2Int8MkldnnPass(
                self._quantized_ops,
                _op_ids_to_skip=self._op_ids_to_skip,
                _scope=inference_scope,
                _place=place,
                _core=core,
                _debug=self._debug)
            if (target == 'quant'):
                graph = self._prepare_for_fp32_mkldnn(graph)
            elif (target == 'int8'):
                graph = quant_transform_pass.apply(graph)
            else:  # target == fp32
                graph = quant_transform_pass.prepare_and_optimize_fp32(graph)

            inference_program = graph.to_program()

            dshape = [3, 224, 224]
            outputs = []
            infer_accs1 = []
            infer_accs5 = []
            batch_acc1 = 0.0
            batch_acc5 = 0.0
            fpses = []
            batch_times = []
            batch_time = 0.0
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

                if (target == 'fp32'):
                    # FP32 models have accuracy measuring layers
                    labels = labels.reshape([-1, 1])
                    start = time.time()
                    out = exe.run(inference_program,
                                  feed={
                                      feed_target_names[0]: images,
                                      feed_target_names[1]: labels
                                  },
                                  fetch_list=fetch_targets)
                    batch_time = (time.time() - start) * 1000  # in miliseconds
                    batch_acc1, batch_acc5 = out[1][0], out[2][0]
                    outputs.append(batch_acc1)
                else:
                    # Quant INT8 models do not have accuracy measuring layers
                    start = time.time()
                    out = exe.run(inference_program,
                                  feed={feed_target_names[0]: images},
                                  fetch_list=fetch_targets)
                    batch_time = (time.time() - start) * 1000  # in miliseconds
                    outputs.append(out[0])
                    # Calculate accuracy result
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

    def _print_performance(self, title, fps, lat):
        _logger.info('{0}: avg fps: {1:.2f}, avg latency: {2:.4f} ms'.format(
            title, fps, lat))

    def _print_accuracy(self, title, acc1, acc5):
        _logger.info(
            '{0}: avg top1 accuracy: {1:.4f}, avg top5 accuracy: {2:.4f}'.
            format(title, acc1, acc5))

    def _summarize_performance(self, int8_fps, int8_lat, fp32_fps, fp32_lat):
        _logger.info('--- Performance summary ---')
        self._print_performance('INT8', int8_fps, int8_lat)
        if fp32_lat >= 0:
            self._print_performance('FP32', fp32_fps, fp32_lat)

    def _summarize_accuracy(self, quant_acc1, quant_acc5, int8_acc1, int8_acc5,
                            fp32_acc1, fp32_acc5):
        _logger.info('--- Accuracy summary ---')
        self._print_accuracy('Quant', quant_acc1, quant_acc5)
        self._print_accuracy('INT8', int8_acc1, int8_acc5)
        if fp32_acc1 >= 0:
            self._print_accuracy('FP32', fp32_acc1, fp32_acc5)

    def _compare_accuracy(self, threshold, quant_acc1, int8_acc1):
        _logger.info(
            'Accepted top1 accuracy drop threshold: {0}. (condition: (Quant_top1_acc - IN8_top1_acc) <= threshold && Quant_top1_acc > 0.5 && INT8_top1_acc > 0.5)'
            .format(threshold))
        # We assume valid accuracy to be at least 0.5
        assert quant_acc1 > 0.5
        assert int8_acc1 > 0.5
        assert quant_acc1 - int8_acc1 <= threshold

    def _strings_from_csv(self, string):
        return set(s.strip() for s in string.split(','))

    def _ints_from_csv(self, string):
        return set(map(int, string.split(',')))

    def test_graph_transformation(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        quant_model_path = test_case_args.quant_model
        assert quant_model_path, 'The Quant model path cannot be empty. Please, use the --quant_model option.'
        data_path = test_case_args.infer_data
        assert data_path, 'The dataset path cannot be empty. Please, use the --infer_data option.'
        fp32_model_path = test_case_args.fp32_model
        batch_size = test_case_args.batch_size
        batch_num = test_case_args.batch_num
        skip_batch_num = test_case_args.skip_batch_num
        acc_diff_threshold = test_case_args.acc_diff_threshold
        self._debug = test_case_args.debug

        self._quantized_ops = set()
        if test_case_args.ops_to_quantize:
            self._quantized_ops = self._strings_from_csv(
                test_case_args.ops_to_quantize)

        self._op_ids_to_skip = set([-1])
        if test_case_args.op_ids_to_skip:
            self._op_ids_to_skip = self._ints_from_csv(
                test_case_args.op_ids_to_skip)

        self._targets = self._strings_from_csv(test_case_args.targets)
        assert self._targets.intersection(
            {'quant', 'int8', 'fp32'}
        ), 'The --targets option, if used, must contain at least one of the targets: "quant", "int8", "fp32".'

        _logger.info('Quant & INT8 prediction run.')
        _logger.info('Quant model: {}'.format(quant_model_path))
        if fp32_model_path:
            _logger.info('FP32 model: {}'.format(fp32_model_path))
        _logger.info('Dataset: {}'.format(data_path))
        _logger.info('Batch size: {}'.format(batch_size))
        _logger.info('Batch number: {}'.format(batch_num))
        _logger.info('Accuracy drop threshold: {}.'.format(acc_diff_threshold))
        _logger.info('Quantized ops: {}.'.format(','.join(
            self._quantized_ops) if self._quantized_ops else 'all quantizable'))
        _logger.info('Op ids to skip quantization: {}.'.format(','.join(
            map(str, self._op_ids_to_skip)) if test_case_args.op_ids_to_skip
                                                               else 'none'))
        _logger.info('Targets: {}.'.format(','.join(self._targets)))

        if 'quant' in self._targets:
            _logger.info('--- Quant prediction start ---')
            val_reader = paddle.batch(
                self._reader_creator(data_path), batch_size=batch_size)
            quant_output, quant_acc1, quant_acc5, quant_fps, quant_lat = self._predict(
                val_reader,
                quant_model_path,
                batch_size,
                batch_num,
                skip_batch_num,
                target='quant')
            self._print_performance('Quant', quant_fps, quant_lat)
            self._print_accuracy('Quant', quant_acc1, quant_acc5)

        if 'int8' in self._targets:
            _logger.info('--- INT8 prediction start ---')
            val_reader = paddle.batch(
                self._reader_creator(data_path), batch_size=batch_size)
            int8_output, int8_acc1, int8_acc5, int8_fps, int8_lat = self._predict(
                val_reader,
                quant_model_path,
                batch_size,
                batch_num,
                skip_batch_num,
                target='int8')
            self._print_performance('INT8', int8_fps, int8_lat)
            self._print_accuracy('INT8', int8_acc1, int8_acc5)

        fp32_acc1 = fp32_acc5 = fp32_fps = fp32_lat = -1
        if 'fp32' in self._targets and fp32_model_path:
            _logger.info('--- FP32 prediction start ---')
            val_reader = paddle.batch(
                self._reader_creator(data_path), batch_size=batch_size)
            fp32_output, fp32_acc1, fp32_acc5, fp32_fps, fp32_lat = self._predict(
                val_reader,
                fp32_model_path,
                batch_size,
                batch_num,
                skip_batch_num,
                target='fp32')
            self._print_performance('FP32', fp32_fps, fp32_lat)
            self._print_accuracy('FP32', fp32_acc1, fp32_acc5)

        if {'int8', 'fp32'}.issubset(self._targets):
            self._summarize_performance(int8_fps, int8_lat, fp32_fps, fp32_lat)
        if {'int8', 'quant'}.issubset(self._targets):
            self._summarize_accuracy(quant_acc1, quant_acc5, int8_acc1,
                                     int8_acc5, fp32_acc1, fp32_acc5)
            self._compare_accuracy(acc_diff_threshold, quant_acc1, int8_acc1)


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
