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

import os
import time
import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

from bert_dygraph_model import PretrainModelLayer
from bert_utils import get_bert_config, get_feed_data_reader

from predictor_utils import PredictorTools

program_translator = ProgramTranslator()
place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace(
)
SEED = 2020
STEP_NUM = 10
PRINT_STEP = 2
MODEL_SAVE_DIR = "./inference"
MODEL_SAVE_PREFIX = "./inference/bert"
MODEL_FILENAME = "bert" + INFER_MODEL_SUFFIX
PARAMS_FILENAME = "bert" + INFER_PARAMS_SUFFIX
DY_STATE_DICT_SAVE_PATH = "./bert.dygraph"


def train(bert_config, data_reader, to_static):
    with fluid.dygraph.guard(place):
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

        data_loader = fluid.io.DataLoader.from_generator(
            capacity=50, iterable=True)
        data_loader.set_batch_generator(
            data_reader.data_generator(), places=place)

        bert = PretrainModelLayer(
            config=bert_config, weight_sharing=False, use_fp16=False)

        optimizer = fluid.optimizer.Adam(parameter_list=bert.parameters())
        step_idx = 0
        speed_list = []
        for input_data in data_loader():
            src_ids, pos_ids, sent_ids, input_mask, mask_label, mask_pos, labels = input_data
            next_sent_acc, mask_lm_loss, total_loss = bert(
                src_ids=src_ids,
                position_ids=pos_ids,
                sentence_ids=sent_ids,
                input_mask=input_mask,
                mask_label=mask_label,
                mask_pos=mask_pos,
                labels=labels)
            total_loss.backward()
            optimizer.minimize(total_loss)
            bert.clear_gradients()

            acc = np.mean(np.array(next_sent_acc.numpy()))
            loss = np.mean(np.array(total_loss.numpy()))
            ppl = np.mean(np.exp(np.array(mask_lm_loss.numpy())))

            if step_idx % PRINT_STEP == 0:
                if step_idx == 0:
                    print("Step: %d, loss: %f, ppl: %f, next_sent_acc: %f" %
                          (step_idx, loss, ppl, acc))
                    avg_batch_time = time.time()
                else:
                    speed = PRINT_STEP / (time.time() - avg_batch_time)
                    speed_list.append(speed)
                    print(
                        "Step: %d, loss: %f, ppl: %f, next_sent_acc: %f, speed: %.3f steps/s"
                        % (step_idx, loss, ppl, acc, speed))
                    avg_batch_time = time.time()

            step_idx += 1
            if step_idx == STEP_NUM:
                if to_static:
                    fluid.dygraph.jit.save(bert, MODEL_SAVE_PREFIX)
                else:
                    fluid.dygraph.save_dygraph(bert.state_dict(),
                                               DY_STATE_DICT_SAVE_PATH)
                break
        return loss, ppl


def train_dygraph(bert_config, data_reader):
    program_translator.enable(False)
    return train(bert_config, data_reader, False)


def train_static(bert_config, data_reader):
    program_translator.enable(True)
    return train(bert_config, data_reader, True)


def predict_static(data):
    paddle.enable_static()
    exe = fluid.Executor(place)
    # load inference model
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(
         MODEL_SAVE_DIR,
         executor=exe,
         model_filename=MODEL_FILENAME,
         params_filename=PARAMS_FILENAME)
    pred_res = exe.run(inference_program,
                       feed=dict(zip(feed_target_names, data)),
                       fetch_list=fetch_targets)

    return pred_res


def predict_dygraph(bert_config, data):
    program_translator.enable(False)
    with fluid.dygraph.guard(place):
        bert = PretrainModelLayer(
            config=bert_config, weight_sharing=False, use_fp16=False)
        model_dict, _ = fluid.dygraph.load_dygraph(DY_STATE_DICT_SAVE_PATH)

        bert.set_dict(model_dict)
        bert.eval()

        input_vars = [fluid.dygraph.to_variable(x) for x in data]
        src_ids, pos_ids, sent_ids, input_mask, mask_label, mask_pos, labels = input_vars
        pred_res = bert(
            src_ids=src_ids,
            position_ids=pos_ids,
            sentence_ids=sent_ids,
            input_mask=input_mask,
            mask_label=mask_label,
            mask_pos=mask_pos,
            labels=labels)
        pred_res = [var.numpy() for var in pred_res]

        return pred_res


def predict_dygraph_jit(data):
    with fluid.dygraph.guard(place):
        bert = fluid.dygraph.jit.load(MODEL_SAVE_PREFIX)
        bert.eval()

        src_ids, pos_ids, sent_ids, input_mask, mask_label, mask_pos, labels = data
        pred_res = bert(src_ids, pos_ids, sent_ids, input_mask, mask_label,
                        mask_pos, labels)
        pred_res = [var.numpy() for var in pred_res]

        return pred_res


def predict_analysis_inference(data):
    output = PredictorTools(MODEL_SAVE_DIR, MODEL_FILENAME, PARAMS_FILENAME,
                            data)
    out = output()
    return out


class TestBert(unittest.TestCase):
    def setUp(self):
        self.bert_config = get_bert_config()
        self.data_reader = get_feed_data_reader(self.bert_config)

    def test_train(self):
        static_loss, static_ppl = train_static(self.bert_config,
                                               self.data_reader)
        dygraph_loss, dygraph_ppl = train_dygraph(self.bert_config,
                                                  self.data_reader)
        self.assertTrue(
            np.allclose(static_loss, dygraph_loss),
            msg="static_loss: {} \n dygraph_loss: {}".format(static_loss,
                                                             dygraph_loss))
        self.assertTrue(
            np.allclose(static_ppl, dygraph_ppl),
            msg="static_ppl: {} \n dygraph_ppl: {}".format(static_ppl,
                                                           dygraph_ppl))

        self.verify_predict()

    def verify_predict(self):
        for data in self.data_reader.data_generator()():
            dygraph_pred_res = predict_dygraph(self.bert_config, data)
            static_pred_res = predict_static(data)
            dygraph_jit_pred_res = predict_dygraph_jit(data)
            predictor_pred_res = predict_analysis_inference(data)

            for dy_res, st_res, dy_jit_res, predictor_res in zip(
                    dygraph_pred_res, static_pred_res, dygraph_jit_pred_res,
                    predictor_pred_res):
                self.assertTrue(
                    np.allclose(st_res, dy_res),
                    "dygraph_res: {},\n static_res: {}".format(
                        dy_res[~np.isclose(st_res, dy_res)],
                        st_res[~np.isclose(st_res, dy_res)]))
                self.assertTrue(
                    np.allclose(st_res, dy_jit_res),
                    "dygraph_jit_res: {},\n static_res: {}".format(
                        dy_jit_res[~np.isclose(st_res, dy_jit_res)],
                        st_res[~np.isclose(st_res, dy_jit_res)]))
                self.assertTrue(
                    np.allclose(st_res, predictor_res),
                    "dygraph_jit_res: {},\n static_res: {}".format(
                        predictor_res[~np.isclose(st_res, predictor_res)],
                        st_res[~np.isclose(st_res, predictor_res)]))
            break


if __name__ == '__main__':
    unittest.main()
