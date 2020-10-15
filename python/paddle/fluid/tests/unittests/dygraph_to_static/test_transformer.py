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

import logging
import os
import time
import unittest

import numpy as np
import paddle
import paddle.fluid as fluid

import transformer_util as util
from transformer_dygraph_model import CrossEntropyCriterion, Transformer, position_encoding_init

trainer_count = 1
place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace(
)
SEED = 10
STEP_NUM = 10


def train_static(args, batch_generator):
    paddle.enable_static()
    paddle.manual_seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # define input and reader
            input_field_names = util.encoder_data_input_fields + \
                                util.decoder_data_input_fields[:-1] + util.label_data_input_fields
            input_descs = util.get_input_descs(args)
            input_slots = [{
                "name": name,
                "shape": input_descs[name][0],
                "dtype": input_descs[name][1]
            } for name in input_field_names]
            input_field = util.InputField(input_slots)
            # Define DataLoader
            data_loader = fluid.io.DataLoader.from_generator(
                input_field.feed_list, capacity=60)
            data_loader.set_batch_generator(batch_generator, places=place)
            # define model
            transformer = Transformer(
                args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
                args.n_layer, args.n_head, args.d_key, args.d_value,
                args.d_model, args.d_inner_hid, args.prepostprocess_dropout,
                args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
                args.postprocess_cmd, args.weight_sharing, args.bos_idx,
                args.eos_idx)
            logits = transformer(*input_field.feed_list[:7])
            # define loss
            criterion = CrossEntropyCriterion(args.label_smooth_eps)
            lbl_word, lbl_weight = input_field.feed_list[7:]
            sum_cost, avg_cost, token_num = criterion(logits, lbl_word,
                                                      lbl_weight)
            # define optimizer
            learning_rate = fluid.layers.learning_rate_scheduler.noam_decay(
                args.d_model, args.warmup_steps, args.learning_rate)
            optimizer = fluid.optimizer.Adam(
                learning_rate=learning_rate,
                beta1=args.beta1,
                beta2=args.beta2,
                epsilon=float(args.eps))
            optimizer.minimize(avg_cost)
            # the best cross-entropy value with label smoothing
            loss_normalizer = -((1. - args.label_smooth_eps) * np.log(
                (1. - args.label_smooth_eps)) + args.label_smooth_eps * np.log(
                    args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))
    step_idx = 0
    total_batch_num = 0
    avg_loss = []
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    for pass_id in range(args.epoch):
        batch_id = 0
        for feed_dict in data_loader:
            outs = exe.run(program=train_prog,
                           feed=feed_dict,
                           fetch_list=[sum_cost.name, token_num.name])
            if step_idx % args.print_step == 0:
                sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[
                    1])
                total_sum_cost = sum_cost_val.sum()
                total_token_num = token_num_val.sum()
                total_avg_cost = total_sum_cost / total_token_num
                avg_loss.append(total_avg_cost)
                if step_idx == 0:
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f" %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)])))
                    avg_batch_time = time.time()
                else:
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f, speed: %.2f steps/s" %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)]),
                         args.print_step / (time.time() - avg_batch_time)))
                    avg_batch_time = time.time()
            batch_id += 1
            step_idx += 1
            total_batch_num = total_batch_num + 1
            if step_idx == STEP_NUM:
                if args.save_dygraph_model_path:
                    model_path = os.path.join(args.save_static_model_path,
                                              "transformer")
                    fluid.save(train_prog, model_path)
                break
    return np.array(avg_loss)


def train_dygraph(args, batch_generator):
    with fluid.dygraph.guard(place):
        if SEED is not None:
            paddle.manual_seed(SEED)
            paddle.framework.random._manual_program_seed(SEED)
        # define data loader
        train_loader = fluid.io.DataLoader.from_generator(capacity=10)
        train_loader.set_batch_generator(batch_generator, places=place)
        # define model
        transformer = Transformer(
            args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
            args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
            args.d_inner_hid, args.prepostprocess_dropout,
            args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
            args.postprocess_cmd, args.weight_sharing, args.bos_idx,
            args.eos_idx)
        # define loss
        criterion = CrossEntropyCriterion(args.label_smooth_eps)
        # define optimizer
        learning_rate = fluid.layers.learning_rate_scheduler.noam_decay(
            args.d_model, args.warmup_steps, args.learning_rate)
        # define optimizer
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=float(args.eps),
            parameter_list=transformer.parameters())
        # the best cross-entropy value with label smoothing
        loss_normalizer = -(
            (1. - args.label_smooth_eps) * np.log(
                (1. - args.label_smooth_eps)) + args.label_smooth_eps *
            np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))
        ce_time = []
        ce_ppl = []
        avg_loss = []
        step_idx = 0
        for pass_id in range(args.epoch):
            pass_start_time = time.time()
            batch_id = 0
            for input_data in train_loader():
                (src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                 trg_slf_attn_bias, trg_src_attn_bias, lbl_word,
                 lbl_weight) = input_data
                logits = transformer(src_word, src_pos, src_slf_attn_bias,
                                     trg_word, trg_pos, trg_slf_attn_bias,
                                     trg_src_attn_bias)
                sum_cost, avg_cost, token_num = criterion(logits, lbl_word,
                                                          lbl_weight)
                avg_cost.backward()
                optimizer.minimize(avg_cost)
                transformer.clear_gradients()
                if step_idx % args.print_step == 0:
                    total_avg_cost = avg_cost.numpy() * trainer_count
                    avg_loss.append(total_avg_cost[0])
                    if step_idx == 0:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                        avg_batch_time = time.time()
                    else:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f steps/s"
                            %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             args.print_step / (time.time() - avg_batch_time)))
                        ce_ppl.append(np.exp([min(total_avg_cost, 100)]))
                        avg_batch_time = time.time()
                batch_id += 1
                step_idx += 1
                if step_idx == STEP_NUM:
                    if args.save_dygraph_model_path:
                        model_dir = os.path.join(args.save_dygraph_model_path)
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        fluid.save_dygraph(
                            transformer.state_dict(),
                            os.path.join(model_dir, "transformer"))
                        fluid.save_dygraph(
                            optimizer.state_dict(),
                            os.path.join(model_dir, "transformer"))
                    break
            time_consumed = time.time() - pass_start_time
            ce_time.append(time_consumed)
        return np.array(avg_loss)


def predict_dygraph(args, batch_generator):
    with fluid.dygraph.guard(place):
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        # define data loader
        test_loader = fluid.io.DataLoader.from_generator(capacity=10)
        test_loader.set_batch_generator(batch_generator, places=place)

        # define model
        transformer = Transformer(
            args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
            args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
            args.d_inner_hid, args.prepostprocess_dropout,
            args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
            args.postprocess_cmd, args.weight_sharing, args.bos_idx,
            args.eos_idx)

        # load the trained model
        model_dict, _ = util.load_dygraph(
            os.path.join(args.save_dygraph_model_path, "transformer"))
        # to avoid a longer length than training, reset the size of position
        # encoding to max_length
        model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
            args.max_length + 1, args.d_model)
        model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
            args.max_length + 1, args.d_model)
        transformer.load_dict(model_dict)

        # set evaluate mode
        transformer.eval()

        step_idx = 0
        speed_list = []
        for input_data in test_loader():
            (src_word, src_pos, src_slf_attn_bias, trg_word,
             trg_src_attn_bias) = input_data
            seq_ids, seq_scores = transformer.beam_search(
                src_word,
                src_pos,
                src_slf_attn_bias,
                trg_word,
                trg_src_attn_bias,
                bos_id=args.bos_idx,
                eos_id=args.eos_idx,
                beam_size=args.beam_size,
                max_len=args.max_out_len)
            seq_ids = seq_ids.numpy()
            seq_scores = seq_scores.numpy()
            if step_idx % args.print_step == 0:
                if step_idx == 0:
                    logging.info(
                        "Dygraph Predict: step_idx: %d, 1st seq_id: %d, 1st seq_score: %.2f"
                        % (step_idx, seq_ids[0][0][0], seq_scores[0][0]))
                    avg_batch_time = time.time()
                else:
                    speed = args.print_step / (time.time() - avg_batch_time)
                    speed_list.append(speed)
                    logging.info(
                        "Dygraph Predict: step_idx: %d, 1st seq_id: %d, 1st seq_score: %.2f, speed: %.3f steps/s"
                        % (step_idx, seq_ids[0][0][0], seq_scores[0][0], speed))
                    avg_batch_time = time.time()

            step_idx += 1
            if step_idx == STEP_NUM:
                break
        logging.info("Dygraph Predict:  avg_speed: %.4f steps/s" %
                     (np.mean(speed_list)))
        return seq_ids, seq_scores


def predict_static(args, batch_generator):
    test_prog = fluid.Program()
    with fluid.program_guard(test_prog):
        paddle.manual_seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        # define input and reader
        input_field_names = util.encoder_data_input_fields + util.fast_decoder_data_input_fields
        input_descs = util.get_input_descs(args, 'test')
        input_slots = [{
            "name": name,
            "shape": input_descs[name][0],
            "dtype": input_descs[name][1]
        } for name in input_field_names]

        input_field = util.InputField(input_slots)
        feed_list = input_field.feed_list
        loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_list, capacity=10)

        # define model
        transformer = Transformer(
            args.src_vocab_size, args.trg_vocab_size, args.max_length + 1,
            args.n_layer, args.n_head, args.d_key, args.d_value, args.d_model,
            args.d_inner_hid, args.prepostprocess_dropout,
            args.attention_dropout, args.relu_dropout, args.preprocess_cmd,
            args.postprocess_cmd, args.weight_sharing, args.bos_idx,
            args.eos_idx)

        out_ids, out_scores = transformer.beam_search(
            *feed_list,
            bos_id=args.bos_idx,
            eos_id=args.eos_idx,
            beam_size=args.beam_size,
            max_len=args.max_out_len)

    # This is used here to set dropout to the test mode.
    test_prog = test_prog.clone(for_test=True)

    # define the executor and program for training
    exe = fluid.Executor(place)

    util.load(test_prog,
              os.path.join(args.save_static_model_path, "transformer"), exe)

    loader.set_batch_generator(batch_generator, places=place)

    step_idx = 0
    speed_list = []
    for feed_dict in loader:
        seq_ids, seq_scores = exe.run(
            test_prog,
            feed=feed_dict,
            fetch_list=[out_ids.name, out_scores.name],
            return_numpy=True)
        if step_idx % args.print_step == 0:
            if step_idx == 0:
                logging.info(
                    "Static Predict: step_idx: %d, 1st seq_id: %d, 1st seq_score: %.2f,"
                    % (step_idx, seq_ids[0][0][0], seq_scores[0][0]))
                avg_batch_time = time.time()
            else:
                speed = args.print_step / (time.time() - avg_batch_time)
                speed_list.append(speed)
                logging.info(
                    "Static Predict: step_idx: %d, 1st seq_id: %d, 1st seq_score: %.2f, speed: %.3f steps/s"
                    % (step_idx, seq_ids[0][0][0], seq_scores[0][0], speed))
                avg_batch_time = time.time()

        step_idx += 1
        if step_idx == STEP_NUM:
            break
    logging.info("Static Predict:  avg_speed: %.4f steps/s" %
                 (np.mean(speed_list)))

    return seq_ids, seq_scores


class TestTransformer(unittest.TestCase):
    def prepare(self, mode='train'):
        args = util.ModelHyperParams()
        batch_generator = util.get_feed_data_reader(args, mode)
        return args, batch_generator

    def _test_train(self):
        args, batch_generator = self.prepare(mode='train')
        static_avg_loss = train_static(args, batch_generator)
        dygraph_avg_loss = train_dygraph(args, batch_generator)
        self.assertTrue(np.allclose(static_avg_loss, dygraph_avg_loss))

    def _test_predict(self):
        args, batch_generator = self.prepare(mode='test')
        static_seq_ids, static_scores = predict_static(args, batch_generator)
        dygraph_seq_ids, dygraph_scores = predict_dygraph(args, batch_generator)

        self.assertTrue(
            np.allclose(static_seq_ids, static_seq_ids),
            msg="static_seq_ids: {} \n dygraph_seq_ids: {}".format(
                static_seq_ids, dygraph_seq_ids))
        self.assertTrue(
            np.allclose(static_scores, dygraph_scores),
            msg="static_scores: {} \n dygraph_scores: {}".format(
                static_scores, dygraph_scores))

    def test_check_result(self):
        self._test_train()
        self._test_predict()


if __name__ == '__main__':
    unittest.main()
