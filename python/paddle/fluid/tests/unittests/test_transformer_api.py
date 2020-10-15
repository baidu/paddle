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

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.nn.layer.transformer import MultiHeadAttention, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder, Transformer

import unittest


def generate_basic_params(mode="attn", self_attention=True):
    batch_size, query_length = [np.random.randint(2, 10) for _ in range(2)]
    d_head, num_heads = [np.random.randint(3, 10) for _ in range(2)]
    attn_dropout = 0.0
    embed_dim = d_head * num_heads
    if mode == "attn":
        if self_attention:
            kdim, vdim = embed_dim, embed_dim
            key_length, value_length = query_length, query_length
        else:
            kdim, vdim = [np.random.randint(5, 20) for _ in range(2)]
            key_length = np.random.randint(2, 10)
            value_length = key_length
        return batch_size, query_length, key_length, value_length, embed_dim, kdim, vdim, num_heads, attn_dropout

    else:
        dropout, act_dropout = 0.0, 0.0
        dim_feedforward = np.random.randint(128, 1024)
        sequence_length = np.random.randint(2, 10)
        if mode == "encoder_layer":
            return batch_size, embed_dim, num_heads, dim_feedforward, dropout, attn_dropout, act_dropout, sequence_length
        elif mode == "decoder_layer":
            target_length = np.random.randint(2, 10)
            return batch_size, embed_dim, num_heads, dim_feedforward, dropout, attn_dropout, act_dropout, sequence_length, target_length


def generate_query_key_value_cache(self_attention,
                                   batch_size,
                                   num_heads,
                                   query_length,
                                   embed_dim,
                                   key_length=None,
                                   value_length=None,
                                   kdim=None,
                                   vdim=None,
                                   cache=None):
    query = np.random.rand(batch_size, query_length,
                           embed_dim).astype("float32")
    attn_mask = np.zeros((batch_size, num_heads, query_length, key_length))
    attn_mask[0][0][0][0] = -1e9

    head_dim = embed_dim // num_heads
    if self_attention:
        key, value = query, query
    else:
        key = np.random.rand(batch_size, key_length, kdim).astype("float32")
        value = np.random.rand(batch_size, value_length, vdim).astype("float32")
    cache_dict = {}
    if cache:
        if not self_attention:
            cache_dict["static_k"] = np.random.rand(
                batch_size, num_heads, key_length, head_dim).astype("float32")
            cache_dict["static_v"] = np.random.rand(
                batch_size, num_heads, value_length, head_dim).astype("float32")
        else:
            cache_dict["k"] = np.random.rand(batch_size, num_heads, key_length,
                                             head_dim).astype("float32")
            cache_dict["v"] = np.random.rand(
                batch_size, num_heads, value_length, head_dim).astype("float32")
    else:
        cache_dict = None
    return query, key, value, attn_mask, cache_dict


def fc(x, weight):
    return np.matmul(x, weight)


def softmax(x):
    np.seterr(invalid='ignore')
    output = np.zeros(x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_curr = x[i, j, k, :]
                e_x = np.exp(x_curr - np.amax(x_curr))
                output[i, j, k, :] = e_x / np.sum(e_x)
    return output


def batch_matmul(x, y):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    retval = np.zeros(
        (x.shape[0], x.shape[1], x.shape[2], y.shape[3]), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            retval[i, j, :, :] = np.matmul(x[i, j, :, :], y[i, j, :, :])
    return retval


def scaled_dot_product_attention(q, k, v, d_key, attn_mask, multi_head_attn):
    k = k.transpose([0, 1, 3, 2])
    qkt = batch_matmul(q, k / np.sqrt(d_key, dtype=np.float64))
    if attn_mask is not None:
        qkt += attn_mask
    weight = softmax(qkt)
    attn_heads = batch_matmul(weight, v)
    attn_heads = attn_heads.transpose((0, 2, 1, 3))
    attn_heads = attn_heads.reshape((attn_heads.shape[0], attn_heads.shape[1],
                                     attn_heads.shape[2] * attn_heads.shape[3]))
    return attn_heads


def cal_qkv(key, value, num_heads, embed_dim, multi_head_attn):
    with fluid.dygraph.guard():
        head_dim = embed_dim // num_heads
        k_weight = multi_head_attn.k_proj.weight.numpy()
        v_weight = multi_head_attn.v_proj.weight.numpy()
        k = fc(key, k_weight)
        v = fc(value, v_weight)
        k = k.reshape((k.shape[0], k.shape[1], num_heads, head_dim))
        k = k.transpose((0, 2, 1, 3))
        v = v.reshape((v.shape[0], v.shape[1], num_heads, head_dim))
        v = v.transpose((0, 2, 1, 3))
        return k, v


def prepare_qkv(query, key, value, num_heads, embed_dim, self_attention,
                multi_head_attn, cache_dict):
    q_weight = multi_head_attn.q_proj.weight.numpy()
    q = fc(query, q_weight)
    q = q.reshape((q.shape[0], q.shape[1], num_heads, embed_dim // num_heads))
    q = q.transpose((0, 2, 1, 3))

    if not self_attention and cache_dict:
        k, v = cache_dict["static_k"], cache_dict["static_v"]
    else:
        k, v = cal_qkv(key, value, num_heads, embed_dim, multi_head_attn)
        if cache_dict is not None:
            k = np.concatenate((cache_dict["k"], k), axis=2)
            v = np.concatenate((cache_dict["v"], v), axis=2)
    return (q, k, v, cache_dict)


def add(x, y=None):
    fluid.enable_dygraph()
    with fluid.dygraph.guard():
        x = x.numpy() if not isinstance(x, np.ndarray) else x
        if y is not None:
            x += y
            return x
        return x


def relu(x):
    compare = x > 0
    return x * compare


def layer_norm(x, normalized_shape, norm, epsilon=1e-05, act=None):
    fluid.enable_dygraph()
    with fluid.dygraph.guard():
        # scale:
        weight = norm.weight.numpy()
        # shift:
        bias = norm.bias.numpy()

        batch_size, src_len, d_model = x.shape
        x = x.reshape((batch_size * src_len, d_model))
        mu = np.mean(x, axis=1, keepdims=True)
        sigma_squar = np.sum(np.square(x - mu), axis=1) / d_model
        x1_up = (x - mu)
        x1_down_1 = sigma_squar + epsilon
        x1_down = np.sqrt(x1_down_1)
        x1_down = x1_down.reshape((x1_down.shape[0], 1))
        x1 = x1_up / x1_down
        x_scaled = weight * x1
        x_scaled_bias = x_scaled + bias
        x_scaled_bias = x_scaled_bias.reshape((batch_size, src_len, d_model))
    return x_scaled_bias


def ffn(src, encoder_layer, ffn_fc1_act="relu"):
    assert ffn_fc1_act == "relu", "only relu is supported"
    fluid.enable_dygraph()
    with fluid.dygraph.guard():
        src = src.numpy() if not isinstance(src, np.ndarray) else src
        w1 = encoder_layer.linear1.weight.numpy()
        w2 = encoder_layer.linear2.weight.numpy()
        # fc1
        x1 = fc(src, w1)
        x1 = relu(x1)
        # fc2
        x2 = fc(x1, w2)
        return x2


class TestTransformer(unittest.TestCase):
    def test_multi_head_attention(self):
        def multihead_attention_test_helper(self_attention, cache):
            paddle.manual_seed(2020)
            paddle.framework.random._manual_program_seed(2020)
            # self_attention|cross_attention, cache|No cache
            with fluid.dygraph.guard(fluid.CPUPlace()):

                # generate params for multi_head_attention
                batch_size, query_length, key_length, value_length, embed_dim, kdim, vdim, num_heads, attn_dropout = generate_basic_params(
                    "attn", self_attention)
                query, key, value, attn_mask, cache_dict = generate_query_key_value_cache(
                    self_attention, batch_size, num_heads, query_length,
                    embed_dim, key_length, value_length, kdim, vdim, cache)
                if cache and self_attention:
                    attn_mask = np.concatenate((attn_mask, attn_mask), axis=3)
                need_weight, param_attr, bias_attr = False, None, None
                # call paddle's function
                multi_head_attn = MultiHeadAttention(
                    embed_dim, num_heads, attn_dropout, kdim, vdim, need_weight,
                    param_attr, bias_attr)
                # construct cache object
                cache_obj = None
                if cache_dict:
                    if 'k' and 'v' in cache_dict:
                        cache_obj = multi_head_attn.Cache(
                            paddle.to_tensor(cache_dict['k']),
                            paddle.to_tensor(cache_dict['v']))
                    elif 'static_k' and 'static_v' in cache_dict:
                        cache_obj = multi_head_attn.StaticCache(
                            paddle.to_tensor(cache_dict['static_k']),
                            paddle.to_tensor(cache_dict['static_v']))
                if attn_mask is not None:
                    attn_output = multi_head_attn(
                        paddle.to_tensor(query),
                        paddle.to_tensor(key),
                        paddle.to_tensor(value),
                        paddle.to_tensor(attn_mask), cache_obj)
                else:
                    attn_output = multi_head_attn(
                        paddle.to_tensor(query),
                        paddle.to_tensor(key),
                        paddle.to_tensor(value), attn_mask, cache_obj)
                attn_output = attn_output[0] if cache_dict else attn_output

                # implementation by numpy
                # compute q, k, v
                q, k, v, _ = prepare_qkv(query, key, value, num_heads,
                                         embed_dim, self_attention,
                                         multi_head_attn, cache_dict)
                # scale dot product attention
                attn_heads = scaled_dot_product_attention(
                    q, k, v, embed_dim // num_heads, attn_mask, multi_head_attn)
                out_proj_weight = multi_head_attn.out_proj.weight.numpy()
                reference = fc(attn_heads, out_proj_weight)

                np.testing.assert_allclose(
                    attn_output.numpy(), reference, atol=1e-6)

        multihead_attention_test_helper(True, True)
        multihead_attention_test_helper(True, False)
        multihead_attention_test_helper(False, True)
        multihead_attention_test_helper(False, False)

    def test_transformer_encoder_layer(self):

        with fluid.dygraph.guard(fluid.CPUPlace()):
            paddle.framework.manual_seed(2020)
            paddle.framework.random._manual_program_seed(2020)

            ffn_fc1_act = "relu"
            # 1.generate basic params
            batch_size, d_model, n_head, dim_feedforward, dropout, attn_dropout, act_dropout, sequence_length = generate_basic_params(
                mode="encoder_layer")
            # 2.generate input for encoder
            src = np.random.rand(batch_size, sequence_length,
                                 d_model).astype("float32")
            residual = src
            src_mask = np.zeros((batch_size, n_head, sequence_length,
                                 sequence_length)).astype("float32")
            src_mask[0][0][0][0] = -np.inf

            # paddle
            encoder_layer = TransformerEncoderLayer(
                d_model, n_head, dim_feedforward, dropout, ffn_fc1_act,
                attn_dropout, act_dropout)

            encoder_output = encoder_layer(
                paddle.to_tensor(src),
                paddle.to_tensor(src_mask))  # paddle.to_tensor(src_mask))
            # 4.numpy:
            # paddle self attention
            self_attn = MultiHeadAttention(
                d_model, n_head, dropout=attn_dropout)
            attn_output = self_attn(
                paddle.to_tensor(src),
                paddle.to_tensor(src),
                paddle.to_tensor(src), paddle.to_tensor(src_mask)).numpy()

            src = attn_output + residual
            src_norm = layer_norm(src, d_model, encoder_layer.norm1)
            residual = src_norm

            ffn_output = ffn(src_norm, encoder_layer, ffn_fc1_act)
            src = residual + ffn_output
            src = layer_norm(src, d_model, encoder_layer.norm2)

            np.testing.assert_allclose(
                encoder_output.numpy(), src, rtol=1e-5, atol=1e-6)

    def test_transformer_decoder_layer(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            paddle.framework.manual_seed(2020)
            activation = "relu"
            normalize_before = False
            batch_size, d_model, n_head, dim_feedforward, dropout, attn_dropout, act_dropout, source_length, target_length = generate_basic_params(
                mode="decoder_layer")
            tgt = np.random.rand(batch_size, target_length,
                                 d_model).astype("float32")
            memory = np.random.rand(batch_size, source_length,
                                    d_model).astype("float32")
            tgt_mask = np.zeros((batch_size, n_head, target_length,
                                 target_length)).astype("float32")
            tgt_mask[0][0][0][0] = -1e9
            memory_mask = np.zeros((batch_size, n_head, target_length,
                                    source_length)).astype("float32")
            memory_mask[0][0][0][0] = -1e9
            for cache in [True, False]:
                self_attn = MultiHeadAttention(
                    d_model, n_head, dropout=attn_dropout)
                cross_attn = MultiHeadAttention(
                    d_model, n_head, dropout=attn_dropout)

                # paddle decoderlayer:
                decoder_layer = TransformerDecoderLayer(
                    d_model, n_head, dim_feedforward, dropout, activation,
                    attn_dropout, act_dropout, normalize_before)
                cache_objs = None
                if cache:
                    cache_objs = decoder_layer.gen_cache(
                        paddle.to_tensor(memory))

                decoder_output = decoder_layer(
                    paddle.to_tensor(tgt),
                    paddle.to_tensor(memory),
                    paddle.to_tensor(tgt_mask),
                    paddle.to_tensor(memory_mask), cache_objs)

                decoder_output = decoder_output[0].numpy(
                ) if cache else decoder_output.numpy()

                # numpy:
                residual = tgt
                # self-attn
                self_attn_cache = cache_objs[
                    0] if cache_objs is not None else None
                tgt = self_attn(
                    paddle.to_tensor(tgt),
                    paddle.to_tensor(tgt),
                    paddle.to_tensor(tgt),
                    paddle.to_tensor(tgt_mask), self_attn_cache)

                tgt = tgt[0].numpy() if cache else tgt.numpy()

                tgt = residual + tgt
                # postprocess
                tgt_norm = layer_norm(tgt, d_model, decoder_layer.norm1)
                residual = tgt_norm
                # cross-attn
                cross_attn_cache = cache_objs[
                    1] if cache_objs is not None else None
                tgt = cross_attn(
                    paddle.to_tensor(tgt_norm),
                    paddle.to_tensor(memory),
                    paddle.to_tensor(memory),
                    paddle.to_tensor(memory_mask), cross_attn_cache)
                tgt = tgt[0].numpy() if cache else tgt.numpy()

                # postprocess
                tgt = tgt + residual
                tgt_norm = layer_norm(tgt, d_model, decoder_layer.norm2)
                residual = tgt_norm
                # FFN
                ffn_output = ffn(tgt_norm, decoder_layer, activation)
                # post process
                tgt = residual + ffn_output
                tgt_norm = layer_norm(tgt, d_model, decoder_layer.norm3)

                np.testing.assert_allclose(
                    decoder_output, tgt_norm, rtol=1e-5, atol=1e-6)

    def test_encoder(self):
        batch_size, d_model, n_head, dim_feedforward, dropout, attn_dropout, act_dropout, sequence_length = generate_basic_params(
            mode="encoder_layer")

        src = np.random.rand(batch_size, sequence_length,
                             d_model).astype("float32")

        src_mask = np.zeros((batch_size, n_head, sequence_length,
                             sequence_length)).astype("float32")
        src_mask[0][0][0][0] = -np.inf
        with fluid.dygraph.guard(fluid.CPUPlace()):
            encoder_layer = TransformerEncoderLayer(d_model, n_head,
                                                    dim_feedforward, dropout)
            num_layers = 6
            encoder = TransformerEncoder(encoder_layer, num_layers)
            # src, src_mask
            enc_output = encoder(
                paddle.to_tensor(src), paddle.to_tensor(src_mask))

    def test_decoder(self):
        batch_size, d_model, n_head, dim_feedforward, dropout, _, _, source_length, target_length = generate_basic_params(
            mode="decoder_layer")
        tgt = np.random.rand(batch_size, target_length,
                             d_model).astype("float32")
        memory = np.random.rand(batch_size, source_length,
                                d_model).astype("float32")
        tgt_mask = np.zeros((batch_size, n_head, target_length,
                             target_length)).astype("float32")
        tgt_mask[0][0][0][0] = -1e9
        memory_mask = np.zeros((batch_size, n_head, target_length,
                                source_length)).astype("float32")
        memory_mask[0][0][0][0] = -1e9
        with fluid.dygraph.guard(fluid.CPUPlace()):
            decoder_layer = TransformerDecoderLayer(d_model, n_head,
                                                    dim_feedforward, dropout)
            num_layers = 6
            decoder = TransformerDecoder(decoder_layer, num_layers)

            output = decoder(
                paddle.to_tensor(tgt),
                paddle.to_tensor(memory),
                paddle.to_tensor(tgt_mask), paddle.to_tensor(memory_mask))

    def test_transformer(self):
        batch_size, d_model, n_head, dim_feedforward, dropout, _, _, source_length, target_length = generate_basic_params(
            mode="decoder_layer")

        # batch_size, source_length, target_length, d_model, n_head = 4, 8, 8, 64, 8
        with fluid.dygraph.guard(fluid.CPUPlace()):
            transformer = Transformer(
                d_model,
                n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout)
            src = paddle.to_tensor(
                np.random.rand(batch_size, source_length, d_model).astype(
                    "float32"))
            tgt = paddle.to_tensor(
                np.random.rand(batch_size, target_length, d_model).astype(
                    "float32"))
            src_mask = np.zeros((batch_size, n_head, source_length,
                                 source_length)).astype("float32")
            src_mask[0][0][0][0] = -np.inf
            src_mask = paddle.to_tensor(src_mask)
            tgt_mask = np.zeros((batch_size, n_head, target_length,
                                 target_length)).astype("float32")
            tgt_mask[0][0][0][0] = -1e9
            memory_mask = np.zeros((batch_size, n_head, target_length,
                                    source_length)).astype("float32")
            memory_mask[0][0][0][0] = -1e9
            tgt_mask, memory_mask = paddle.to_tensor(
                tgt_mask), paddle.to_tensor(memory_mask)
            trans_output = transformer(src, tgt, src_mask, tgt_mask,
                                       memory_mask)

    def test_transformer_attr_1(self):
        batch_size, d_model, n_head, dim_feedforward, dropout, _, _, source_length, target_length = generate_basic_params(
            mode="decoder_layer")

        # batch_size, source_length, target_length, d_model, n_head = 4, 8, 8, 64, 8
        with fluid.dygraph.guard(fluid.CPUPlace()):
            transformer = Transformer(
                d_model,
                n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                weight_attr=[None],
                bias_attr=[False])
            src = paddle.to_tensor(
                np.random.rand(batch_size, source_length, d_model).astype(
                    "float32"))
            tgt = paddle.to_tensor(
                np.random.rand(batch_size, target_length, d_model).astype(
                    "float32"))
            src_mask = np.zeros((batch_size, n_head, source_length,
                                 source_length)).astype("float32")
            src_mask[0][0][0][0] = -np.inf
            src_mask = paddle.to_tensor(src_mask)
            tgt_mask = np.zeros((batch_size, n_head, target_length,
                                 target_length)).astype("float32")
            tgt_mask[0][0][0][0] = -1e9
            memory_mask = np.zeros((batch_size, n_head, target_length,
                                    source_length)).astype("float32")
            memory_mask[0][0][0][0] = -1e9
            tgt_mask, memory_mask = paddle.to_tensor(
                tgt_mask), paddle.to_tensor(memory_mask)
            trans_output = transformer(src, tgt, src_mask, tgt_mask,
                                       memory_mask)

    def test_transformer_attr_2(self):
        batch_size, d_model, n_head, dim_feedforward, dropout, _, _, source_length, target_length = generate_basic_params(
            mode="decoder_layer")

        # batch_size, source_length, target_length, d_model, n_head = 4, 8, 8, 64, 8
        with fluid.dygraph.guard(fluid.CPUPlace()):
            transformer = Transformer(
                d_model,
                n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                weight_attr=[None, None],
                bias_attr=[False, False])
            src = paddle.to_tensor(
                np.random.rand(batch_size, source_length, d_model).astype(
                    "float32"))
            tgt = paddle.to_tensor(
                np.random.rand(batch_size, target_length, d_model).astype(
                    "float32"))
            src_mask = np.zeros((batch_size, n_head, source_length,
                                 source_length)).astype("float32")
            src_mask[0][0][0][0] = -np.inf
            src_mask = paddle.to_tensor(src_mask)
            tgt_mask = np.zeros((batch_size, n_head, target_length,
                                 target_length)).astype("float32")
            tgt_mask[0][0][0][0] = -1e9
            memory_mask = np.zeros((batch_size, n_head, target_length,
                                    source_length)).astype("float32")
            memory_mask[0][0][0][0] = -1e9
            tgt_mask, memory_mask = paddle.to_tensor(
                tgt_mask), paddle.to_tensor(memory_mask)
            trans_output = transformer(src, tgt, src_mask, tgt_mask,
                                       memory_mask)

    def test_transformer_attr_3(self):
        batch_size, d_model, n_head, dim_feedforward, dropout, _, _, source_length, target_length = generate_basic_params(
            mode="decoder_layer")

        # batch_size, source_length, target_length, d_model, n_head = 4, 8, 8, 64, 8
        with fluid.dygraph.guard(fluid.CPUPlace()):
            transformer = Transformer(
                d_model,
                n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                weight_attr=[None, None, None],
                bias_attr=[False, False, True])
            src = paddle.to_tensor(
                np.random.rand(batch_size, source_length, d_model).astype(
                    "float32"))
            tgt = paddle.to_tensor(
                np.random.rand(batch_size, target_length, d_model).astype(
                    "float32"))
            src_mask = np.zeros((batch_size, n_head, source_length,
                                 source_length)).astype("float32")
            src_mask[0][0][0][0] = -np.inf
            src_mask = paddle.to_tensor(src_mask)
            tgt_mask = np.zeros((batch_size, n_head, target_length,
                                 target_length)).astype("float32")
            tgt_mask[0][0][0][0] = -1e9
            memory_mask = np.zeros((batch_size, n_head, target_length,
                                    source_length)).astype("float32")
            memory_mask[0][0][0][0] = -1e9
            tgt_mask, memory_mask = paddle.to_tensor(
                tgt_mask), paddle.to_tensor(memory_mask)
            trans_output = transformer(src, tgt, src_mask, tgt_mask,
                                       memory_mask)

    def test_transformer_attr_boolean(self):
        batch_size, d_model, n_head, dim_feedforward, dropout, _, _, source_length, target_length = generate_basic_params(
            mode="decoder_layer")

        # batch_size, source_length, target_length, d_model, n_head = 4, 8, 8, 64, 8
        with fluid.dygraph.guard(fluid.CPUPlace()):
            transformer = Transformer(
                d_model,
                n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bias_attr=False)
            src = paddle.to_tensor(
                np.random.rand(batch_size, source_length, d_model).astype(
                    "float32"))
            tgt = paddle.to_tensor(
                np.random.rand(batch_size, target_length, d_model).astype(
                    "float32"))
            src_mask = np.zeros((batch_size, n_head, source_length,
                                 source_length)).astype("float32")
            src_mask[0][0][0][0] = -np.inf
            src_mask = paddle.to_tensor(src_mask)
            tgt_mask = np.zeros((batch_size, n_head, target_length,
                                 target_length)).astype("float32")
            tgt_mask[0][0][0][0] = -1e9
            memory_mask = np.zeros((batch_size, n_head, target_length,
                                    source_length)).astype("float32")
            memory_mask[0][0][0][0] = -1e9
            tgt_mask, memory_mask = paddle.to_tensor(
                tgt_mask), paddle.to_tensor(memory_mask)
            trans_output = transformer(src, tgt, src_mask, tgt_mask,
                                       memory_mask)

    def test_generate_square_subsequent_mask(self):
        length = 5
        d_model, n_head, dim_feedforward = 8, 4, 64
        transformer = Transformer(
            d_model, n_head, dim_feedforward=dim_feedforward)
        mask = transformer.generate_square_subsequent_mask(length)


if __name__ == "__main__":
    unittest.main()
