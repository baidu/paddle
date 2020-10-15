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
import unittest

paddle.disable_static()
SEED = 2020
np.random.seed(SEED)
paddle.manual_seed(SEED)


class Generator(fluid.dygraph.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = paddle.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = fluid.layers.tanh(x)
        return x


class Discriminator(fluid.dygraph.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convd = paddle.nn.Conv2d(6, 3, 1)

    def forward(self, x):
        x = self.convd(x)
        return x


class TestRetainGraph(unittest.TestCase):
    def cal_gradient_penalty(self,
                             netD,
                             real_data,
                             fake_data,
                             edge_data=None,
                             type='mixed',
                             constant=1.0,
                             lambda_gp=10.0):
        if lambda_gp > 0.0:
            if type == 'real':
                interpolatesv = real_data
            elif type == 'fake':
                interpolatesv = fake_data
            elif type == 'mixed':
                alpha = paddle.rand((real_data.shape[0], 1))
                alpha = paddle.expand(alpha, [
                    real_data.shape[0],
                    np.prod(real_data.shape) // real_data.shape[0]
                ])
                alpha = paddle.reshape(alpha, real_data.shape)
                interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
            else:
                raise NotImplementedError('{} not implemented'.format(type))
            interpolatesv.stop_gradient = False
            real_data.stop_gradient = True
            fake_AB = paddle.concat((real_data.detach(), interpolatesv), 1)
            disc_interpolates = netD(fake_AB)

            outs = paddle.fluid.layers.fill_constant(disc_interpolates.shape,
                                        disc_interpolates.dtype, 1.0)
            gradients = paddle.grad(
                outputs=disc_interpolates,
                inputs=fake_AB,
                grad_outputs=outs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)

            gradients = paddle.reshape(gradients[0], [real_data.shape[0], -1])

            gradient_penalty = paddle.mean((paddle.norm(
                gradients + 1e-16, 2, 1) - constant)**
                                                  2) * lambda_gp  # added eps
            return gradient_penalty, gradients
        else:
            return 0.0, None

    def run_retain(self, need_retain):
        g = Generator()
        d = Discriminator()

        optim_g = paddle.optimizer.Adam(parameters=g.parameters())
        optim_d = paddle.optimizer.Adam(parameters=d.parameters())

        gan_criterion = paddle.nn.MSELoss()
        l1_criterion = paddle.nn.L1Loss()

        A = np.random.rand(2, 3, 32, 32).astype('float32')
        B = np.random.rand(2, 3, 32, 32).astype('float32')

        realA = paddle.to_tensor(A)
        realB = paddle.to_tensor(B)
        fakeB = g(realA)

        optim_d.clear_gradients()
        fake_AB = paddle.concat((realA, fakeB), 1)
        G_pred_fake = d(fake_AB.detach())

        false_target = paddle.fluid.layers.fill_constant(G_pred_fake.shape, 'float32', 0.0)

        G_gradient_penalty, _ = self.cal_gradient_penalty(
            d, realA, fakeB, lambda_gp=10.0)
        loss_d = gan_criterion(G_pred_fake, false_target) + G_gradient_penalty

        loss_d.backward(retain_graph=need_retain)
        optim_d.minimize(loss_d)

        optim_g.clear_gradients()
        fake_AB = paddle.concat((realA, fakeB), 1)
        G_pred_fake = d(fake_AB)
        true_target = paddle.fluid.layers.fill_constant(G_pred_fake.shape, 'float32', 1.0)
        loss_g = l1_criterion(fakeB, realB) + gan_criterion(G_pred_fake,
                                                            true_target)

        loss_g.backward()
        optim_g.minimize(loss_g)

    def test_retain(self):
        self.run_retain(need_retain=True)
        self.assertRaises(
            fluid.core.EnforceNotMet, self.run_retain, need_retain=False)


if __name__ == '__main__':
    unittest.main()
