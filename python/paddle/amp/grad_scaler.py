#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.dygraph.amp import AmpScaler

__all__ = []


class GradScaler(AmpScaler):
    """
    GradScaler is used for Auto-Mixed-Precision training in dynamic graph mode. 
    It controls the scaling of loss, helps avoiding numerical overflow.
    The object of this class has two methods `scale()`, `minimize()`.

    `scale()` is used to multiply the loss by a scale ratio.
    `minimize()` is similar as `optimizer.minimize()`, performs parameters updating.

    Commonly, it is used together with `paddle.amp.auto_cast` to achieve Auto-Mixed-Precision in 
    dynamic graph mode.

    Args:
        enable(bool, optional): Enable loss scaling or not. Default is True.
        init_loss_scaling (float, optional): The initial loss scaling factor. Default is 2**15.
        incr_ratio(float, optional): The multiplier to use when increasing the loss 
                        scaling. Default is 2.0.
        decr_ratio(float, optional): The less-than-one-multiplier to use when decreasing 
                        the loss scaling. Default is 0.5.
        incr_every_n_steps(int, optional): Increases loss scaling every n consecutive 
                                steps with finite gradients. Default is 1000.
        decr_every_n_nan_or_inf(int, optional): Decreases loss scaling every n 
                                    accumulated steps with nan or inf gradients. Default is 2.
        use_dynamic_loss_scaling(bool, optional): Whether to use dynamic loss scaling. If False, fixed loss_scaling is used. If True, the loss scaling is updated dynamicly. Default is True.
    Returns:
        An GradScaler object.

    Examples:

        .. code-block:: python

            import paddle

            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
            optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            data = paddle.rand([10, 3, 32, 32])

            with paddle.amp.auto_cast():
                conv = model(data)
                loss = paddle.mean(conv)
                
            scaled = scaler.scale(loss)  # scale the loss 
            scaled.backward()            # do backward
            scaler.minimize(optimizer, scaled)  # update parameters     
            optimizer.clear_grad()
    """

    def __init__(self,
                 enable=True,
                 init_loss_scaling=2.**15,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 use_dynamic_loss_scaling=True):
        super(GradScaler, self).__init__(enable, init_loss_scaling, incr_ratio,
                                         decr_ratio, incr_every_n_steps,
                                         decr_every_n_nan_or_inf,
                                         use_dynamic_loss_scaling)

    def scale(self, var):
        """
        Multiplies a Tensor by the scale factor and returns scaled outputs.  
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.

        Args:
            var (Tensor):  The tensor to scale.
        Returns:
            The scaled tensor or original tensor.
        
        Examples:

            .. code-block:: python

                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])

                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)

                scaled = scaler.scale(loss)  # scale the loss 
                scaled.backward()            # do backward
                scaler.minimize(optimizer, scaled)  # update parameters  
                optimizer.clear_grad()
        """
        return super(GradScaler, self).scale(var)

    def minimize(self, optimizer, *args, **kwargs):
        """
        This function is similar as `optimizer.minimize()`, which performs parameters updating.
        
        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, it first unscales the scaled gradients of parameters, then updates the parameters.

        Finally, the loss scaling ratio is updated.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
            args:  Arguments, which will be forward to `optimizer.minimize()`.
            kwargs: Keyword arguments, which will be forward to `optimizer.minimize()`.

        Examples:

            .. code-block:: python

                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])

                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)

                scaled = scaler.scale(loss)  # scale the loss 
                scaled.backward()            # do backward
                scaler.minimize(optimizer, scaled)  # update parameters  
                optimizer.clear_grad()
        """
        return super(GradScaler, self).minimize(optimizer, *args, **kwargs)
