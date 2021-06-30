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
    The object of this class has some methods such as `scale()`, `unscale()`, `step()`, `update()`.

    `scale()` is used to multiply the loss by a scale ratio.
    `unscale()` + `step()` is similar as `Optimizer.minimize()`, performs parameters updating.
    `update()` is used to update the loss scaling.

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
            # update parameters 
            scaler.unscale(optimizer)
            scaler.step(optimizer, scaled)
            scaler.update()     
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
                # update parameters 
                scaler.unscale(optimizer)
                scaler.step(optimizer, scaled)
                scaler.update() 
                optimizer.clear_grad()
        """
        return super(GradScaler, self).scale(var)

    def unscale(self, optimizer):
        """
        Unscale the gradients of parameters, multiplies the gradients of parameters by 1/scale.  
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
        Returns:
            The unscaled parameters or original parameters.
        
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
                # update parameters 
                scaler.unscale(optimizer)
                scaler.step(optimizer, scaled)
                scaler.update() 
                optimizer.clear_grad()
        """
        return super(GradScaler, self).unscale(optimizer)

    def step(self, optimizer, *args, **kwargs):
        """
        This function is similar as `Optimizer.minimize()`, which performs parameters updating.

        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, it first unscales the scaled gradients of parameters, then updates the parameters.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
            args:  Arguments, which will be forward to `optimizer.minimize()`.
            kwargs: Keyword arguments, which will be forward to `Optimizer.minimize()`.
        
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
                # update parameters 
                scaler.unscale(optimizer)
                scaler.step(optimizer, scaled)
                scaler.update() 
                optimizer.clear_grad()
        """
        return super(GradScaler, self).step(optimizer, *args, **kwargs)

    def update(self):
        """
        Updates the loss_scaling.
        
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
                # update parameters 
                scaler.unscale(optimizer)
                scaler.step(optimizer, scaled)
                scaler.update() 
                optimizer.clear_grad()
        """
        return super(GradScaler, self).update()

    def get_enable(self):
        """
        Get the parameter of `_enable`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                enable = scaler.get_enable()
        """
        return super(GradScaler, self).get_enable()

    def get_use_dynamic_loss_scaling(self):
        """
        Get the parameter of `_use_dynamic_loss_scaling`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                use_dynamic_loss_scaling = scaler.get_use_dynamic_loss_scaling()
        """
        return super(GradScaler, self).get_use_dynamic_loss_scaling()

    def get_init_loss_scaling(self):
        """
        Get the parameter of `_init_loss_scaling`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                init_loss_scaling = scaler.get_init_loss_scaling()
        """
        return super(GradScaler, self).get_init_loss_scaling()

    def set_init_loss_scaling(self, new_init_loss_scaling):
        """
        Set the parameter of `_init_loss_scaling` by `new_init_loss_scaling`.

        Args:
            new_init_loss_scaling(int):  The new_init_loss_scaling used to update _init_loss_scaling.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                new_init_loss_scaling = 1000
                scaler.set_init_loss_scaling(new_init_loss_scaling)
        """
        super(GradScaler, self).set_init_loss_scaling(new_init_loss_scaling)

    def get_incr_ratio(self):
        """
        Get the parameter of `_incr_ratio`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                incr_ratio = scaler.get_incr_ratio()
        """
        return super(GradScaler, self).get_incr_ratio()

    def set_incr_ratio(self, new_incr_ratio):
        """
        Set the parameter of `_incr_ratio` by `new_incr_ratio`, `new_incr_ratio` should > 1.0.

        Args:
            new_incr_ratio(float):  The new_incr_ratio used to update _incr_ratio.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                new_incr_ratio = 3.0
                scaler.set_incr_ratio(new_incr_ratio)
        """
        super(GradScaler, self).set_incr_ratio(new_incr_ratio)

    def get_decr_ratio(self):
        """
        Get the parameter of `_decr_ratio`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                decr_ratio = scaler.get_decr_ratio()
        """
        return super(GradScaler, self).get_decr_ratio()

    def set_decr_ratio(self, new_decr_ratio):
        """
        Set the parameter of `_decr_ratio` by `new_incr_ratio`, `new_decr_ratio` should < 1.0.

        Args:
            new_decr_ratio(float):  The new_decr_ratio used to update _decr_ratio.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                new_decr_ratio = 0.1
                scaler.set_decr_ratio(new_decr_ratio)
        """
        super(GradScaler, self).set_decr_ratio(new_decr_ratio)

    def get_incr_every_n_steps(self):
        """
        Get the parameter of `_incr_every_n_steps`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                incr_every_n_steps = scaler.get_incr_every_n_steps()
        """
        return super(GradScaler, self).get_incr_every_n_steps()

    def set_incr_every_n_steps(self, new_incr_every_n_steps):
        """
        Set the parameter of `_incr_every_n_steps` by `new_incr_every_n_steps`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                new_incr_every_n_steps = 2000
                scaler.set_incr_every_n_steps(new_incr_every_n_steps)
        """
        super(GradScaler, self).set_incr_every_n_steps(new_incr_every_n_steps)

    def get_decr_every_n_nan_or_inf(self):
        """
        Get the parameter of `_decr_every_n_nan_or_inf`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                decr_every_n_nan_or_inf = scaler.get_decr_every_n_nan_or_inf()
        """
        return super(GradScaler, self).get_decr_every_n_nan_or_inf()

    def set_decr_every_n_nan_or_inf(self, new_decr_every_n_nan_or_inf):
        """
        Set the parameter of `_decr_every_n_nan_or_inf` by `new_decr_every_n_nan_or_inf`.
        
        Examples:

            .. code-block:: python

                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                new_decr_every_n_nan_or_inf = 3
                scaler.set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)
        """
        super(GradScaler,
              self).set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)
