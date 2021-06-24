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
import paddle
import paddle.nn as nn
from . import models  # noqa: F401
from . import transforms  # noqa: F401
from . import datasets  # noqa: F401
from . import ops  # noqa: F401
from .image import set_image_backend  # noqa: F401
from .image import get_image_backend  # noqa: F401
from .image import image_load  # noqa: F401
from .models import LeNet as models_LeNet
import paddle.utils.deprecated as deprecated

__all__ = [  #noqa
    'set_image_backend', 'get_image_backend', 'image_load'
]


class LeNet(models_LeNet):
    """LeNet model from
    `"LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.`_

    Args:
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 10.

    Examples:
        .. code-block:: python

            from paddle.vision.models import LeNet

            model = LeNet()
    """

    @deprecated(
        since="2.0.0",
        update_to="paddle.vision.models.LeNet",
        level=1,
        reason="Please use new API in models, paddle.vision.LeNet will be removed in future"
    )
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__(num_classes=10)
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2))

        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84), nn.Linear(84, num_classes))
