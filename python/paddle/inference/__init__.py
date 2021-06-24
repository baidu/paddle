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

from ..fluid.inference import Config  # noqa: F401
from ..fluid.inference import DataType  # noqa: F401
from ..fluid.inference import PlaceType  # noqa: F401
from ..fluid.inference import PrecisionType  # noqa: F401
from ..fluid.inference import Tensor  # noqa: F401
from ..fluid.inference import Predictor  # noqa: F401
from ..fluid.inference import create_predictor  # noqa: F401
from ..fluid.inference import get_version  # noqa: F401
from ..fluid.inference import get_num_bytes_of_data_type  # noqa: F401
from ..fluid.inference import PredictorPool  # noqa: F401

__all__ = [  # noqa
    'Config',
    'DataType',
    'PlaceType',
    'PrecisionType',
    'Tensor',
    'Predictor',
    'create_predictor',
    'get_version',
    'get_num_bytes_of_data_type',
    'PredictorPool'
]
