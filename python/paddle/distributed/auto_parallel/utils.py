#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = []


class ProcessMesh:
    def __init__(self, mesh, process_group=None):
        """
        A class to describe the logical topology of all processes.

        mesh (list): a list to describe process topology
        process_group (list): a list of processes belonging to this group
        Examples:
            dp_degree=pp_degree=mp_degree=2
            mesh = ProcessMesh([dp_degree, pp_degree, mp_degree])
        """
        process_num = np.prod(mesh)
        if process_group is None:
            process_group = list(range(process_num))
        assert len(process_group) == process_num
        self.process_group = process_group
        self.mesh = mesh

    def get_mesh(self):
        return self.mesh

    def get_process_group(self):
        return self.process_group
