#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Defination of device workers."""

from __future__ import print_function

__all__ = [
    'DeviceWorker', 'Hogwild', 'DownpourSGD', 'Section', 'DownpourSGDOPT'
]


class DeviceWorker(object):
    """
    DeviceWorker is an abstract class, which generates worker desc.
    This class is an inner class that we do computation logics within
    the implementation. For example, execution of a program or a graph.
    """

    def __init__(self):
        """Init."""
        self._program = None
        self._infer = None

    def _set_infer(self, infer=False):
        """
        set inference flag for current device worker

        Args:
            infer(bool): whether to do inference
        """
        self._infer = infer

    def _set_fleet_desc(self, fleet_desc):
        """
        Set fleet desc.

        Args:
            fleet_desc(PSParameter): pslib.PSParameter object
        """
        self._fleet_desc = fleet_desc

    def _set_program(self, program):
        """
        Set program.

        Args:
            program(Program): a Program object
        """
        self._program = program

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc.

        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        raise NotImplementedError(
            "DeviceWorker does not implement gen_worker_desc, "
            "please use Hogwild or DownpourSGD, etc.")


class Hogwild(DeviceWorker):
    """
    Hogwild is a kind of SGD algorithm.

    """

    def __init__(self):
        """Init."""
        super(Hogwild, self).__init__()

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc, which device worker is HogwildWorker.

        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        trainer_desc.device_worker_name = "HogwildWorker"
        if self._infer:
            # just ignore feed op for inference model
            trainer_desc.hogwild_param.skip_ops.extend(["feed"])

        dense_table_set = set()
        program_id = str(id(self._program))
        if self._program == None:
            print("program of current device worker is not configured")
            exit(-1)
        opt_info = self._program._fleet_opt
        # when opt_info is None or empty dict, it should return
        if not opt_info:
            return

        from paddle.fluid.incubate.fleet.parameter_server import version

        if version.is_transpiler() and "fleet_desc" not in opt_info:
            return

        program_configs = opt_info["program_configs"]
        downpour = trainer_desc.downpour_param
        hogwild = trainer_desc.hogwild_param

        for pid in program_configs:
            if pid == program_id:
                pc = downpour.program_config.add()
                pc.program_id = program_id
                for i in program_configs[program_id]["push_sparse"]:
                    pc.push_sparse_table_id.extend([i])
                for i in program_configs[program_id]["push_dense"]:
                    pc.push_dense_table_id.extend([i])
                    dense_table_set.add(i)
                for i in program_configs[program_id]["pull_sparse"]:
                    pc.pull_sparse_table_id.extend([i])
                for i in program_configs[program_id]["pull_dense"]:
                    pc.pull_dense_table_id.extend([i])
                    dense_table_set.add(i)
                break

        trainer_desc.device_worker_name = "HogwildWorker"
        pull_thread = trainer_desc.pull_dense_param
        pull_thread.device_num = trainer_desc.thread_num
        if opt_info.get("program_id_to_worker") is None:
            raise ValueError("opt_info must have program_id_to_worker")
        prog_id_to_worker = opt_info["program_id_to_worker"]
        if prog_id_to_worker.get(program_id) is None:
            raise ValueError("%s not found in program_id_to_worker" %
                             program_id)
        worker = opt_info["program_id_to_worker"][program_id]
        for i in worker.get_desc().dense_table:
            if i.table_id in dense_table_set:
                dense_table = pull_thread.dense_table.add()
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.table_id = \
                    i.table_id
        sparse_len = len(worker.get_desc().sparse_table)
        for i in range(sparse_len):
            sparse_table = downpour.sparse_table.add()
            sparse_table.table_id = worker.get_desc().sparse_table[i].table_id
            sparse_table.sparse_key_name.extend(worker.get_desc().sparse_table[
                i].slot_key)
            sparse_table.sparse_value_name.extend(worker.get_desc()
                                                  .sparse_table[i].slot_value)
            sparse_table.sparse_grad_name.extend(worker.get_desc().sparse_table[
                i].slot_gradient)
            sparse_table.fea_dim = \
                self._fleet_desc.server_param.downpour_server_param.downpour_table_param[
                    i].accessor.fea_dim
            # not use emb_dim
            sparse_table.emb_dim = -1
            # not use hard code click
            sparse_table.label_var_name = ""
        if opt_info["stat_var_names"]:
            for i in opt_info["stat_var_names"]:
                hogwild.stat_var_names.extend([i])
                downpour.stat_var_names.extend([i])

        for i in worker.get_desc().dense_table:
            if i.table_id in dense_table_set:
                dense_table = downpour.dense_table.add()
                dense_table.table_id = i.table_id
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.dense_grad_name.extend(
                    i.dense_gradient_variable_name)
        hogwild.skip_ops.extend(worker.get_desc().skip_op)
        if self._infer:
            hogwild.skip_ops.extend(
                ["push_sparse", "push_sparse_v2", "push_dense"])


class DownpourSGD(DeviceWorker):
    """
    DownpourSGD is a kind of distributed SGD algorithm.
    """

    def __init__(self):
        """
        Init.
        initialize downpourSGD device worker
        """
        super(DownpourSGD, self).__init__()

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc, which device worker is DownpourWorker.

        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        dense_table_set = set()
        program_id = str(id(self._program))
        if self._program == None:
            print("program of current device worker is not configured")
            exit(-1)
        opt_info = self._program._fleet_opt
        program_configs = opt_info["program_configs"]
        downpour = trainer_desc.downpour_param

        for pid in program_configs:
            if pid == program_id:
                pc = downpour.program_config.add()
                pc.program_id = program_id
                for i in program_configs[program_id]["push_sparse"]:
                    pc.push_sparse_table_id.extend([i])
                for i in program_configs[program_id]["push_dense"]:
                    pc.push_dense_table_id.extend([i])
                    dense_table_set.add(i)
                for i in program_configs[program_id]["pull_sparse"]:
                    pc.pull_sparse_table_id.extend([i])
                for i in program_configs[program_id]["pull_dense"]:
                    pc.pull_dense_table_id.extend([i])
                    dense_table_set.add(i)
                # code for partial push dense table such as multitask
                if "cond2denseid" in program_configs[program_id]:
                    cond2denseid = program_configs[program_id]["cond2denseid"]
                    for key, value in cond2denseid.items():
                        mc_map = pc.partial_pushdense_condtable_map.add()
                        mc_map.key = key
                        mc_map.value = value
                break

        trainer_desc.device_worker_name = opt_info.get("worker_class",
                                                       "DownpourWorker")
        pull_thread = trainer_desc.pull_dense_param
        pull_thread.device_num = trainer_desc.thread_num
        if opt_info.get("program_id_to_worker") is None:
            raise ValueError("opt_info must have program_id_to_worker")
        prog_id_to_worker = opt_info["program_id_to_worker"]
        if prog_id_to_worker.get(program_id) is None:
            raise ValueError("%s not found in program_id_to_worker" %
                             program_id)
        worker = opt_info["program_id_to_worker"][program_id]
        for i in worker.get_desc().dense_table:
            if i.table_id in dense_table_set:
                dense_table = pull_thread.dense_table.add()
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.table_id = \
                    i.table_id
        sparse_len = len(worker.get_desc().sparse_table)
        for i in range(sparse_len):
            sparse_table = downpour.sparse_table.add()
            sparse_table.table_id = worker.get_desc().sparse_table[i].table_id
            sparse_table.sparse_key_name.extend(worker.get_desc().sparse_table[
                i].slot_key)
            sparse_table.sparse_value_name.extend(worker.get_desc()
                                                  .sparse_table[i].slot_value)
            sparse_table.sparse_grad_name.extend(worker.get_desc().sparse_table[
                i].slot_gradient)
            if opt_info["use_cvm"] or "no_cvm" in opt_info and opt_info[
                    "no_cvm"] == True:
                sparse_table.emb_dim = \
                    self._fleet_desc.server_param.downpour_server_param.downpour_table_param[
                        i].accessor.fea_dim
                sparse_table.fea_dim = sparse_table.emb_dim
            else:
                sparse_table.emb_dim = \
                    self._fleet_desc.server_param.downpour_server_param.downpour_table_param[
                        i].accessor.fea_dim - 2
                sparse_table.fea_dim = sparse_table.emb_dim + 2
            # TODO(guru4elephant): hard code here, need to improve
            sparse_table.label_var_name = "click"
        if opt_info["stat_var_names"]:
            for i in opt_info["stat_var_names"]:
                downpour.stat_var_names.extend([i])

        for i in worker.get_desc().dense_table:
            if i.table_id in dense_table_set:
                dense_table = downpour.dense_table.add()
                dense_table.table_id = i.table_id
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.dense_grad_name.extend(
                    i.dense_gradient_variable_name)
        downpour.skip_ops.extend(worker.get_desc().skip_op)
        if self._infer:
            downpour.push_dense = False
            downpour.push_sparse = False


class DownpourSGDOPT(DeviceWorker):
    """
    DownpourSGDOPT is a kind of distributed SGD algorithm.
    """

    def __init__(self):
        """
        Init.
        initialize downpourSGDOPT device worker
        """
        super(DownpourSGDOPT, self).__init__()

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc, which device worker is DownpourWorker.

        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        dense_table_set = set()
        program_id = str(id(self._program))
        if self._program == None:
            print("program of current device worker is not configured")
            exit(-1)
        opt_info = self._program._fleet_opt
        program_configs = opt_info["program_configs"]
        downpour = trainer_desc.downpour_param

        for pid in program_configs:
            if pid == program_id:
                pc = downpour.program_config.add()
                pc.program_id = program_id
                for i in program_configs[program_id]["push_sparse"]:
                    pc.push_sparse_table_id.extend([i])
                for i in program_configs[program_id]["push_dense"]:
                    pc.push_dense_table_id.extend([i])
                    dense_table_set.add(i)
                for i in program_configs[program_id]["pull_sparse"]:
                    pc.pull_sparse_table_id.extend([i])
                for i in program_configs[program_id]["pull_dense"]:
                    pc.pull_dense_table_id.extend([i])
                    dense_table_set.add(i)
                break

        trainer_desc.device_worker_name = "DownpourWorkerOpt"
        pull_thread = trainer_desc.pull_dense_param
        pull_thread.device_num = trainer_desc.thread_num
        if opt_info.get("program_id_to_worker") is None:
            raise ValueError("opt_info must have program_id_to_worker")
        prog_id_to_worker = opt_info["program_id_to_worker"]
        if prog_id_to_worker.get(program_id) is None:
            raise ValueError("%s not found in program_id_to_worker" %
                             program_id)
        worker = opt_info["program_id_to_worker"][program_id]
        for i in worker.get_desc().dense_table:
            if i.table_id in dense_table_set:
                dense_table = pull_thread.dense_table.add()
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.table_id = \
                    i.table_id
        sparse_len = len(worker.get_desc().sparse_table)
        for i in range(sparse_len):
            sparse_table = downpour.sparse_table.add()
            sparse_table.table_id = worker.get_desc().sparse_table[i].table_id
            sparse_table.sparse_key_name.extend(worker.get_desc().sparse_table[
                i].slot_key)
            sparse_table.sparse_value_name.extend(worker.get_desc()
                                                  .sparse_table[i].slot_value)
            sparse_table.sparse_grad_name.extend(worker.get_desc().sparse_table[
                i].slot_gradient)
            if opt_info["use_cvm"] or "no_cvm" in opt_info and opt_info[
                    "no_cvm"] == True:
                sparse_table.emb_dim = \
                    self._fleet_desc.server_param.downpour_server_param.downpour_table_param[
                        i].accessor.fea_dim
                sparse_table.fea_dim = sparse_table.emb_dim
            else:
                sparse_table.emb_dim = \
                    self._fleet_desc.server_param.downpour_server_param.downpour_table_param[
                        i].accessor.fea_dim - 2
                sparse_table.fea_dim = sparse_table.emb_dim + 2
            # TODO(guru4elephant): hard code here, need to improve
            sparse_table.label_var_name = "click"
        if "local_tables" in opt_info and sparse_table.table_id in opt_info[
                "local_tables"]:
            sparse_table.is_local = True
        if "async_tables" in opt_info and sparse_table.table_id in opt_info[
                "async_tables"]:
            sparse_table.is_async = True
        if opt_info["stat_var_names"]:
            for i in opt_info["stat_var_names"]:
                downpour.stat_var_names.extend([i])

        for i in worker.get_desc().dense_table:
            if i.table_id in dense_table_set:
                dense_table = downpour.dense_table.add()
                dense_table.table_id = i.table_id
                dense_table.dense_value_name.extend(i.dense_variable_name)
                dense_table.dense_grad_name.extend(
                    i.dense_gradient_variable_name)
        downpour.skip_ops.extend(worker.get_desc().skip_op)
        if self._infer:
            downpour.push_dense = False
            downpour.push_sparse = False


class Section(DeviceWorker):
    """SectionWorker."""

    def __init__(self):
        """Init."""
        super(Section, self).__init__()

    def _gen_worker_desc(self, trainer_desc):
        """
        Generator worker desc, which device worker is SectionWorker.
        Args:
            trainer_desc(TrainerDesc): a TrainerDesc object
        """
        from google.protobuf import text_format
        from . import core
        trainer_desc.device_worker_name = "SectionWorker"
        pipeline_opt = self._program._pipeline_opt
        section_param = trainer_desc.section_param
        section_param.num_microbatches = pipeline_opt["num_microbatches"]
        section_param.start_cpu_core_id = pipeline_opt["start_cpu_core_id"]
        for i, program in enumerate(pipeline_opt["section_program_list"]):
            cfg = section_param.section_config.add()
            cfg.program_desc.ParseFromString(program["program"]._get_desc()
                                             .serialize_to_string())
            # TODO: why does not work
            # cfg.program_desc.CopyFrom(program.program._get_desc())
            place = pipeline_opt["place_list"][i]
            place_id = pipeline_opt["place_id_list"][i]
            if isinstance(place, core.CPUPlace):
                cfg.place = cfg.CPUPlace
            elif isinstance(place, core.CUDAPlace):
                cfg.place = cfg.CUDAPlace
            elif isinstance(place, core.CUDAPinnedPlace):
                cfg.place = cfg.CUDAPinnedPlace
            else:
                raise NotImplementedError(
                    "SectionWorker only supports CPUPlace, CUDAPlace and CUDAPinnedPlace now."
                )
            cfg.place_id = place_id


class DeviceWorkerFactory(object):
    def _create_device_worker(self, worker_type):
        classname = worker_type.capitalize()
        return globals()[classname]()
