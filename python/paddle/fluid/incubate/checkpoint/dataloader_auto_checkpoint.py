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

import six
from . import auto_checkpoint as acp

g_ranges = {}

CONST_GENERATOR_BEGIN = "begin"
CONST_GENERATOR_END = "end"


class TrainEpochRangeWrapper(object):
    def __init__(self, name):
        self._running_status = CONST_GENERATOR_BEGIN
        self._epoch_no = -1
        self._checkpoint_epoch_no = None

        self._train_epoch_range = acp.TrainEpochRange(
            -1, name, checkpoint_inter=acp._get_checker().save_checkpoint_inter)

        if self._train_epoch_range.restored_from == acp.CONST_CHECKPOINT:
            self._checkpoint_epoch_no = self._train_epoch_range._checkpoint_epoch_no
            # can't assign data because it will be go throughed
            #self._epoch_no = self._train_epoch_range._epoch_no

    def save_checkpoint(self):
        if self.beyond_restored():
            self._train_epoch_range.save_checkpoint()

    def __str__(self):
        return "epoch_no:{} checkpoint_epoch_no:{} running_status:{}".format(
            self._epoch_no, self._checkpoint_epoch_no, self._running_status)

    def increment_epoch_no(self):
        self._epoch_no += 1
        self._train_epoch_range._epoch_no = self._epoch_no

    def check(self):
        assert len(
            self._train_epoch_range._exe_status
        ) <= 1, "data loader checkpoint must contain one exe when running"

    def beyond_restored(self):
        if self.is_restored:
            return self._epoch_no > self._checkpoint_epoch_no

        return True

    @property
    def is_restored(self):
        return self._checkpoint_epoch_no is not None

    def contain(self, exe_name, program_name):
        key = acp._get_running_key(exe_name, program_name)

        e = self._train_epoch_range._exe_status
        if key not in e:
            return False

        return True


logger = acp._get_logger(20)


def _check_env():
    checker = acp._get_checker()
    if not checker.valid():
        return False

    if acp.g_acp_type == acp.CONST_ACP_TYPE:
        return False

    return True


def _current(name):
    init = False

    acp.g_acp_type = acp.CONST_DACP_TYPE

    if name not in g_ranges:
        g_ranges[name] = TrainEpochRangeWrapper(name)
        init = True
    t = g_ranges[name]
    t.check()

    acp.g_train_epoch_range = t._train_epoch_range

    return t, init


def _init_checkpoint(name):
    if not _check_env():
        return False

    t, init = _current(name)
    t.increment_epoch_no()

    logger.info("acp_type:{}".format(acp.g_acp_type))
    if t.is_restored:
        logger.info("begin dataloader epoch_no:{} checkpoint_epoch_no:{}".
                    format(t._epoch_no, t._checkpoint_epoch_no))
    else:
        logger.info("begin dataloader epoch_no:{}".format(t._epoch_no))

    return True


def _beyond_restored(name):
    t, init = _current(name)
    assert not init, "internal error, {} must be initted".format(name)

    if not t.beyond_restored():
        return False

    return True


def _ignore_epoch(name):
    if not _init_checkpoint(name):
        return False

    if _beyond_restored(name):
        return False

    return True


def _end(name):
    if not _check_env():
        return False

    try:
        # check
        assert name in g_ranges, \
            "internal error: g_ranges must contain the name:{}, now:{}".format(name, g_ranges.keys())
        t = g_ranges[name]
        t._runing_status = CONST_GENERATOR_END
        assert t._train_epoch_range == acp.g_train_epoch_range, "interal error, current running range must equal"

        t.save_checkpoint()

        if not t.is_restored:
            logger.info("end dataloader epoch_no:{}".format(t._epoch_no))
        else:
            logger.info("end dataloader epoch_no:{} checkpoint_epoch_no:{}".
                        format(t._epoch_no, t._checkpoint_epoch_no))
    finally:
        # important
        acp.g_train_epoch_range = None

    return True


def _is_restoring(executor, program):
    if acp.g_acp_type != acp.CONST_DACP_TYPE:
        return False

    if len(g_ranges) < 1:
        return False

    for n, range_wrapper in six.iteritems(g_ranges):  # ranges
        if not range_wrapper.is_restored:
            continue

        if not range_wrapper.contain(executor._auto_checkpoint_name,
                                     program._auto_checkpoint_name):
            continue

        if range_wrapper.beyond_restored():
            return False
        else:
            logger.info(
                "range_wrapper:{} is restoring save_model may be canceled".
                format(range_wrapper))
            return True

    return False
