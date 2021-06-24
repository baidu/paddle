# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import argparse
import json
import six
import sys
import re
import os
import glob
import unittest
import pandas
import tempfile
import platform
import pandas as pd


class ChromeTraceFormatter(object):
    def __init__(self):
        self._events = []
        self._metadata = []

    def _create_event(self, ph, category, name, pid, tid, timestamp):
        """Creates a new Chrome Trace event.

        For details of the file format, see:
        https://github.com/catapult-project/catapult/blob/master/tracing/README.md

        Args:
          ph:  The type of event - usually a single character.
          category: The event category as a string.
          name:  The event name as a string.
          pid:  Identifier of the process generating this event as an integer.
          tid:  Identifier of the thread generating this event as an integer.
          timestamp:  The timestamp of this event as a long integer.

        Returns:
          A JSON compatible event object.
        """
        event = {}
        event['ph'] = ph
        event['cat'] = category
        event['name'] = name
        event['pid'] = pid
        event['tid'] = tid
        event['ts'] = timestamp
        return event

    def emit_pid(self, name, pid):
        """Adds a process metadata event to the trace.

        Args:
          name:  The process name as a string.
          pid:  Identifier of the process as an integer.
        """
        event = {}
        event['name'] = 'process_name'
        event['ph'] = 'M'
        event['pid'] = pid
        event['args'] = {'name': name}
        self._metadata.append(event)

    def emit_region(self, timestamp, duration, pid, tid, category, name, args):
        """Adds a region event to the trace.

        Args:
          timestamp:  The start timestamp of this region as a long integer.
          duration:  The duration of this region as a long integer.
          pid:  Identifier of the process generating this event as an integer.
          tid:  Identifier of the thread generating this event as an integer.
          category: The event category as a string.
          name:  The event name as a string.
          args:  A JSON-compatible dictionary of event arguments.
        """
        event = self._create_event('X', category, name, pid, tid, timestamp)
        event['dur'] = duration
        event['args'] = args
        self._events.append(event)

    def emit_counter(self, category, name, pid, timestamp, counter, value):
        """Emits a record for a single counter.

        Args:
            category: The event category as string
            name: The event name as string
            pid: Identifier of the process generating this event as integer
            timestamp: The timestamps of this event as long integer
            counter: Name of the counter as string
            value: Value of the counter as integer
            tid: Thread id of the allocation as integer
        """
        event = self._create_event('C', category, name, pid, 0, timestamp)
        event['args'] = {counter: value}
        self._events.append(event)

    def format_to_string(self, pretty=False):
        """Formats the chrome trace to a string.

        Args:
          pretty: (Optional.)  If True, produce human-readable JSON output.

        Returns:
          A JSON-formatted string in Chrome Trace format.
        """
        trace = {}
        trace['traceEvents'] = self._metadata + self._events
        if pretty:
            return json.dumps(trace, indent=4, separators=(',', ': '))
        else:
            return json.dumps(trace, separators=(',', ':'))

    def clear(self):
        self._events = []
        self._metadata = []


if __name__ == "__main__":
    pass
