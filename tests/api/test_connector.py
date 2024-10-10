#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

from unittest.mock import patch

import pytest

import piquasso as pq


def test_BaseConnector_cannot_be_instantiated():
    with pytest.raises(Exception):
        pq.api.connector.BaseConnector()


def test_BaseConnector_with_overriding_defaults():
    """
    NOTE: This test basically tests Python itself, but it is left here for us to
    remember that the `BaseConnector` class defaults need to be able to overridden for
    any plugin which might want to use e.g. different permanent or hafnian calculation.
    """

    def plugin_loop_hafnian():
        return 43

    class PluginConnector(pq.api.connector.BaseConnector):
        def __init__(self) -> None:
            super().__init__()

            self.loop_hafnian = plugin_loop_hafnian

    p = patch.multiple(PluginConnector, __abstractmethods__=set())

    p.start()
    plugin_connector = PluginConnector()
    p.stop()

    assert plugin_connector.loop_hafnian is plugin_loop_hafnian

    assert plugin_connector.loop_hafnian() == 43


def test_BaseConnector_repr():

    class PluginConnector(pq.api.connector.BaseConnector):
        def __init__(self) -> None:
            super().__init__()

    p = patch.multiple(PluginConnector, __abstractmethods__=set())

    p.start()
    plugin_connector = PluginConnector()
    p.stop()

    assert repr(plugin_connector) == str(plugin_connector) == "PluginConnector()"
