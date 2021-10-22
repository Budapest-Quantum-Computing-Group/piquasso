#
# Copyright 2021 Budapest Quantum Computing Group
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

import pytest

import piquasso as pq
from piquasso.api.errors import PiquassoException


@pytest.fixture
def MyGaussianState():
    class _MyGaussianState(pq.GaussianState):
        pass

    _MyGaussianState.__name__ = "MyGaussianState"

    return _MyGaussianState


@pytest.fixture
def MyBeamsplitter():
    class _MyBeamsplitter(pq.Beamsplitter):
        pass

    _MyBeamsplitter.__name__ = "MyBeamsplitter"

    return _MyBeamsplitter


def test_use_plugin(MyGaussianState, MyBeamsplitter):
    class Plugin:
        classes = {
            "GaussianState": MyGaussianState,
            "Beamsplitter": MyBeamsplitter,
        }

    pq.registry.use_plugin(Plugin, override=True)

    with pq.Program() as program:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    GaussianState = pq.registry.get_class("GaussianState")

    state = GaussianState(d=3)
    state.apply(program)

    assert state.__class__ is MyGaussianState
    assert pq.registry.get_class("Beamsplitter") is MyBeamsplitter


def test_use_plugin_with_reimport(MyGaussianState, MyBeamsplitter):
    class Plugin:
        classes = {
            "GaussianState": MyGaussianState,
            "Beamsplitter": MyBeamsplitter,
        }

    pq.registry.use_plugin(Plugin, override=True)

    with pq.Program() as program:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    GaussianState = pq.registry.get_class("GaussianState")

    state = GaussianState(d=3)
    state.apply(program)

    import piquasso  # noqa: F401

    assert state.__class__ is MyGaussianState
    assert pq.registry.get_class("Beamsplitter") is MyBeamsplitter


def test_untouched_classes_remain_to_be_accessible(
    MyGaussianState,
    MyBeamsplitter,
):
    class Plugin:
        classes = {
            "GaussianState": MyGaussianState,
            "Beamsplitter": MyBeamsplitter,
        }

    pq.registry.use_plugin(Plugin, override=True)

    with pq.Program() as program:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3)

    GaussianState = pq.registry.get_class("GaussianState")

    state = GaussianState(d=3)
    state.apply(program)

    assert pq.Beamsplitter is not MyBeamsplitter
    assert pq.registry.get_class("Beamsplitter") is MyBeamsplitter
    assert pq.registry.get_class("Phaseshifter") is pq.instructions.gates.Phaseshifter


def test_overriding_items_in_registry_raises_PiquassoException():
    class Plugin:
        classes = {
            "GaussianState": MyGaussianState,
            "Beamsplitter": MyBeamsplitter,
        }

    with pytest.raises(PiquassoException):
        pq.registry.use_plugin(Plugin)
