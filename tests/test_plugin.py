#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

import pytest

import piquasso as pq


@pytest.fixture
def MyGaussianBackend():
    # TODO: better way to expose API
    class _MyGaussianBackend(pq.gaussian.backend.GaussianBackend):
        pass

    _MyGaussianBackend.__name__ = "MyGaussianBackend"

    return _MyGaussianBackend


@pytest.fixture
def MyGaussianState(MyGaussianBackend):
    class _MyGaussianState(pq.GaussianState):
        _backend_class = MyGaussianBackend

    _MyGaussianState.__name__ = "MyGaussianState"

    return _MyGaussianState


@pytest.fixture
def MyBeamsplitter():
    class _MyBeamsplitter(pq.B):
        pass

    _MyBeamsplitter.__name__ = "MyBeamsplitter"

    return _MyBeamsplitter


def test_use_plugin(MyGaussianState, MyBeamsplitter, MyGaussianBackend):
    class Plugin:
        classes = {
            "GaussianState": MyGaussianState,
            "B": MyBeamsplitter,
        }

    pq.use(Plugin)

    program = pq.Program(state=pq.GaussianState.create_vacuum(3))

    with program:
        pq.Q(0, 1) | pq.B(theta=np.pi/3)

    program.execute()

    assert program.state.__class__ is MyGaussianState
    assert program.backend.__class__ is MyGaussianBackend
    assert pq.B is MyBeamsplitter


def test_use_plugin_with_reimport(MyGaussianState, MyBeamsplitter, MyGaussianBackend):
    class Plugin:
        classes = {
            "GaussianState": MyGaussianState,
            "B": MyBeamsplitter,
        }

    pq.use(Plugin)

    program = pq.Program(state=pq.GaussianState.create_vacuum(3))

    with program:
        pq.Q(0, 1) | pq.B(theta=np.pi/3)

    program.execute()

    import piquasso  # noqa: F401

    assert program.state.__class__ is MyGaussianState
    assert program.backend.__class__ is MyGaussianBackend
    assert pq.B is MyBeamsplitter


def test_untouched_classes_remain_to_be_accessible(
    MyGaussianState,
    MyBeamsplitter,
    MyGaussianBackend,
):
    class Plugin:
        classes = {
            "GaussianState": MyGaussianState,
            "B": MyBeamsplitter,
        }

    pq.use(Plugin)

    program = pq.Program(state=pq.GaussianState.create_vacuum(3))

    with program:
        pq.Q(0, 1) | pq.B(theta=np.pi/3)

    program.execute()

    assert pq.B is MyBeamsplitter
    assert pq.R is pq.operations.R
