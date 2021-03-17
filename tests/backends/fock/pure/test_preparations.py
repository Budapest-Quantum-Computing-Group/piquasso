#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq


def test_create_number_state():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState.create_vacuum(d=2, cutoff=2)
        pq.Q(1) | pq.Create()

        pq.Q(0, 1) | pq.B(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


def test_create_and_annihilate_number_state():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState.create_vacuum(d=2, cutoff=2)
        pq.Q(1) | pq.Create()
        pq.Q(1) | pq.Annihilate()

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [1, 0, 0, 0, 0, 0],
    )


def test_create_annihilate_and_create():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState.create_vacuum(d=2, cutoff=2)
        pq.Q(1) | pq.Create()
        pq.Q(1) | pq.Annihilate()

        pq.Q(1) | pq.Create()

        pq.Q(0, 1) | pq.B(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


def test_overflow_with_zero_norm_raises_RuntimeError():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=2)
        pq.Q(2) | pq.Number(1) * np.sqrt(2/5)
        pq.Q(1) | pq.Number(1) * np.sqrt(3/5)

        pq.Q(1, 2) | pq.Create()

    with pytest.raises(RuntimeError) as error:
        program.execute()

    assert error.value.args[0] == "The norm of the state is 0."


def test_creation_on_multiple_modes():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=3)
        pq.Q(2) | pq.Number(1) * np.sqrt(2/5)
        pq.Q(1) | pq.Number(1) * np.sqrt(3/5)

        pq.Q(1, 2) | pq.Create()

    program.execute()

    assert np.isclose(program.state.norm, 1)

    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 2/5, 0, 3/5, 0, 0, 0, 0, 0, 0
        ],
    )


def test_state_is_renormalized_after_overflow():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=2)
        pq.Q(2) | pq.Number(1) * np.sqrt(2/6)
        pq.Q(1) | pq.Number(1) * np.sqrt(3/6)
        pq.Q(2) | pq.Number(2) * np.sqrt(1/6)

        pq.Q(2) | pq.Create()

    program.execute()

    assert np.isclose(program.state.norm, 1)

    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0, 0, 0,
            0.4, 0.6, 0, 0, 0, 0
        ],
    )
