#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_create_number_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=2) | pq.Vacuum()

        pq.Q(1) | pq.Create()

        pq.Q(0, 1) | pq.B(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_create_and_annihilate_number_state(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=2) | pq.Vacuum()

        pq.Q(1) | pq.Create()
        pq.Q(1) | pq.Annihilate()

    program.execute()

    assert np.isclose(program.state.norm, 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [1, 0, 0, 0, 0, 0],
    )


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_create_annihilate_and_create(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=2, cutoff=2) | pq.Vacuum()

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


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_overflow_with_zero_norm_raises_RuntimeError(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=2)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 2/5
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 0), bra=(0, 1, 0)) * 3/5

        pq.Q(1, 2) | pq.Create()

    with pytest.raises(RuntimeError) as error:
        program.execute()

    assert error.value.args[0] == "The norm of the state is 0."


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_creation_on_multiple_modes(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 2/5
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 0), bra=(0, 1, 0)) * 3/5

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


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_state_is_renormalized_after_overflow(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=2)

        pq.Q() | (2/6) * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1))
        pq.Q() | (3/6) * pq.DensityMatrix(ket=(0, 1, 0), bra=(0, 1, 0))
        pq.Q() | (1/6) * pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))

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