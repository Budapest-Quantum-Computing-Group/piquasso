#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import numpy as np

import piquasso as pq


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_on_one_mode(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(1, 0, 1)) * (- 3j)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (- 1j)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)) * (- 2j)

        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (1, ) or outcome == (2, )

    if outcome == (1, ):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                1/3 * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
                4j * pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)),
                -2j * pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)),
                -4j * pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)),
                2 / 3 * pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)),
                2j * pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)),
            ]
        )

    elif outcome == (2, ):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))
            ]
        )

    assert program.state == expected_state


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_on_two_modes(StateClass):
    with pq.Program() as program:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(1, 0, 1)) * (- 3j)

        pq.Q() | pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (- 1j)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 0, 1)) * (- 2j)

        pq.Q() | pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(1, 2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (0, 1) or outcome == (1, 1) or outcome == (0, 2)

    if outcome == (0, 1):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
                pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 1)) * (-6j),
                pq.DensityMatrix(ket=(1, 0, 1), bra=(0, 0, 1)) * 6j,
            ]
        )

    elif outcome == (1, 1):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 1, 1), bra=(0, 1, 1)),
            ]
        )

    elif outcome == (0, 2):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 2), bra=(0, 0, 2))
            ]
        )

    assert program.state == expected_state


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_on_all_modes(StateClass):
    with pq.Program() as preparation:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 0)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(0, 0, 1)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (0, 0, 0) or outcome == (0, 0, 1) or outcome == (1, 0, 0)

    if outcome == (0, 0, 0):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)),
            ]
        )

    elif outcome == (0, 0, 1):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)),
            ]
        )

    elif outcome == (1, 0, 0):
        expected_state = StateClass.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)),
            ]
        )

    assert program.state == expected_state


@pytest.mark.parametrize("StateClass", [pq.FockState, pq.PNCFockState])
def test_measure_particle_number_with_multiple_shots(StateClass):
    shots = 4

    with pq.Program() as preparation:
        pq.Q() | StateClass(d=3, cutoff=3)

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 0), bra=(0, 0, 0)) / 4

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(1, 0, 0)) / 2

        pq.Q() | pq.DensityMatrix(ket=(0, 0, 1), bra=(1, 0, 0)) * np.sqrt(1/8)
        pq.Q() | pq.DensityMatrix(ket=(1, 0, 0), bra=(0, 0, 1)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.MeasureParticleNumber(shots=shots)

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == shots
