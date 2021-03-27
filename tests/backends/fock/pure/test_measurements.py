#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

import piquasso as pq


def test_measure_particle_number_on_one_mode():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=3)

        pq.Q() | pq.StateVector(0, 1, 1) * np.sqrt(2/6)

        pq.Q(2) | pq.StateVector(1) * np.sqrt(1/6)
        pq.Q(2) | pq.StateVector(2) * np.sqrt(3/6)

        pq.Q(2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    sample = results[0].samples[0]
    assert sample == (1, ) or sample == (2, )

    if sample == (1, ):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                0.5773502691896258 * pq.StateVector(0, 0, 1),
                0.816496580927726 * pq.StateVector(0, 1, 1),
            ]
        )

    elif sample == (2, ):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.StateVector(0, 0, 2)
            ]
        )

    assert program.state == expected_state


def test_measure_particle_number_on_two_modes():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=3)

        pq.Q(1, 2) | pq.StateVector(1, 1) * np.sqrt(2/6)
        pq.Q(1, 2) | pq.StateVector(0, 1) * np.sqrt(1/6)
        pq.Q(1, 2) | pq.StateVector(0, 2) * np.sqrt(3/6)

        pq.Q(1, 2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    sample = results[0].samples[0]
    assert sample == (0, 1) or sample == (1, 1) or sample == (0, 2)

    if sample == (0, 1):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.StateVector(0, 0, 1)
            ]
        )

    elif sample == (1, 1):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.StateVector(0, 1, 1)
            ]
        )

    elif sample == (0, 2):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=3,
            number_preparations=[
                pq.StateVector(0, 0, 2)
            ]
        )

    assert program.state == expected_state


def test_measure_particle_number_on_all_modes():
    state = pq.PureFockState(
        state_vector=[
            0.5,
            0.5, 0, np.sqrt(1/2),
        ],
        d=3,
        cutoff=2,
    )

    program = pq.Program(state=state)

    with program:
        pq.Q() | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    sample = results[0].samples[0]
    assert sample == (0, 0, 0) or sample == (1, 0, 0) or sample == (0, 0, 1)

    if sample == (0, 0, 0):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.StateVector(0, 0, 0),
            ]
        )

    elif sample == (0, 0, 1):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.StateVector(0, 0, 1),
            ]
        )

    elif sample == (1, 0, 0):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.StateVector(1, 0, 0),
            ]
        )

    assert program.state == expected_state


def test_measure_particle_number_with_multiple_shots():
    shots = 4

    state = pq.PureFockState(
        state_vector=[
            0.5,
            0.5, 0, np.sqrt(1/2),
        ],
        d=3,
        cutoff=2,
    )

    program = pq.Program(state=state)

    with program:
        pq.Q() | pq.MeasureParticleNumber(shots=shots)

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results[0].samples) == shots
