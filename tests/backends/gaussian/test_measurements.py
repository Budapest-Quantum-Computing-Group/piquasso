#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest

import numpy as np

import piquasso as pq


@pytest.fixture
def program():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(0) | pq.D(r=2, phi=np.pi/3)
        pq.Q(1) | pq.D(r=1, phi=np.pi/4)
        pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.S(r=1 / 2, phi=np.pi/4)
        pq.Q(2) | pq.S(r=3 / 4)

    return program


def test_measure_homodyne(program):
    with program:
        pq.Q(0) | pq.MeasureHomodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].operation.modes == (0, )

    program.state.validate()


def test_measure_homodyne_on_multiple_modes(program):
    with program:
        pq.Q(0, 1) | pq.MeasureHomodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].operation.modes == (0, 1)

    program.state.validate()


def test_measure_homodyne_with_multiple_shots(program):
    shots = 4

    with program:
        pq.Q(0, 1) | pq.MeasureHomodyne(shots=shots)

    results = program.execute()

    assert len(results) == shots


def test_measure_heterodyne(program):
    with program:
        pq.Q(0) | pq.MeasureHeterodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].operation.modes == (0, )

    program.state.validate()


def test_measure_heterodyne_on_multiple_modes(program):
    with program:
        pq.Q(0, 1) | pq.MeasureHeterodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].operation.modes == (0, 1)

    program.state.validate()


def test_measure_heterodyne_with_multiple_shots(program):
    shots = 4

    with program:
        pq.Q(0, 1) | pq.MeasureHeterodyne(shots=shots)

    results = program.execute()

    assert len(results) == shots


def test_measure_dyne(program):
    detection_covariance = np.array(
        [
            [1, 2],
            [2, 1],
        ]
    )

    with program:
        pq.Q(0) | pq.MeasureDyne(detection_covariance)

    results = program.execute()

    assert len(results) == 1
    assert results[0].operation.modes == (0, )

    program.state.validate()


def test_measure_dyne_with_multiple_shots(program):
    shots = 4

    detection_covariance = np.array(
        [
            [1, 2],
            [2, 1],
        ]
    )

    with program:
        pq.Q(0) | pq.MeasureDyne(detection_covariance, shots=shots)

    results = program.execute()

    assert len(results) == shots


def test_measure_particle_number_on_one_modes(program):
    with program:
        pq.Q(0) | pq.MeasureParticleNumber(cutoff=3)

    results = program.execute()

    assert results


def test_measure_particle_number_on_two_modes(program):
    with program:
        pq.Q(0, 1) | pq.MeasureParticleNumber(cutoff=3)

    results = program.execute()

    assert results


def test_measure_particle_number_on_all_modes(program):
    with program:
        pq.Q() | pq.MeasureParticleNumber(cutoff=3)

    results = program.execute()

    assert results
