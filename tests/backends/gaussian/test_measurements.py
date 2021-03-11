#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

import piquasso as pq

from piquasso._math.linalg import is_selfadjoint, is_symmetric


def test_measure_homodyne():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=3)

        pq.Q(0) | pq.D(r=2, phi=np.pi/3)
        pq.Q(1) | pq.D(r=1, phi=np.pi/4)
        pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.S(amp=1 / 2, theta=np.pi/4)
        pq.Q(2) | pq.S(amp=3 / 4)

        pq.Q(0) | pq.MeasureHomodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].measurement.modes == (0, )

    assert is_symmetric(program.state.G)
    assert is_selfadjoint(program.state.C)


def test_measure_homodyne_on_multiple_modes():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=3)

        pq.Q(0) | pq.D(r=2, phi=np.pi/3)
        pq.Q(1) | pq.D(r=1, phi=np.pi/4)
        pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.S(amp=1 / 2, theta=np.pi/4)
        pq.Q(2) | pq.S(amp=3 / 4)

        pq.Q(0, 1) | pq.MeasureHomodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].measurement.modes == (0, 1)

    assert is_symmetric(program.state.G)
    assert is_selfadjoint(program.state.C)


def test_measure_heterodyne():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=3)

        pq.Q(0) | pq.D(r=2, phi=np.pi/3)
        pq.Q(1) | pq.D(r=1, phi=np.pi/4)
        pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.S(amp=1 / 2, theta=np.pi/4)
        pq.Q(2) | pq.S(amp=3 / 4)

        pq.Q(0) | pq.MeasureHeterodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].measurement.modes == (0, )

    assert is_symmetric(program.state.G)
    assert is_selfadjoint(program.state.C)


def test_measure_heterodyne_on_multiple_modes():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=3)

        pq.Q(0) | pq.D(r=2, phi=np.pi/3)
        pq.Q(1) | pq.D(r=1, phi=np.pi/4)
        pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.S(amp=1 / 2, theta=np.pi/4)
        pq.Q(2) | pq.S(amp=3 / 4)

        pq.Q(0, 1) | pq.MeasureHeterodyne()

    results = program.execute()

    assert len(results) == 1
    assert results[0].measurement.modes == (0, 1)

    assert is_symmetric(program.state.G)
    assert is_selfadjoint(program.state.C)


def test_measure_dyne():
    detection_covariance = np.array(
        [
            [1, 2],
            [2, 1],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=3)

        pq.Q(0) | pq.D(r=2, phi=np.pi/3)
        pq.Q(1) | pq.D(r=1, phi=np.pi/4)
        pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

        pq.Q(1) | pq.S(amp=1 / 2, theta=np.pi/4)
        pq.Q(2) | pq.S(amp=3 / 4)

        pq.Q(0) | pq.MeasureDyne(detection_covariance)

    results = program.execute()

    assert len(results) == 1
    assert results[0].measurement.modes == (0, )

    assert is_symmetric(program.state.G)
    assert is_selfadjoint(program.state.C)
