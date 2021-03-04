#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

import piquasso as pq


def test_5050_beamsplitter():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [1, 0],
                [0, 0, 0],
            ],
            d=2,
            cutoff=2,
        )
        pq.Q(0, 1) | pq.B(theta=np.pi / 4, phi=np.pi / 3)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.5, 0.5, 0, 0, 0],
    )


def test_beamsplitter():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [1, 0],
                [0, 0, 0],
            ],
            d=2,
            cutoff=2,
        )

        pq.Q(0, 1) | pq.B(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


def test_beamsplitter_multiple_particles():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [0.5, 0],
                [0.5, 0, np.sqrt(1/2)],
            ],
            d=2,
            cutoff=2,
        )

        pq.Q(0, 1) | pq.B(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.16362712, 0.08637288,
            0.24672554, 0.17929466, 0.32397979
        ],
    )


def test_beamsplitter_leaves_vacuum_unchanged():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0.5],
                [np.sqrt(1/2), 0],
                [0.5, 0, 0],
            ],
            d=2,
            cutoff=2,
        )

        pq.Q(0, 1) | pq.B(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0.25,
            0.32725425, 0.17274575,
            0.10709534, 0.11306356, 0.02984109
        ],
    )


def test_multiple_beamsplitters():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            d=3,
            cutoff=2,
        )

        pq.Q(0, 1) | pq.B(theta=np.pi / 4, phi=np.pi / 5)
        pq.Q(1, 2) | pq.B(theta=np.pi / 6, phi=1.5 * np.pi)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.5, 0.375, 0.125,
            0, 0, 0, 0, 0, 0
        ],
    )


def test_multiple_beamsplitters_with_multiple_particles():
    with pq.Program() as program:

        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [0.5, 0, 0],
                [0.5, np.sqrt(1/2), 0, 0, 0, 0],
            ],
            d=3,
            cutoff=2,
        )

        pq.Q(0, 1) | pq.B(theta=np.pi / 4, phi=np.pi / 5)
        pq.Q(1, 2) | pq.B(theta=np.pi / 6, phi=1.5 * np.pi)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.125, 0.09375, 0.03125,
            0.234375, 0.15625, 0.1875, 0.109375, 0.0625, 0
        ],
    )


def test_phaseshift():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [0.5, 0],
                [np.sqrt(1/2), 0.5, 0],
            ],
            d=2,
            cutoff=2,
        )

        pq.Q(0) | pq.R(phi=np.pi / 3)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.25, 0, 0.5, 0.25, 0],
    )


def test_fourier():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [0.5, 0],
                [np.sqrt(1/2), 0.5, 0],
            ],
            d=2,
            cutoff=2,
        )

        pq.Q(0) | pq.F()

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.25, 0, 0.5, 0.25, 0],
    )


def test_mach_zehnder():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [0.5, 0],
                [np.sqrt(1/2), 0.5, 0],
            ],
            d=2,
            cutoff=2,
        )

        pq.Q(0, 1) | pq.MZ(int_=np.pi/3, ext=np.pi/4)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.0625, 0.1875, 0.04845345, 0.09690689, 0.60463966],
    )


def test_beamsplitters_and_phaseshifters_with_multiple_particles():
    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [0.5, 0, 0],
                [0.5, np.sqrt(1/2), 0, 0, 0, 0],
            ],
            d=3,
            cutoff=2,
        )

        pq.Q(0) | pq.R(phi=np.pi/3)
        pq.Q(1) | pq.R(phi=np.pi/3)

        pq.Q(0, 1) | pq.B(theta=np.pi / 4, phi=np.pi / 5)
        pq.Q(1, 2) | pq.B(theta=np.pi / 6, phi=1.5 * np.pi)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.125, 0.09375, 0.03125,
            0.43324878, 0.02366748, 0.1875, 0.04308374, 0.0625, 0
        ],
    )


def test_passive_transform():

    T = np.array(
        [
            [0.5, 0.53033009 + 0.53033009j, 0.21650635 + 0.375j],
            [-0.61237244 + 0.61237244j,  0.4330127, 0.24148146 + 0.06470476j],
            [0, -0.48296291 + 0.12940952j, 0.8660254]
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.PNCFockState.from_pure(
            coefficients=[
                [0],
                [0.5, 0, 0],
                [0.5, np.sqrt(1/2), 0, 0, 0, 0],
            ],
            d=3,
            cutoff=2,
        )

        pq.Q(0, 1, 2) | pq.PassiveTransform(T=T)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [
            0,
            0.0625, 0.1875, 0,
            0.01443139, 0.10696977, 0.32090931, 0.0192306, 0.11538358, 0.17307537
        ],
    )
