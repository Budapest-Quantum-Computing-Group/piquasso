#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq


def test_5050_beamsplitter():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=2, cutoff=2)
        pq.Q(1) | pq.Number(1)

        pq.Q(0, 1) | pq.B(theta=np.pi / 4, phi=np.pi / 3)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.5, 0.5, 0, 0, 0],
    )


def test_beamsplitter():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=2, cutoff=2)
        pq.Q(1) | 1 * pq.Number(1)

        pq.Q(0, 1) | pq.B(theta=np.pi / 5, phi=np.pi / 6)

    program.execute()

    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.6545085, 0.3454915, 0, 0, 0],
    )


def test_beamsplitter_multiple_particles():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=2, cutoff=2)
        pq.Q(1) | pq.Number(1) / 2
        pq.Q(1) | pq.Number(2) / 2
        pq.Q(0) | pq.Number(2) / np.sqrt(2)

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
        pq.Q() | pq.PureFockState(d=2, cutoff=2)
        pq.Q() | pq.Number(0, 0) / 2
        pq.Q() | pq.Number(0, 1) / np.sqrt(2)
        pq.Q() | pq.Number(0, 2) / 2

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
        pq.Q() | pq.PureFockState(d=3, cutoff=2)
        pq.Q(2) | pq.Number(1) * 1

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
        pq.Q() | pq.PureFockState(d=3, cutoff=2)
        pq.Q() | pq.Number(0, 0, 1) / 2
        pq.Q() | pq.Number(0, 0, 2) / 2
        pq.Q() | pq.Number(0, 1, 1) / np.sqrt(2)

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
        pq.Q() | pq.PureFockState(d=2, cutoff=2)
        pq.Q() | pq.Number(0, 1) / 2
        pq.Q() | pq.Number(0, 2) / np.sqrt(2)
        pq.Q() | pq.Number(1, 1) / 2

        pq.Q(0) | pq.R(phi=np.pi / 3)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.25, 0, 0.5, 0.25, 0],
    )


def test_fourier():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=2, cutoff=2)
        pq.Q() | pq.Number(0, 1) / 2
        pq.Q() | pq.Number(0, 2) / np.sqrt(2)
        pq.Q() | pq.Number(1, 1) / 2

        pq.Q(0) | pq.F()

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.25, 0, 0.5, 0.25, 0],
    )


def test_mach_zehnder():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=2, cutoff=2)
        pq.Q() | pq.Number(0, 1) / 2
        pq.Q() | pq.Number(0, 2) / np.sqrt(2)
        pq.Q() | pq.Number(1, 1) / 2

        pq.Q(0, 1) | pq.MZ(int_=np.pi/3, ext=np.pi/4)

    program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert np.allclose(
        program.state.fock_probabilities,
        [0, 0.0625, 0.1875, 0.04845345, 0.09690689, 0.60463966],
    )


def test_beamsplitters_and_phaseshifters_with_multiple_particles():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=2)
        pq.Q() | pq.Number(0, 0, 1) / 2
        pq.Q() | pq.Number(0, 0, 2) / 2
        pq.Q() | pq.Number(0, 1, 1) / np.sqrt(2)

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
        pq.Q() | pq.PureFockState(d=3, cutoff=2)
        pq.Q() | pq.Number(0, 0, 1) / 2
        pq.Q() | pq.Number(0, 0, 2) / 2
        pq.Q() | pq.Number(0, 1, 1) / np.sqrt(2)

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


def test_measure_particle_number_on_one_mode():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=2)

        pq.Q() | pq.Number(0, 1, 1) * np.sqrt(2/6)

        pq.Q(2) | pq.Number(1) * np.sqrt(1/6)
        pq.Q(2) | pq.Number(2) * np.sqrt(3/6)

        pq.Q(2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (1, ) or outcome == (2, )

    if outcome == (1, ):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                0.5773502691896258 * pq.Number(0, 0, 1),
                0.816496580927726 * pq.Number(0, 1, 1),
            ]
        )

    elif outcome == (2, ):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(0, 0, 2)
            ]
        )

    assert program.state == expected_state


def test_measure_particle_number_on_two_modes():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=2)

        pq.Q(1, 2) | pq.Number(1, 1) * np.sqrt(2/6)
        pq.Q(1, 2) | pq.Number(0, 1) * np.sqrt(1/6)
        pq.Q(1, 2) | pq.Number(0, 2) * np.sqrt(3/6)

        pq.Q(1, 2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (0, 1) or outcome == (1, 1) or outcome == (0, 2)

    if outcome == (0, 1):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(0, 0, 1)
            ]
        )

    elif outcome == (1, 1):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(0, 1, 1)
            ]
        )

    elif outcome == (0, 2):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(0, 0, 2)
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
        cutoff=1,
    )

    program = pq.Program(state=state)

    with program:
        pq.Q() | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (0, 0, 0) or outcome == (1, 0, 0) or outcome == (0, 0, 1)

    if outcome == (0, 0, 0):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=1,
            number_preparations=[
                pq.Number(0, 0, 0),
            ]
        )

    elif outcome == (0, 0, 1):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=1,
            number_preparations=[
                pq.Number(0, 0, 1),
            ]
        )

    elif outcome == (1, 0, 0):
        expected_state = pq.PureFockState.from_number_preparations(
            d=3, cutoff=1,
            number_preparations=[
                pq.Number(1, 0, 0),
            ]
        )

    assert program.state == expected_state


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


def test_kerr():
    xi = np.pi / 3

    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=3)
        pq.Q() | pq.Number(0, 2, 1)

        pq.Q(1) | pq.K(xi=xi)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 1

    assert np.isclose(nonzero_elements[0][0], - np.exp(1j * np.pi/3))
    assert nonzero_elements[0][1] == (0, 2, 1)


def test_cross_kerr():
    with pq.Program() as program:
        pq.Q() | pq.PureFockState(d=3, cutoff=3)
        pq.Q() | pq.Number(0, 2, 1)

        pq.Q(1, 2) | pq.CK(xi=np.pi / 2)

    program.execute()

    # TODO: Better way of presenting the resulting state.
    nonzero_elements = list(program.state.nonzero_elements)

    assert len(nonzero_elements) == 1

    assert np.isclose(nonzero_elements[0][0], -1)
    assert nonzero_elements[0][1] == (0, 2, 1)
