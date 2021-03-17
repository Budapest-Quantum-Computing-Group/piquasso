#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import pytest
import numpy as np

import piquasso as pq


@pytest.fixture
def generate_random_gaussian_state():
    def func(*, d):
        with pq.Program() as random_state_generator:
            pq.Q() | pq.GaussianState.create_vacuum(d=d)

            for i in range(d):
                alpha = 2 * (np.random.rand() + 1j * np.random.rand()) - 1
                displacement_gate = pq.D(alpha=alpha)

                r = 2 * np.random.rand() - 1
                theta = np.random.rand() * 2 * np.pi
                squeezing_gate = pq.S(r, theta=theta)

                pq.Q(i) | displacement_gate | squeezing_gate

        random_state_generator.execute()

        random_state = random_state_generator.state

        random_state.validate()

        return random_state

    return func


def test_squeezing(program, gaussian_state_assets):
    amp = -0.6
    theta = 0.7

    with program:
        pq.Q(1) | pq.S(amp, theta)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_phaseshift(program, gaussian_state_assets):
    with program:
        pq.Q(0) | pq.R(phi=np.pi/3)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_beamsplitter(program, gaussian_state_assets):
    theta = np.pi/4
    phi = np.pi/3

    with program:
        pq.Q(0, 1) | pq.B(theta=theta, phi=phi)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_with_alpha(program, gaussian_state_assets):
    alpha = 3 - 4j

    with program:
        pq.Q(1) | pq.D(alpha=alpha)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_with_r_and_phi(program, gaussian_state_assets):
    r = 5
    phi = np.pi / 4

    with program:
        pq.Q(1) | pq.D(r=r, phi=phi)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_position_displacement(program, gaussian_state_assets):
    x = 4

    with program:
        pq.Q(1) | pq.X(x)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_momentum_displacement(program, gaussian_state_assets):
    p = 5

    with program:
        pq.Q(1) | pq.Z(p)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement(program, gaussian_state_assets):
    alpha = 2 + 3j
    r = 4
    phi = np.pi/3

    with program:
        pq.Q(0) | pq.D(r=r, phi=phi)
        pq.Q(1) | pq.D(alpha=alpha)
        pq.Q(2) | pq.D(r=r, phi=phi) | pq.D(alpha=alpha)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_and_squeezing(program, gaussian_state_assets):
    r = 4
    phi = np.pi/3

    with program:
        pq.Q(0) | pq.D(r=r, phi=phi)
        pq.Q(1) | pq.S(amp=-0.6, theta=0.7)
        pq.Q(2) | pq.S(amp=0.8)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_two_mode_squeezing(program, gaussian_state_assets):
    r = 4
    phi = np.pi/3

    with program:
        pq.Q(1, 2) | pq.S2(r=r, phi=phi)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_controlled_X_gate(program, gaussian_state_assets):
    s = 2

    with program:
        pq.Q(1, 2) | pq.CX(s=s)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_controlled_Z_gate(program, gaussian_state_assets):
    s = 2

    with program:
        pq.Q(1, 2) | pq.CZ(s=s)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_fourier(program, gaussian_state_assets):
    with program:
        pq.Q(0) | pq.F()

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_mach_zehnder(program, gaussian_state_assets):
    int_ = np.pi/3
    ext = np.pi/4

    with program:
        pq.Q(1, 2) | pq.MZ(int_=int_, ext=ext)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_quadratic_phase(program, gaussian_state_assets):
    with program:
        pq.Q(0) | pq.P(0)
        pq.Q(1) | pq.P(2)
        pq.Q(2) | pq.P(1)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_leaves_the_covariance_invariant(program):
    r = 1
    phi = 1

    initial_cov = program.state.cov

    with program:
        pq.Q(0) | pq.D(r=r, phi=phi)

    program.execute()
    program.state.validate()

    final_cov = program.state.cov

    assert np.allclose(initial_cov, final_cov)


def test_passive_transform_for_1_modes(program, gaussian_state_assets):
    alpha = np.exp(1j * np.pi/3)

    T = np.array(
        [
            [alpha]
        ],
        dtype=complex,
    )

    with program:
        pq.Q(1) | pq.Interferometer(T)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_passive_transform_2_modes(program, gaussian_state_assets):
    theta = np.pi/4
    phi = np.pi/3

    random_unitary = np.array(
        [
            [  np.cos(theta),   np.sin(theta) * np.exp(1j * phi)],
            [- np.sin(theta) * np.exp(- 1j * phi), - np.cos(theta)],
        ],
        dtype=complex,
    )

    with program:
        pq.Q(0, 1) | pq.Interferometer(random_unitary)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_passive_transform_for_all_modes(program, gaussian_state_assets):
    random_unitary = np.array(
        [
            [-0.000299 - 0.248251j, -0.477889 - 0.502114j, -0.1605961 - 0.657331j],
            [-0.642609 - 0.628177j, -0.057561 - 0.133947j,  0.2663486 + 0.316625j],
            [ 0.355304 + 0.067664j, -0.703395 + 0.059033j,  0.5225116 + 0.312911j],
        ],
        dtype=complex,
    )

    with program:
        pq.Q(0, 1, 2) | pq.Interferometer(random_unitary)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_gaussian_transform_for_1_modes(program, gaussian_state_assets):
    alpha = np.exp(1j * np.pi/3)
    beta = np.exp(1j * np.pi/4)

    passive_transform = np.array(
        [
            [alpha]
        ],
        dtype=complex,
    )
    active_transform = np.array(
        [
            [beta]
        ],
        dtype=complex,
    )

    with program:
        pq.Q(1) | pq.GaussianTransform(P=passive_transform, A=active_transform)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_leaves_the_covariance_invariant_for_complex_program():
    with pq.Program() as initialization:
        pq.Q() | pq.GaussianState.create_vacuum(d=3)

        pq.Q(0) | pq.D(alpha=np.exp(1j * np.pi/4))
        pq.Q(1) | pq.D(alpha=1)
        pq.Q(2) | pq.D(alpha=1j)

        pq.Q(0) | pq.S(amp=1, theta=np.pi/2)
        pq.Q(1) | pq.S(amp=2, theta=np.pi/3)
        pq.Q(2) | pq.S(amp=2, theta=np.pi/4)

    initialization.execute()
    initialization.state.validate()

    initial_cov = initialization.state.cov

    with pq.Program() as program:
        pq.Q(0, 1, 2) | initialization.state

        pq.Q(0) | pq.D(r=1, phi=1)

    program.execute()
    program.state.validate()

    final_cov = program.state.cov

    assert np.allclose(initial_cov, final_cov)


def test_displaced_vacuum_stays_valid():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=3)
        pq.Q(0) | pq.D(r=2, phi=np.pi/3)

    program.execute()
    program.state.validate()


def test_multiple_displacements_leave_the_covariance_invariant():
    vacuum = pq.GaussianState.create_vacuum(d=3)

    initial_cov = vacuum.cov

    with pq.Program() as program:
        pq.Q() | vacuum

        pq.Q(0) | pq.D(r=2, phi=np.pi/3)
        pq.Q(1) | pq.D(r=1, phi=np.pi/4)
        pq.Q(2) | pq.D(r=1 / 2, phi=np.pi/6)

    program.execute()
    program.state.validate()

    assert np.allclose(program.state.cov, initial_cov)


def test_apply_passive(generate_random_gaussian_state, generate_unitary_matrix):
    state = generate_random_gaussian_state(d=5)

    expected_m = state.m.copy()
    expected_C = state.C.copy()
    expected_G = state.G.copy()

    T = generate_unitary_matrix(3)

    # TODO: The algorithm is re-implemented here, unfortunately, it should be removed.

    expected_m[(0, 1, 3), ] = T @ expected_m[(0, 1, 3), ]

    columns = np.array(
        [
            [0, 1, 3],
            [0, 1, 3],
            [0, 1, 3],
        ]
    )

    rows = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
        ]
    )

    index = rows, columns

    expected_C[index] = T.conjugate() @ expected_C[index] @ T.transpose()
    expected_C[(0, 1, 3), 2] = T.conjugate() @ expected_C[(0, 1, 3), 2]
    expected_C[(0, 1, 3), 4] = T.conjugate() @ expected_C[(0, 1, 3), 4]
    expected_C[:, (0, 1, 3)] = np.conj(expected_C[(0, 1, 3), :]).transpose()

    expected_G[index] = T @ expected_G[index] @ T.transpose()
    expected_G[(0, 1, 3), 2] = T @ expected_G[(0, 1, 3), 2]
    expected_G[(0, 1, 3), 4] = T @ expected_G[(0, 1, 3), 4]
    expected_G[:, (0, 1, 3)] = expected_G[(0, 1, 3), :].transpose()

    with pq.Program(state=state) as program:
        pq.Q(0, 1, 3) | pq.Interferometer(T)

    program.execute()
    program.state.validate()

    assert np.allclose(program.state.m, expected_m)
    assert np.allclose(program.state.C, expected_C)
    assert np.allclose(program.state.G, expected_G)


def test_complex_circuit(gaussian_state_assets):
    with pq.Program() as program:
        pq.Q() | pq.GaussianState.create_vacuum(d=5)

        pq.Q(0) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(1) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(2) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(3) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(4) | pq.S(amp=0.1) | pq.D(alpha=1)

        pq.Q(0, 1) | pq.B(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.B(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.B(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.B(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.B(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.B(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.B(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.B(3.340269832485504, 3.289367083610399)

        pq.Q(0) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(1) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(2) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(3) | pq.S(amp=0.1) | pq.D(alpha=1)
        pq.Q(4) | pq.S(amp=0.1) | pq.D(alpha=1)

        pq.Q(0, 1) | pq.B(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.B(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.B(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.B(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.B(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.B(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.B(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.B(3.340269832485504, 3.289367083610399)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state
