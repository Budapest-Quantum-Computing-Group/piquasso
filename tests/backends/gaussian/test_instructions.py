#
# Copyright 2021-2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np

import piquasso as pq

from scipy.linalg import polar, sinhm, coshm, expm


@pytest.fixture
def generate_random_gaussian_state():
    def func(*, d):
        with pq.Program() as random_state_generator:
            for i in range(d):
                alpha = 2 * (np.random.rand() + 1j * np.random.rand()) - 1
                displacement_gate = pq.Displacement(
                    r=np.abs(alpha), phi=np.angle(alpha)
                )

                r = 2 * np.random.rand() - 1
                phi = np.random.rand() * 2 * np.pi
                squeezing_gate = pq.Squeezing(r, phi=phi)

                pq.Q(i) | displacement_gate | squeezing_gate

        simulator = pq.GaussianSimulator(d=d)
        state = simulator.execute(random_state_generator).state
        state.validate()

        return state

    return func


def test_squeezing(state, gaussian_state_assets):
    r = -0.6
    phi = 0.7

    with pq.Program() as program:
        pq.Q(1) | pq.Squeezing(r, phi)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_phaseshift(state, gaussian_state_assets):
    with pq.Program() as program:
        pq.Q(0) | pq.Phaseshifter(phi=np.pi / 3)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_phaseshift_modes_are_shifted_from_original(state):
    original = state.copy()

    angle = np.pi / 3

    with pq.Program() as program:
        pq.Q(0) | pq.Phaseshifter(phi=angle)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    assert state.reduced((0,)) == original.rotated(-angle).reduced((0,))
    assert state.reduced((1, 2)) == original.reduced((1, 2))


def test_beamsplitter(state, gaussian_state_assets):
    theta = np.pi / 4
    phi = 2 * np.pi / 3

    with pq.Program() as program:
        pq.Q(0, 1) | pq.Beamsplitter(theta=theta, phi=phi)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_displacement_with_alpha(state, gaussian_state_assets):
    alpha = 3 - 4j

    with pq.Program() as program:
        pq.Q(1) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_displacement_with_r_and_phi(state, gaussian_state_assets):
    r = 5
    phi = np.pi / 4

    with pq.Program() as program:
        pq.Q(1) | pq.Displacement(r=r, phi=phi)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_displacement_on_multiple_modes(state, gaussian_state_assets):
    alpha = 3 - 4j

    with pq.Program() as program:
        for i in [0, 1]:
            pq.Q(i) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_displacement_on_all_modes(state, gaussian_state_assets):
    alpha = 3 - 4j

    with pq.Program() as program:
        for i in range(state.d):
            pq.Q(i) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_position_displacement(state, gaussian_state_assets):
    x = 2

    with pq.Program() as program:
        pq.Q(1) | pq.PositionDisplacement(x)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_momentum_displacement(state, gaussian_state_assets):
    p = 2.5

    with pq.Program() as program:
        pq.Q(1) | pq.MomentumDisplacement(p)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_displacement(state, gaussian_state_assets):
    alpha = 2 + 3j
    r = 4
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=r, phi=phi)
        pq.Q(1) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))
        pq.Q(2) | pq.Displacement(r=r, phi=phi) | pq.Displacement(
            r=np.abs(alpha), phi=np.angle(alpha)
        )

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_displacement_and_squeezing(state, gaussian_state_assets):
    r = 4
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=r, phi=phi)
        pq.Q(1) | pq.Squeezing(r=-0.6, phi=0.7)
        pq.Q(2) | pq.Squeezing(r=0.8)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_two_mode_squeezing(state, gaussian_state_assets):
    r = 4
    phi = np.pi / 3

    with pq.Program() as program:
        pq.Q(1, 2) | pq.Squeezing2(r=r, phi=phi)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_controlled_X_gate(state, gaussian_state_assets):
    s = 2

    with pq.Program() as program:
        pq.Q(1, 2) | pq.ControlledX(s=s)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_controlled_Z_gate(state, gaussian_state_assets):
    s = 2

    with pq.Program() as program:
        pq.Q(1, 2) | pq.ControlledZ(s=s)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_fourier(state, gaussian_state_assets):
    with pq.Program() as program:
        pq.Q(0) | pq.Fourier()

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_mach_zehnder(state, gaussian_state_assets):
    int_ = np.pi / 3
    ext = np.pi / 4

    with pq.Program() as program:
        pq.Q(1, 2) | pq.MachZehnder(int_=int_, ext=ext)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_quadratic_phase(state, gaussian_state_assets):
    with pq.Program() as program:
        pq.Q(0) | pq.QuadraticPhase(0)
        pq.Q(1) | pq.QuadraticPhase(2)
        pq.Q(2) | pq.QuadraticPhase(1)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_displacement_leaves_the_covariance_invariant(state):
    r = 1
    phi = 1

    initial_covariance_matrix = state.xpxp_covariance_matrix

    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=r, phi=phi)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    final_covariance_matrix = state.xpxp_covariance_matrix

    assert np.allclose(initial_covariance_matrix, final_covariance_matrix)


def test_interferometer_for_1_modes(state, gaussian_state_assets):
    alpha = np.exp(1j * np.pi / 3)

    T = np.array(
        [[alpha]],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q(1) | pq.Interferometer(T)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_interferometer_for_2_modes(state, gaussian_state_assets):
    theta = np.pi / 4
    phi = np.pi / 3

    random_unitary = np.array(
        [
            [np.cos(theta), np.sin(theta) * np.exp(1j * phi)],
            [-np.sin(theta) * np.exp(-1j * phi), -np.cos(theta)],
        ],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q(0, 1) | pq.Interferometer(random_unitary)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_interferometer_for_all_modes(state, gaussian_state_assets):
    self_adjoint = np.array(
        [
            [1, 2j, 3 + 4j],
            [-2j, 2, 5],
            [3 - 4j, 5, 6],
        ],
        dtype=complex,
    )
    unitary = expm(1j * self_adjoint)

    with pq.Program() as program:
        pq.Q(0, 1, 2) | pq.Interferometer(unitary)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_GaussianTransform_for_1_modes(state, gaussian_state_assets):
    r = 0.4

    alpha = np.exp(1j * np.pi / 3)
    beta = np.exp(1j * np.pi / 4)

    passive = np.array(
        [[alpha * np.cosh(r)]],
        dtype=complex,
    )
    active = np.array(
        [[beta * np.sinh(r)]],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q(1) | pq.GaussianTransform(passive=passive, active=active)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_GaussianTransform_with_general_squeezing_matrix():
    d = 3

    squeezing_matrix = np.array(
        [
            [0.1, 0.2j, 0.3],
            [0.2j, 0.2, 0.4],
            [0.3, 0.4, 0.1j],
        ],
        dtype=complex,
    )
    U, r = polar(squeezing_matrix)

    theta = np.pi / 3

    global_phase = np.array(
        [
            [np.cos(theta), -np.sin(theta) * np.exp(1j * np.pi / 4), 0],
            [np.sin(theta) * np.exp(-1j * np.pi / 4), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    passive = global_phase @ coshm(r)
    active = global_phase @ sinhm(r) @ U.conj()

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(all) | pq.GaussianTransform(passive=passive, active=active)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state

    state.validate()

    probabilities = state.fock_probabilities

    assert all(probability >= 0 for probability in probabilities)

    assert sum(probabilities) <= 1.0 or np.isclose(sum(probabilities), 1.0)


def test_GaussianTransform_raises_InvalidParameter_for_nonsymplectic_matrix():
    d = 3

    zero_matrix = np.zeros(shape=(d,) * 2)

    with pq.Program():
        pq.Q() | pq.Vacuum()

        with pytest.raises(pq.api.exceptions.InvalidParameter):
            pq.Q(all) | pq.GaussianTransform(passive=zero_matrix, active=zero_matrix)


def test_graph_embedding(state):
    adjacency_matrix = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.Graph(adjacency_matrix)

    simulator = pq.GaussianSimulator(d=state.d)
    state = simulator.execute(program).state
    state.validate()


def test_displacement_leaves_the_covariance_invariant_for_complex_program():
    d = 3

    alphas = [np.exp(1j * np.pi / 4), 1, 1j]

    with pq.Program() as initialization:
        for i, alpha in enumerate(alphas):
            pq.Q(i) | pq.Displacement(r=np.abs(alpha), phi=np.angle(alpha))

        pq.Q(0) | pq.Squeezing(r=1, phi=np.pi / 2)
        pq.Q(1) | pq.Squeezing(r=2, phi=np.pi / 3)
        pq.Q(2) | pq.Squeezing(r=2, phi=np.pi / 4)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(initialization).state
    state.validate()

    initial_covariance_matrix = state.xpxp_covariance_matrix

    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=1, phi=1)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()

    final_covariance_matrix = state.xpxp_covariance_matrix

    assert np.allclose(initial_covariance_matrix, final_covariance_matrix)


def test_displaced_vacuum_stays_valid():
    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=2, phi=np.pi / 3)

    simulator = pq.GaussianSimulator(d=3)
    state = simulator.execute(program).state
    state.validate()


def test_multiple_displacements_leave_the_covariance_invariant():
    d = 3

    simulator = pq.GaussianSimulator(d=d)

    vacuum = simulator.create_initial_state()

    initial_covariance_matrix = vacuum.xpxp_covariance_matrix

    with pq.Program() as program:
        pq.Q(0) | pq.Displacement(r=2, phi=np.pi / 3)
        pq.Q(1) | pq.Displacement(r=1, phi=np.pi / 4)
        pq.Q(2) | pq.Displacement(r=1, phi=np.pi / 6)

    state = simulator.execute(program).state
    state.validate()

    assert np.allclose(state.xpxp_covariance_matrix, initial_covariance_matrix)


@pytest.mark.monkey
def test_random_interferometer(generate_random_gaussian_state, generate_unitary_matrix):
    d = 5

    state = generate_random_gaussian_state(d=d)

    T = generate_unitary_matrix(3)

    with pq.Program() as program:
        pq.Q(0, 1, 3) | pq.Interferometer(T)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program, initial_state=state).state
    state.validate()


def test_complex_circuit(gaussian_state_assets):
    d = 5

    with pq.Program() as program:
        for i in range(5):
            pq.Q(i) | pq.Squeezing(r=0.1) | pq.Displacement(r=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

        for i in range(5):
            pq.Q(i) | pq.Squeezing(r=0.1) | pq.Displacement(r=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

    simulator = pq.GaussianSimulator(d=d)
    state = simulator.execute(program).state
    state.validate()

    expected_state = gaussian_state_assets.load()
    assert state == expected_state


def test_program_stacking_with_measurement():
    with pq.Program() as preparation:
        pq.Q(0, 1) | pq.Squeezing2(r=1, phi=np.pi / 4)
        pq.Q(2, 3) | pq.Squeezing2(r=2, phi=np.pi / 3)

    with pq.Program() as interferometer:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4, phi=np.pi / 3)
        pq.Q(1) | pq.Phaseshifter(phi=np.pi / 2)
        pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 6)

    with pq.Program() as program:
        pq.Q(0, 1, 2, 3, 4) | preparation

        pq.Q(0, 1, 2) | interferometer
        pq.Q(2, 3, 4) | interferometer

        pq.Q(3) | pq.HeterodyneMeasurement()

    simulator = pq.GaussianSimulator(d=5)
    state = simulator.execute(program).state
    state.validate()


def test_complex_one_mode_scenario():
    with pq.Program() as program:
        pq.Q(0) | pq.Squeezing(r=np.log(2))
        pq.Q(0) | pq.Displacement(r=1)
        pq.Q(0) | pq.Phaseshifter(np.pi / 4)

    simulator = pq.GaussianSimulator(d=1, config=pq.Config(cutoff=4))
    state = simulator.execute(program).state

    assert np.allclose(
        state.fock_probabilities,
        [
            0.1615172143957243,
            0.41348406885305417,
            0.31024226541130745,
            0.03980473302825398,
        ],
    )
