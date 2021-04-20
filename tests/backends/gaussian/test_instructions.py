#
# Copyright 2021 Budapest Quantum Computing Group
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


@pytest.fixture
def generate_random_gaussian_state():
    def func(*, d):
        with pq.Program() as random_state_generator:
            pq.Q() | pq.GaussianState(d=d)

            for i in range(d):
                alpha = 2 * (np.random.rand() + 1j * np.random.rand()) - 1
                displacement_gate = pq.Displacement(alpha=alpha)

                r = 2 * np.random.rand() - 1
                phi = np.random.rand() * 2 * np.pi
                squeezing_gate = pq.Squeezing(r, phi=phi)

                pq.Q(i) | displacement_gate | squeezing_gate

        random_state_generator.execute()

        random_state = random_state_generator.state

        random_state.validate()

        return random_state

    return func


def test_squeezing(program, gaussian_state_assets):
    r = -0.6
    phi = 0.7

    with program:
        pq.Q(1) | pq.Squeezing(r, phi)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_phaseshift(program, gaussian_state_assets):
    with program:
        pq.Q(0) | pq.Phaseshifter(phi=np.pi/3)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_phaseshift_on_multiple_modes(program):
    with program.copy() as separate_instructions:
        pq.Q(0) | pq.Phaseshifter(phi=np.pi / 3)
        pq.Q(1) | pq.Phaseshifter(phi=np.pi / 3)

    with program as single_instruction:
        pq.Q(0, 1) | pq.Phaseshifter(phi=np.pi/3)

    separate_instructions.execute()
    separate_instructions.state.validate()

    single_instruction.execute()
    single_instruction.state.validate()

    assert np.allclose(separate_instructions.state.mean, single_instruction.state.mean)


def test_phaseshift_modes_are_shifted_from_original(program):
    original = program.copy()

    angle = np.pi/3

    with program:
        pq.Q(0) | pq.Phaseshifter(phi=angle)

    program.execute()
    program.state.validate()

    assert program.state.reduced((0,)) == original.state.rotated(-angle).reduced((0,))
    assert program.state.reduced((1, 2)) == original.state.reduced((1, 2))


def test_beamsplitter(program, gaussian_state_assets):
    theta = np.pi/4
    phi = np.pi/3

    with program:
        pq.Q(0, 1) | pq.Beamsplitter(theta=theta, phi=phi)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_with_alpha(program, gaussian_state_assets):
    alpha = 3 - 4j

    with program:
        pq.Q(1) | pq.Displacement(alpha=alpha)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_with_r_and_phi(program, gaussian_state_assets):
    r = 5
    phi = np.pi / 4

    with program:
        pq.Q(1) | pq.Displacement(r=r, phi=phi)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_on_multiple_modes(program, gaussian_state_assets):
    alpha = 3 - 4j

    with program:
        pq.Q(0, 1) | pq.Displacement(alpha=alpha)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_on_all_modes(program, gaussian_state_assets):
    alpha = 3 - 4j

    with program:
        pq.Q() | pq.Displacement(alpha=alpha)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_position_displacement(program, gaussian_state_assets):
    x = 4

    with program:
        pq.Q(1) | pq.PositionDisplacement(x)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_momentum_displacement(program, gaussian_state_assets):
    p = 5

    with program:
        pq.Q(1) | pq.MomentumDisplacement(p)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement(program, gaussian_state_assets):
    alpha = 2 + 3j
    r = 4
    phi = np.pi/3

    with program:
        pq.Q(0) | pq.Displacement(r=r, phi=phi)
        pq.Q(1) | pq.Displacement(alpha=alpha)
        pq.Q(2) | pq.Displacement(r=r, phi=phi) | pq.Displacement(alpha=alpha)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_and_squeezing(program, gaussian_state_assets):
    r = 4
    phi = np.pi/3

    with program:
        pq.Q(0) | pq.Displacement(r=r, phi=phi)
        pq.Q(1) | pq.Squeezing(r=-0.6, phi=0.7)
        pq.Q(2) | pq.Squeezing(r=0.8)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_two_mode_squeezing(program, gaussian_state_assets):
    r = 4
    phi = np.pi/3

    with program:
        pq.Q(1, 2) | pq.Squeezing2(r=r, phi=phi)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_controlled_X_gate(program, gaussian_state_assets):
    s = 2

    with program:
        pq.Q(1, 2) | pq.ControlledX(s=s)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_controlled_Z_gate(program, gaussian_state_assets):
    s = 2

    with program:
        pq.Q(1, 2) | pq.ControlledZ(s=s)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_fourier(program, gaussian_state_assets):
    with program:
        pq.Q(0) | pq.Fourier()

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_mach_zehnder(program, gaussian_state_assets):
    int_ = np.pi/3
    ext = np.pi/4

    with program:
        pq.Q(1, 2) | pq.MachZehnder(int_=int_, ext=ext)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_quadratic_phase(program, gaussian_state_assets):
    with program:
        pq.Q(0) | pq.QuadraticPhase(0)
        pq.Q(1) | pq.QuadraticPhase(2)
        pq.Q(2) | pq.QuadraticPhase(1)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_displacement_leaves_the_covariance_invariant(program):
    r = 1
    phi = 1

    initial_cov = program.state.cov

    with program:
        pq.Q(0) | pq.Displacement(r=r, phi=phi)

    program.execute()
    program.state.validate()

    final_cov = program.state.cov

    assert np.allclose(initial_cov, final_cov)


def test_interferometer_for_1_modes(program, gaussian_state_assets):
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


def test_interferometer_for_2_modes(program, gaussian_state_assets):
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


def test_interferometer_for_all_modes(program, gaussian_state_assets):
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


def test_graph_embedding(program, gaussian_state_assets):
    adjacency_matrix = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )

    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=3)
        pq.Q() | pq.Graph(adjacency_matrix)

    program.execute()
    program.state.validate()


def test_generate_subgraphs_from_adjacency_matrix(program, gaussian_state_assets):
    adjacency_matrix = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    shots = 2

    pq.constants.seed(40)

    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=3)
        pq.Q() | pq.Graph(adjacency_matrix)

        pq.Q() | pq.ParticleNumberMeasurement(shots=shots)

    results = program.execute()
    program.state.validate()

    subgraphs = results[0].to_subgraph_nodes()

    assert len(subgraphs) == shots
    assert subgraphs == [[1, 2], [1, 2]]

    pq.constants.seed()  # Teardown! NOTE: this deserves a fixture.


def test_displacement_leaves_the_covariance_invariant_for_complex_program():
    with pq.Program() as initialization:
        pq.Q() | pq.GaussianState(d=3)

        pq.Q(all) | pq.Displacement(alpha=[np.exp(1j * np.pi/4), 1, 1j])

        pq.Q(all) | pq.Squeezing(r=[1, 2, 2], phi=[np.pi/2, np.pi/3, np.pi/4])

    initialization.execute()
    initialization.state.validate()

    initial_cov = initialization.state.cov

    with pq.Program() as program:
        pq.Q(0, 1, 2) | initialization.state

        pq.Q(0) | pq.Displacement(r=1, phi=1)

    program.execute()
    program.state.validate()

    final_cov = program.state.cov

    assert np.allclose(initial_cov, final_cov)


def test_displaced_vacuum_stays_valid():
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=3)
        pq.Q(0) | pq.Displacement(r=2, phi=np.pi/3)

    program.execute()
    program.state.validate()


def test_multiple_displacements_leave_the_covariance_invariant():
    vacuum = pq.GaussianState(d=3)

    initial_cov = vacuum.cov

    with pq.Program() as program:
        pq.Q() | vacuum

        pq.Q(all) | pq.Displacement(r=[2, 1, 1], phi=[np.pi/3, np.pi/4, np.pi/6])

    program.execute()
    program.state.validate()

    assert np.allclose(program.state.cov, initial_cov)


@pytest.mark.monkey
def test_random_interferometer(generate_random_gaussian_state, generate_unitary_matrix):
    state = generate_random_gaussian_state(d=5)

    T = generate_unitary_matrix(3)

    with pq.Program(state=state) as program:
        pq.Q(0, 1, 3) | pq.Interferometer(T)

    program.execute()
    program.state.validate()


def test_complex_circuit(gaussian_state_assets):
    with pq.Program() as program:
        pq.Q() | pq.GaussianState(d=5)

        pq.Q(all) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

        pq.Q(all) | pq.Squeezing(r=0.1) | pq.Displacement(alpha=1)

        pq.Q(0, 1) | pq.Beamsplitter(0.0959408065906761, 0.06786053071484363)
        pq.Q(2, 3) | pq.Beamsplitter(0.7730047654405018, 1.453770233324797)
        pq.Q(1, 2) | pq.Beamsplitter(1.0152680371119776, 1.2863559998816205)
        pq.Q(3, 4) | pq.Beamsplitter(1.3205517879465705, 0.5236836466492961)
        pq.Q(0, 1) | pq.Beamsplitter(4.394480318177715, 4.481575657714487)
        pq.Q(2, 3) | pq.Beamsplitter(2.2300919706807534, 1.5073556513699888)
        pq.Q(1, 2) | pq.Beamsplitter(2.2679037068773673, 1.9550229282085838)
        pq.Q(3, 4) | pq.Beamsplitter(3.340269832485504, 3.289367083610399)

    program.execute()
    program.state.validate()

    expected_state = gaussian_state_assets.load()
    assert program.state == expected_state


def test_program_stacking_with_measurement():
    with pq.Program() as preparation:
        pq.Q() | pq.GaussianState(d=5)

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

    program.execute()
