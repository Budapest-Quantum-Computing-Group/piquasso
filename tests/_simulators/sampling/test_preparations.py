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


def test_initial_state_raises_InvalidState_for_nonnormalized_input_state():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0]) * 0.5

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        state.validate()

    assert error.value.args[0] == "The state is not normalized: norm=0.25"


def test_initial_state_for_nonnormalized_input_state_validate_False():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0]) * 0.5

    simulator = pq.SamplingSimulator(d=5, config=pq.Config(validate=False))
    state = simulator.execute(program).state

    state.validate()


def test_initial_state_raises_InvalidState_for_occupation_numbers_of_differing_length():
    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0]) * 1 / np.sqrt(2)
        pq.Q() | pq.StateVector([1, 1, 1]) * 1 / np.sqrt(2)

    simulator = pq.SamplingSimulator(d=5)

    with pytest.raises(pq.api.exceptions.InvalidState) as error:
        simulator.execute(program).state

    assert error.value.args[0] == (
        "The occupation numbers '(1, 1, 1)' are not well-defined on '5' modes: "
        "instruction=StateVector(occupation_numbers=(1, 1, 1), "
        "coefficient=0.7071067811865475, modes=(0, 1, 2, 3, 4))"
    )


def test_interferometer_init():
    with pq.Program() as program:
        pass

    simulator = pq.SamplingSimulator(d=5)

    state = simulator.execute(program).state

    expected_interferometer = np.diag(np.ones(state.d, dtype=complex))
    assert np.allclose(state.interferometer, expected_interferometer)


def test_multiple_interferometer_on_neighbouring_modes():
    U = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0, 1, 2) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_multiple_interferometer_on_gaped_modes():
    U = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(0, 1, 4) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 2, 0, 0, 3],
            [4, 5, 0, 0, 6],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [7, 8, 0, 0, 9],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_multiple_interferometer_on_reversed_gaped_modes():
    U = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=complex,
    )

    with pq.Program() as program:
        pq.Q() | pq.StateVector([1, 1, 1, 0, 0])

        pq.Q(4, 3, 1) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=5)
    state = simulator.execute(program).state

    expected_interferometer = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 9, 0, 8, 7],
            [0, 0, 1, 0, 0],
            [0, 6, 0, 5, 4],
            [0, 3, 0, 2, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(state.interferometer, expected_interferometer)


def test_probability_distribution():
    U = np.array(
        [
            [1, 0, 0],
            [0, -0.54687158 + 0.07993182j, 0.32028583 - 0.76938896j],
            [0, 0.78696803 + 0.27426941j, 0.42419041 - 0.35428818j],
        ],
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=3, config=pq.Config(cutoff=3))
    state = simulator.execute(program).state

    assert np.allclose(
        state.fock_probabilities,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.30545762, 0.69454239, 0.0, 0.0, 0.0],
    )


def test_get_particle_detection_probability():
    U = np.array(
        [
            [1, 0, 0],
            [0, -0.54687158 + 0.07993182j, 0.32028583 - 0.76938896j],
            [0, 0.78696803 + 0.27426941j, 0.42419041 - 0.35428818j],
        ],
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=3)
    state = simulator.execute(program).state

    probability = state.get_particle_detection_probability(occupation_number=(1, 1, 0))

    assert np.allclose(probability, 0.30545762086020883)


def test_get_particle_detection_probability_complex_scenario():
    U = np.array(
        [
            [
                -0.25022099 + 0.32110177j,
                -0.69426529 - 0.49960543j,
                -0.28233272 + 0.15153042j,
            ],
            [
                -0.69028768 + 0.23351228j,
                0.14839865 + 0.49185272j,
                -0.43153658 - 0.13714903j,
            ],
            [
                0.41073351 + 0.36681879j,
                -0.06655274 + 0.00442722j,
                -0.22146243 - 0.80202711j,
            ],
        ],
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 2])
        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=3)
    state = simulator.execute(program).state

    probability = state.get_particle_detection_probability(occupation_number=(2, 1, 0))

    assert np.allclose(probability, 0.038483056956364094)


def test_get_particle_detection_probability_on_different_subspace():
    U = np.array(
        [
            [1, 0, 0],
            [0, -0.54687158 + 0.07993182j, 0.32028583 - 0.76938896j],
            [0, 0.78696803 + 0.27426941j, 0.42419041 - 0.35428818j],
        ],
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([1, 1, 0])
        pq.Q(all) | pq.Interferometer(U)

    simulator = pq.SamplingSimulator(d=3)
    state = simulator.execute(program).state

    different_particle_subspace_occupation_number = (3, 1, 0)

    probability = state.get_particle_detection_probability(
        occupation_number=different_particle_subspace_occupation_number
    )

    assert np.allclose(probability, 0.0)


def test_multiple_StateVector_instructions_state_vector():
    initial_occupation_numbers = np.array(
        [
            [1, 1, 0, 1],
            [2, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])
    d = initial_occupation_numbers.shape[1]

    with pq.Program() as program:
        for idx in range(len(coefficients)):
            pq.Q() | pq.StateVector(
                initial_occupation_numbers[idx], coefficient=coefficients[idx]
            )

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter5050()

        pq.Q(0, 3) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 7)

    config = pq.Config(cutoff=4)

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(
        sampling_state.state_vector,
        [
            0.63245553,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.15523032 + 0.15523032j,
            -0.07833273 - 0.13567627j,
            -0.07833273 - 0.13567627j,
            -0.21290502 + 0.07449869j,
            -0.16199069 + 0.04340527j,
            -0.22908942 + 0.06138433j,
            -0.00850609 - 0.11350585j,
            -0.16199069 + 0.04340527j,
            -0.00850609 - 0.11350585j,
            -0.11515274 - 0.0129746j,
            -0.02450333 - 0.0894709j,
            0.03791189 - 0.08560374j,
            0.03791189 - 0.08560374j,
            0.00074147 - 0.00404062j,
            0.14143668 - 0.02975653j,
            0.20002168 - 0.04208209j,
            -0.03276254 + 0.02788141j,
            0.14143668 - 0.02975653j,
            -0.03276254 + 0.02788141j,
            -0.04646917 + 0.06369578j,
            0.0,
            0.0,
            -0.19316271 - 0.04756413j,
            0.0,
            -0.27317333 - 0.06726584j,
            -0.09056537 + 0.02373235j,
            0.0,
            -0.19316271 - 0.04756413j,
            -0.09056537 + 0.02373235j,
            -0.05941314 + 0.0318212j,
        ],
    )


def test_multiple_StateVector_instructions_get_particle_detection_probability():
    initial_occupation_numbers = np.array(
        [
            [1, 1, 0, 1],
            [2, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])
    d = initial_occupation_numbers.shape[1]

    with pq.Program() as program:
        for idx in range(len(coefficients)):
            pq.Q() | pq.StateVector(
                initial_occupation_numbers[idx], coefficient=coefficients[idx]
            )

        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 4)

        pq.Q(1) | pq.Phaseshifter(np.pi / 3)

        pq.Q(1, 2) | pq.Beamsplitter5050()

        pq.Q(0, 3) | pq.Beamsplitter(theta=np.pi / 5, phi=np.pi / 7)

    config = pq.Config(cutoff=4)

    sampling_simulator = pq.SamplingSimulator(d=d, config=config)
    sampling_state = sampling_simulator.execute(program).state
    sampling_state.validate()

    assert np.allclose(
        sampling_state.get_particle_detection_probability((1, 1, 0, 1)),
        0.0018507569323483858,
    )
