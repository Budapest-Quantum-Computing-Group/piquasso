#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

import numpy as np
import pytest

import piquasso as pq

from scipy.special import comb


def test_state_vector(generate_unitary_matrix):
    input_state = np.array([2, 1, 3, 1, 1], dtype=int)

    unitary = generate_unitary_matrix(5)

    program = pq.Program(
        instructions=[
            pq.NumberState(input_state),
            pq.Interferometer(unitary),
        ]
    )

    simulator = pq.SamplingSimulator(d=len(unitary), config=pq.Config(cutoff=9))

    result = simulator.execute(program)

    state = result.state

    state.validate()

    state_vector = state.state_vector

    assert np.isclose(np.sum(np.abs(state_vector) ** 2), 1.0)


def test_validate_not_normalized():
    input_state = [1, 1]

    program = pq.Program(
        instructions=[
            pq.NumberState(input_state),
            pq.Beamsplitter5050().on_modes(0, 1),
            pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
        ]
    )

    simulator = pq.SamplingSimulator(config=pq.Config(cutoff=3))

    result = simulator.execute(program)

    with pytest.raises(
        pq.api.exceptions.InvalidState, match="The state is not normalized."
    ):
        result.state.validate()


class TestGetParticleDetectionProbability:
    def test_get_particle_detection_probability_simple(self):
        input_state = np.array([1, 1], dtype=int)

        unitary = np.array([[1, 0], [0, 1]], dtype=complex)

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Interferometer(unitary),
            ]
        )

        simulator = pq.SamplingSimulator(d=len(unitary), config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        occupation_number = np.array([1, 1], dtype=int)

        probability = state.get_particle_detection_probability(occupation_number)

        assert np.isclose(probability, 1.0)

    def test_get_particle_detection_probability_beamsplitter(self):
        input_state = np.array([1, 1], dtype=int)

        theta = np.pi / 5

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta).on_modes(0, 1),
            ]
        )

        simulator = pq.SamplingSimulator(d=len(input_state), config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        occupation_number = np.array([2, 0], dtype=int)

        probability = state.get_particle_detection_probability(occupation_number)

        assert np.isclose(probability, np.sin(2 * theta) ** 2 / 2)

    def test_get_particle_detection_probability_uniform_lossy(self):
        input_state = np.array([1, 1], dtype=int)

        theta = np.pi / 5

        transmissivity = 0.5

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta).on_modes(0, 1),
                pq.UniformLoss(transmissivity=transmissivity).on_modes(0, 1),
            ]
        )

        simulator = pq.SamplingSimulator(d=len(input_state), config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        occupation_number = np.array([2, 0], dtype=int)

        probability = state.get_particle_detection_probability(occupation_number)

        expected_probability = transmissivity**4 * np.sin(2 * theta) ** 2 / 2

        assert np.isclose(probability, expected_probability)

    def test_get_particle_detection_probability_uniform_lossy_three_modes(self):
        input_state = np.array([1, 1, 1], dtype=int)

        theta_01 = np.pi / 5
        theta_12 = np.pi / 7
        transmissivity = 0.6

        lossless_program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta_01).on_modes(0, 1),
                pq.Beamsplitter(theta=theta_12).on_modes(1, 2),
            ]
        )

        lossy_program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta_01).on_modes(0, 1),
                pq.Beamsplitter(theta=theta_12).on_modes(1, 2),
                pq.UniformLoss(transmissivity=transmissivity).on_modes(0, 1, 2),
            ]
        )

        simulator = pq.SamplingSimulator(
            d=len(input_state),
            config=pq.Config(cutoff=4),
        )

        lossless_state = simulator.execute(lossless_program).state
        lossy_state = simulator.execute(lossy_program).state

        detected_occupation = np.array([1, 1, 0], dtype=int)

        probability = lossy_state.get_particle_detection_probability(
            detected_occupation
        )

        possible_no_loss_outputs = [
            np.array([2, 1, 0], dtype=int),
            np.array([1, 2, 0], dtype=int),
            np.array([1, 1, 1], dtype=int),
        ]

        expected_probability = 0.0

        def _uniform_loss_transition_probability(
            no_loss_occupation_number,
            detected_occupation_number,
            transmissivity,
        ):
            probability = 1.0

            survival_probability = transmissivity**2
            loss_probability = 1.0 - survival_probability

            for no_loss, detected in zip(
                no_loss_occupation_number,
                detected_occupation_number,
            ):
                no_loss = int(no_loss)
                detected = int(detected)

                if detected > no_loss:
                    return 0.0

                lost = no_loss - detected

                probability *= (
                    comb(no_loss, detected)
                    * survival_probability**detected
                    * loss_probability**lost
                )

            return probability

        for no_loss_output in possible_no_loss_outputs:
            loss_probability = _uniform_loss_transition_probability(
                no_loss_occupation_number=no_loss_output,
                detected_occupation_number=detected_occupation,
                transmissivity=transmissivity,
            )

            ideal_probability = lossless_state.get_particle_detection_probability(
                no_loss_output
            )

            expected_probability += loss_probability * ideal_probability

        assert np.isclose(probability, expected_probability)

    def test_postselection_simple(self):
        input_state = np.array([1, 1], dtype=int)

        theta = np.pi / 5

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta).on_modes(0, 1),
                pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
            ]
        )

        simulator = pq.SamplingSimulator(d=len(input_state), config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        occupation_number = np.array([1], dtype=int)

        probability = state.get_particle_detection_probability(occupation_number)

        assert np.isclose(probability, np.cos(2 * theta) ** 2)

    def test_postselection_with_uniform_loss(self):
        input_state = np.array([1, 1], dtype=int)

        theta = np.pi / 5
        transmissivity = 0.6

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta).on_modes(0, 1),
                pq.UniformLoss(transmissivity=transmissivity).on_modes(0, 1),
                pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
            ]
        )

        simulator = pq.SamplingSimulator(
            d=len(input_state),
            config=pq.Config(cutoff=3),
        )

        result = simulator.execute(program)

        state = result.state

        occupation_number = np.array([0], dtype=int)

        probability = state.get_particle_detection_probability(occupation_number)

        survival_probability = transmissivity**2

        expected_probability = survival_probability * (1.0 - survival_probability)

        assert np.isclose(probability, expected_probability)

    def test_postselection_with_nonuniform_loss(self):
        input_state = np.array([1, 1], dtype=int)

        theta = np.pi / 5

        transmissivity_0 = 0.6
        transmissivity_1 = 0.8

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta).on_modes(0, 1),
                pq.UniformLoss(transmissivity=transmissivity_0).on_modes(0),
                pq.UniformLoss(transmissivity=transmissivity_1).on_modes(1),
                pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
            ]
        )

        simulator = pq.SamplingSimulator(
            d=len(input_state),
            config=pq.Config(cutoff=3),
        )

        result = simulator.execute(program)

        state = result.state

        occupation_number = np.array([0], dtype=int)

        probability = state.get_particle_detection_probability(occupation_number)

        survival_probability_0 = transmissivity_0**2
        survival_probability_1 = transmissivity_1**2

        expected_probability = (
            survival_probability_0
            * (1.0 - survival_probability_1)
            * np.cos(2 * theta) ** 2
            + survival_probability_0
            * (1.0 - survival_probability_0)
            * np.sin(2 * theta) ** 2
        )

        assert np.isclose(probability, expected_probability)

    def test_postselection_with_nonuniform_loss_multiple_input_number_states(self):
        theta = np.pi / 5

        transmissivity_0 = 0.6
        transmissivity_1 = 0.8

        coefficient_11 = np.sqrt(0.3)
        coefficient_20 = np.sqrt(0.5)

        program = pq.Program(
            instructions=[
                coefficient_11 * pq.NumberState([1, 1]),
                coefficient_20 * pq.NumberState([2, 0]),
                pq.Beamsplitter(theta=theta).on_modes(0, 1),
                pq.UniformLoss(transmissivity=transmissivity_0).on_modes(0),
                pq.UniformLoss(transmissivity=transmissivity_1).on_modes(1),
                pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
            ]
        )

        simulator = pq.SamplingSimulator(
            d=2,
            config=pq.Config(cutoff=3),
        )

        result = simulator.execute(program)

        state = result.state

        occupation_number = np.array([0], dtype=int)

        probability = state.get_particle_detection_probability(occupation_number)

        survival_probability_0 = transmissivity_0**2
        survival_probability_1 = transmissivity_1**2

        output_amplitude_11 = coefficient_11 * np.cos(
            2 * theta
        ) + coefficient_20 * np.sin(2 * theta) / np.sqrt(2)

        output_amplitude_20 = (
            -coefficient_11 * np.sin(2 * theta) / np.sqrt(2)
            + coefficient_20 * np.cos(theta) ** 2
        )

        expected_probability = (
            survival_probability_0
            * (1.0 - survival_probability_1)
            * np.abs(output_amplitude_11) ** 2
            + 2
            * survival_probability_0
            * (1.0 - survival_probability_0)
            * np.abs(output_amplitude_20) ** 2
        )

        assert np.isclose(probability, expected_probability)


class TestMarginalProbabilities:
    def test_marginal_probabilities_simple(self):
        input_state = np.array([1, 1], dtype=int)

        unitary = np.array([[1, 0], [0, 1]], dtype=complex)

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Interferometer(unitary),
            ]
        )

        simulator = pq.SamplingSimulator(d=len(unitary), config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        modes = [0]

        probabilities = state.get_marginal_probabilities(modes)

        expected = {(0,): 0.0, (1,): 1.0, (2,): 0.0}
        assert set(probabilities) == set(expected)

        for outcome, expected_value in expected.items():
            assert np.isclose(probabilities[outcome], expected_value)

    def test_marginal_probabilities_simple_beamsplitter(self):
        input_state = np.array([1, 1], dtype=int)

        theta = np.pi / 5

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=theta).on_modes(0, 1),
            ]
        )

        simulator = pq.SamplingSimulator(d=2, config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        modes = [0]

        probabilities = state.get_marginal_probabilities(modes)

        expected_probabilities = {
            (0,): np.sin(2 * theta) ** 2 / 2,
            (1,): np.cos(2 * theta) ** 2,
            (2,): np.sin(2 * theta) ** 2 / 2,
        }

        for outcome, expected in expected_probabilities.items():
            assert np.isclose(probabilities[outcome], expected)

    def test_marginal_probabilities_3_modes(self):
        input_state = np.array([1, 1, 0], dtype=int)

        alpha = np.pi / 5
        beta = np.pi / 4

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=alpha).on_modes(0, 1),
                pq.Beamsplitter(theta=beta).on_modes(1, 2),
            ]
        )

        simulator = pq.SamplingSimulator(d=3, config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        modes = [1]

        probabilities = state.get_marginal_probabilities(modes)

        sin_2alpha_sq = np.sin(2 * alpha) ** 2
        cos_2alpha_sq = np.cos(2 * alpha) ** 2

        sin_beta_sq = np.sin(beta) ** 2
        cos_beta_sq = np.cos(beta) ** 2

        p0 = (
            0.5 * sin_2alpha_sq
            + sin_beta_sq * cos_2alpha_sq
            + 0.5 * sin_2alpha_sq * sin_beta_sq**2
        )

        p1 = cos_beta_sq * cos_2alpha_sq + sin_2alpha_sq * cos_beta_sq * sin_beta_sq

        p2 = 0.5 * sin_2alpha_sq * cos_beta_sq**2

        for outcome, expected in {(0,): p0, (1,): p1, (2,): p2}.items():
            assert np.isclose(probabilities[outcome], expected)

    def test_marginal_probabilities_4_modes_selected_2_modes(self):
        input_state = np.array([1, 1, 0, 0], dtype=int)

        alpha = np.pi / 5
        beta = np.pi / 4
        gamma = np.pi / 7

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=alpha).on_modes(0, 1),
                pq.Beamsplitter(theta=beta).on_modes(1, 2),
                pq.Beamsplitter(theta=gamma).on_modes(2, 3),
            ]
        )

        simulator = pq.SamplingSimulator(d=4, config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        modes = [1, 3]

        probabilities = state.get_marginal_probabilities(modes)

        S = np.sin(2 * alpha) ** 2
        C = np.cos(2 * alpha) ** 2

        sin_beta_sq = np.sin(beta) ** 2
        cos_beta_sq = np.cos(beta) ** 2

        sin_gamma_sq = np.sin(gamma) ** 2
        cos_gamma_sq = np.cos(gamma) ** 2

        expected = np.zeros((3, 3))

        expected[0, 0] = (
            0.5 * S
            + sin_beta_sq * cos_gamma_sq * C
            + 0.5 * S * sin_beta_sq**2 * cos_gamma_sq**2
        )

        expected[0, 1] = (
            sin_beta_sq * sin_gamma_sq * C
            + S * sin_beta_sq**2 * cos_gamma_sq * sin_gamma_sq
        )

        expected[0, 2] = 0.5 * S * sin_beta_sq**2 * sin_gamma_sq**2

        expected[1, 0] = cos_beta_sq * C + S * cos_beta_sq * sin_beta_sq * cos_gamma_sq

        expected[1, 1] = S * cos_beta_sq * sin_beta_sq * sin_gamma_sq

        expected[2, 0] = 0.5 * S * cos_beta_sq**2

        for outcome, probability in probabilities.items():
            assert np.isclose(probability, expected[outcome])

    def test_marginal_probabilities_4_modes_postselected(self):
        input_state = np.array([1, 1, 0, 0], dtype=int)

        alpha = np.pi / 5
        beta = np.pi / 4
        gamma = np.pi / 7

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=alpha).on_modes(0, 1),
                pq.Beamsplitter(theta=beta).on_modes(1, 2),
                pq.Beamsplitter(theta=gamma).on_modes(2, 3),
                pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
            ]
        )

        simulator = pq.SamplingSimulator(d=4, config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        modes = [1]

        probabilities = state.get_marginal_probabilities(modes)

        postselection_probability = np.cos(2 * alpha) ** 2

        expected = {
            (0,): postselection_probability * np.sin(beta) ** 2,
            (1,): postselection_probability * np.cos(beta) ** 2,
        }

        for outcome, probability in probabilities.items():
            assert np.isclose(probability, expected[outcome])

    @pytest.mark.monkey
    def test_marginal_probabilities_sum_to_1_for_random(self, generate_unitary_matrix):
        input_state = np.array([2, 1, 3, 1, 1], dtype=int)

        unitary = generate_unitary_matrix(5)

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Interferometer(unitary),
            ]
        )

        simulator = pq.SamplingSimulator(d=len(unitary), config=pq.Config(cutoff=9))

        result = simulator.execute(program)

        state = result.state

        modes = [0, 1, 2]

        probabilities = state.get_marginal_probabilities(modes)

        assert np.isclose(np.sum(list(probabilities.values())), 1.0)

    def test_uniform_lossy_state_marginal_probabilities_trivial(self):
        input_state = np.array([1, 0], dtype=int)

        transmissivity = 0.5
        survival_probability = transmissivity**2

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.UniformLoss(transmissivity=transmissivity).on_modes(0, 1),
            ]
        )

        simulator = pq.SamplingSimulator(d=2, config=pq.Config(cutoff=2))

        result = simulator.execute(program)

        probabilities = result.state.get_marginal_probabilities(modes=[0])

        expected = {
            (0,): 1.0 - survival_probability,
            (1,): survival_probability,
        }

        assert set(probabilities) == set(expected)

        for outcome, expected_probability in expected.items():
            assert np.isclose(probabilities[outcome], expected_probability)

        assert np.isclose(sum(probabilities.values()), 1.0)

    def test_uniform_lossy_state_marginal_probabilities_trivial_multiphoton(self):
        input_state = np.array([2, 1, 0], dtype=int)

        transmissivity = 0.5
        survival_probability = transmissivity**2

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.UniformLoss(transmissivity=transmissivity).on_modes(0, 1, 2),
            ]
        )

        simulator = pq.SamplingSimulator(d=3, config=pq.Config(cutoff=4))

        result = simulator.execute(program)

        probabilities = result.state.get_marginal_probabilities(modes=[0, 1])

        expected = {
            (0, 0): (1.0 - survival_probability) ** 3,
            (0, 1): (1.0 - survival_probability) ** 2 * survival_probability,
            (1, 0): 2 * survival_probability * (1.0 - survival_probability) ** 2,
            (1, 1): 2 * survival_probability**2 * (1.0 - survival_probability),
            (2, 0): survival_probability**2 * (1.0 - survival_probability),
            (2, 1): survival_probability**3,
        }

        for outcome, expected_probability in expected.items():
            assert np.isclose(probabilities[outcome], expected_probability)

        assert np.isclose(sum(probabilities.values()), 1.0)

    def test_uniform_lossy_state_marginal_probabilities_with_beamsplitter(self):
        input_state = np.array([1, 1], dtype=int)

        transmissivity = 0.5
        survival_probability = transmissivity**2

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter(theta=np.pi / 4, phi=0.0).on_modes(0, 1),
                pq.UniformLoss(transmissivity=transmissivity).on_modes(0, 1),
            ]
        )

        simulator = pq.SamplingSimulator(d=2, config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        probabilities = result.state.get_marginal_probabilities(modes=[0])

        expected = {
            (0,): 0.5 + 0.5 * (1.0 - survival_probability) ** 2,
            (1,): 0.5 * 2.0 * survival_probability * (1.0 - survival_probability),
            (2,): 0.5 * survival_probability**2,
        }

        for outcome, expected_probability in expected.items():
            assert np.isclose(probabilities[outcome], expected_probability)

        assert np.isclose(sum(probabilities.values()), 1.0)

    def test_non_uniformly_lossy_state_raises_NotImplementedCalculation(
        self,
    ):
        input_state = np.array([1, 1], dtype=int)

        lossy_interferometer = np.diag([0.5, 0.25])

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.LossyInterferometer(lossy_interferometer),
            ]
        )

        simulator = pq.SamplingSimulator(d=2, config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        with pytest.raises(
            pq.api.exceptions.NotImplementedCalculation,
            match=(
                "Marginal probability calculation is not implemented for non-uniformly "
                "lossy states."
            ),
        ):
            _ = result.state.get_marginal_probabilities(modes=[0])

    def test_multiple_occupation_numbers_raises_NotImplementedCalculation(self):
        unitary = np.array([[1, 0], [0, 1]], dtype=complex)

        program = pq.Program(
            instructions=[
                (
                    pq.NumberState([0, 2]) * np.sqrt(0.5)
                    + pq.NumberState([2, 0]) * np.sqrt(0.5)
                ),
                pq.Interferometer(unitary),
            ]
        )

        simulator = pq.SamplingSimulator(d=len(unitary), config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        modes = [0]

        with pytest.raises(
            pq.api.exceptions.NotImplementedCalculation,
            match=(
                "Marginal probability calculation is not implemented for states with "
                "multiple input occupation numbers."
            ),
        ):
            _ = state.get_marginal_probabilities(modes)

    def test_postselecting_on_same_mode_raises_PiquassoException(self):
        input_state = np.array([1, 1, 0], dtype=int)

        program = pq.Program(
            instructions=[
                pq.NumberState(input_state),
                pq.Beamsplitter5050().on_modes(0, 1),
                pq.PostSelectPhotons(photon_counts=[1]).on_modes(0),
            ]
        )

        simulator = pq.SamplingSimulator(d=3, config=pq.Config(cutoff=3))

        result = simulator.execute(program)

        state = result.state

        modes = [0]

        with pytest.raises(
            pq.api.exceptions.PiquassoException,
            match=(
                "Marginal probabilities cannot be calculated for postselected modes."
            ),
        ):
            _ = state.get_marginal_probabilities(modes)
