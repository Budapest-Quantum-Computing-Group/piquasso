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

import numpy as np

from piquasso.api.errors import InvalidState
from piquasso._math.fock import cutoff_cardinality

from ..state import BaseFockState
from ..general.state import FockState

from .circuit import PureFockCircuit


class PureFockState(BaseFockState):
    """A simulated pure Fock state.

    If no mixed states are needed for a Fock simulation, then this state is the most
    appropriate currently.

    Args:
        state_vector (numpy.ndarray, optional): The initial state vector.
        d (int): The number of modes.
        cutoff (int): The Fock space cutoff.
    """

    circuit_class = PureFockCircuit

    def __init__(self, state_vector=None, *, d, cutoff):
        super().__init__(d=d, cutoff=cutoff)

        self._state_vector = (
            np.array(state_vector)
            if state_vector is not None
            else self._get_empty()
        )

    def _get_empty(self):
        return np.zeros(shape=(self._space.cardinality, ), dtype=complex)

    def _apply_vacuum(self):
        self._state_vector = self._get_empty()
        self._state_vector[0] = 1.0

    def _apply_passive_linear(self, operator, modes):
        index = self._get_operator_index(modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        fock_operator = self._space.get_passive_fock_operator(embedded_operator)

        self._state_vector = fock_operator @ self._state_vector

    def _get_probability_map(self, *, modes):
        probability_map = {}

        for index, basis in self._space.basis:
            coefficient = self._state_vector[index]

            subspace_basis = basis.on_modes(modes=modes)

            if subspace_basis in probability_map:
                probability_map[subspace_basis] += coefficient ** 2
            else:
                probability_map[subspace_basis] = coefficient ** 2

        return probability_map

    @staticmethod
    def _get_normalization(probability_map, sample):
        return np.sqrt(1 / probability_map[sample])

    def _project_to_subspace(self, *, subspace_basis, modes, normalization):
        projected_state_vector = self._get_projected_state_vector(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        self._state_vector = projected_state_vector * normalization

    def _get_projected_state_vector(self, *, subspace_basis, modes):
        new_state_vector = self._get_empty()

        index = self._space.get_projection_operator_indices_for_pure(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        new_state_vector[index] = self._state_vector[index]

        return new_state_vector

    def _add_occupation_number_basis(self, coefficient, occupation_numbers, modes=None):
        if modes:
            occupation_numbers = self._space.get_occupied_basis(
                modes=modes, occupation_numbers=occupation_numbers
            )

        index = self._space.index(occupation_numbers)

        self._state_vector[index] = coefficient

    def _apply_creation_operator(self, modes):
        operator = self._space.get_creation_operator(modes)

        self._state_vector = operator @ self._state_vector

        self.normalize()

    def _apply_annihilation_operator(self, modes):
        operator = self._space.get_annihilation_operator(modes)

        self._state_vector = operator @ self._state_vector

        self.normalize()

    def _apply_kerr(self, xi, mode):
        for index, basis in self._space.basis:
            number = basis[mode]
            coefficient = np.exp(
                1j * xi * number * (2 * number + 1)
            )
            self._state_vector[index] *= coefficient

    def _apply_cross_kerr(self, xi, modes):
        for index, basis in self._space.basis:
            coefficient = np.exp(
                1j * xi * basis[modes[0]] * basis[modes[1]]
            )
            self._state_vector[index] *= coefficient

    def _apply_linear(
        self,
        passive_block,
        active_block,
        displacement,
        modes
    ):
        operator = self._space.get_linear_fock_operator(
            modes=modes, auxiliary_modes=self._get_auxiliary_modes(modes),
            passive_block=passive_block,
            active_block=active_block,
            displacement=displacement,
        )

        self._state_vector = operator @ self._state_vector

        self.normalize()

    @property
    def nonzero_elements(self):
        for index, basis in self._space.basis:
            coefficient = self._state_vector[index]
            if coefficient != 0:
                yield coefficient, basis

    def __repr__(self):
        return " + ".join([
            str(coefficient) + str(basis)
            for coefficient, basis in self.nonzero_elements
        ])

    def __eq__(self, other):
        return np.allclose(self._state_vector, other._state_vector)

    def get_density_matrix(self, cutoff):
        cutoff = cutoff or self.cutoff

        cardinality = cutoff_cardinality(d=self.d, cutoff=cutoff)

        state_vector = self._state_vector[:cardinality]

        return np.outer(state_vector, state_vector)

    def _as_mixed(self):
        return FockState.from_fock_state(self)

    def reduced(self, modes):
        return self._as_mixed().reduced(modes)

    def get_particle_detection_probability(self, occupation_number):
        index = self._space.index(occupation_number)

        return np.real(
            self._state_vector[index].conjugate() * self._state_vector[index]
        )

    def get_fock_probabilities(self, cutoff=None):
        cutoff = cutoff or self._space.cutoff

        cardinality = cutoff_cardinality(d=self._space.d, cutoff=cutoff)

        return (self._state_vector * self._state_vector.conjugate()).real[:cardinality]

    @property
    def fock_probabilities(self):
        return self.get_fock_probabilities()

    def normalize(self):
        if np.isclose(self.norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self._state_vector = self._state_vector / np.sqrt(self.norm)

    def validate(self):
        sum_of_probabilities = sum(self.fock_probabilities)

        if not np.isclose(sum_of_probabilities, 1.0):
            raise InvalidState(
                f"The sum of probabilities is {sum_of_probabilities}, "
                "instead of 1.0:\n"
                f"fock_probabilities={self.fock_probabilities}"
            )
