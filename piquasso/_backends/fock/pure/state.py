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

from typing import Tuple, Dict, Mapping, Generator, Any

import numpy as np

from piquasso.api.config import Config
from piquasso.api.instruction import Instruction
from piquasso.api.errors import InvalidState
from piquasso._math.fock import cutoff_cardinality, FockBasis

from ..state import BaseFockState
from ..general.state import FockState


class PureFockState(BaseFockState):
    r"""A simulated pure Fock state.

    If no mixed states are needed for a Fock simulation, then this state is the most
    appropriate currently, since it does not store an entire density matrix, only a
    state vector with size

    .. math::
        {d + c - 1 \choose c - 1},

    where :math:`c \in \mathbb{N}` is the Fock space cutoff.

    Args:
        state_vector (numpy.ndarray, optional): The initial state vector.
        d (int): The number of modes.
        cutoff (int): The Fock space cutoff.
    """

    _instruction_map = {
        "StateVector": "_state_vector_instruction",
        **BaseFockState._instruction_map
    }

    def __init__(self, *, d: int, config: Config = None) -> None:
        super().__init__(d=d, config=config)

        self._state_vector = self._get_empty()

    def _get_empty(self) -> np.ndarray:
        return np.zeros(shape=(self._space.cardinality, ), dtype=complex)

    def _vacuum(self, *_args: Any, **_kwargs: Any) -> None:
        self._state_vector = self._get_empty()
        self._state_vector[0] = 1.0

    def _passive_linear(self, instruction: Instruction) -> None:
        operator: np.ndarray = instruction._all_params["passive_block"]

        index = self._get_operator_index(instruction.modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        fock_operator = self._space.get_passive_fock_operator(embedded_operator)

        self._state_vector = fock_operator @ self._state_vector

    def _get_probability_map(
        self, *, modes: Tuple[int, ...]
    ) -> Dict[FockBasis, float]:
        probability_map: Dict[FockBasis, float] = {}

        for index, basis in self._space.basis:
            coefficient = float(self._state_vector[index])

            subspace_basis = basis.on_modes(modes=modes)

            if subspace_basis in probability_map:
                probability_map[subspace_basis] += coefficient ** 2
            else:
                probability_map[subspace_basis] = coefficient ** 2

        return probability_map

    @staticmethod
    def _get_normalization(
        probability_map: Mapping[FockBasis, float], sample: FockBasis
    ) -> float:
        return np.sqrt(1 / probability_map[sample])

    def _project_to_subspace(
        self, *, subspace_basis: FockBasis, modes: Tuple[int, ...], normalization: float
    ) -> None:
        projected_state_vector = self._get_projected_state_vector(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        self._state_vector = projected_state_vector * normalization

    def _get_projected_state_vector(
        self, *, subspace_basis: FockBasis, modes: Tuple[int, ...]
    ) -> np.ndarray:
        new_state_vector = self._get_empty()

        index = self._space.get_projection_operator_indices_for_pure(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        new_state_vector[index] = self._state_vector[index]

        return new_state_vector

    def _add_occupation_number_basis(  # type: ignore
        self,
        coefficient: complex,
        occupation_numbers: Tuple[int, ...],
        modes: Tuple[int, ...] = None
    ) -> None:
        if modes:
            occupation_numbers = self._space.get_occupied_basis(
                modes=modes, occupation_numbers=occupation_numbers
            )

        index = self._space.index(occupation_numbers)

        self._state_vector[index] = coefficient

    def _create(self, instruction: Instruction) -> None:
        operator = self._space.get_creation_operator(instruction.modes)

        self._state_vector = operator @ self._state_vector

        self.normalize()

    def _annihilate(self, instruction: Instruction) -> None:
        operator = self._space.get_annihilation_operator(instruction.modes)

        self._state_vector = operator @ self._state_vector

        self.normalize()

    def _kerr(self, instruction: Instruction) -> None:
        mode = instruction.modes[0]
        xi = instruction._all_params["xi"]

        for index, basis in self._space.basis:
            number = basis[mode]
            coefficient = np.exp(
                1j * xi * number * (2 * number + 1)
            )
            self._state_vector[index] *= coefficient

    def _cross_kerr(self, instruction: Instruction) -> None:
        modes = instruction.modes
        xi = instruction._all_params["xi"]

        for index, basis in self._space.basis:
            coefficient = np.exp(
                1j * xi * basis[modes[0]] * basis[modes[1]]
            )
            self._state_vector[index] *= coefficient

    def _linear(self, instruction: Instruction) -> None:
        operator = self._space.get_linear_fock_operator(
            modes=instruction.modes,
            cache_size=self._config.cache_size,
            auxiliary_modes=self._get_auxiliary_modes(instruction.modes),
            passive_block=instruction._all_params["passive_block"],
            active_block=instruction._all_params["active_block"],
            displacement=instruction._all_params["displacement_vector"],
        )

        self._state_vector = operator @ self._state_vector

        self.normalize()

    def _state_vector_instruction(self, instruction: Instruction) -> None:
        self._add_occupation_number_basis(
            **instruction._all_params,
            modes=instruction.modes,
        )

    @property
    def nonzero_elements(
        self
    ) -> Generator[Tuple[complex, FockBasis], Any, None]:
        for index, basis in self._space.basis:
            coefficient: complex = self._state_vector[index]
            if coefficient != 0:
                yield coefficient, basis

    def __repr__(self) -> str:
        return " + ".join([
            str(coefficient) + str(basis)
            for coefficient, basis in self.nonzero_elements
        ])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PureFockState):
            return False
        return np.allclose(self._state_vector, other._state_vector)

    @property
    def density_matrix(self) -> np.ndarray:
        cardinality = cutoff_cardinality(d=self.d, cutoff=self._config.cutoff)

        state_vector = self._state_vector[:cardinality]

        return np.outer(state_vector, state_vector)

    def _as_mixed(self) -> FockState:
        return FockState.from_fock_state(self)

    def reduced(self, modes: Tuple[int, ...]) -> FockState:
        return self._as_mixed().reduced(modes)

    def get_particle_detection_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        index = self._space.index(occupation_number)

        return np.real(
            self._state_vector[index].conjugate() * self._state_vector[index]
        )

    @property
    def fock_probabilities(self) -> np.ndarray:
        cardinality = cutoff_cardinality(d=self._space.d, cutoff=self._config.cutoff)

        return (self._state_vector * self._state_vector.conjugate()).real[:cardinality]

    def normalize(self) -> None:
        if np.isclose(self.norm, 0):
            raise InvalidState("The norm of the state is 0.")

        self._state_vector = self._state_vector / np.sqrt(self.norm)

    def validate(self) -> None:
        sum_of_probabilities = sum(self.fock_probabilities)

        if not np.isclose(sum_of_probabilities, 1.0):
            raise InvalidState(
                f"The sum of probabilities is {sum_of_probabilities}, "
                "instead of 1.0:\n"
                f"fock_probabilities={self.fock_probabilities}"
            )
