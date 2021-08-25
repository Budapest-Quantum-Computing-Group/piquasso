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
from typing import Tuple, Dict, List, Mapping, Generator, Any

import numpy as np
import numpy.typing as npt

from piquasso._math.fock import FockBasis, FockOperatorBasis
from piquasso.api.errors import InvalidState
from piquasso._math.combinatorics import partitions

from ..state import BaseFockState
from ..general.state import FockState

from .circuit import PNCFockCircuit

from scipy.linalg import block_diag


class PNCFockState(BaseFockState):
    """Object to represent a particle number conserving bosonic state in the Fock basis.

    Note:
        If you only work with pure states, it is advised to use
        :class:`~piquasso._backends.fock.pure.state.PureFockState` instead.

    Args:
        representation (numpy.ndarray, optional): The initial representation.
        d (int): The number of modes.
        cutoff (int): The Fock space cutoff.
    """
    circuit_class = PNCFockCircuit

    def __init__(
        self,
        representation: List[npt.NDArray[np.complex128]] = None,
        *,
        d: int,
        cutoff: int
    ) -> None:
        super().__init__(d=d, cutoff=cutoff)

        self._representation = (
            representation
            if representation is not None
            else self._get_empty()
        )

    def _get_empty(self) -> List[npt.NDArray[np.complex128]]:  # type: ignore
        return [
            np.zeros(shape=(self._space._symmetric_cardinality(n), ) * 2, dtype=complex)
            for n in range(self._space.cutoff)
        ]

    def _apply_vacuum(self) -> None:
        self._representation = self._get_empty()
        self._representation[0][0, 0] = 1.0

    def _apply_passive_linear(
        self, operator: npt.NDArray[np.complex128], modes: Tuple[int, ...]
    ) -> None:
        index = self._get_operator_index(modes)

        embedded_operator = np.identity(self._space.d, dtype=complex)

        embedded_operator[index] = operator

        for n, subrep in enumerate(self._representation):
            tensorpower_operator = self._space.symmetric_tensorpower(
                embedded_operator, n
            )
            self._representation[n] = (
                tensorpower_operator @ subrep
                @ tensorpower_operator.conjugate().transpose()
            )

    def _get_probability_map(
        self, *, modes: Tuple[int, ...]
    ) -> Dict[FockBasis, float]:
        probability_map: Dict[FockBasis, float] = {}

        for n, subrep in enumerate(self._representation):
            for index, basis in self._space.subspace_operator_basis_diagonal_on_modes(
                modes=modes, n=n
            ):
                coefficient = float(self._representation[n][index])

                subspace_basis = basis.ket.on_modes(modes=modes)

                if subspace_basis in probability_map:
                    probability_map[subspace_basis] += coefficient
                else:
                    probability_map[subspace_basis] = coefficient

        return probability_map

    @staticmethod
    def _get_normalization(
        probability_map: Mapping[FockBasis, float], sample: FockBasis
    ) -> float:
        return 1 / probability_map[sample]

    def _project_to_subspace(
        self, *, subspace_basis: FockBasis, modes: Tuple[int, ...], normalization: float
    ) -> None:
        projected_representation = self._get_projected(
            subspace_basis=subspace_basis,
            modes=modes,
        )

        for n, subrep in enumerate(projected_representation):
            self._representation[n] = subrep * normalization

    def _get_projected(
        self, *, subspace_basis: FockBasis, modes: Tuple[int, ...]
    ) -> List[npt.NDArray[np.complex128]]:
        new_representation = self._get_empty()

        for n, subrep in enumerate(self._representation):
            index = self._space.get_projection_operator_indices_on_subspace(
                subspace_basis=subspace_basis,
                modes=modes,
                n=n,
            )

            if index:
                new_representation[n][index] = subrep[index]

        return new_representation

    def _hacky_apply_operator(self, operator: npt.NDArray[np.complex128]) -> None:
        """
        HACK: Here we switch to the full representation for a brief moment. I'm sure
        there's a better way.
        """

        density_matrix = block_diag(*self._representation)

        density_matrix = operator @ density_matrix @ operator.conjugate().transpose()

        for n, subrep in enumerate(self._representation):
            begin, end = self._space.get_subspace_indices(n)

            self._representation[n] = density_matrix[begin:end, begin:end]

    def _apply_creation_operator(self, modes: Tuple[int, ...]) -> None:
        operator = self._space.get_creation_operator(modes)

        self._hacky_apply_operator(operator)

        self.normalize()

    def _apply_annihilation_operator(self, modes: Tuple[int, ...]) -> None:
        operator = self._space.get_annihilation_operator(modes)

        self._hacky_apply_operator(operator)

        self.normalize()

    def _add_occupation_number_basis(self, *, ket, bra, coefficient):
        ket_n = sum(ket)
        bra_n = sum(bra)

        if ket_n != bra_n:
            return

        n = ket_n

        subspace_basis = self._space.get_subspace_basis(n)

        index = subspace_basis.index(ket)
        dual_index = subspace_basis.index(bra)

        self._representation[n][index, dual_index] = coefficient

    def _apply_kerr(self, xi: complex, mode: int) -> None:
        for n, subrep in enumerate(self._representation):
            for index, (basis, dual_basis) in (
                self._space.enumerate_subspace_operator_basis(n)
            ):
                number = basis[mode]
                dual_number = dual_basis[mode]

                coefficient = np.exp(
                    1j * xi * (
                        number * (2 * number + 1)
                        - dual_number * (2 * dual_number + 1)
                    )
                )

                self._representation[n][index] *= coefficient

    def _apply_cross_kerr(self, xi: complex, modes: Tuple[int, int]) -> None:
        for n, subrep in enumerate(self._representation):
            for index, (basis, dual_basis) in (
                self._space.enumerate_subspace_operator_basis(n)
            ):
                coefficient = np.exp(
                    1j * xi * (
                        basis[modes[0]] * basis[modes[1]]
                        - dual_basis[modes[0]] * dual_basis[modes[1]]
                    )
                )

                self._representation[n][index] *= coefficient

    def _apply_linear(
        self,
        passive_block: npt.NDArray[np.complex128],
        active_block: npt.NDArray[np.complex128],
        displacement: npt.NDArray[np.complex128],
        modes: Tuple[int, ...]
    ) -> None:
        operator = self._space.get_linear_fock_operator(
            modes=modes,
            auxiliary_modes=self._get_auxiliary_modes(modes),
            passive_block=passive_block,
            active_block=active_block,
            displacement=displacement,
        )

        self._hacky_apply_operator(operator)

        self.normalize()

    @property
    def nonzero_elements(
        self
    ) -> Generator[Tuple[complex, FockOperatorBasis], Any, None]:
        for n, subrep in enumerate(self._representation):
            for index, basis in self._space.enumerate_subspace_operator_basis(n):
                coefficient = self._representation[n][index]
                if coefficient != 0:
                    yield coefficient, basis

    def __repr__(self) -> str:
        return " + ".join(
            [
                str(coefficient) + str(basis)
                for coefficient, basis in self.nonzero_elements
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PNCFockState):
            return False
        return all([
            np.allclose(subrep, other._representation[n])
            for n, subrep in enumerate(self._representation)
        ])

    def get_density_matrix(
        self, cutoff: int = None
    ) -> npt.NDArray[np.complex128]:
        cutoff = cutoff or self.cutoff

        return block_diag(*self._representation[:cutoff])

    def _as_mixed(self) -> FockState:
        return FockState.from_fock_state(self)

    def reduced(self, modes: Tuple[int, ...]) -> FockState:
        return self._as_mixed().reduced(modes)

    def get_particle_detection_probability(
        self, occupation_number: Tuple[int, ...]
    ) -> float:
        number_of_particles = sum(occupation_number)

        subrep_probabilities = np.diag(self._representation[number_of_particles])

        basis = partitions(self.d, number_of_particles)

        index = basis.index(occupation_number)

        return subrep_probabilities[index].real

    def get_fock_probabilities(self, cutoff: int = None) -> npt.NDArray[np.float64]:
        cutoff = cutoff or self.cutoff

        ret = []

        for subrep in self._representation[:cutoff]:
            probability = np.real(np.diag(subrep))
            ret.extend(probability)

        return np.array(ret, dtype=float)

    @property
    def fock_probabilities(self) -> npt.NDArray[np.float64]:
        return self.get_fock_probabilities()

    def normalize(self) -> None:
        if np.isclose(self.norm, 0):
            raise InvalidState("The norm of the state is 0.")

        norm = self.norm

        for n, subrep in enumerate(self._representation):
            self._representation[n] = subrep / norm

    def validate(self) -> None:
        sum_of_probabilities = sum(self.fock_probabilities)

        if not np.isclose(sum_of_probabilities, 1.0):
            raise InvalidState(
                f"The sum of probabilities is {sum_of_probabilities}, "
                "instead of 1.0:\n"
                f"fock_probabilities={self.fock_probabilities}"
            )
