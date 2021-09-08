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

import abc
import random
from typing import Tuple, Generator, Any, Mapping, List

from piquasso.instructions.preparations import StateVector
from piquasso._math.fock import FockBasis
from piquasso.api.instruction import Instruction
from piquasso.api.result import Result
from piquasso.api.state import State

from piquasso._math import fock


class BaseFockState(State, abc.ABC):
    _instruction_map = {
        "Interferometer": "_passive_linear",
        "Beamsplitter": "_passive_linear",
        "Phaseshifter": "_passive_linear",
        "MachZehnder": "_passive_linear",
        "Fourier": "_passive_linear",
        "Kerr": "_kerr",
        "CrossKerr": "_cross_kerr",
        "Squeezing": "_linear",
        "QuadraticPhase": "_linear",
        "Displacement": "_linear",
        "PositionDisplacement": "_linear",
        "MomentumDisplacement": "_linear",
        "Squeezing2": "_linear",
        "GaussianTransform": "_linear",
        "ParticleNumberMeasurement": "_particle_number_measurement",
        "Vacuum": "_vacuum",
        "Create": "_create",
        "Annihilate": "_annihilate",
    }

    def __init__(self, *, d: int, cutoff: int) -> None:
        super().__init__()
        self._space = fock.FockSpace(
            d=d,
            cutoff=cutoff,
        )

    @classmethod
    def from_number_preparations(
        cls, *, d: int, cutoff: int, number_preparations: List[StateVector]
    ) -> "BaseFockState":
        """
        NOTE: Here is a small coupling between :class:`Instruction` and :class:`State`.
        This is the only case (so far) where the user could specify instructions
        directly.

        Is this needed?
        """

        self = cls(d=d, cutoff=cutoff)

        for number_preparation in number_preparations:
            self._add_occupation_number_basis(**number_preparation.params)

        return self

    @property
    def d(self) -> int:
        return self._space.d

    @property
    def cutoff(self) -> int:
        return self._space.cutoff

    @property
    def norm(self) -> int:
        return sum(self.fock_probabilities)

    def _particle_number_measurement(
        self, instruction: Instruction
    ) -> None:
        probability_map = self._get_probability_map(
            modes=instruction.modes,
        )

        samples = random.choices(
            population=list(probability_map.keys()),
            weights=list(probability_map.values()),
            k=self.shots,
        )

        # NOTE: We choose the last sample for multiple shots.
        sample = samples[-1]

        normalization = self._get_normalization(probability_map, sample)

        self._project_to_subspace(
            subspace_basis=sample,
            modes=instruction.modes,
            normalization=normalization,
        )

        self.result = Result(
            instruction=instruction,
            samples=samples  # type: ignore
        )

    def _as_code(self) -> str:
        return (
            f"pq.Q() | pq.{self.__class__.__name__}(d={self.d}, cutoff={self.cutoff})"
        )

    @abc.abstractmethod
    def _get_empty(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _vacuum(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def _passive_linear(self, instruction: Instruction) -> None:
        pass

    @abc.abstractmethod
    def _get_probability_map(
        self, *, modes: Tuple[int, ...]
    ) -> Mapping[FockBasis, float]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _get_normalization(
        probability_map: Mapping[FockBasis, float], sample: FockBasis
    ) -> float:
        pass

    @abc.abstractmethod
    def _project_to_subspace(
        self, *, subspace_basis: FockBasis, modes: Tuple[int, ...], normalization: float
    ) -> None:
        pass

    @abc.abstractmethod
    def _create(self, instruction: Instruction) -> None:
        pass

    @abc.abstractmethod
    def _annihilate(self, instruction: Instruction) -> None:
        pass

    @abc.abstractmethod
    def _add_occupation_number_basis(
        self, *, ket: Tuple[int, ...], bra: Tuple[int, ...], coefficient: complex
    ) -> None:
        pass

    @abc.abstractmethod
    def _kerr(self, instruction: Instruction) -> None:
        pass

    @abc.abstractmethod
    def _cross_kerr(self, instruction: Instruction) -> None:
        pass

    @abc.abstractmethod
    def _linear(self, instruction: Instruction) -> None:
        pass

    @property
    @abc.abstractmethod
    def nonzero_elements(
        self
    ) -> Generator[Tuple[complex, tuple], Any, None]:
        pass

    @abc.abstractmethod
    def get_density_matrix(self, cutoff: int) -> np.ndarray:
        """The density matrix of the state in terms of Fock basis vectors."""
        pass

    @property
    def density_matrix(self) -> np.ndarray:
        """The density matrix of the state in terms of Fock basis vectors."""
        return self.get_density_matrix(cutoff=self.cutoff)

    @abc.abstractmethod
    def reduced(self, modes: Tuple[int, ...]) -> "BaseFockState":
        """Reduces the state to a subsystem corresponding to the specified modes."""
        pass

    @abc.abstractmethod
    def get_fock_probabilities(self, cutoff: int) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def fock_probabilities(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def normalize(self) -> None:
        pass

    @abc.abstractmethod
    def validate(self) -> None:
        pass

    def reset(self) -> None:
        """
        Resets this object to a vacuum state.
        """

        self._vacuum()
