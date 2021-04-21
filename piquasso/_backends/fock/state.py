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

import abc
import random

from piquasso.api.state import State

from piquasso._math import fock


class BaseFockState(State, abc.ABC):
    def __init__(self, *, d, cutoff):
        self._space = fock.FockSpace(
            d=d,
            cutoff=cutoff,
        )

    @classmethod
    def from_number_preparations(cls, *, d, cutoff, number_preparations):
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
    def d(self):
        return self._space.d

    @property
    def cutoff(self):
        return self._space.cutoff

    @property
    def norm(self):
        return sum(self.fock_probabilities)

    def _particle_number_measurement(self, modes, shots):
        probability_map = self._get_probability_map(
            modes=modes,
        )

        samples = random.choices(
            population=list(probability_map.keys()),
            weights=probability_map.values(),
            k=shots,
        )

        # NOTE: We choose the last sample for multiple shots.
        sample = samples[-1]

        normalization = self._get_normalization(probability_map, sample)

        self._project_to_subspace(
            subspace_basis=sample,
            modes=modes,
            normalization=normalization,
        )

        return samples

    @abc.abstractclassmethod
    def _get_empty(cls):
        pass

    @abc.abstractmethod
    def _apply_passive_linear(self, operator, modes):
        pass

    @abc.abstractmethod
    def _get_probability_map(*, modes, shots):
        pass

    @abc.abstractmethod
    def _get_normalization(sample):
        pass

    @abc.abstractmethod
    def _project_to_subspace(*, subspace_basis, modes, normalization):
        pass

    @abc.abstractmethod
    def _apply_creation_operator(self, modes):
        pass

    @abc.abstractmethod
    def _apply_annihilation_operator(self, modes):
        pass

    @abc.abstractmethod
    def _apply_kerr(self, xi, mode):
        pass

    @abc.abstractmethod
    def _apply_cross_kerr(self, xi, modes):
        pass

    @abc.abstractmethod
    def _apply_linear(
        self,
        passive_block,
        active_block,
        displacement,
        modes,
    ):
        pass

    @property
    @abc.abstractmethod
    def nonzero_elements(self):
        pass

    @abc.abstractmethod
    def get_fock_probabilities(self, cutoff):
        pass

    @property
    @abc.abstractmethod
    def fock_probabilities(self):
        pass

    @abc.abstractmethod
    def normalize(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass
