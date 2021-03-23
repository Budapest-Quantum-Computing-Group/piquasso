#
# Copyright (C) 2020 by TODO - All rights reserved.
#

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

    @property
    def d(self):
        return self._space.d

    @property
    def cutoff(self):
        return self._space.cutoff

    @property
    def norm(self):
        return sum(self.fock_probabilities)

    def _measure_particle_number(self, modes, shots):
        if not modes:
            modes = tuple(range(self._space.d))

        probability_map = self._get_probability_map(
            modes=modes,
        )

        outcomes = random.choices(
            population=list(probability_map.keys()),
            weights=probability_map.values(),
            k=shots,
        )

        # NOTE: We choose the last outcome for multiple shots.
        outcome = outcomes[-1]

        normalization = self._get_normalization(probability_map, outcome)

        self._project_to_subspace(
            subspace_basis=outcome,
            modes=modes,
            normalization=normalization,
        )

        return outcomes

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
    def _get_normalization(outcome):
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

    @property
    @abc.abstractmethod
    def nonzero_elements(self):
        pass

    @property
    @abc.abstractmethod
    def fock_probabilities(self):
        pass

    @abc.abstractmethod
    def normalize(self):
        pass
