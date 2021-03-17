#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc

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

    @abc.abstractmethod
    def _apply_passive_linear(self, operator, modes):
        pass

    @abc.abstractmethod
    def _measure_particle_number(self, modes):
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
