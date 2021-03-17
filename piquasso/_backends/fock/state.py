#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import abc

from piquasso.api.state import State


class BaseFockState(State, abc.ABC):
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
