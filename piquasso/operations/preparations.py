#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.core.registry import _register
from piquasso.api.operation import Operation


@_register
class Number(Operation):
    r"""State preparation with Fock basis vectors."""

    def __init__(self, *occupation_numbers, coefficient=1.0):
        super().__init__(occupation_numbers, coefficient)

    def __mul__(self, coefficient):
        self.params = (self.params[0], self.params[1] * coefficient)
        return self

    __rmul__ = __mul__


@_register
class DMNumber(Operation):
    r"""State preparation with Fock basis vectors."""

    def __init__(self, *, ket, bra, coefficient=1.0):
        super().__init__(ket, bra, coefficient)

    def __mul__(self, coefficient):
        self.params = (*self.params[:2], self.params[2] * coefficient)
        return self

    __rmul__ = __mul__


@_register
class Create(Operation):
    r"""Create a particle on a mode."""

    def __init__(self):
        pass


@_register
class Annihilate(Operation):
    r"""Annihilate a particle on a mode."""

    def __init__(self):
        pass
