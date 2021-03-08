#
# Copyright (C) 2020 by TODO - All rights reserved.
#


from piquasso.core.mixins import _WeightMixin
from piquasso.core.registry import _register
from piquasso.api.operation import Operation


@_register
class Number(Operation, _WeightMixin):
    r"""State preparation with Fock basis vectors."""

    def __init__(self, *occupation_numbers, coefficient=1.0):
        super().__init__(occupation_numbers, coefficient)


@_register
class DMNumber(Operation, _WeightMixin):
    r"""State preparation with Fock basis vectors."""

    def __init__(self, *, ket, bra, coefficient=1.0):
        super().__init__(ket, bra, coefficient)


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
