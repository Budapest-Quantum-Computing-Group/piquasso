#
# Copyright (C) 2020 by TODO - All rights reserved.
#


from piquasso.core.mixins import _WeightMixin
from piquasso.core.registry import _register
from piquasso.api.instruction import Instruction


@_register
class Vacuum(Instruction):
    r"""Prepare the system in a vacuum state."""

    def __init__(self):
        pass


@_register
class Mean(Instruction):
    r"""Set the first canonical moment of the state."""

    def __init__(self, mean):
        super().__init__(mean=mean)


@_register
class Covariance(Instruction):
    r"""Sets the covariance matrix of the state."""

    def __init__(self, cov):
        super().__init__(cov=cov)


@_register
class StateVector(Instruction, _WeightMixin):
    r"""State preparation with Fock basis vectors."""

    def __init__(self, *occupation_numbers, coefficient=1.0):
        super().__init__(occupation_numbers=occupation_numbers, coefficient=coefficient)


@_register
class DensityMatrix(Instruction, _WeightMixin):
    r"""State preparation with density matrix elements."""

    def __init__(self, ket=None, bra=None, coefficient=1.0):
        super().__init__(ket=ket, bra=bra, coefficient=coefficient)


@_register
class Create(Instruction):
    r"""Create a particle on a mode."""

    def __init__(self):
        pass


@_register
class Annihilate(Instruction):
    r"""Annihilate a particle on a mode."""

    def __init__(self):
        pass
