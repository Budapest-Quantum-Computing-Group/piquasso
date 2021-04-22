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


from piquasso.core.mixins import _WeightMixin
from piquasso.api.instruction import Instruction


class Vacuum(Instruction):
    r"""Prepare the system in a vacuum state."""

    def __init__(self):
        pass


class Mean(Instruction):
    r"""Set the first canonical moment of the state."""

    def __init__(self, mean):
        super().__init__(mean=mean)


class Covariance(Instruction):
    r"""Sets the covariance matrix of the state."""

    def __init__(self, cov):
        super().__init__(cov=cov)


class StateVector(Instruction, _WeightMixin):
    r"""State preparation with Fock basis vectors."""

    def __init__(self, *occupation_numbers, coefficient=1.0):
        super().__init__(occupation_numbers=occupation_numbers, coefficient=coefficient)


class DensityMatrix(Instruction, _WeightMixin):
    r"""State preparation with density matrix elements."""

    def __init__(self, ket=None, bra=None, coefficient=1.0):
        super().__init__(ket=ket, bra=bra, coefficient=coefficient)


class Create(Instruction):
    r"""Create a particle on a mode."""

    def __init__(self):
        pass


class Annihilate(Instruction):
    r"""Annihilate a particle on a mode."""

    def __init__(self):
        pass
