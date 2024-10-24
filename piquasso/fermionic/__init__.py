#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

r"""
=================
Fermionic package
=================

Piquasso package for Fermionic quantum computation.

Note:
    This feature is still experimental.

Basic notations
---------------

Let :math:`a_k` and :math:`a_k^\dagger` denote the Dirac operators. These fulfill the
CAR algebra, i.e.,

.. math::
    \{ a_i, a_j^\dagger \} &= I \delta_{ij}, \\\\
    \{a_i, a_j \} &= \{ a_i^\dagger, a_j^\dagger \} = 0.

Let us define :math:`\mathbf{\alpha}` as

.. math::
    \mathbf{\alpha} = [a_1^\dagger, \dots, a_d^\dagger, a_1, \dots, a_d].

The Majorana operators are defined as

.. math::

    x_k &:= \frac{a_k + a_k^\dagger}{\sqrt{2}}, \\\\
    p_k &:= \frac{a_k - a_k^\dagger}{i\sqrt{2}},

where :math:`a_k` and :math:`a_k^\dagger` denote the Dirac operators.

:math:`\mathbf{m}` denotes the vector of Majorana operators, i.e.,

.. math::

    \mathbf{m} := [x_1, \dots, x_d, p_1, \dots, p_d].


.. automodule:: piquasso.fermionic.instructions
   :members:
   :show-inheritance:

.. automodule:: piquasso.fermionic.gaussian
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

"""

from . import gaussian

from .gaussian import GaussianState, GaussianSimulator
from .instructions import GaussianHamiltonian, ParentHamiltonian


__all__ = [
    # Gaussian module
    "gaussian",
    # States
    "GaussianState",
    # Simulators
    "GaussianSimulator",
    # Preparations
    "ParentHamiltonian",
    # Gates
    "GaussianHamiltonian",
]
