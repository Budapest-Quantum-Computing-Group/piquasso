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
Fermionic package
=================

Piquasso package for Fermionic quantum computation.

Note:
    This feature is still experimental.

Basic notations
---------------

Let :math:`f_k` and :math:`f_k^\dagger` denote the Dirac operators. These fulfill the
CAR algebra, i.e.,

.. math::
    \{ f_i, f_j^\dagger \} &= I \delta_{ij}, \\\\
    \{ f_i, f_j \} &= \{ f_i^\dagger, f_j^\dagger \} = 0.

Let us define :math:`\mathbf{f}` as

.. math::
    \mathbf{f} = [f_1^\dagger, \dots, f_d^\dagger, f_1, \dots, f_d].

The Majorana operators are defined as

.. math::

    x_k &:= f_k + f_k^\dagger, \\\\
    p_k &:= -i (f_k - f_k^\dagger),

where :math:`f_k` and :math:`f_k^\dagger` denote the Dirac operators.

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
